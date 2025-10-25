"""Diffusion-based Safe Score Matching agent that integrates the safety framework."""

from typing import Optional

import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.ssm.model import QNetwork, DiffusionScoreModel, SafetyQNetwork
from agents.ssm.utils import (
    soft_gate,
    soft_update,
    cosine_beta_schedule,
    vp_beta_schedule,
    ddpm_sampler,
)


class SSMAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.tau = args.tau
        self.T = args.T
        self.M_q = args.M_q
        self.M_safe = args.M_safe
        self.ddpm_temperature = args.ddpm_temperature
        self.beta_schedule = args.beta_schedule
        self.value_samples = args.safety_value_samples
        self.kappa = args.safe_gate_kappa
        self.g_alpha = args.safe_gate_alpha
        self.alpha_coef = args.safety_alpha_coef
        self.temporal_weight = args.safety_temporal_weight
        self.semantic_weight = args.safety_semantic_weight
        self.terminal_safe_value = args.safety_terminal_value

        action_dim = action_space.shape[0]

        # Reward critics
        self.critic_1 = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.critic_2 = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.target_critic_1 = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.target_critic_2 = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Safety critic (Q_h)
        self.safety_q = SafetyQNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.safety_q_target = SafetyQNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.safety_q_target.load_state_dict(self.safety_q.state_dict())

        # Diffusion score model (policy)
        self.score_model = DiffusionScoreModel(
            state_dim=num_inputs,
            action_dim=action_dim,
            hidden_dim=args.hidden_size,
            time_dim=args.time_dim,
        ).to(self.device)

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.score_model.parameters(), lr=args.lr)
        self.critic_optim_1 = torch.optim.Adam(self.critic_1.parameters(), lr=args.lr)
        self.critic_optim_2 = torch.optim.Adam(self.critic_2.parameters(), lr=args.lr)
        self.safety_optim = torch.optim.Adam(self.safety_q.parameters(), lr=args.qc_lr)

        # Diffusion schedules
        if self.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(self.T)
        elif self.beta_schedule == "vp":
            self.betas = vp_beta_schedule(self.T)
        else:
            self.betas = torch.linspace(1e-4, 2e-2, self.T)

        self.betas = self.betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(self.device)

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------
    def _alpha_function(self, value: torch.Tensor) -> torch.Tensor:
        return self.alpha_coef * torch.relu(value)

    def _h_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        return cost - 3.0

    def estimate_value(
        self,
        state: torch.Tensor,
        num_samples: Optional[int] = None,
        use_target: bool = False,
    ) -> torch.Tensor:
        """Approximate V_h(s) by Monte-Carlo averaging of Q_h(s, a)."""

        if num_samples is None:
            num_samples = self.value_samples
        network = self.safety_q_target if use_target else self.safety_q
        values = []
        for _ in range(num_samples):
            actions = self.sample_action(state, guidance=True)
            q_val = network(state, actions)
            values.append(q_val)
        value = torch.stack(values, dim=0).mean(dim=0)
        if use_target:
            return value.detach()
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, state, eval: bool = False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.sample_action(state_tensor, guidance=True)
        return action.cpu().numpy()[0]

    def sample_action(self, state: torch.Tensor, guidance: bool = False) -> torch.Tensor:
        guidance_fn = self._guidance if guidance else None
        return ddpm_sampler(
            self.score_model,
            state,
            self.T,
            self.alphas,
            self.alpha_hats,
            self.betas,
            guidance_fn=guidance_fn,
            device=self.device,
        )

    def _guidance(self, state: torch.Tensor, action: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        """Return gradient-based guidance for safe high-value denoising."""

        q1 = self.critic_1(state, action).sum()
        q2 = self.critic_2(state, action).sum()
        dq_da_1 = torch.autograd.grad(q1, action, create_graph=False)[0]
        dq_da_2 = torch.autograd.grad(q2, action, create_graph=False)[0]
        reward_grad = 0.5 * (dq_da_1 + dq_da_2)

        safety_q_val = self.safety_q(state, action)
        dq_safe = torch.autograd.grad(safety_q_val.sum(), action, create_graph=False)[0]
        gate = soft_gate(safety_q_val.detach(), kappa=self.kappa, alpha=self.g_alpha)
        guidance = self.M_q * gate * reward_grad + self.M_safe * (1 - gate) * dq_safe
        return guidance

    # ------------------------------------------------------------------
    # Critics update
    # ------------------------------------------------------------------
    def update_reward_critic(self, state, action, reward, next_state, mask):
        with torch.no_grad():
            next_action = self.sample_action(next_state, guidance=True)
            next_q1 = self.target_critic_1(next_state, next_action)
            next_q2 = self.target_critic_2(next_state, next_action)
            target_v = torch.min(next_q1, next_q2)
            q_target = reward + mask * self.discount * target_v

        cur_q1 = self.critic_1(state, action)
        cur_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(cur_q1, q_target) + F.mse_loss(cur_q2, q_target)

        self.critic_optim_1.zero_grad()
        self.critic_optim_2.zero_grad()
        critic_loss.backward()
        self.critic_optim_1.step()
        self.critic_optim_2.step()
        return critic_loss.item()

    def update_safety_q(self, state, action, cost, next_state, mask):
        current_q = self.safety_q(state, action)
        current_v = self.estimate_value(state, use_target=False)

        with torch.no_grad():
            next_v = self.estimate_value(next_state, use_target=True)
            next_v = mask * next_v + (1 - mask) * self.terminal_safe_value
            alpha_term = self._alpha_function(current_v.detach())
            delta_v = next_v - current_v.detach()
            hat_q = torch.relu(delta_v + alpha_term)
            stage_violation = torch.relu(cost)
            target_q = torch.maximum(stage_violation, hat_q)

        temporal_loss = F.mse_loss(torch.relu(current_q), target_q)
        h_target = self._h_from_cost(cost)
        semantic_loss = F.mse_loss(current_v, h_target)

        loss = self.temporal_weight * temporal_loss + self.semantic_weight * semantic_loss
        self.safety_optim.zero_grad()
        loss.backward()
        self.safety_optim.step()
        return loss.item(), temporal_loss.item(), semantic_loss.item()

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------
    def update_actor(self, state, action):
        B = state.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device).float()
        noise = torch.randn_like(action)
        alpha_hat = self.alpha_hats[t.long()].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_hat) * action + torch.sqrt(1 - alpha_hat) * noise
        noisy_action.requires_grad_(True)

        q1 = self.critic_1(state, noisy_action).sum()
        q2 = self.critic_2(state, noisy_action).sum()
        dq_da_1 = torch.autograd.grad(q1, noisy_action, create_graph=True)[0]
        dq_da_2 = torch.autograd.grad(q2, noisy_action, create_graph=True)[0]
        reward_grad = 0.5 * (dq_da_1 + dq_da_2).detach()

        safety_q_val = self.safety_q(state, noisy_action)
        dq_safe = torch.autograd.grad(safety_q_val.sum(), noisy_action, create_graph=True)[0].detach()
        gate = soft_gate(safety_q_val.detach(), kappa=self.kappa, alpha=self.g_alpha)

        target_score = -self.M_q * gate * reward_grad - self.M_safe * (1 - gate) * dq_safe
        eps_pred = self.score_model(state, noisy_action, t.unsqueeze(1))
        actor_loss = ((target_score - eps_pred) ** 2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss.item()

    def update_parameters(self, memory, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory
        state = torch.FloatTensor(state_batch).to(self.device)
        next_state = torch.FloatTensor(next_state_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch[:, 0]).unsqueeze(1).to(self.device)
        cost = torch.FloatTensor(reward_batch[:, 1]).unsqueeze(1).to(self.device)
        mask = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        critic_loss = self.update_reward_critic(state, action, reward, next_state, mask)
        safety_loss, temporal_loss, semantic_loss = self.update_safety_q(state, action, cost, next_state, mask)
        actor_loss = self.update_actor(state, action)

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        soft_update(self.safety_q_target, self.safety_q, self.tau)

        return {
            "loss_q": critic_loss,
            "loss_safety": safety_loss,
            "loss_temporal": temporal_loss,
            "loss_semantic": semantic_loss,
            "loss_actor": actor_loss,
        }

    def save_model(self, path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.score_model.state_dict(), path / "score_model.pt")
        torch.save(self.critic_1.state_dict(), path / "critic1.pt")
        torch.save(self.critic_2.state_dict(), path / "critic2.pt")
        torch.save(self.safety_q.state_dict(), path / "safety_q.pt")