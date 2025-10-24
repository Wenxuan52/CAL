"""Gauss-policy variant of the SSM agent with local safety updates."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.guass_test.model import GaussianPolicy, QNetwork, SafetyValueNetwork
from agents.guass_test.utils import soft_gate, soft_update


class GuassTestAgent(Agent):
    """Safe Score Matching agent variant for the Gaussian policy baseline."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.tau = args.tau
        self.update_counter = 0

        action_dim = action_space.shape[0]

        # Reward critic networks
        self.critic = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety network shared for Q and V estimates
        self.safety_value = SafetyValueNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.safety_value_target = SafetyValueNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.safety_value_target.load_state_dict(self.safety_value.state_dict())

        # Policy
        self.policy = GaussianPolicy(args, num_inputs, action_dim, args.hidden_size, action_space).to(self.device)

        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(np.prod(action_space.shape))

        # Safety guidance parameters
        self.beta = getattr(args, "safe_beta", 1.0)
        self.kappa = getattr(args, "safe_thresh", 0.0)
        self.g_alpha = getattr(args, "gate_alpha", 5.0)

        # Gauss-specific hyper-parameters
        self.alpha_coef = getattr(args, "guass_alpha_coef", 0.5)
        self.temporal_weight = getattr(args, "guass_temporal_weight", 1.0)
        self.semantic_weight = getattr(args, "guass_semantic_weight", 0.5)
        self.terminal_safe_value = getattr(args, "guass_terminal_value", 0.0)

        # Optimisers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_optimizer = torch.optim.Adam(self.safety_value.parameters(), lr=getattr(args, "qc_lr", args.lr))

        print(
            "[GuassTest] device={} (alpha_coef={}, temporal_w={}, semantic_w={})".format(
                self.device, self.alpha_coef, self.temporal_weight, self.semantic_weight
            )
        )

        self.train()
        self.critic_target.train()
        self.safety_value_target.train()

    def train(self, training: bool = True):
        self.training = training
        self.critic.train(training)
        self.safety_value.train(training)
        self.policy.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, eval=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval:
            _, _, action = self.policy.sample(state_tensor)
        else:
            action, _, _ = self.policy.sample(state_tensor)
        return action.detach().cpu().numpy()[0]

    # ---------------------------------------------------------------------
    # Reward critic update
    # ---------------------------------------------------------------------
    def update_reward_critic(self, state, action, reward, next_state, mask):
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            q_target = reward + mask * self.discount * target_v
        current_q1, current_q2 = self.critic(state, action)
        loss_q = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()
        return loss_q.item()

    # ---------------------------------------------------------------------
    # Safety critic update
    # ---------------------------------------------------------------------
    def _alpha_function(self, value: torch.Tensor) -> torch.Tensor:
        return self.alpha_coef * torch.relu(value)

    def _h_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        return -cost

    def update_safety_value(self, state, action, cost, next_state, mask):
        current_v = self.safety_value(state, action)

        with torch.no_grad():
            next_action, _, _ = self.policy.sample(next_state)
            next_v = self.safety_value_target(next_state, next_action)
            next_v = mask * next_v + (1 - mask) * self.terminal_safe_value
            alpha_term = self._alpha_function(current_v.detach())
            delta_v = next_v - current_v.detach()
            hat_q = torch.relu(delta_v + alpha_term)
            stage_violation = torch.relu(cost)
            target_q = torch.maximum(stage_violation, hat_q)

        current_q = torch.relu(current_v)
        temporal_loss = F.mse_loss(current_q, target_q)

        h_target = self._h_from_cost(cost)
        semantic_loss = F.mse_loss(current_v, h_target)

        loss = self.temporal_weight * temporal_loss + self.semantic_weight * semantic_loss

        self.safety_optimizer.zero_grad()
        loss.backward()
        self.safety_optimizer.step()

        return loss.item(), temporal_loss.item(), semantic_loss.item()

    # ---------------------------------------------------------------------
    # Actor update
    # ---------------------------------------------------------------------
    def update_actor(self, state):
        action, log_prob, _ = self.policy.sample(state)
        q1, q2 = self.critic(state, action)
        q_min = torch.min(q1, q2)

        qc_value = self.safety_value(state, action)
        qc_positive = torch.relu(qc_value)
        gate = soft_gate(qc_value, kappa=self.kappa, alpha=self.g_alpha)

        loss_pi = (
            self.alpha.detach() * log_prob
            - gate * q_min
            + (1 - gate) * self.beta * F.softplus(qc_positive)
        ).mean()

        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return loss_pi.item()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state = torch.FloatTensor(state_batch).to(self.device)
        next_state = torch.FloatTensor(next_state_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch[:, 0]).unsqueeze(1).to(self.device)
        cost = torch.FloatTensor(reward_batch[:, 1]).unsqueeze(1).to(self.device)
        mask = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        q_loss = self.update_reward_critic(state, action, reward, next_state, mask)
        safety_loss, temporal_loss, semantic_loss = self.update_safety_value(
            state, action, cost, next_state, mask
        )
        pi_loss = self.update_actor(state)

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.safety_value_target, self.safety_value, self.tau)

        return {
            "loss_q": q_loss,
            "loss_safety": safety_loss,
            "loss_temporal": temporal_loss,
            "loss_semantic": semantic_loss,
            "loss_pi": pi_loss,
        }

    def act(self, obs, sample=False):
        return self.select_action(obs, eval=not sample)

    def save_model(self, save_dir, suffix=""):
        actor_path = save_dir / f"guass_test_actor_{suffix}.pth"
        critic_path = save_dir / f"guass_test_critic_{suffix}.pth"
        safety_path = save_dir / f"guass_test_safety_{suffix}.pth"

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.safety_value.state_dict(), safety_path)

