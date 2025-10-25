"""Gaussian policy agent with multi-step safety rollouts."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.guass_ms.model import GaussianPolicy, QNetwork, SafetyValueNetwork
from agents.guass_ms.utils import soft_gate, soft_update


class GuassMSAgent(Agent):
    """Safe Score Matching agent variant with multi-step safety propagation."""

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

        # Multi-step rollout parameters
        self.rollout_n = max(1, int(getattr(args, "guass_ms_rollout", 1)))
        self.rollout_eta = getattr(args, "guass_ms_eta", 1.0)
        self.rollout_gamma_h = getattr(args, "guass_ms_gamma_h", 0.99)

        # Optimisers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_optimizer = torch.optim.Adam(self.safety_value.parameters(), lr=getattr(args, "qc_lr", args.lr))

        print(
            "[GuassMS] device={} (alpha_coef={}, temporal_w={}, semantic_w={}, rollout_n={})".format(
                self.device, self.alpha_coef, self.temporal_weight, self.semantic_weight, self.rollout_n
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

    # ------------------------------------------------------------------
    # Reward critic update
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Safety critic update
    # ------------------------------------------------------------------
    def _alpha_function(self, value: torch.Tensor) -> torch.Tensor:
        return self.alpha_coef * torch.relu(value)

    def _h_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        return cost - 10.0

    def update_safety_value(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        cost_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        rollout_n: Optional[int] = None,
    ):
        batch_size, seq_len = state_seq.shape[0], state_seq.shape[1]
        rollout_len = min(rollout_n or self.rollout_n, seq_len)

        state_seq = state_seq[:, :rollout_len]
        action_seq = action_seq[:, :rollout_len]
        next_state_seq = next_state_seq[:, :rollout_len]
        cost_seq = cost_seq[:, :rollout_len]
        mask_seq = mask_seq[:, :rollout_len]

        state_flat = state_seq.reshape(batch_size * rollout_len, -1)
        action_flat = action_seq.reshape(batch_size * rollout_len, -1)

        value_seq = self.safety_value(state_flat, action_flat).view(batch_size, rollout_len, 1)
        current_v = value_seq[:, 0]

        with torch.no_grad():
            detached_value_seq = value_seq.detach()
            target_value_seq = self.safety_value_target(state_flat, action_flat).view(batch_size, rollout_len, 1)

            cost_tensor = cost_seq
            if cost_tensor.dim() == 2:
                cost_tensor = cost_tensor.unsqueeze(-1)
            mask_tensor = mask_seq
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(-1)
            mask_tensor = mask_tensor.unsqueeze(-1) if mask_tensor.dim() == 2 else mask_tensor

            next_state_tensor = next_state_seq
            if next_state_tensor.dim() == 2:
                next_state_tensor = next_state_tensor.unsqueeze(1)

            terminal_tensor = torch.full((batch_size, 1), self.terminal_safe_value, device=self.device)

            last_next_state = next_state_tensor[:, rollout_len - 1]
            next_action_last, _, _ = self.policy.sample(last_next_state)
            bootstrap_v = self.safety_value_target(last_next_state, next_action_last)
            bootstrap_v = (
                mask_tensor[:, rollout_len - 1] * bootstrap_v
                + (1 - mask_tensor[:, rollout_len - 1]) * terminal_tensor
            )

            next_v_steps = []
            for step in range(rollout_len):
                if step == rollout_len - 1:
                    next_v = bootstrap_v
                else:
                    next_v = (
                        mask_tensor[:, step] * target_value_seq[:, step + 1]
                        + (1 - mask_tensor[:, step]) * terminal_tensor
                    )
                next_v_steps.append(next_v)

            prev_hat = torch.zeros_like(bootstrap_v)
            eta_gamma = self.rollout_eta * self.rollout_gamma_h

            for step in reversed(range(rollout_len)):
                delta = (
                    next_v_steps[step]
                    - detached_value_seq[:, step]
                    + self._alpha_function(detached_value_seq[:, step])
                )
                future = mask_tensor[:, step] * eta_gamma * prev_hat
                hat_candidate = torch.relu(delta + future)
                stage_violation = torch.relu(cost_tensor[:, step])
                prev_hat = torch.maximum(stage_violation, hat_candidate)

            hat_q = prev_hat

        current_q = torch.relu(current_v)
        temporal_loss = F.mse_loss(current_q, hat_q)

        first_cost = cost_seq[:, 0]
        if first_cost.dim() == 1:
            first_cost = first_cost.unsqueeze(-1)
        h_target = self._h_from_cost(first_cost)
        semantic_loss = F.mse_loss(current_v, h_target)

        loss = self.temporal_weight * temporal_loss + self.semantic_weight * semantic_loss

        self.safety_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_value.parameters(), 10.0)
        self.safety_optimizer.step()

        return loss.item(), temporal_loss.item(), semantic_loss.item()

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        if state_batch.ndim == 3:
            # Sequence batch (multi-step)
            state_seq = torch.FloatTensor(state_batch).to(self.device)
            action_seq = torch.FloatTensor(action_batch).to(self.device)
            reward_seq = torch.FloatTensor(reward_batch).to(self.device)
            next_state_seq = torch.FloatTensor(next_state_batch).to(self.device)
            mask_seq = torch.FloatTensor(mask_batch).to(self.device)
            if mask_seq.dim() == 3 and mask_seq.size(-1) == 1:
                mask_seq = mask_seq.squeeze(-1)

            state = state_seq[:, 0]
            next_state = next_state_seq[:, 0]
            action = action_seq[:, 0]
            reward = reward_seq[:, 0, 0].unsqueeze(1)
            cost = reward_seq[:, 0, 1].unsqueeze(1)
            mask = mask_seq[:, 0].unsqueeze(1)
            cost_seq = reward_seq[:, :, 1].unsqueeze(-1)
        else:
            state = torch.FloatTensor(state_batch).to(self.device)
            next_state = torch.FloatTensor(next_state_batch).to(self.device)
            action = torch.FloatTensor(action_batch).to(self.device)
            reward = torch.FloatTensor(reward_batch[:, 0]).unsqueeze(1).to(self.device)
            cost = torch.FloatTensor(reward_batch[:, 1]).unsqueeze(1).to(self.device)
            mask = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

            state_seq = state.unsqueeze(1)
            action_seq = action.unsqueeze(1)
            cost_seq = cost.unsqueeze(1)
            next_state_seq = next_state.unsqueeze(1)
            mask_seq = mask

        q_loss = self.update_reward_critic(state, action, reward, next_state, mask)
        safety_loss, temporal_loss, semantic_loss = self.update_safety_value(
            state_seq, action_seq, cost_seq, next_state_seq, mask_seq, rollout_n=self.rollout_n
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
