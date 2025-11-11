from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.cal.model import QNetwork, QcEnsemble
from agents.cal.utils import soft_update


@dataclass
class NoiseScheduleState:
    value: float
    decay: float
    minimum: float

    def update(self) -> float:
        self.value = max(self.value * self.decay, self.minimum)
        return self.value


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim: int, action_space, hidden_layers: Iterable[int]):
        super().__init__()
        layers = []
        last_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_space.shape[0]))
        self.model = nn.Sequential(*layers)

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.model(state)
        return torch.tanh(raw) * self.action_scale + self.action_bias


class DEMAgent(Agent):
    """Safe RL agent that uses a deterministic energy-matching style policy."""

    def __init__(self, num_inputs: int, action_space, args):
        super().__init__()
        device_str = args.dem_config.device
        if device_str is None or device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        self.device = torch.device(device_str)

        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.policy_target_update_frequency = args.policy_target_update_frequency
        self.policy_update_delay = max(int(args.policy_update_delay), 1)
        self.args = args

        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float32)

        self.update_counter = 0
        self.policy_update_counter = 0

        # Safety-related parameters
        self.c = args.c
        self.k = args.k

        # Critics
        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.safety_critics = QcEnsemble(num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(
            num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size
        ).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # Deterministic policy and target
        hidden_layers = args.actor_hidden_layers
        self.policy = DeterministicPolicy(num_inputs, action_space, hidden_layers).to(self.device)
        self.policy_target = DeterministicPolicy(num_inputs, action_space, hidden_layers).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.actor_lr, weight_decay=args.dem_actor_weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.safety_critic_optimizer = torch.optim.Adam(
            self.safety_critics.parameters(), lr=args.safety_critic_lr
        )
        self.log_lam = torch.tensor(np.log(0.6931), device=self.device, requires_grad=True)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.dem_lam_lr)

        self.rect = torch.tensor(0.0, device=self.device)
        self.noise_schedule = NoiseScheduleState(
            value=args.exploration_noise_std,
            decay=args.dem_action_noise_decay,
            minimum=args.dem_action_noise_min,
        )

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()
        self.policy_target.train()

        # Target cost budget
        if args.safetygym:
            if self.safety_discount < 1:
                factor = (1 - self.safety_discount ** args.epoch_length) / (1 - self.safety_discount)
                self.target_cost = args.cost_lim * factor / args.epoch_length
            else:
                self.target_cost = args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)

    def train(self, training: bool = True):
        self.training = training
        self.policy.train(training)
        self.policy_target.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)

    @property
    def lam(self):
        return self.log_lam.exp()

    def select_action(self, state, eval: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state_tensor)
            if not eval:
                noise_scale = self.noise_schedule.value
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.max(torch.min(action, self.action_high), self.action_low)
                self.noise_schedule.update()
            else:
                if not self.args.dem_config.deterministic_eval:
                    noise_scale = self.noise_schedule.value
                    noise = torch.randn_like(action) * noise_scale
                    action = action + noise
                    action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.detach().cpu().numpy()[0]

    def _compute_cost_statistics(self, qc_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if qc_values.shape[0] == 1:
            mean = qc_values[0]
            std = torch.zeros_like(mean)
        else:
            std, mean = torch.std_mean(qc_values, dim=0)
        return mean, std

    def update_critic(self, state, action, reward, cost, next_state, mask):
        with torch.no_grad():
            next_action = self.policy_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.args.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm)
        self.critic_optimizer.step()

        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)
            if self.args.safetygym:
                mask_cost = torch.ones_like(mask)
            else:
                mask_cost = mask
            if self.args.intrgt_max:
                qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
                next_QC_random_max = next_QCs[qc_idxs].max(dim=0, keepdim=True).values
                next_QC = next_QC_random_max.repeat(self.args.qc_ens_size, 1, 1)
            else:
                next_QC = next_QCs
            target_QCs = cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) + (
                mask_cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) * self.safety_discount * next_QC
            )

        current_QCs = self.safety_critics(state, action)
        safety_critic_loss = F.mse_loss(current_QCs, target_QCs.detach())
        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        if self.args.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.safety_critics.parameters(), self.args.grad_clip_norm)
        self.safety_critic_optimizer.step()

    def update_actor(self, state, action_taken):
        new_action = self.policy(state)
        actor_Q1, actor_Q2 = self.critic(state, new_action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_QCs = self.safety_critics(state, new_action)
        actor_mean, actor_std = self._compute_cost_statistics(actor_QCs)
        actor_QC = actor_mean + self.args.k * actor_std

        with torch.no_grad():
            current_QCs = self.safety_critics(state, action_taken)
            current_mean, current_std = self._compute_cost_statistics(current_QCs)
            current_QC = current_mean + self.args.k * current_std

        self.rect = self.c * torch.mean(self.target_cost - current_QC)
        self.rect = torch.clamp(self.rect.detach(), max=self.lam.item())

        lam = self.lam.detach()
        actor_loss = torch.mean(-actor_Q + (lam - self.rect) * actor_QC)

        if self.args.dem_use_entropy_regularization:
            entropy_bonus = -self.args.dem_entropy_coef * torch.mean(new_action.pow(2))
            actor_loss = actor_loss + entropy_bonus

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.args.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
        self.actor_optimizer.step()

        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.lam * (self.target_cost - current_QC).detach())
        lam_loss.backward()
        self.log_lam_optimizer.step()

    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        cost_batch = torch.as_tensor(reward_batch[:, 1], dtype=torch.float32, device=self.device).unsqueeze(1)
        reward_batch = torch.as_tensor(reward_batch[:, 0], dtype=torch.float32, device=self.device).unsqueeze(1)
        mask_batch = torch.as_tensor(mask_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.update_critic(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch)

        self.policy_update_counter += 1
        if self.policy_update_counter % self.policy_update_delay == 0:
            self.update_actor(state_batch, action_batch)
        if self.policy_update_counter % self.policy_target_update_frequency == 0:
            soft_update(self.policy_target, self.policy, self.critic_tau)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

    def save_model(self, save_dir, suffix: str = ""):
        actor_path = save_dir / f"dem_actor_{suffix}.pth"
        critic_path = save_dir / f"dem_critics_{suffix}.pth"
        safety_path = save_dir / f"dem_safety_critics_{suffix}.pth"

        print(
            f"[DEM] Saving models to:\n  {actor_path}\n  {critic_path}\n  {safety_path}"
        )
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.safety_critics.state_dict(), safety_path)

    def load_model(self, actor_path, critic_path, safety_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.policy_target.load_state_dict(self.policy.state_dict())
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
        if safety_path is not None:
            self.safety_critics.load_state_dict(torch.load(safety_path, map_location=self.device))
            self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())
