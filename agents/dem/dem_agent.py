from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.cal.model import QNetwork, QcEnsemble
from agents.cal.utils import soft_update
from agents.dem.energies.base_energy_function import BaseEnergyFunction
from agents.dem.models.components.lambda_weighter import BasicLambdaWeighter
from agents.dem.models.components.mlp import Block, TimeConder
from agents.dem.models.components.noise_schedules import (
    GeometricNoiseSchedule,
    LinearNoiseSchedule,
    PowerNoiseSchedule,
    QuadraticNoiseSchedule,
    SubLinearNoiseSchedule,
)
from agents.dem.models.components.sde_integration import integrate_sde
from agents.dem.models.components.sdes import VEReverseSDE


def _build_noise_schedule(config) -> SubLinearNoiseSchedule:
    schedule_type = (config.type or "geometric").lower()
    if schedule_type == "linear":
        return LinearNoiseSchedule(beta=config.beta)
    if schedule_type == "quadratic":
        return QuadraticNoiseSchedule(beta=config.beta)
    if schedule_type == "power":
        return PowerNoiseSchedule(beta=config.beta, power=config.power)
    if schedule_type == "sublinear":
        return SubLinearNoiseSchedule(beta=config.beta)
    if schedule_type == "geometric":
        return GeometricNoiseSchedule(sigma_min=config.sigma_min, sigma_max=config.sigma_max)
    raise ValueError(f"Unsupported DEM noise schedule type: {config.type}")


class PolicyEnergyFunction(BaseEnergyFunction):
    """Energy induced by critics and safety constraints."""

    def __init__(
        self,
        states: torch.Tensor,
        critic: QNetwork,
        safety_critics: QcEnsemble,
        lam_value: torch.Tensor,
        rect_value: torch.Tensor,
        target_cost: float,
        args,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        energy_regularization: float = 0.0,
        action_penalty: float = 0.0,
    ) -> None:
        super().__init__(dimensionality=action_low.numel())
        self.states = states
        self.critic = critic
        self.safety_critics = safety_critics
        self.lam_value = lam_value
        self.rect_value = rect_value
        self.target_cost = target_cost
        self.args = args
        self.action_low = action_low
        self.action_high = action_high
        self.energy_regularization = energy_regularization
        self.action_penalty = action_penalty

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size = actions.size(0)
        states = self.states
        if states.size(0) != batch_size:
            if states.size(0) == 1:
                states = states.expand(batch_size, -1)
            else:
                raise ValueError("Mismatched batch sizes for energy evaluation")

        q1, q2 = self.critic(states, actions)
        q_values = torch.min(q1, q2).squeeze(-1)

        qc_values = self.safety_critics(states, actions)
        if qc_values.dim() == 3:
            qc_std, qc_mean = torch.std_mean(qc_values, dim=0, unbiased=False)
            qc_std = qc_std.squeeze(-1)
            qc_mean = qc_mean.squeeze(-1)
        else:
            qc_mean = qc_values.squeeze(-1)
            qc_std = torch.zeros_like(qc_mean)
        qc_estimate = qc_mean + self.args.k * qc_std

        effective_lam = torch.clamp(self.lam_value - self.rect_value, min=0.0)
        energy = -q_values + effective_lam * (qc_estimate - self.target_cost)

        if self.energy_regularization > 0:
            energy = energy + self.energy_regularization * (actions.pow(2).sum(dim=-1))

        if self.action_penalty > 0:
            lower_violation = F.relu(self.action_low - actions)
            upper_violation = F.relu(actions - self.action_high)
            boundary_penalty = (lower_violation.pow(2) + upper_violation.pow(2)).sum(dim=-1)
            energy = energy + self.action_penalty * boundary_penalty

        return energy


class StateConditionedScoreNet(nn.Module):
    """State-conditioned score network using DEM building blocks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        hidden_layers: int,
        time_layers: int,
    ) -> None:
        super().__init__()
        self.state_action_proj = nn.Linear(state_dim + action_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.time_encoder = TimeConder(channel=hidden_size, out_dim=hidden_size, num_layers=max(time_layers, 1))
        self.blocks = nn.ModuleList(
            [Block(hidden_size, hidden_size, add_t_emb=True, concat_t_emb=False) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if times.dim() == 1:
            times = times.unsqueeze(-1)
        times = times.to(actions.dtype)
        t_emb = self.time_encoder(times)
        x = torch.cat([states, actions], dim=-1)
        x = self.state_action_proj(x)
        x = F.gelu(self.norm(x))
        for block in self.blocks:
            x = block(x, t_emb)
        return self.output_layer(x)


class DEMAgent(Agent):
    """Safe RL agent that trains a DEM score model for action sampling."""

    def __init__(self, num_inputs: int, action_space, args) -> None:
        super().__init__()
        self.args = args
        device_str = args.dem_config.device
        if device_str is None or device_str == "auto":
            device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.policy_update_delay = max(int(args.policy_update_delay), 1)
        self.update_counter = 0
        self.total_env_steps = 0

        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

        # Critics
        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.safety_critics = QcEnsemble(num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size).to(
            self.device
        )
        self.safety_critic_targets = QcEnsemble(num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size).to(
            self.device
        )
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # Score network and diffusion machinery
        cfg = args.dem_config
        self.score_model = StateConditionedScoreNet(
            num_inputs,
            self.action_dim,
            hidden_size=cfg.dem_score_hidden_size,
            hidden_layers=cfg.dem_score_hidden_layers,
            time_layers=cfg.dem_score_time_layers,
        ).to(self.device)
        self.score_optimizer = torch.optim.Adam(
            self.score_model.parameters(), lr=cfg.dem_score_lr, weight_decay=args.dem_actor_weight_decay
        )

        self.noise_schedule = _build_noise_schedule(cfg.dem_noise_schedule)
        self.lambda_weighter = BasicLambdaWeighter(self.noise_schedule, epsilon=cfg.dem_lambda_epsilon)
        self.num_integration_steps = cfg.dem_num_integration_steps
        self.time_range = cfg.dem_time_range
        self.prior_std = cfg.dem_prior_std
        self.training_diffusion_scale = cfg.dem_diffusion_scale
        self.eval_diffusion_scale = cfg.dem_eval_diffusion_scale
        self.negative_time = cfg.dem_negative_time
        self.negative_time_steps = cfg.dem_negative_time_steps
        self.energy_regularization = cfg.dem_energy_regularization
        self.action_penalty = cfg.dem_action_penalty

        # Optimizers
        self.actor_optimizer = self.score_optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.safety_critic_lr)

        self.log_lam = torch.tensor(np.log(0.6931), device=self.device, requires_grad=True)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.dem_lam_lr)

        self.rect = torch.tensor(0.0, device=self.device)

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Target cost
        if args.safetygym:
            if self.safety_discount < 1:
                factor = (1 - self.safety_discount ** args.epoch_length) / (1 - self.safety_discount)
                self.target_cost = args.cost_lim * factor / args.epoch_length
            else:
                self.target_cost = args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget:", self.target_cost)

    def train(self, training: bool = True) -> None:
        self.training = training
        self.critic.train(training)
        self.safety_critics.train(training)
        self.score_model.train(training)

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lam.exp()

    def _random_action(self) -> np.ndarray:
        low = self.action_low.cpu().numpy()
        high = self.action_high.cpu().numpy()
        return np.random.uniform(low, high)

    def _exploration_scale(self, eval_mode: bool) -> float:
        schedule = self.args.dem_config.dem_exploration_schedule
        if eval_mode:
            return float(self.eval_diffusion_scale)
        if schedule.type == "linear_decay" and schedule.steps:
            ratio = min(self.total_env_steps / max(float(schedule.steps), 1.0), 1.0)
            final = schedule.final if schedule.final is not None else schedule.initial
            return float(schedule.initial + (final - schedule.initial) * ratio)
        if schedule.type == "constant" or schedule.steps is None:
            return float(schedule.initial)
        if schedule.type == "exponential_decay" and schedule.final is not None:
            ratio = min(self.total_env_steps / max(float(schedule.steps or 1), 1.0), 1.0)
            return float(schedule.initial * ((schedule.final / max(schedule.initial, 1e-6)) ** ratio))
        return float(schedule.initial)

    def _build_energy(self, states: torch.Tensor) -> PolicyEnergyFunction:
        return PolicyEnergyFunction(
            states=states,
            critic=self.critic,
            safety_critics=self.safety_critics,
            lam_value=self.lam.detach(),
            rect_value=self.rect.detach(),
            target_cost=self.target_cost,
            args=self.args,
            action_low=self.action_low,
            action_high=self.action_high,
            energy_regularization=self.energy_regularization,
            action_penalty=self.action_penalty,
        )

    def _sample_actions(
        self, states: torch.Tensor, diffusion_scale: float, deterministic: bool = False
    ) -> torch.Tensor:
        batch_size = states.size(0)
        states = states.to(self.device).detach()
        prior = torch.randn(batch_size, self.action_dim, device=self.device) * self.prior_std
        energy_function = self._build_energy(states)

        def score_fn(t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t.expand(actions.size(0))
            return self.score_model(states, actions, t)

        sde = VEReverseSDE(score_fn, self.noise_schedule)
        trajectory = integrate_sde(
            sde,
            prior,
            self.num_integration_steps,
            energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=True,
            no_grad=True,
            time_range=self.time_range,
            negative_time=self.negative_time and deterministic,
            num_negative_time_steps=self.negative_time_steps,
            clipper=None,
        )
        actions = trajectory[-1]
        high = self.action_high.unsqueeze(0)
        low = self.action_low.unsqueeze(0)
        actions = torch.max(torch.min(actions, high), low)
        return actions

    def select_action(self, state, eval: bool = False) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not eval:
            self.total_env_steps += 1
        if not eval and self.total_env_steps <= self.args.init_exploration_steps:
            return self._random_action()

        diffusion_scale = self._exploration_scale(eval)
        with torch.no_grad():
            action_tensor = self._sample_actions(state_tensor, diffusion_scale, deterministic=eval)
        return action_tensor.squeeze(0).cpu().numpy()

    def _compute_cost_statistics(self, qc_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if qc_values.shape[0] == 1:
            mean = qc_values[0]
            std = torch.zeros_like(mean)
        else:
            std, mean = torch.std_mean(qc_values, dim=0, unbiased=False)
        return mean.squeeze(-1), std.squeeze(-1)

    def update_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        next_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._sample_actions(next_state, self.training_diffusion_scale, deterministic=False)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + mask * self.discount * target_V

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.args.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm)
        self.critic_optimizer.step()

        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)
            mask_cost = torch.ones_like(mask) if self.args.safetygym else mask
            if getattr(self.args, "intrgt_max", False):
                qc_idxs = np.random.choice(self.args.qc_ens_size, getattr(self.args, "M", 1))
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

    def update_score(self, state: torch.Tensor, action: torch.Tensor) -> None:
        batch_size = state.size(0)
        state = state.to(self.device).detach()
        action = action.to(self.device).detach()
        t = torch.rand(batch_size, device=self.device)
        noise_variance = self.noise_schedule.h(t)
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.as_tensor(noise_variance, dtype=t.dtype, device=self.device)
        else:
            noise_variance = noise_variance.to(self.device, dtype=t.dtype)
        noise_scale = torch.sqrt(noise_variance.unsqueeze(-1) + 1e-6)
        noise = torch.randn_like(action)
        noised_action = action + noise_scale * noise
        noised_action.requires_grad_(True)

        detached_state = state
        energy_function = self._build_energy(detached_state)
        energies = energy_function(noised_action)
        target_grad = -torch.autograd.grad(energies.sum(), noised_action, create_graph=False)[0]

        predicted_score = self.score_model(detached_state, noised_action, t)
        weights = self.lambda_weighter(t).unsqueeze(-1)
        loss = ((predicted_score - target_grad) ** 2).sum(dim=-1) * weights.squeeze(-1)
        loss = loss.mean()

        self.score_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.score_model.parameters(), self.args.grad_clip_norm)
        self.score_optimizer.step()

    def update_parameters(self, memory, updates: int) -> None:
        self.update_counter += 1
        state_batch, action_batch, reward_cost_batch, next_state_batch, mask_batch = memory

        state_batch = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        reward_cost_batch = torch.as_tensor(reward_cost_batch, dtype=torch.float32, device=self.device)
        reward_batch = reward_cost_batch[:, 0].unsqueeze(-1)
        cost_batch = reward_cost_batch[:, 1].unsqueeze(-1)
        mask_batch = torch.as_tensor(mask_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.update_critic(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch)

        with torch.no_grad():
            current_QCs = self.safety_critics(state_batch, action_batch)
            mean_cost, std_cost = self._compute_cost_statistics(current_QCs)
            current_QC = mean_cost + self.args.k * std_cost

        self.rect = self.args.c * torch.mean(self.target_cost - current_QC)
        self.rect = torch.clamp(self.rect.detach(), max=self.lam.item())

        lam = self.lam
        lam_loss = torch.mean(lam * (self.target_cost - current_QC.detach()))
        self.log_lam_optimizer.zero_grad()
        lam_loss.backward()
        self.log_lam_optimizer.step()

        if self.update_counter % self.policy_update_delay == 0:
            self.update_score(state_batch, action_batch)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

    def save_model(self, save_dir: Optional[Path], suffix: str = "latest") -> None:
        if save_dir is None:
            return
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.critic.state_dict(), save_path / f"dem_critic_{suffix}.pth")
        torch.save(self.critic_target.state_dict(), save_path / f"dem_critic_target_{suffix}.pth")
        torch.save(self.safety_critics.state_dict(), save_path / f"dem_safety_critics_{suffix}.pth")
        torch.save(self.score_model.state_dict(), save_path / f"dem_score_{suffix}.pth")
        torch.save({"log_lam": self.log_lam.detach().cpu()}, save_path / f"dem_aux_{suffix}.pth")

    def load_model(self, actor_path, critic_path, safety_path):
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
        if safety_path is not None:
            self.safety_critics.load_state_dict(torch.load(safety_path, map_location=self.device))
            self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())
        if actor_path is not None:
            self.score_model.load_state_dict(torch.load(actor_path, map_location=self.device))
