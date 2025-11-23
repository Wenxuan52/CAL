"""Safety-Gym integration of the original DEM algorithm.

This module adapts the score-based DEM components so they can be optimised inside the
existing replay-driven loop used throughout the code base.  The implementation keeps
the replay/Q-learning structure already provided by the framework, but upgrades the
actor into a denoising diffusion policy that is trained with energy matching losses
similar to those used in the reference code base.  Only reward optimisation is
considered – costs reported by Safety-Gym are ignored when updating the agent.
"""

from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.dem.config import DEMAgentConfig, ExplorationSchedule, NoiseScheduleConfig


def build_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a feed-forward network with configurable hidden layers."""

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DiffusionPolicy(nn.Module):
    """Score network predicting the noise direction used by DEM."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int],
        time_embed_dim: int,
        time_layers: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.state_encoder = build_mlp(state_dim, hidden_layers, hidden_layers[-1], activation)
        self.action_encoder = build_mlp(action_dim, [hidden_layers[-1]], hidden_layers[-1], activation)
        from agents.dem.models.components.mlp import TimeConder

        self.time_encoder = TimeConder(time_embed_dim, hidden_layers[-1], time_layers)

        joint_input_dim = hidden_layers[-1] * 3
        self.output = build_mlp(joint_input_dim, hidden_layers, action_dim, activation)

    def forward(
        self,
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(noisy_action)
        time_feat = self.time_encoder(sigma.view(-1, 1))
        joint = torch.cat([state_feat, action_feat, time_feat], dim=-1)
        return self.output(joint)


class TwinCritic(nn.Module):
    """Twin-critic architecture mirroring the DEM energy evaluation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = build_mlp(input_dim, hidden_layers, 1)
        self.q2 = build_mlp(input_dim, hidden_layers, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu)


class DEMAgent(Agent):
    """Replay-based DEM agent with diffusion policy optimisation."""

    def __init__(self, num_inputs, action_space, args) -> None:
        super().__init__()

        self.args = args
        self.config: Optional[DEMAgentConfig] = getattr(args, "dem_config", None)
        if self.config is None:
            from agents.dem.config import load_dem_config

            self.config = load_dem_config(args.env_name)
        assert self.config is not None

        preferred_device = self.config.device
        if preferred_device and preferred_device.startswith("cuda") and not torch.cuda.is_available():
            preferred_device = "cpu"

        device_name = preferred_device or ("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        self.discount = float(args.gamma)
        self.tau = float(args.tau)
        self.total_it = 0
        self.env_steps = 0

        action_dim = int(action_space.shape[0])
        critic_hidden_layers: Sequence[int] = (
            tuple(self.config.actor_hidden_layers) or (self.config.hidden_size, self.config.hidden_size)
        )

        self.policy = DiffusionPolicy(
            num_inputs,
            action_dim,
            critic_hidden_layers,
            self.config.dem_score_hidden_size,
            self.config.dem_score_time_layers,
        ).to(self.device)
        self.policy_target = DiffusionPolicy(
            num_inputs,
            action_dim,
            critic_hidden_layers,
            self.config.dem_score_hidden_size,
            self.config.dem_score_time_layers,
        ).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.critic = TwinCritic(num_inputs, action_dim, critic_hidden_layers).to(self.device)
        self.critic_target = TwinCritic(num_inputs, action_dim, critic_hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.actor_lr,
            weight_decay=self.config.dem_actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        self.grad_clip_norm: Optional[float] = self.config.grad_clip_norm
        self.policy_update_delay = max(int(self.config.policy_target_update_frequency), 1)
        self.critic_target_update_frequency = max(int(self.config.critic_target_update_frequency), 1)

        self.noise_schedule = self.config.dem_noise_schedule
        self.exploration_schedule = self.config.dem_exploration_schedule
        self.deterministic_eval = bool(self.config.deterministic_eval)

        self.init_random_steps = int(self.config.init_exploration_steps)
        self.min_pool_size = int(self.config.min_pool_size)

        self._action_low = np.asarray(action_space.low, dtype=np.float32)
        self._action_high = np.asarray(action_space.high, dtype=np.float32)
        self._action_low_tensor = torch.as_tensor(self._action_low, device=self.device)
        self._action_high_tensor = torch.as_tensor(self._action_high, device=self.device)

        self.train()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def train(self, training: bool = True):  # type: ignore[override]
        self.training = training
        self.policy.train(training)
        self.critic.train(training)
        self.policy_target.train(training)
        self.critic_target.train(training)

    def select_action(self, state, eval_t: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if (not eval_t and self.env_steps < self.init_random_steps) or self.min_pool_size <= 0:
            action = self._sample_uniform_action()
        else:
            with torch.no_grad():
                action_tensor = self._generate_actions(self.policy, state_tensor, deterministic=eval_t)
            action = action_tensor.cpu().numpy()[0]

            if not eval_t:
                noise_std = self._compute_exploration_std()
                if noise_std > 0.0:
                    action += np.random.normal(0.0, noise_std, size=action.shape)
            elif not self.deterministic_eval:
                eval_noise = float(self.config.dem_eval_diffusion_scale)
                if eval_noise > 0.0:
                    action += np.random.normal(0.0, eval_noise, size=action.shape)

            action = np.clip(action, self._action_low, self._action_high)

        if not eval_t:
            self.env_steps += 1

        return action

    def update_parameters(self, memory, updates):
        self.total_it += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device)
        if reward.ndim == 2 and reward.shape[1] > 1:
            reward = reward[:, 0]
        reward = reward.unsqueeze(-1)
        mask = torch.as_tensor(mask_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_action = self._generate_actions(self.policy_target, next_state, deterministic=True)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + mask * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        if self.total_it % self.policy_update_delay == 0:
            sigma = self._sample_noise_levels(action.shape[0])
            epsilon = torch.randn_like(action)
            noisy_action = action + sigma * epsilon
            predicted_noise = self.policy(state, noisy_action, sigma)
            denoised_action = noisy_action - sigma * predicted_noise

            q_val = self.critic.q1_only(state, denoised_action)
            denoise_loss = F.mse_loss(predicted_noise, epsilon)

            actor_loss = (
                self.config.dem_lambda_epsilon * denoise_loss
                - q_val.mean()
                + self.config.dem_action_penalty * (denoised_action.pow(2).mean())
            )
            if self.config.dem_energy_regularization > 0.0:
                actor_loss = actor_loss + self.config.dem_energy_regularization * (predicted_noise.pow(2).mean())

            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.policy_optimizer.step()

            self._soft_update(self.policy_target, self.policy)

        if self.total_it % self.critic_target_update_frequency == 0:
            self._soft_update(self.critic_target, self.critic)

    def save_model(self, save_dir, suffix: str = ""):
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        policy_path = save_dir / f"dem_policy{('_' + suffix) if suffix else ''}.pth"
        critic_path = save_dir / f"dem_critic{('_' + suffix) if suffix else ''}.pth"
        config_path = save_dir / f"dem_config{('_' + suffix) if suffix else ''}.yaml"

        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.critic.state_dict(), critic_path)

        try:
            import yaml

            with config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(asdict(self.config), fh)
        except Exception:
            torch.save(self.config, config_path.with_suffix(".pt"))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _sample_uniform_action(self) -> np.ndarray:
        return np.random.uniform(self._action_low, self._action_high)

    def _compute_exploration_std(self) -> float:
        schedule: ExplorationSchedule = self.exploration_schedule
        initial = float(schedule.initial)

        if schedule.type == "linear" and schedule.final is not None and schedule.steps:
            progress = min(self.env_steps / max(schedule.steps, 1), 1.0)
            return float(initial + progress * (schedule.final - initial))
        if schedule.type == "exponential" and schedule.final is not None and schedule.steps:
            progress = min(self.env_steps / max(schedule.steps, 1), 1.0)
            ratio = (schedule.final / max(initial, 1e-6)) if initial > 0 else 1.0
            return float(initial * (ratio ** progress))

        return initial

    def _sample_noise_levels(self, batch_size: int) -> torch.Tensor:
        cfg: NoiseScheduleConfig = self.noise_schedule
        u = torch.rand(batch_size, 1, device=self.device)

        if cfg.type == "geometric":
            sigma = cfg.sigma_min * (cfg.sigma_max / max(cfg.sigma_min, 1e-6)) ** u
        elif cfg.type == "linear":
            sigma = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * u
        elif cfg.type == "polynomial":
            sigma = (
                cfg.sigma_min ** cfg.power
                + u * (cfg.sigma_max ** cfg.power - cfg.sigma_min ** cfg.power)
            ) ** (1.0 / cfg.power)
        else:
            sigma = torch.full_like(u, cfg.sigma_max)

        sigma = sigma.clamp(min=1e-6)
        return sigma

    def _generate_actions(
        self,
        policy: DiffusionPolicy,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_dim = self._action_low_tensor.shape[0]
        current = torch.randn(batch_size, action_dim, device=self.device)
        current = current * float(self.config.dem_prior_std)

        steps = max(int(self.config.dem_num_integration_steps), 1)
        time_horizon = float(self.config.dem_time_range)
        step_size = time_horizon / steps

        for i in range(steps):
            t = time_horizon - i * step_size
            sigma = torch.full((batch_size, 1), t, device=self.device)
            predicted_noise = policy(state, current, sigma)
            current = current - step_size * self.config.dem_diffusion_scale * predicted_noise
            current = current.clamp(self._action_low_tensor, self._action_high_tensor)

        if not deterministic:
            exploration = float(self.config.dem_diffusion_scale)
            if exploration > 0.0:
                current = current + torch.randn_like(current) * exploration * 0.05

        return current.clamp(self._action_low_tensor, self._action_high_tensor)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data * tau)


__all__ = ["DEMAgent"]