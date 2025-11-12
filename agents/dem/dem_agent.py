"""Implementation of a lightweight DEM agent compatible with the project training loop.

This agent adapts the core ideas of DEM (energy matching with deterministic policies)
to the on-policy interaction interface used throughout the repository.  The original
DEM code base is organised around Lightning modules and Hydra configs which are not
directly compatible with the replay-based training loop employed here.  The goal of
this module is therefore to provide a self-contained PyTorch agent that mirrors the
expected `Agent` interface (`select_action`, `update_parameters`, `train`, etc.) while
loading all hyper-parameters from the dedicated DEM configuration dataclass.

Only reward optimisation is considered in this integration as Safety-Gym specific
cost handling is outside the scope of the original DEM algorithm.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.dem.config import DEMAgentConfig, ExplorationSchedule


def build_mlp(input_dim: int, hidden_layers: Iterable[int], output_dim: int) -> nn.Sequential:
    """Create a simple feed-forward network with ReLU activations."""

    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy with squashing to the environment action bounds."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Iterable[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.net = build_mlp(state_dim, hidden_layers, action_dim)

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        self.register_buffer("action_scale", torch.from_numpy(action_scale))
        self.register_buffer("action_bias", torch.from_numpy(action_bias))

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raw_action = self.net(state)
        squashed = torch.tanh(raw_action)
        return squashed * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Twin-critic architecture (TD3 style)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: Iterable[int]) -> None:
        super().__init__()
        self.q1 = build_mlp(state_dim + action_dim, hidden_layers, 1)
        self.q2 = build_mlp(state_dim + action_dim, hidden_layers, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        q1 = self.q1(xu)
        q2 = self.q2(xu)
        return q1, q2

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu)


class DEMAgent(Agent):
    """Replay-based DEM agent that follows the generic agent interface."""

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

        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = float(self.config.exploration_noise_std)
        self.noise_clip = float(self.config.noise_clip)
        self.policy_update_delay = int(self.config.policy_update_delay)
        self.critic_target_update_frequency = int(self.config.critic_target_update_frequency)
        self.total_it = 0
        self.env_steps = 0

        action_dim = action_space.shape[0]
        hidden_layers = tuple(self.config.actor_hidden_layers) or (self.config.hidden_size, self.config.hidden_size)
        critic_hidden_layers = hidden_layers or (self.config.hidden_size, self.config.hidden_size)

        self.actor = Actor(num_inputs, action_dim, hidden_layers, action_space.low, action_space.high).to(self.device)
        self.actor_target = Actor(num_inputs, action_dim, hidden_layers, action_space.low, action_space.high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(num_inputs, action_dim, critic_hidden_layers).to(self.device)
        self.critic_target = Critic(num_inputs, action_dim, critic_hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            weight_decay=self.config.dem_actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        self.grad_clip_norm: Optional[float] = self.config.grad_clip_norm

        self.deterministic_eval = bool(self.config.deterministic_eval)

        # Pre-compute numpy bounds for clipping actions.
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
        self.actor.train(training)
        self.critic.train(training)
        self.actor_target.train(training)
        self.critic_target.train(training)

    # For compatibility with samplers
    def select_action(self, state, eval_t: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        action_np = action.cpu().numpy()[0]

        if not eval_t:
            noise_std = self._compute_exploration_std()
            if noise_std > 0.0:
                action_np += np.random.normal(0.0, noise_std, size=action_np.shape)
            self.env_steps += 1
        elif not self.deterministic_eval:
            # Optional evaluation noise controlled via configuration.
            noise_std = self.config.dem_eval_diffusion_scale
            if noise_std > 0.0:
                action_np += np.random.normal(0.0, noise_std, size=action_np.shape)

        return np.clip(action_np, self._action_low, self._action_high)

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
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(self._action_low_tensor, self._action_high_tensor)

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
            actor_action = self.actor(state)
            actor_loss = -self.critic.q1_only(state, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

        if self.total_it % self.critic_target_update_frequency == 0:
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.actor_target, self.actor)

    def save_model(self, save_dir, suffix: str = ""):
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        actor_path = save_dir / f"dem_actor{('_' + suffix) if suffix else ''}.pth"
        critic_path = save_dir / f"dem_critic{('_' + suffix) if suffix else ''}.pth"
        config_path = save_dir / f"dem_config{('_' + suffix) if suffix else ''}.yaml"

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        try:
            import yaml

            with config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(asdict(self.config), fh)
        except Exception:
            # Fallback: store configuration as torch object if YAML unavailable
            torch.save(self.config, config_path.with_suffix(".pt"))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _compute_exploration_std(self) -> float:
        schedule: ExplorationSchedule = self.config.dem_exploration_schedule
        initial = float(schedule.initial)

        if schedule.type == "linear" and schedule.final is not None and schedule.steps:
            progress = min(self.env_steps / max(schedule.steps, 1), 1.0)
            return float(initial + progress * (schedule.final - initial))
        if schedule.type == "exponential" and schedule.final is not None and schedule.steps:
            progress = min(self.env_steps / max(schedule.steps, 1), 1.0)
            ratio = (schedule.final / initial) if initial > 0 else 1.0
            return float(initial * (ratio ** progress))

        return initial

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


__all__ = ["DEMAgent"]
