from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class ExplorationSchedule:
    type: str = "constant"
    initial: float = 1.0
    final: Optional[float] = None
    steps: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplorationSchedule":
        if data is None:
            return cls()
        return cls(
            type=data.get("type", "constant"),
            initial=float(data.get("initial", 1.0)),
            final=(None if data.get("final") is None else float(data["final"])),
            steps=(None if data.get("steps") is None else int(data["steps"]))
        )


@dataclass
class NoiseScheduleConfig:
    type: str = "geometric"
    beta: float = 1.0
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    power: float = 2.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NoiseScheduleConfig":
        if data is None:
            return cls()
        return cls(
            type=data.get("type", "geometric"),
            beta=float(data.get("beta", 1.0)),
            sigma_min=float(data.get("sigma_min", 0.1)),
            sigma_max=float(data.get("sigma_max", 1.0)),
            power=float(data.get("power", 2.0)),
        )


@dataclass
class DEMAgentConfig:
    hidden_size: int
    actor_hidden_layers: Tuple[int, ...]
    actor_lr: float
    critic_lr: float
    safety_critic_lr: float
    exploration_noise_std: float
    noise_clip: float
    policy_target_update_frequency: int
    critic_target_update_frequency: int
    tau: float
    qc_ens_size: int
    k: float
    c: float
    min_pool_size: int
    num_train_repeat: int
    max_train_repeat_per_step: int
    policy_train_batch_size: int
    init_exploration_steps: int
    replay_size: int
    dem_action_noise_decay: float
    dem_action_noise_min: float
    dem_log_std_clip: Tuple[float, float]
    dem_use_entropy_regularization: bool
    dem_entropy_coef: float
    dem_lam_lr: float
    dem_actor_weight_decay: float
    dem_exploration_schedule: ExplorationSchedule
    dem_noise_schedule: NoiseScheduleConfig
    dem_score_hidden_size: int
    dem_score_hidden_layers: int
    dem_score_time_layers: int
    dem_score_lr: float
    dem_num_integration_steps: int
    dem_time_range: float
    dem_prior_std: float
    dem_diffusion_scale: float
    dem_eval_diffusion_scale: float
    dem_lambda_epsilon: float
    dem_negative_time: bool
    dem_negative_time_steps: int
    dem_energy_regularization: float
    dem_action_penalty: float
    device: Optional[str]
    grad_clip_norm: Optional[float]
    policy_update_delay: int
    deterministic_eval: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DEMAgentConfig":
        schedule = ExplorationSchedule.from_dict(data.get("dem_exploration_schedule"))
        noise_schedule = NoiseScheduleConfig.from_dict(data.get("dem_noise_schedule"))
        log_std_clip = data.get("dem_log_std_clip", [-5.0, 2.0])
        return cls(
            hidden_size=int(data.get("hidden_size", 256)),
            actor_hidden_layers=tuple(data.get("actor_hidden_layers", [256, 256])),
            actor_lr=float(data.get("actor_lr", 3e-4)),
            critic_lr=float(data.get("critic_lr", 3e-4)),
            safety_critic_lr=float(data.get("safety_critic_lr", 3e-4)),
            exploration_noise_std=float(data.get("exploration_noise_std", 0.2)),
            noise_clip=float(data.get("noise_clip", 0.5)),
            policy_target_update_frequency=int(data.get("policy_target_update_frequency", 2)),
            critic_target_update_frequency=int(data.get("critic_target_update_frequency", 2)),
            tau=float(data.get("tau", 0.005)),
            qc_ens_size=int(data.get("qc_ens_size", 4)),
            k=float(data.get("k", 1.0)),
            c=float(data.get("c", 10.0)),
            min_pool_size=int(data.get("min_pool_size", 1000)),
            num_train_repeat=int(data.get("num_train_repeat", 10)),
            max_train_repeat_per_step=int(data.get("max_train_repeat_per_step", 5)),
            policy_train_batch_size=int(data.get("policy_train_batch_size", 256)),
            init_exploration_steps=int(data.get("init_exploration_steps", 5000)),
            replay_size=int(data.get("replay_size", 1_000_000)),
            dem_action_noise_decay=float(data.get("dem_action_noise_decay", 1.0)),
            dem_action_noise_min=float(data.get("dem_action_noise_min", 0.05)),
            dem_log_std_clip=tuple(float(v) for v in log_std_clip),
            dem_use_entropy_regularization=bool(data.get("dem_use_entropy_regularization", False)),
            dem_entropy_coef=float(data.get("dem_entropy_coef", 0.0)),
            dem_lam_lr=float(data.get("dem_lam_lr", data.get("actor_lr", 3e-4))),
            dem_actor_weight_decay=float(data.get("dem_actor_weight_decay", 0.0)),
            dem_exploration_schedule=schedule,
            dem_noise_schedule=noise_schedule,
            dem_score_hidden_size=int(data.get("dem_score_hidden_size", data.get("hidden_size", 256))),
            dem_score_hidden_layers=int(data.get("dem_score_hidden_layers", 3)),
            dem_score_time_layers=int(data.get("dem_score_time_layers", 2)),
            dem_score_lr=float(data.get("dem_score_lr", data.get("actor_lr", 3e-4))),
            dem_num_integration_steps=int(data.get("dem_num_integration_steps", 32)),
            dem_time_range=float(data.get("dem_time_range", 1.0)),
            dem_prior_std=float(data.get("dem_prior_std", 1.0)),
            dem_diffusion_scale=float(data.get("dem_diffusion_scale", 1.0)),
            dem_eval_diffusion_scale=float(data.get("dem_eval_diffusion_scale", 0.0)),
            dem_lambda_epsilon=float(data.get("dem_lambda_epsilon", 1e-3)),
            dem_negative_time=bool(data.get("dem_negative_time", False)),
            dem_negative_time_steps=int(data.get("dem_negative_time_steps", 0)),
            dem_energy_regularization=float(data.get("dem_energy_regularization", 0.0)),
            dem_action_penalty=float(data.get("dem_action_penalty", 0.0)),
            device=data.get("device"),
            grad_clip_norm=(None if data.get("grad_clip_norm") is None else float(data["grad_clip_norm"])),
            policy_update_delay=int(data.get("policy_update_delay", 1)),
            deterministic_eval=bool(data.get("deterministic_eval", True)),
        )


def load_dem_config(env_name: str, config_name: str = "safetygym.yaml") -> DEMAgentConfig:
    """Load DEM agent configuration for a specific environment."""
    config_path = Path(__file__).resolve().parent / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"DEM configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    general = raw_cfg.get("general", {})
    environments = raw_cfg.get("environments", {})
    env_specific = environments.get(env_name, {})

    merged: Dict[str, Any] = {**general, **env_specific}
    return DEMAgentConfig.from_dict(merged)


def apply_dem_config_to_args(args: Any, dem_config: DEMAgentConfig) -> None:
    """Mutate an argparse.Namespace with DEM-specific configuration values."""
    for field_name in dem_config.__dataclass_fields__:
        value = getattr(dem_config, field_name)
        setattr(args, field_name, value)

    # store config for later reference
    setattr(args, "dem_config", dem_config)
