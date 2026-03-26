#!/usr/bin/env python3
"""
Boundary multimodality analysis for saved ALGD (v5) checkpoints.

Example:
python boundary_multimodality.py \
  --env PointButton2 \
  --results_folder results/Safexp-PointButton2-v0/pointbutton2_algd/2025-12-06_15-07_seed3240/
"""

import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import gym
import numpy as np
import torch
import safety_gym  # noqa: F401, required for Safety-Gym env registration

from agents.algd.algd_v5 import ALGDAgent
from env.constraints import get_threshold


ENV_ALIASES = {
    "pointbutton2": "Safexp-PointButton2-v0",
    "carbutton2": "Safexp-CarButton2-v0",
    "pointpush1": "Safexp-PointPush1-v0",
    "safexp-pointbutton2-v0": "Safexp-PointButton2-v0",
    "safexp-carbutton2-v0": "Safexp-CarButton2-v0",
    "safexp-pointpush1-v0": "Safexp-PointPush1-v0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Boundary multimodality for saved ALGD checkpoints")
    parser.add_argument("--env", type=str, required=True, help="PointButton2 / CarButton2 / PointPush1 (or full env id)")
    parser.add_argument("--results_folder", type=str, required=True, help="Folder containing actor_.pth/critics_.pth/safetycritics_.pth")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_boundary_states", type=int, default=300)
    parser.add_argument("--max_env_steps", type=int, default=200000)
    parser.add_argument("--boundary_margin", type=float, default=1.0, help="near-boundary: |Qc_risk - threshold| <= margin")

    parser.add_argument("--actions_per_state", type=int, default=64)
    parser.add_argument("--kmeans_iters", type=int, default=20)
    parser.add_argument("--min_cluster_frac", type=float, default=0.2)
    parser.add_argument("--separation_coef", type=float, default=2.0)

    # Model hyperparameters used for building ALGDAgent (must match training).
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--qc_ens_size", type=int, default=4)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--safety_gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--qc_lr", type=float, default=3e-4)
    parser.add_argument("--critic_target_update_frequency", type=int, default=2)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--intrgt_max", action="store_true", default=False)

    # ALGD v5 extras
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--diffusion_T", type=int, default=5)
    parser.add_argument("--actor_loss_coef", type=float, default=1.0)
    parser.add_argument("--score_coef", type=float, default=0.1)
    parser.add_argument("--score_mc_samples", type=int, default=4)
    parser.add_argument("--score_sigma_scale", type=float, default=1.0)
    parser.add_argument("--score_beta", type=float, default=1.0)
    parser.add_argument("--use_aug_lag", action="store_true", default=True)
    parser.add_argument("--guidance_scale", type=float, default=0.05)
    parser.add_argument("--guidance_normalize", action="store_true", default=True)
    parser.add_argument("--profile_score_mc", action="store_true", default=False)
    parser.add_argument("--profile_warmup", type=int, default=50)
    parser.add_argument("--profile_every", type=int, default=1)

    return parser.parse_args()


def canonical_env_name(env_arg: str) -> str:
    key = env_arg.strip().lower()
    if key in ENV_ALIASES:
        return ENV_ALIASES[key]
    if env_arg.startswith("Safexp-") and env_arg.endswith("-v0"):
        return env_arg
    raise ValueError(f"Unsupported --env={env_arg}. Use PointButton2/CarButton2/PointPush1 or full Safexp-*-v0.")


def to_agent_args(cli: argparse.Namespace, env_name: str) -> SimpleNamespace:
    # Keep only fields ALGDAgent reads.
    return SimpleNamespace(
        gamma=cli.gamma,
        safety_gamma=cli.safety_gamma,
        tau=cli.tau,
        c=cli.c,
        rho=cli.rho,
        diffusion_T=cli.diffusion_T,
        actor_loss_coef=cli.actor_loss_coef,
        score_coef=cli.score_coef,
        score_mc_samples=cli.score_mc_samples,
        score_sigma_scale=cli.score_sigma_scale,
        score_beta=cli.score_beta,
        use_aug_lag=cli.use_aug_lag,
        hidden_size=cli.hidden_size,
        qc_ens_size=cli.qc_ens_size,
        guidance_scale=cli.guidance_scale,
        guidance_normalize=cli.guidance_normalize,
        lr=cli.lr,
        qc_lr=cli.qc_lr,
        k=cli.k,
        M=cli.M,
        intrgt_max=cli.intrgt_max,
        safetygym=True,
        epoch_length=400,
        critic_target_update_frequency=cli.critic_target_update_frequency,
        cost_lim=get_threshold(env_name, constraint="safetygym"),
        profile_score_mc=cli.profile_score_mc,
        profile_warmup=cli.profile_warmup,
        profile_every=cli.profile_every,
        env_name=env_name,
        seed=cli.seed,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kmeans2(actions: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Very light k=2 kmeans, returns (centers[2,act_dim], labels[N])."""
    n = actions.shape[0]
    idx0, idx1 = np.random.choice(n, size=2, replace=False)
    centers = np.stack([actions[idx0], actions[idx1]], axis=0)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d0 = np.sum((actions - centers[0]) ** 2, axis=1)
        d1 = np.sum((actions - centers[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(2):
            mask = labels == k
            if np.any(mask):
                centers[k] = actions[mask].mean(axis=0)
            else:
                # Re-init empty cluster to a random point
                centers[k] = actions[np.random.randint(0, n)]

    return centers, labels


def is_multimodal(
    actions: np.ndarray,
    min_cluster_frac: float,
    separation_coef: float,
    kmeans_iters: int,
) -> bool:
    centers, labels = kmeans2(actions, iters=kmeans_iters)
    n = actions.shape[0]

    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    min_n = int(np.ceil(min_cluster_frac * n))
    if n0 < min_n or n1 < min_n:
        return False

    a0 = actions[labels == 0]
    a1 = actions[labels == 1]

    var0 = float(np.mean(np.sum((a0 - centers[0]) ** 2, axis=1))) if len(a0) > 0 else 0.0
    var1 = float(np.mean(np.sum((a1 - centers[1]) ** 2, axis=1))) if len(a1) > 0 else 0.0
    center_dist = float(np.linalg.norm(centers[0] - centers[1]))

    within_scale = float(np.sqrt(var0 + var1 + 1e-12))
    return center_dist > separation_coef * within_scale


@torch.no_grad()
def qc_risk(agent: ALGDAgent, s_np: np.ndarray, a_np: np.ndarray) -> float:
    s = torch.as_tensor(s_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
    a = torch.as_tensor(a_np, dtype=torch.float32, device=agent.device).unsqueeze(0)

    qcs = agent.safety_critics(s, a)  # [E,1,1]
    qc_std, qc_mean = torch.std_mean(qcs, dim=0)
    if agent.args.qc_ens_size == 1:
        qc_std = torch.zeros_like(qc_mean)
    risk = qc_mean + agent.args.k * qc_std
    return float(risk.squeeze().item())


@torch.no_grad()
def sample_actions(agent: ALGDAgent, s_np: np.ndarray, n: int) -> np.ndarray:
    s = torch.as_tensor(s_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
    s_rep = s.repeat(n, 1)
    a = agent.policy.sample(s_rep)
    return a.detach().cpu().numpy()


def collect_near_boundary_states(
    env,
    agent: ALGDAgent,
    threshold: float,
    margin: float,
    num_states: int,
    max_env_steps: int,
) -> List[np.ndarray]:
    states = []  # type: List[np.ndarray]

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    steps = 0
    while steps < max_env_steps and len(states) < num_states:
        action = agent.select_action(obs, eval=False)
        risk = qc_risk(agent, obs, action)
        if abs(risk - threshold) <= margin:
            states.append(np.array(obs, dtype=np.float32, copy=True))

        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_obs, _, done, _ = step_out

        obs = next_obs
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

        steps += 1

    return states


def main() -> None:
    cli = parse_args()
    env_name = canonical_env_name(cli.env)

    results_dir = Path(cli.results_folder)
    actor_path = results_dir / "actor_.pth"
    critics_path = results_dir / "critics_.pth"
    safetycritics_path = results_dir / "safetycritics_.pth"

    missing = [str(p) for p in [actor_path, critics_path, safetycritics_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))

    set_seed(cli.seed)

    env = gym.make(env_name)
    try:
        env.reset(seed=cli.seed)
    except TypeError:
        env.seed(cli.seed)

    args = to_agent_args(cli, env_name)

    obs_dim = env.observation_space.shape[0]
    agent = ALGDAgent(obs_dim, env.action_space, args)
    agent.load_model(str(actor_path), str(critics_path), str(safetycritics_path))
    agent.train(False)

    threshold = float(args.cost_lim)

    near_states = collect_near_boundary_states(
        env=env,
        agent=agent,
        threshold=threshold,
        margin=cli.boundary_margin,
        num_states=cli.num_boundary_states,
        max_env_steps=cli.max_env_steps,
    )

    if len(near_states) == 0:
        print("Boundary multimodality: 0.00% (0/0) -- no near-boundary states found.")
        return

    multimodal_count = 0
    for s in near_states:
        acts = sample_actions(agent, s, n=cli.actions_per_state)
        if is_multimodal(
            acts,
            min_cluster_frac=cli.min_cluster_frac,
            separation_coef=cli.separation_coef,
            kmeans_iters=cli.kmeans_iters,
        ):
            multimodal_count += 1

    ratio = 100.0 * multimodal_count / len(near_states)
    print(f"Boundary multimodality: {ratio:.2f}% ({multimodal_count}/{len(near_states)})")


if __name__ == "__main__":
    main()
