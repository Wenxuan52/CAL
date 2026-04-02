#!/usr/bin/env python3
"""
Post-hoc analysis for ONE ALGD checkpoint under ONE rho setting.

This script:
1) Loads one ALGD checkpoint triplet (actor/critic/safety critic).
2) Rolls out env and collects near-boundary / away-boundary states.
3) For each state, samples one action and computes curvature-related metrics:
   - lambda_min_hess_L
   - lambda_min_hess_LA
   - rho_grad_Qc_norm_sq
   - kappa_L
   - kappa_A
   - dom_ratio
4) Computes multimodality on both regions with kmeans2-style test.
5) Saves per-state CSV + summary CSV and prints concise terminal summary.

No argparse is used. Edit USER CONFIG section only.
"""

import csv
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import gym
import numpy as np
import torch
import safety_gym  # noqa: F401, required for Safety-Gym env registration

from agents.algd.algd_v5 import ALGDAgent
from env.constraints import get_threshold


# =============================================================================
# USER CONFIG (EDIT HERE)
# =============================================================================

ENV_NAME = "Safexp-PointButton1-v0"
RESULTS_FOLDER = "results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.5/2025-12-31_11-28_seed8477"
SEED = 8477
RHO = 0.5

NUM_NEAR_STATES = 50
NUM_AWAY_STATES = 50
MAX_ENV_STEPS = 250000
BOUNDARY_MARGIN = 0.05

ACTIONS_PER_STATE = 128
KMEANS_ITERS = 20
KMEANS_RESTARTS = 10
MIN_CLUSTER_FRAC = 0.15
SEPARATION_COEF = 1.75

# Agent hparams (match training/checkpoint as closely as possible)
HIDDEN_SIZE = 256
QC_ENS_SIZE = 4
K = 1.0
M = 4
GAMMA = 0.99
SAFETY_GAMMA = 0.99
TAU = 0.005
LR = 3e-4
QC_LR = 3e-4
CRITIC_TARGET_UPDATE_FREQUENCY = 2
C = 10.0
DIFFUSION_T = 5
ACTOR_LOSS_COEF = 1.0
SCORE_COEF = 0.1
SCORE_MC_SAMPLES = 4
SCORE_SIGMA_SCALE = 1.0
SCORE_BETA = 1.0
USE_AUG_LAG = True
GUIDANCE_SCALE = 0.05
GUIDANCE_NORMALIZE = True

# output folder/name prefix (saved inside {REPO_ROOT}/temp_results)
OUTPUT_DIRNAME = "temp_results_"
PER_STATE_CSV_PREFIX = "algd_rho_curvature_multimodality_per_state"
SUMMARY_CSV_PREFIX = "algd_rho_curvature_multimodality_summary"


# =============================================================================
# Utilities (style-aligned with boundary_multimodality.py)
# =============================================================================

ENV_ALIASES = {
    "pointbutton2": "Safexp-PointButton2-v0",
    "carbutton2": "Safexp-CarButton2-v0",
    "pointpush1": "Safexp-PointPush1-v0",
    "safexp-pointbutton2-v0": "Safexp-PointButton2-v0",
    "safexp-carbutton2-v0": "Safexp-CarButton2-v0",
    "safexp-pointpush1-v0": "Safexp-PointPush1-v0",
}


def canonical_env_name(env_arg: str) -> str:
    key = env_arg.strip().lower()
    if key in ENV_ALIASES:
        return ENV_ALIASES[key]
    if env_arg.startswith("Safexp-") and env_arg.endswith("-v0"):
        return env_arg
    raise ValueError(f"Unsupported env name: {env_arg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed: int = None):
    if seed is None:
        obs = env.reset()
    else:
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()

    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


def step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        next_obs, reward, done, info = step_out

    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    return next_obs, reward, done, info


def get_attr_first(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Could not find any of attributes {names} in {type(obj).__name__}")


def safe_eval_mode(agent) -> None:
    try:
        agent.train(False)
    except Exception:
        pass

    for attr_name in ["policy", "actor", "critic", "critics", "safety_critics", "safety_critic"]:
        if hasattr(agent, attr_name):
            try:
                getattr(agent, attr_name).eval()
            except Exception:
                pass


def get_device(agent) -> torch.device:
    if hasattr(agent, "device"):
        return torch.device(agent.device)
    policy = get_attr_first(agent, ["policy", "actor"])
    return next(policy.parameters()).device


def make_agent_args(env_id: str, seed: int, qc_ens_size: int) -> SimpleNamespace:
    merged = dict(
        safetygym=True,
        epoch_length=400,
        cost_lim=get_threshold(env_id, constraint="safetygym"),
        env_name=env_id,
        seed=seed,
        hidden_size=HIDDEN_SIZE,
        qc_ens_size=qc_ens_size,
        k=K,
        M=min(M, qc_ens_size),
        gamma=GAMMA,
        safety_gamma=SAFETY_GAMMA,
        tau=TAU,
        lr=LR,
        qc_lr=QC_LR,
        critic_target_update_frequency=CRITIC_TARGET_UPDATE_FREQUENCY,
        c=C,
        intrgt_max=False,
        rho=RHO,
        diffusion_T=DIFFUSION_T,
        actor_loss_coef=ACTOR_LOSS_COEF,
        score_coef=SCORE_COEF,
        score_mc_samples=SCORE_MC_SAMPLES,
        score_sigma_scale=SCORE_SIGMA_SCALE,
        score_beta=SCORE_BETA,
        use_aug_lag=USE_AUG_LAG,
        guidance_scale=GUIDANCE_SCALE,
        guidance_normalize=GUIDANCE_NORMALIZE,
        profile_score_mc=False,
        profile_warmup=50,
        profile_every=1,
    )
    return SimpleNamespace(**merged)


def infer_qc_ensemble_size_from_checkpoint(safety_ckpt_path: Path) -> int:
    state = torch.load(safety_ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        return 1

    for key in ("nn1.weight", "module.nn1.weight"):
        if key in state and hasattr(state[key], "shape") and len(state[key].shape) >= 1:
            return int(state[key].shape[0])
    return 1


def resolve_checkpoint_file(results_folder: Path, patterns: Sequence[str], kind: str) -> Path:
    folder = Path(results_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Results folder does not exist: {folder}")

    for pat in patterns:
        p = folder / pat
        if p.exists():
            return p

    matches = []
    for pat in patterns:
        matches.extend(sorted(folder.glob(pat)))

    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: (len(x.name), -x.stat().st_mtime))
        return matches[0]

    raise FileNotFoundError(f"Could not find {kind} checkpoint in {folder}")


def find_checkpoint_triplet(results_folder: str) -> Tuple[Path, Path, Path]:
    folder = Path(results_folder)
    actor_path = resolve_checkpoint_file(
        folder,
        patterns=["actor_.pth", "actor.pth", "policy_.pth", "policy.pth", "actor*.pth", "policy*.pth"],
        kind="actor/policy",
    )
    critic_path = resolve_checkpoint_file(
        folder,
        patterns=["critics_.pth", "critic_.pth", "critics.pth", "critic.pth", "qf_.pth", "qf.pth", "critic*.pth", "critics*.pth"],
        kind="critic",
    )
    safety_path = resolve_checkpoint_file(
        folder,
        patterns=[
            "safetycritics_.pth",
            "safety_critics_.pth",
            "safetycritic_.pth",
            "safetycritics.pth",
            "safety_critics.pth",
            "safetycritic.pth",
            "safetycritic*.pth",
            "safety_critic*.pth",
            "safetycritics*.pth",
        ],
        kind="safety critic",
    )
    return actor_path, critic_path, safety_path


def load_agent_checkpoints(agent, results_folder: str) -> Tuple[Path, Path, Path]:
    actor_path, critic_path, safety_path = find_checkpoint_triplet(results_folder)

    if hasattr(agent, "load_model"):
        for cast_to_str in [True, False]:
            try:
                a = str(actor_path) if cast_to_str else actor_path
                c = str(critic_path) if cast_to_str else critic_path
                s = str(safety_path) if cast_to_str else safety_path
                agent.load_model(a, c, s)
                safe_eval_mode(agent)
                return actor_path, critic_path, safety_path
            except TypeError:
                pass
            except Exception:
                break

    device = get_device(agent)
    actor_state = torch.load(actor_path, map_location=device)
    critic_state = torch.load(critic_path, map_location=device)
    safety_state = torch.load(safety_path, map_location=device)

    get_attr_first(agent, ["policy", "actor"]).load_state_dict(actor_state, strict=True)
    get_attr_first(agent, ["critic", "critics"]).load_state_dict(critic_state, strict=True)
    get_attr_first(agent, ["safety_critics", "safety_critic", "safetycritic"]).load_state_dict(safety_state, strict=True)
    safe_eval_mode(agent)
    return actor_path, critic_path, safety_path


def reduce_qc_ensemble(qcs: torch.Tensor, ens_size: int, batch_n: int) -> torch.Tensor:
    qcs = torch.as_tensor(qcs)

    if qcs.ndim == 3:
        if qcs.shape[0] == ens_size:
            return qcs
        if qcs.shape[1] == ens_size:
            return qcs.transpose(0, 1)
    elif qcs.ndim == 2:
        if qcs.shape[0] == ens_size:
            return qcs.unsqueeze(-1)
        if qcs.shape[1] == ens_size:
            return qcs.transpose(0, 1).unsqueeze(-1)
    elif qcs.ndim == 1:
        return qcs.view(1, batch_n, 1)

    if qcs.numel() % max(1, ens_size * batch_n) == 0:
        trailing = qcs.numel() // (ens_size * batch_n)
        return qcs.view(ens_size, batch_n, trailing)

    raise ValueError(f"Unsupported safety critic output shape: {tuple(qcs.shape)}")


def extract_actions_from_sample_output(sample_out: Any, batch_n: int) -> np.ndarray:
    candidate = sample_out
    if isinstance(sample_out, (tuple, list)):
        tensor_candidates = [x for x in sample_out if isinstance(x, (torch.Tensor, np.ndarray))]
        if not tensor_candidates:
            raise ValueError("policy.sample returned a tuple/list without tensor/array actions")
        candidate = None
        for x in tensor_candidates:
            x0 = np.asarray(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
            if x0.ndim >= 1 and x0.shape[0] == batch_n:
                candidate = x
                break
        if candidate is None:
            candidate = tensor_candidates[0]

    acts = candidate.detach().cpu().numpy() if isinstance(candidate, torch.Tensor) else np.asarray(candidate)
    if acts.ndim == 1:
        acts = acts[None, :]
    return acts.astype(np.float32, copy=False)


@torch.no_grad()
def sample_actions(agent, s_np: np.ndarray, n: int) -> np.ndarray:
    device = get_device(agent)
    s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
    s_rep = s.repeat(n, 1)

    if hasattr(agent, "policy") and hasattr(agent.policy, "sample"):
        try:
            return extract_actions_from_sample_output(agent.policy.sample(s_rep), batch_n=n)
        except Exception:
            pass

    if hasattr(agent, "actor") and hasattr(agent.actor, "sample"):
        try:
            return extract_actions_from_sample_output(agent.actor.sample(s_rep), batch_n=n)
        except Exception:
            pass

    acts = []
    for _ in range(n):
        if hasattr(agent, "select_action"):
            try:
                a = agent.select_action(s_np, eval=False)
            except TypeError:
                a = agent.select_action(s_np)
            acts.append(np.asarray(a, dtype=np.float32))
        else:
            raise AttributeError(f"{type(agent).__name__} has neither policy.sample nor select_action")
    return np.stack(acts, axis=0)


def select_rollout_action(agent, obs: np.ndarray) -> np.ndarray:
    if hasattr(agent, "select_action"):
        try:
            return np.asarray(agent.select_action(obs, eval=False), dtype=np.float32)
        except TypeError:
            try:
                return np.asarray(agent.select_action(obs), dtype=np.float32)
            except Exception:
                pass
    return sample_actions(agent, obs, n=1)[0]


@torch.no_grad()
def qc_risk(agent, s_np: np.ndarray, a_np: np.ndarray) -> float:
    if a_np.ndim == 1:
        a_np = a_np[None, :]

    device = get_device(agent)
    n = a_np.shape[0]
    s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0).repeat(n, 1)
    a = torch.as_tensor(a_np, dtype=torch.float32, device=device)

    safety_model = get_attr_first(agent, ["safety_critics", "safety_critic", "safetycritic"])
    qcs = safety_model(s, a)
    if isinstance(qcs, (tuple, list)):
        qcs = qcs[0]

    qcs = reduce_qc_ensemble(qcs, ens_size=getattr(agent.args, "qc_ens_size", 1), batch_n=n)
    qc_std, qc_mean = torch.std_mean(qcs, dim=0)
    if getattr(agent.args, "qc_ens_size", 1) == 1:
        qc_std = torch.zeros_like(qc_mean)

    risk = qc_mean + getattr(agent.args, "k", 1.0) * qc_std
    return float(risk.view(-1)[0].item())


def kmeans2_once(actions: np.ndarray, iters: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
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

        for k_idx in range(2):
            mask = labels == k_idx
            if np.any(mask):
                centers[k_idx] = actions[mask].mean(axis=0)
            else:
                centers[k_idx] = actions[np.random.randint(0, n)]

    d0 = np.sum((actions - centers[0]) ** 2, axis=1)
    d1 = np.sum((actions - centers[1]) ** 2, axis=1)
    labels = (d1 < d0).astype(np.int64)
    inertia = float(np.sum(np.where(labels == 0, d0, d1)))
    return centers, labels, inertia


def kmeans2_best(actions: np.ndarray, iters: int = 20, restarts: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    best_centers = None
    best_labels = None
    best_inertia = None

    for _ in range(restarts):
        centers, labels, inertia = kmeans2_once(actions, iters=iters)
        if best_inertia is None or inertia < best_inertia:
            best_centers = centers
            best_labels = labels
            best_inertia = inertia

    return best_centers, best_labels


def is_multimodal(
    actions: np.ndarray,
    min_cluster_frac: float,
    separation_coef: float,
    kmeans_iters: int,
    kmeans_restarts: int,
) -> bool:
    centers, labels = kmeans2_best(actions, iters=kmeans_iters, restarts=kmeans_restarts)
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


def sanitize_float(x: Any, fallback: float = float("nan")) -> float:
    try:
        y = float(x)
    except Exception:
        return fallback
    if not math.isfinite(y):
        return fallback
    return y


def get_dual_lambda(agent) -> torch.Tensor:
    """
    Try to retrieve ALGD dual variable lambda from common attribute names.

    If attribute looks like log-lambda, automatically exponentiate.
    If not found, raise a clear error for manual patching.

    TODO(user): If your ALGDAgent stores lambda with another name, add it here.
    """
    candidates = [
        "log_lam",
        "log_lambda",
        "lam",
        "lambda_param",
        "lambda_value",
        "dual_lambda",
        "dual_var",
        "log_dual",
    ]

    for name in candidates:
        if not hasattr(agent, name):
            continue
        raw = getattr(agent, name)

        if callable(raw):
            try:
                raw = raw()
            except Exception:
                continue

        if isinstance(raw, torch.Tensor):
            lam_t = raw
        else:
            try:
                lam_t = torch.as_tensor(raw, dtype=torch.float32, device=get_device(agent))
            except Exception:
                continue

        lname = name.lower()
        if "log" in lname and ("lam" in lname or "lambda" in lname):
            lam_t = lam_t.exp()

        return lam_t.reshape(-1)[0]

    raise RuntimeError(
        "Cannot find dual lambda in agent. Please edit get_dual_lambda() and map your agent's dual variable attribute explicitly."
    )


def scalar_hessian_wrt_action(scalar: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Explicit Hessian of a scalar wrt 1D action tensor."""
    if scalar.ndim != 0:
        scalar = scalar.reshape(-1).sum()

    grad = torch.autograd.grad(scalar, action, create_graph=True, retain_graph=True)[0]
    act_dim = action.numel()
    rows = []
    for i in range(act_dim):
        g_i = grad[i]
        second = torch.autograd.grad(g_i, action, create_graph=False, retain_graph=True, allow_unused=False)[0]
        rows.append(second)
    H = torch.stack(rows, dim=0)
    H = 0.5 * (H + H.T)
    return H


def build_loaded_agent(env, env_id: str, seed: int, results_folder: str):
    _, _, safety_path = find_checkpoint_triplet(results_folder)
    inferred_qc_ens_size = infer_qc_ensemble_size_from_checkpoint(safety_path)

    args = make_agent_args(env_id=env_id, seed=seed, qc_ens_size=inferred_qc_ens_size)
    obs_dim = env.observation_space.shape[0]
    agent = ALGDAgent(obs_dim, env.action_space, args)
    safe_eval_mode(agent)

    actor_path, critic_path, safety_path = load_agent_checkpoints(agent, results_folder)
    return agent, actor_path, critic_path, safety_path


def collect_near_and_away_states(
    env,
    agent,
    threshold: float,
    lambda_value: float,
    rho_value: float,
    num_near: int,
    num_away: int,
    max_env_steps: int,
    boundary_margin: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    near_states: List[np.ndarray] = []
    away_states: List[np.ndarray] = []

    obs = reset_env(env, seed=seed)
    steps = 0

    while steps < max_env_steps and (len(near_states) < num_near or len(away_states) < num_away):
        action = select_rollout_action(agent, obs)
        qc_used = qc_risk(agent, obs, action)
        gap = qc_used - threshold
        active_val = lambda_value + rho_value * gap
        # near region uses ACTIVE boundary condition:
        #   abs(Qc_used - h) <= BOUNDARY_MARGIN  and  lambda + rho*(Qc_used - h) > 0
        is_near = abs(gap) <= boundary_margin and active_val > 0.0
        if is_near and len(near_states) < num_near:
            near_states.append(np.array(obs, dtype=np.float32, copy=True))
        # away region uses complement of near condition:
        #   NOT( abs(Qc_used - h) <= BOUNDARY_MARGIN and lambda + rho*(Qc_used - h) > 0 )
        if (not is_near) and len(away_states) < num_away:
            away_states.append(np.array(obs, dtype=np.float32, copy=True))

        next_obs, _, done, _ = step_env(env, action)
        obs = next_obs
        if done:
            obs = reset_env(env)

        steps += 1

    return near_states, away_states, steps


def select_lambda_for_hess_la(eig_l_a: torch.Tensor, region_name: str, eps: float = 1e-12) -> torch.Tensor:
    eig_sorted = torch.sort(eig_l_a.reshape(-1)).values
    if eig_sorted.numel() == 0:
        return torch.as_tensor(float("nan"), dtype=eig_l_a.dtype, device=eig_l_a.device)

    if region_name == "near":
        mid_idx = eig_sorted.numel() // 2
        return eig_sorted[mid_idx]

    nonzero = eig_sorted[torch.abs(eig_sorted) > eps]
    if nonzero.numel() == 0:
        return eig_sorted[0]
    return nonzero[0]


def state_curvature_metrics(agent, s_np: np.ndarray, threshold: float, region_name: str) -> Dict[str, float]:
    device = get_device(agent)
    s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)

    a_np = sample_actions(agent, s_np, n=1)[0]
    a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    a = a.clone().detach().requires_grad_(True)

    s_rep = s
    a_batch = a.unsqueeze(0)

    q1, q2 = agent.critic(s_rep, a_batch)
    q_min = torch.min(q1, q2).reshape(())

    qcs = agent.safety_critics(s_rep, a_batch)
    qcs = reduce_qc_ensemble(qcs, ens_size=getattr(agent.args, "qc_ens_size", 1), batch_n=1)
    qc_std, qc_mean = torch.std_mean(qcs, dim=0)
    if getattr(agent.args, "qc_ens_size", 1) == 1:
        qc_std = torch.zeros_like(qc_mean)
    qc_risk_t = (qc_mean + getattr(agent.args, "k", 1.0) * qc_std).reshape(())

    lam = get_dual_lambda(agent).to(device=device, dtype=torch.float32)
    rho = torch.as_tensor(float(getattr(agent, "rho", RHO)), dtype=torch.float32, device=device)
    h = torch.as_tensor(float(threshold), dtype=torch.float32, device=device)

    # L = -Q + lambda * (Qc - h)
    L = -q_min + lam * (qc_risk_t - h)

    # L_A piecewise active/inactive
    penalty_arg = lam + rho * (qc_risk_t - h)
    if float(penalty_arg.detach().cpu().item()) > 0.0:
        LA = -q_min + (((penalty_arg ** 2) - (lam ** 2)) / (2.0 * rho))
    else:
        LA = -q_min

    grad_qc = torch.autograd.grad(qc_risk_t, a, create_graph=True, retain_graph=True)[0]
    grad_qc_norm = torch.linalg.norm(grad_qc)
    u = grad_qc / (grad_qc_norm + 1e-12)

    hess_L = scalar_hessian_wrt_action(L, a)
    hess_LA = scalar_hessian_wrt_action(LA, a)
    hess_Qc = scalar_hessian_wrt_action(qc_risk_t, a)

    eig_L = torch.linalg.eigvalsh(hess_L)
    eig_LA = torch.linalg.eigvalsh(hess_LA)
    eig_Qc = torch.linalg.eigvalsh(hess_Qc)

    lambda_min_hess_L = eig_L.min()
    lambda_min_hess_LA = select_lambda_for_hess_la(eig_LA, region_name=region_name)
    hess_Qc_opnorm = torch.max(torch.abs(eig_Qc))

    kappa_L = (u @ (hess_L @ u))
    kappa_A = (u @ (hess_LA @ u))

    rho_grad_Qc_norm_sq = rho * (grad_qc_norm ** 2)
    dom_ratio = rho * (grad_qc_norm ** 2) / (rho * torch.abs(qc_risk_t - h) * hess_Qc_opnorm + 1e-12)

    out = {
        "qc_risk": sanitize_float(qc_risk_t.detach().cpu().item()),
        "threshold": sanitize_float(threshold),
        "boundary_gap": sanitize_float((qc_risk_t - h).detach().cpu().item()),
        "abs_qc_minus_h": sanitize_float(torch.abs(qc_risk_t - h).detach().cpu().item()),
        "lambda_value": sanitize_float(lam.detach().cpu().item()),
        "Q_value": sanitize_float(q_min.detach().cpu().item()),
        "Qc_value": sanitize_float(qc_risk_t.detach().cpu().item()),  # risk-form Qc used in ALGD actor objective
        "grad_Qc_norm": sanitize_float(grad_qc_norm.detach().cpu().item()),
        "rho_grad_Qc_norm_sq": sanitize_float(rho_grad_Qc_norm_sq.detach().cpu().item()),
        "lambda_min_hess_L": sanitize_float(lambda_min_hess_L.detach().cpu().item()),
        "lambda_min_hess_LA": sanitize_float(lambda_min_hess_LA.detach().cpu().item()),
        "kappa_L": sanitize_float(kappa_L.detach().cpu().item()),
        "kappa_A": sanitize_float(kappa_A.detach().cpu().item()),
        "hess_Qc_opnorm": sanitize_float(hess_Qc_opnorm.detach().cpu().item()),
        "dom_ratio": sanitize_float(dom_ratio.detach().cpu().item()),
    }
    return out


def summarize_numeric(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def analyze_region(
    region_name: str,
    states: List[np.ndarray],
    agent,
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for i, s in enumerate(states):
        try:
            metrics = state_curvature_metrics(agent, s, threshold=threshold, region_name=region_name)
        except RuntimeError as e:
            # Basic numeric protection: record NaNs for problematic state
            print(f"[WARN] region={region_name} state_idx={i} curvature computation failed: {e}")
            metrics = {
                "qc_risk": float("nan"),
                "threshold": threshold,
                "boundary_gap": float("nan"),
                "abs_qc_minus_h": float("nan"),
                "lambda_value": float("nan"),
                "Q_value": float("nan"),
                "Qc_value": float("nan"),
                "grad_Qc_norm": float("nan"),
                "rho_grad_Qc_norm_sq": float("nan"),
                "lambda_min_hess_L": float("nan"),
                "lambda_min_hess_LA": float("nan"),
                "kappa_L": float("nan"),
                "kappa_A": float("nan"),
                "hess_Qc_opnorm": float("nan"),
                "dom_ratio": float("nan"),
            }

        acts = sample_actions(agent, s, n=ACTIONS_PER_STATE)
        mm_flag = is_multimodal(
            acts,
            min_cluster_frac=MIN_CLUSTER_FRAC,
            separation_coef=SEPARATION_COEF,
            kmeans_iters=KMEANS_ITERS,
            kmeans_restarts=KMEANS_RESTARTS,
        )

        row = {
            "region": region_name,
            "state_idx": i,
            **metrics,
            "is_multimodal": int(mm_flag),
        }
        rows.append(row)

    return rows


def write_per_state_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "region",
        "state_idx",
        "qc_risk",
        "threshold",
        "boundary_gap",
        "abs_qc_minus_h",
        "lambda_value",
        "Q_value",
        "Qc_value",
        "grad_Qc_norm",
        "rho_grad_Qc_norm_sq",
        "lambda_min_hess_L",
        "lambda_min_hess_LA",
        "kappa_L",
        "kappa_A",
        "hess_Qc_opnorm",
        "dom_ratio",
        "is_multimodal",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_cols = [
        "qc_risk",
        "boundary_gap",
        "abs_qc_minus_h",
        "lambda_value",
        "Q_value",
        "Qc_value",
        "grad_Qc_norm",
        "rho_grad_Qc_norm_sq",
        "lambda_min_hess_L",
        "lambda_min_hess_LA",
        "kappa_L",
        "kappa_A",
        "hess_Qc_opnorm",
        "dom_ratio",
    ]

    out = []
    for region in ["near", "away"]:
        r_rows = [r for r in rows if r["region"] == region]
        rec = {
            "region": region,
            "num_states": len(r_rows),
            "multimodality_rate": float(np.mean([r["is_multimodal"] for r in r_rows])) if r_rows else float("nan"),
        }
        for c in numeric_cols:
            mean, std = summarize_numeric([r[c] for r in r_rows])
            rec[f"{c}_mean"] = mean
            rec[f"{c}_std"] = std
        out.append(rec)
    return out


def write_summary_csv(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    env_id = canonical_env_name(ENV_NAME)
    set_seed(SEED)
    repo_root = Path(__file__).resolve().parent

    env = gym.make(env_id)
    reset_env(env, seed=SEED)

    print("=" * 90)
    print("ALGD rho curvature + multimodality post-hoc analysis")
    print("=" * 90)
    print(f"env_id         : {env_id}")
    print(f"results_folder : {RESULTS_FOLDER}")
    print(f"seed           : {SEED}")

    agent, actor_path, critic_path, safety_path = build_loaded_agent(
        env=env,
        env_id=env_id,
        seed=SEED,
        results_folder=RESULTS_FOLDER,
    )
    print(f"checkpoint actor : {actor_path}")
    print(f"checkpoint critic: {critic_path}")
    print(f"checkpoint safety: {safety_path}")

    raw_threshold = float(get_threshold(env_id, constraint="safetygym"))
    args_cost_lim = float(getattr(agent.args, "cost_lim", raw_threshold))
    dual_lambda = float(get_dual_lambda(agent).detach().cpu().item())
    rho_value = float(getattr(agent, "rho", RHO))
    threshold = float(getattr(agent, "target_cost", args_cost_lim))

    print(f"raw get_threshold : {raw_threshold:.6f}")
    print(f"agent.args.cost_lim: {args_cost_lim:.6f}")
    print(f"dual lambda       : {dual_lambda:.6f}")
    print(f"rho               : {rho_value:.6f}")
    print(f"final threshold h : {threshold:.6f}")
    print(
        "state split rules -> "
        "near: abs(Qc_used - h) <= BOUNDARY_MARGIN and lambda + rho*(Qc_used - h) > 0 ; "
        "away: NOT(near)"
    )

    near_states, away_states, used_steps = collect_near_and_away_states(
        env=env,
        agent=agent,
        threshold=threshold,
        lambda_value=dual_lambda,
        rho_value=rho_value,
        num_near=NUM_NEAR_STATES,
        num_away=NUM_AWAY_STATES,
        max_env_steps=MAX_ENV_STEPS,
        boundary_margin=BOUNDARY_MARGIN,
        seed=SEED,
    )

    if len(near_states) < NUM_NEAR_STATES:
        print(
            f"[WARN] near-boundary states insufficient: collected={len(near_states)} < target={NUM_NEAR_STATES}. "
            f"Try increasing MAX_ENV_STEPS or BOUNDARY_MARGIN."
        )
    if len(away_states) < NUM_AWAY_STATES:
        print(
            f"[WARN] away-boundary states insufficient: collected={len(away_states)} < target={NUM_AWAY_STATES}. "
            f"Try increasing MAX_ENV_STEPS."
        )

    print(
        f"Collected states in {used_steps} env steps: near={len(near_states)}, away={len(away_states)}"
    )
    print(f"near states collected: {len(near_states)}")
    print(f"away states collected: {len(away_states)}")

    near_rows = analyze_region("near", near_states, agent, threshold)
    away_rows = analyze_region("away", away_states, agent, threshold)
    all_rows = near_rows + away_rows

    env_tag = env_id.replace("-", "_").replace("/", "_")
    rho_tag = str(RHO).replace(".", "p")
    out_dir = repo_root / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    per_state_csv_path = out_dir / f"{PER_STATE_CSV_PREFIX}_{env_tag}_rho{rho_tag}_seed{SEED}.csv"
    summary_csv_path = out_dir / f"{SUMMARY_CSV_PREFIX}_{env_tag}_rho{rho_tag}_seed{SEED}.csv"

    write_per_state_csv(per_state_csv_path, all_rows)
    summary_rows = build_summary_rows(all_rows)
    write_summary_csv(summary_csv_path, summary_rows)

    print("\n" + "-" * 90)
    print("Summary")
    print("-" * 90)
    for rec in summary_rows:
        print(
            f"region={rec['region']:<4} | n={rec['num_states']:>3} | multimodality_rate={100.0 * rec['multimodality_rate']:.2f}% "
            f"| lambda_min_hess_L_mean={rec['lambda_min_hess_L_mean']:.6f} "
            f"| lambda_min_hess_LA_mean={rec['lambda_min_hess_LA_mean']:.6f} "
            f"| rho_grad_Qc_norm_sq_mean={rec['rho_grad_Qc_norm_sq_mean']:.6f} "
            f"| dom_ratio_mean={rec['dom_ratio_mean']:.6f}"
        )

    print("\nSaved:")
    print(f"  per-state CSV: {per_state_csv_path}")
    print(f"  summary CSV  : {summary_csv_path}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
