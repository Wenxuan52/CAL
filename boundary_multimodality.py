#!/usr/bin/env python3
"""
Boundary multimodality comparison on fixed boundary states.

What this script does
---------------------
For each task:
1) Use one designated reference checkpoint to roll out the environment and collect a shared pool
   of candidate states.
2) Use the reference checkpoint's safety critic to select a fixed set of top-K boundary-like states.
3) Evaluate every ALGD and SAC+AugLag checkpoint on that same fixed state set.
4) Report per-seed boundary multimodality and mean±std for each method.

Notes
-----
- No command-line arguments are used. Edit the CONFIG section below.
- Replace every results_folder in CONFIG with your actual checkpoint directory.
- Each results_folder should contain actor / critic / safety critic checkpoints.
"""

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import gym
import numpy as np
import torch
import safety_gym  # noqa: F401, required for Safety-Gym env registration

from agents.algd.algd_v5 import ALGDAgent
from agents.sacauglag.sacauglag import SACAugLagAgent
from env.constraints import get_threshold


# =============================================================================
# CONFIG: EDIT THIS SECTION ONLY
# =============================================================================

GLOBAL_CONFIG: Dict[str, Any] = {
    # Shared fixed-boundary-state construction
    "candidate_pool_size": 1000,
    "candidate_stride": 10,
    "num_boundary_states": 50,
    "max_env_steps": 200000,
    "probe_actions_per_state": 8,

    # Multimodality test
    "actions_per_state": 128,
    "kmeans_iters": 20,
    "kmeans_restarts": 10,
    "min_cluster_frac": 0.15,
    "separation_coef": 1.75,

    # Rollout seed for constructing shared fixed states
    "state_collection_seed": 0,

    # Common model hyperparameters (override per method / per task if needed)
    "common_hparams": {
        "hidden_size": 256,
        "qc_ens_size": 4,
        "k": 1.0,
        "M": 4,
        "gamma": 0.99,
        "safety_gamma": 0.99,
        "tau": 0.005,
        "lr": 3e-4,
        "qc_lr": 3e-4,
        "critic_target_update_frequency": 2,
        "c": 10.0,
        "intrgt_max": False,
    },

    # ALGD-specific hyperparameters
    "algd_hparams": {
        "rho": 1.0,
        "diffusion_T": 5,
        "actor_loss_coef": 1.0,
        "score_coef": 0.1,
        "score_mc_samples": 4,
        "score_sigma_scale": 1.0,
        "score_beta": 1.0,
        "use_aug_lag": True,
        "guidance_scale": 0.05,
        "guidance_normalize": True,
        "profile_score_mc": False,
        "profile_warmup": 50,
        "profile_every": 1,
    },

    # SAC+AugLag-specific hyperparameters
    # Add / modify entries here if your SACAugLagAgent constructor expects more args.
    "sacauglag_hparams": {
        "use_aug_lag": True,
    },
}


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "PointButton2": {
        "env_id": "Safexp-PointButton2-v0",

        # One designated reference checkpoint used ONLY to construct the fixed boundary states
        # shared by ALGD and SAC+AugLag on this task.
        "fixed_boundary_reference": {
            "method": "algd",  # "algd" or "sacauglag"
            "seed": 3240,
            "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_algd/2025-12-06_15-07_seed3240/",
        },

        "algd_runs": [
            {"seed": 3240, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_algd/2025-12-06_15-07_seed3240/"},
            {"seed": 2017, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_algd_ablationENS16/2025-12-29_23-52_seed2017"},
            {"seed": 1282, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_algd_ablationENS16/2025-12-30_00-28_seed1282"},
        ],

        "sacauglag_runs": [
            {"seed": 6613, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_sacauglag/2025-12-13_10-38_seed6613"},
            {"seed": 6614, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_sacauglag/2025-12-13_10-38_seed6613"},
            {"seed": 6615, "results_folder": "results/Safexp-PointButton2-v0/pointbutton2_sacauglag/2025-12-13_10-38_seed6613"},
        ],
    },

    "PointPush1": {
        "env_id": "Safexp-PointPush1-v0",

        "fixed_boundary_reference": {
            "method": "algd",
            "seed": 3240,
            "results_folder": "results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC16/2025-12-26_16-17_seed5767",
        },

        "algd_runs": [
            {"seed": 6576, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_algd_ablationENS16/2025-12-29_23-10_seed6576"},
            {"seed": 283, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_algd_ablationENS16/2025-12-29_23-41_seed283"},
            {"seed": 5767, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC16/2025-12-26_16-17_seed5767"},
        ],

        "sacauglag_runs": [
            {"seed": 7114, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_sacauglag/2025-12-14_06-56_seed7114"},
            {"seed": 7115, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_sacauglag/2025-12-14_06-56_seed7114"},
            {"seed": 7116, "results_folder": "results/Safexp-PointPush1-v0/pointpush1_sacauglag/2025-12-14_06-56_seed7114"},
        ],
    },
}


# =============================================================================
# Utility helpers
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


def get_device(agent) -> torch.device:
    if hasattr(agent, "device"):
        return torch.device(agent.device)
    policy = get_attr_first(agent, ["policy", "actor"])
    try:
        return next(policy.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def validate_config() -> None:
    missing = []
    for task_name, task_cfg in TASK_CONFIGS.items():
        ref = task_cfg["fixed_boundary_reference"]["results_folder"]
        if "REPLACE_WITH_YOUR" in ref:
            missing.append(f"{task_name} fixed_boundary_reference")
        for method_key in ["algd_runs", "sacauglag_runs"]:
            for run in task_cfg[method_key]:
                if "REPLACE_WITH_YOUR" in run["results_folder"]:
                    missing.append(f"{task_name} {method_key} seed={run['seed']}")
    if missing:
        raise ValueError(
            "Please replace placeholder paths in TASK_CONFIGS before running:\n- " + "\n- ".join(missing)
        )


def make_agent_args(method: str, env_id: str, seed: int) -> SimpleNamespace:
    common = dict(GLOBAL_CONFIG["common_hparams"])
    method_specific = dict(GLOBAL_CONFIG["algd_hparams"] if method == "algd" else GLOBAL_CONFIG["sacauglag_hparams"])

    merged = {}
    merged.update(common)
    merged.update(method_specific)

    merged.update(
        dict(
            safetygym=True,
            epoch_length=400,
            cost_lim=get_threshold(env_id, constraint="safetygym"),
            env_name=env_id,
            seed=seed,
        )
    )

    # Ensure optional fields exist even if a class expects them.
    defaults = {
        "rho": 1.0,
        "diffusion_T": 5,
        "actor_loss_coef": 1.0,
        "score_coef": 0.1,
        "score_mc_samples": 4,
        "score_sigma_scale": 1.0,
        "score_beta": 1.0,
        "use_aug_lag": True,
        "guidance_scale": 0.05,
        "guidance_normalize": True,
        "profile_score_mc": False,
        "profile_warmup": 50,
        "profile_every": 1,
    }
    for k, v in defaults.items():
        merged.setdefault(k, v)

    return SimpleNamespace(**merged)


def instantiate_agent(method: str, env, seed: int):
    env_id = env.spec.id if getattr(env, "spec", None) is not None else None
    if env_id is None:
        raise ValueError("Environment spec.id is required to infer the task name.")

    args = make_agent_args(method=method, env_id=env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    agent_cls = ALGDAgent if method == "algd" else SACAugLagAgent
    agent = agent_cls(obs_dim, env.action_space, args)
    safe_eval_mode(agent)
    return agent


def resolve_checkpoint_file(results_folder: Path, patterns: Sequence[str], kind: str) -> Path:
    results_folder = Path(results_folder)
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder does not exist: {results_folder}")

    for pat in patterns:
        exact = results_folder / pat
        if exact.exists():
            return exact

    matches = []
    for pat in patterns:
        matches.extend(sorted(results_folder.glob(pat)))

    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact-looking short filenames first, then newest modified.
        matches = sorted(matches, key=lambda p: (len(p.name), -p.stat().st_mtime))
        return matches[0]

    raise FileNotFoundError(f"Could not find {kind} checkpoint in {results_folder}")


def find_checkpoint_triplet(results_folder: str) -> Tuple[Path, Path, Path]:
    folder = Path(results_folder)

    actor_path = resolve_checkpoint_file(
        folder,
        patterns=[
            "actor_.pth", "actor.pth", "policy_.pth", "policy.pth",
            "actor*.pth", "policy*.pth",
        ],
        kind="actor/policy",
    )
    critic_path = resolve_checkpoint_file(
        folder,
        patterns=[
            "critics_.pth", "critic_.pth", "critics.pth", "critic.pth",
            "qf_.pth", "qf.pth", "critic*.pth", "critics*.pth", "qf*.pth",
        ],
        kind="critic",
    )
    safety_path = resolve_checkpoint_file(
        folder,
        patterns=[
            "safetycritics_.pth", "safety_critics_.pth", "safetycritic_.pth",
            "safetycritics.pth", "safety_critics.pth", "safetycritic.pth",
            "safetycritic*.pth", "safety_critic*.pth", "safetycritics*.pth",
        ],
        kind="safety critic",
    )
    return actor_path, critic_path, safety_path


def load_agent_checkpoints(agent, results_folder: str) -> Tuple[Path, Path, Path]:
    actor_path, critic_path, safety_path = find_checkpoint_triplet(results_folder)

    # Try the class's own loader first.
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
                # Fall back to manual state_dict loading below.
                break

    device = get_device(agent)
    actor_state = torch.load(actor_path, map_location=device)
    critic_state = torch.load(critic_path, map_location=device)
    safety_state = torch.load(safety_path, map_location=device)

    policy_module = get_attr_first(agent, ["policy", "actor"])
    critic_module = get_attr_first(agent, ["critic", "critics"])
    safety_module = get_attr_first(agent, ["safety_critics", "safety_critic", "safetycritic"])

    policy_module.load_state_dict(actor_state, strict=True)
    critic_module.load_state_dict(critic_state, strict=True)
    safety_module.load_state_dict(safety_state, strict=True)
    safe_eval_mode(agent)
    return actor_path, critic_path, safety_path


def reduce_qc_ensemble(qcs: torch.Tensor, ens_size: int, batch_n: int) -> torch.Tensor:
    qcs = torch.as_tensor(qcs)

    if qcs.ndim == 3:
        # [E, N, 1] or [N, E, 1]
        if qcs.shape[0] == ens_size:
            return qcs
        if qcs.shape[1] == ens_size:
            return qcs.transpose(0, 1)
    elif qcs.ndim == 2:
        # [E, N] or [N, E]
        if qcs.shape[0] == ens_size:
            return qcs.unsqueeze(-1)
        if qcs.shape[1] == ens_size:
            return qcs.transpose(0, 1).unsqueeze(-1)
    elif qcs.ndim == 1:
        # [N] -> treat as no ensemble
        return qcs.view(1, batch_n, 1)

    # Best-effort fallback: reshape into [E, N, -1] if possible
    if qcs.numel() % max(1, ens_size * batch_n) == 0:
        trailing = qcs.numel() // (ens_size * batch_n)
        return qcs.view(ens_size, batch_n, trailing)

    raise ValueError(f"Unsupported safety critic output shape: {tuple(qcs.shape)}")


def extract_actions_from_sample_output(sample_out: Any, batch_n: int) -> np.ndarray:
    candidate = sample_out
    if isinstance(sample_out, (tuple, list)):
        tensor_candidates = [x for x in sample_out if isinstance(x, (torch.Tensor, np.ndarray))]
        if not tensor_candidates:
            raise ValueError("policy.sample returned a tuple/list without tensor/array actions.")
        # Prefer the first candidate whose first dim matches batch_n.
        candidate = None
        for x in tensor_candidates:
            x0 = np.asarray(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
            if x0.ndim >= 1 and x0.shape[0] == batch_n:
                candidate = x
                break
        if candidate is None:
            candidate = tensor_candidates[0]

    if isinstance(candidate, torch.Tensor):
        acts = candidate.detach().cpu().numpy()
    else:
        acts = np.asarray(candidate)

    if acts.ndim == 1:
        acts = acts[None, :]
    return acts.astype(np.float32, copy=False)


@torch.no_grad()
def sample_actions(agent, s_np: np.ndarray, n: int) -> np.ndarray:
    device = get_device(agent)
    s = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
    s_rep = s.repeat(n, 1)

    # Preferred path: directly sample from the stochastic policy.
    if hasattr(agent, "policy") and hasattr(agent.policy, "sample"):
        try:
            out = agent.policy.sample(s_rep)
            return extract_actions_from_sample_output(out, batch_n=n)
        except Exception:
            pass

    if hasattr(agent, "actor") and hasattr(agent.actor, "sample"):
        try:
            out = agent.actor.sample(s_rep)
            return extract_actions_from_sample_output(out, batch_n=n)
        except Exception:
            pass

    # Fallback: repeatedly call select_action
    acts = []
    for _ in range(n):
        if hasattr(agent, "select_action"):
            try:
                a = agent.select_action(s_np, eval=False)
            except TypeError:
                a = agent.select_action(s_np)
            acts.append(np.asarray(a, dtype=np.float32))
        else:
            raise AttributeError(f"{type(agent).__name__} has neither policy.sample nor select_action.")
    return np.stack(acts, axis=0)


def select_rollout_action(agent, obs: np.ndarray) -> np.ndarray:
    if hasattr(agent, "select_action"):
        try:
            action = agent.select_action(obs, eval=False)
            return np.asarray(action, dtype=np.float32)
        except TypeError:
            try:
                action = agent.select_action(obs)
                return np.asarray(action, dtype=np.float32)
            except Exception:
                pass
    return sample_actions(agent, obs, n=1)[0]


@torch.no_grad()
def qc_risks(agent, s_np: np.ndarray, a_np: np.ndarray) -> np.ndarray:
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
    return risk.view(-1).detach().cpu().numpy()


def collect_candidate_states(
    env,
    agent,
    candidate_pool_size: int,
    candidate_stride: int,
    max_env_steps: int,
    seed: int,
) -> List[np.ndarray]:
    states: List[np.ndarray] = []
    obs = reset_env(env, seed=seed)
    steps = 0

    while steps < max_env_steps and len(states) < candidate_pool_size:
        if steps % candidate_stride == 0:
            states.append(np.array(obs, dtype=np.float32, copy=True))

        action = select_rollout_action(agent, obs)
        next_obs, _, done, _ = step_env(env, action)
        obs = next_obs

        if done:
            obs = reset_env(env)

        steps += 1

    return states


def select_topk_boundary_states(
    candidate_states: List[np.ndarray],
    agent,
    threshold: float,
    probe_actions_per_state: int,
    num_boundary_states: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    if len(candidate_states) == 0:
        return [], np.array([], dtype=np.float32)

    scores = []
    for s in candidate_states:
        probe_actions = sample_actions(agent, s, n=probe_actions_per_state)
        risks = qc_risks(agent, s, probe_actions)
        score = float(np.min(np.abs(risks - threshold)))
        scores.append(score)

    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)
    topk = min(num_boundary_states, len(candidate_states))
    idx = order[:topk]

    selected_states = [candidate_states[i] for i in idx]
    selected_scores = scores[idx]
    return selected_states, selected_scores


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

        for k in range(2):
            mask = labels == k
            if np.any(mask):
                centers[k] = actions[mask].mean(axis=0)
            else:
                centers[k] = actions[np.random.randint(0, n)]

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


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    mean = float(arr.mean()) if len(arr) > 0 else float("nan")
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


# =============================================================================
# Core evaluation
# =============================================================================

def build_loaded_agent(method: str, env, seed: int, results_folder: str):
    agent = instantiate_agent(method=method, env=env, seed=seed)
    actor_path, critic_path, safety_path = load_agent_checkpoints(agent, results_folder)
    return agent, actor_path, critic_path, safety_path


def construct_fixed_boundary_states(task_name: str, task_cfg: Dict[str, Any], env):
    env_id = canonical_env_name(task_cfg["env_id"])
    ref_cfg = task_cfg["fixed_boundary_reference"]
    method = ref_cfg["method"]
    seed = ref_cfg["seed"]
    results_folder = ref_cfg["results_folder"]

    print(f"[{task_name}] Constructing fixed boundary states from reference: method={method}, seed={seed}")
    ref_agent, actor_path, critic_path, safety_path = build_loaded_agent(
        method=method,
        env=env,
        seed=seed,
        results_folder=results_folder,
    )
    print(f"[{task_name}] Reference checkpoints:")
    print(f"  actor/policy: {actor_path}")
    print(f"  critic      : {critic_path}")
    print(f"  safetycritic: {safety_path}")

    threshold = float(getattr(ref_agent.args, "cost_lim", get_threshold(env_id, constraint="safetygym")))

    candidate_states = collect_candidate_states(
        env=env,
        agent=ref_agent,
        candidate_pool_size=GLOBAL_CONFIG["candidate_pool_size"],
        candidate_stride=GLOBAL_CONFIG["candidate_stride"],
        max_env_steps=GLOBAL_CONFIG["max_env_steps"],
        seed=GLOBAL_CONFIG["state_collection_seed"],
    )
    if len(candidate_states) == 0:
        raise RuntimeError(f"[{task_name}] No candidate states collected from reference agent.")

    boundary_states, boundary_scores = select_topk_boundary_states(
        candidate_states=candidate_states,
        agent=ref_agent,
        threshold=threshold,
        probe_actions_per_state=GLOBAL_CONFIG["probe_actions_per_state"],
        num_boundary_states=GLOBAL_CONFIG["num_boundary_states"],
    )
    if len(boundary_states) == 0:
        raise RuntimeError(f"[{task_name}] No boundary states selected from reference agent.")

    print(f"[{task_name}] Fixed boundary states ready.")
    print(
        f"[{task_name}] Candidate states={len(candidate_states)}, "
        f"selected boundary states={len(boundary_states)}"
    )
    print(
        f"[{task_name}] Boundary score stats (smaller is closer): "
        f"min={boundary_scores.min():.4f}, median={np.median(boundary_scores):.4f}, max={boundary_scores.max():.4f}"
    )
    return boundary_states, boundary_scores


def evaluate_agent_on_fixed_states(agent, boundary_states: List[np.ndarray]) -> Tuple[float, int, int]:
    multimodal_count = 0

    for s in boundary_states:
        acts = sample_actions(agent, s, n=GLOBAL_CONFIG["actions_per_state"])
        flag = is_multimodal(
            acts,
            min_cluster_frac=GLOBAL_CONFIG["min_cluster_frac"],
            separation_coef=GLOBAL_CONFIG["separation_coef"],
            kmeans_iters=GLOBAL_CONFIG["kmeans_iters"],
            kmeans_restarts=GLOBAL_CONFIG["kmeans_restarts"],
        )
        if flag:
            multimodal_count += 1

    total = len(boundary_states)
    ratio = 100.0 * multimodal_count / total
    return ratio, multimodal_count, total


def evaluate_method_runs(
    task_name: str,
    method: str,
    runs: List[Dict[str, Any]],
    env,
    boundary_states: List[np.ndarray],
) -> List[Dict[str, Any]]:
    results = []
    pretty_name = "ALGD" if method == "algd" else "SAC+AugLag"

    print(f"[{task_name}] Evaluating {pretty_name} on shared fixed boundary states...")
    for run in runs:
        seed = run["seed"]
        results_folder = run["results_folder"]
        agent, actor_path, critic_path, safety_path = build_loaded_agent(
            method=method,
            env=env,
            seed=seed,
            results_folder=results_folder,
        )
        ratio, count, total = evaluate_agent_on_fixed_states(agent, boundary_states)

        record = {
            "task": task_name,
            "method": method,
            "seed": seed,
            "results_folder": results_folder,
            "actor_path": str(actor_path),
            "critic_path": str(critic_path),
            "safety_path": str(safety_path),
            "ratio": ratio,
            "count": count,
            "total": total,
        }
        results.append(record)
        print(f"  seed={seed}: Boundary multimodality = {ratio:.2f}% ({count}/{total})")

    return results


def print_task_summary(task_name: str, algd_results: List[Dict[str, Any]], sac_results: List[Dict[str, Any]]) -> None:
    algd_ratios = [r["ratio"] for r in algd_results]
    sac_ratios = [r["ratio"] for r in sac_results]

    algd_mean, algd_std = summarize(algd_ratios)
    sac_mean, sac_std = summarize(sac_ratios)

    print(f"\n[{task_name}] Summary")
    print(f"  ALGD       : {algd_mean:.2f}% ± {algd_std:.2f}% over {len(algd_results)} seeds")
    print(f"  SAC+AugLag : {sac_mean:.2f}% ± {sac_std:.2f}% over {len(sac_results)} seeds")


def print_final_summary(all_results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    for task_name, task_results in all_results.items():
        algd_ratios = [r["ratio"] for r in task_results["algd"]]
        sac_ratios = [r["ratio"] for r in task_results["sacauglag"]]

        algd_mean, algd_std = summarize(algd_ratios)
        sac_mean, sac_std = summarize(sac_ratios)

        print(
            f"{task_name:<14} | "
            f"ALGD {algd_mean:6.2f}% ± {algd_std:5.2f}% | "
            f"SAC+AugLag {sac_mean:6.2f}% ± {sac_std:5.2f}%"
        )


def main() -> None:
    validate_config()
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for task_name, task_cfg in TASK_CONFIGS.items():
        print("\n" + "=" * 90)
        print(f"TASK: {task_name}")
        print("=" * 90)

        env_id = canonical_env_name(task_cfg["env_id"])
        set_seed(GLOBAL_CONFIG["state_collection_seed"])

        env = gym.make(env_id)
        reset_env(env, seed=GLOBAL_CONFIG["state_collection_seed"])

        boundary_states, _ = construct_fixed_boundary_states(task_name, task_cfg, env)

        algd_results = evaluate_method_runs(
            task_name=task_name,
            method="algd",
            runs=task_cfg["algd_runs"],
            env=env,
            boundary_states=boundary_states,
        )
        sac_results = evaluate_method_runs(
            task_name=task_name,
            method="sacauglag",
            runs=task_cfg["sacauglag_runs"],
            env=env,
            boundary_states=boundary_states,
        )

        print_task_summary(task_name, algd_results, sac_results)
        all_results[task_name] = {
            "algd": algd_results,
            "sacauglag": sac_results,
        }

        try:
            env.close()
        except Exception:
            pass

    print_final_summary(all_results)


if __name__ == "__main__":
    main()