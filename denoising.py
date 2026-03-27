import os
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import torch

from agents.algd.algd_v5 import ALGDAgent
from env.constraints import get_threshold


# ============================================================
# Editable paths / core config
# ============================================================
ENV_NAME = "Hopper-v3"
SEED = 123
SAFETYGYM = False
CONSTRAINT_TYPE = "safetygym" if "Safe" in ENV_NAME else "velocity"

# Checkpoints saved by ALGDAgent.save_model(...)
ACTOR_CHECKPOINT_PATH = "./actor_checkpoint.pth"
CRITICS_CHECKPOINT_PATH = "./critics_checkpoint.pth"
SAFETY_CRITICS_CHECKPOINT_PATH = "./safetycritics_checkpoint.pth"

# Output CSV path (repo root)
OUTPUT_CSV_PATH = f"denoising_{SEED}.csv"

# Optional: if you have a persisted lambda value, set here.
# If None, use agent default (exp(log(0.6931)) from ALGDAgent init).
LAMBDA_OVERRIDE = None


# ============================================================
# Editable analysis config
# ============================================================
NUM_STATES_TO_ANALYZE = 64
STATE_SELECTION_MODE = "random"  # "random" or "boundary_near"
NUM_CANDIDATE_STATES = 1024
MAX_ROLLOUT_STEPS = 5000

DIFFUSION_STEPS_OVERRIDE = None  # None -> use policy.T from checkpointed architecture
USE_DETERMINISTIC_POLICY_FOR_STATE_ROLLOUT = True


# ============================================================
# Minimal args required by ALGDAgent
# ============================================================
def build_args(env_name: str, seed: int, safetygym: bool):
    cost_lim = get_threshold(env_name, constraint=("safetygym" if safetygym else "velocity"))
    args = SimpleNamespace(
        agent="algd",
        env_name=env_name,
        seed=seed,
        safetygym=safetygym,
        constraint_type=("safetygym" if safetygym else "velocity"),
        epoch_length=(400 if safetygym else 1000),
        gamma=0.99,
        safety_gamma=0.99,
        tau=0.005,
        critic_target_update_frequency=2,
        hidden_size=256,
        lr=3e-4,
        qc_lr=3e-4,
        k=1.0,
        qc_ens_size=4,
        M=4,
        intrgt_max=False,
        c=10.0,
        rho=1.0,
        diffusion_T=5,
        actor_loss_coef=1.0,
        score_coef=0.1,
        score_mc_samples=4,
        score_sigma_scale=1.0,
        score_beta=1.0,
        use_aug_lag=True,
        guidance_scale=0.05,
        guidance_normalize=True,
        profile_score_mc=False,
        cost_lim=cost_lim,
    )
    return args


def make_env(env_name: str, seed: int, safetygym: bool):
    env = gym.make(env_name)
    if safetygym:
        env.seed(seed)
    else:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    return env


def get_state_dim(env, env_name: str):
    s_dim = env.observation_space.shape[0]
    if env_name == "Ant-v3":
        s_dim = 27
    elif env_name == "Humanoid-v3":
        s_dim = 45
    return int(s_dim)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_candidate_states(env, agent, n_candidates: int, max_steps: int, eval_policy: bool):
    states = []
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for _ in range(max_steps):
        act = agent.select_action(state, eval=eval_policy)
        step_out = env.step(act)
        if len(step_out) == 5:
            next_state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, _, done, _ = step_out

        states.append(np.asarray(state, dtype=np.float32))
        state = next_state

        if done:
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        if len(states) >= n_candidates:
            break

    if len(states) == 0:
        raise RuntimeError("No states collected from rollout.")
    return np.stack(states, axis=0)


def select_states(candidate_states: np.ndarray, agent: ALGDAgent, mode: str, n_select: int):
    if candidate_states.shape[0] < n_select:
        raise ValueError(f"Need at least {n_select} candidates, got {candidate_states.shape[0]}")

    if mode == "random":
        idx = np.random.choice(candidate_states.shape[0], size=n_select, replace=False)
        return candidate_states[idx]

    if mode == "boundary_near":
        device = agent.device
        state_t = torch.as_tensor(candidate_states, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_t = agent.policy.sample_deterministic(state_t)
            _, _, qc_risk = agent.compute_L(state_t, action_t)
            dist = torch.abs(qc_risk.squeeze(-1) - agent.target_cost)
            topk = torch.topk(dist, k=n_select, largest=False).indices
        return candidate_states[topk.detach().cpu().numpy()]

    raise ValueError(f"Unknown STATE_SELECTION_MODE: {mode}")


def scalar_hessian(fn, a_vec: torch.Tensor):
    return torch.autograd.functional.hessian(fn, a_vec)


def analyze_denoise_geometry(agent: ALGDAgent, states_np: np.ndarray, steps_override=None):
    device = agent.device
    state = torch.as_tensor(states_np, dtype=torch.float32, device=device)

    steps = int(agent.policy.T if steps_override is None else steps_override)
    _, actions_by_step, _ = agent.policy.sample_with_full_trajectory(state, steps=steps)

    rows = []
    rho = float(agent.rho)

    for denoise_step in range(steps):
        action_step = actions_by_step[denoise_step].detach()

        grad_qc_norm_sq_vals = []
        hess_qc_trace_vals = []
        rho_grad_qc_norm_sq_vals = []
        grad_la_norm_vals = []
        hess_la_trace_vals = []
        lambda_min_hess_la_vals = []
        lambda_min_hess_l_vals = []
        hess_spectral_radius_vals = []

        for i in range(state.shape[0]):
            s_i = state[i:i + 1]
            a_i = action_step[i].detach().clone().requires_grad_(True)

            def qc_scalar(a_vec):
                _, _, qc_val = agent.compute_L(s_i, a_vec.unsqueeze(0))
                return qc_val.squeeze()

            def la_scalar(a_vec):
                la_val, _, _ = agent.compute_LA(s_i, a_vec.unsqueeze(0))
                return la_val.squeeze()

            def l_scalar(a_vec):
                l_val, _, _ = agent.compute_L(s_i, a_vec.unsqueeze(0))
                return l_val.squeeze()

            qc_val = qc_scalar(a_i)
            la_val = la_scalar(a_i)

            grad_qc = torch.autograd.grad(qc_val, a_i, create_graph=True, retain_graph=True)[0]
            grad_la = torch.autograd.grad(la_val, a_i, create_graph=True, retain_graph=True)[0]

            hess_qc = scalar_hessian(qc_scalar, a_i)
            hess_la = scalar_hessian(la_scalar, a_i)
            hess_l = scalar_hessian(l_scalar, a_i)

            eig_la = torch.linalg.eigvalsh(hess_la)
            eig_l = torch.linalg.eigvalsh(hess_l)

            grad_qc_norm_sq = torch.dot(grad_qc, grad_qc)
            grad_qc_norm_sq_vals.append(float(grad_qc_norm_sq.detach().cpu().item()))
            hess_qc_trace_vals.append(float(torch.trace(hess_qc).detach().cpu().item()))
            rho_grad_qc_norm_sq_vals.append(float((rho * grad_qc_norm_sq).detach().cpu().item()))

            grad_la_norm_vals.append(float(torch.norm(grad_la, p=2).detach().cpu().item()))
            hess_la_trace_vals.append(float(torch.trace(hess_la).detach().cpu().item()))
            lambda_min_hess_la_vals.append(float(eig_la.min().detach().cpu().item()))
            lambda_min_hess_l_vals.append(float(eig_l.min().detach().cpu().item()))

            lambda_max_abs = torch.maximum(torch.abs(eig_la.min()), torch.abs(eig_la.max()))
            hess_spectral_radius_vals.append(float(lambda_max_abs.detach().cpu().item()))

        rows.append(
            {
                "env_name": agent.args.env_name,
                "denoise_step": int(denoise_step),
                "num_states": int(state.shape[0]),
                "grad_Qc_norm_sq": float(np.mean(grad_qc_norm_sq_vals)),
                "hess_Qc_trace": float(np.mean(hess_qc_trace_vals)),
                "rho_grad_Qc_norm_sq": float(np.mean(rho_grad_qc_norm_sq_vals)),
                "grad_LA_norm": float(np.mean(grad_la_norm_vals)),
                "hess_LA_trace": float(np.mean(hess_la_trace_vals)),
                "lambda_min_hess_LA": float(np.mean(lambda_min_hess_la_vals)),
                "lambda_min_hess_L": float(np.mean(lambda_min_hess_l_vals)),
                "hess_spectral_radius": float(np.mean(hess_spectral_radius_vals)),
            }
        )

    return rows


def save_rows_to_csv(rows, output_csv_path: str):
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "env_name",
        "denoise_step",
        "num_states",
        "grad_Qc_norm_sq",
        "hess_Qc_trace",
        "rho_grad_Qc_norm_sq",
        "grad_LA_norm",
        "hess_LA_trace",
        "lambda_min_hess_LA",
        "lambda_min_hess_L",
        "hess_spectral_radius",
    ]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] Denoising geometry CSV: {out.resolve()}")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_global_seeds(SEED)

    env = make_env(ENV_NAME, SEED, SAFETYGYM)
    args = build_args(ENV_NAME, SEED, SAFETYGYM)

    s_dim = get_state_dim(env, ENV_NAME)
    agent = ALGDAgent(s_dim, env.action_space, args)

    agent.load_model(
        ACTOR_CHECKPOINT_PATH,
        CRITICS_CHECKPOINT_PATH,
        SAFETY_CRITICS_CHECKPOINT_PATH,
    )

    if LAMBDA_OVERRIDE is not None:
        with torch.no_grad():
            agent.log_lam.copy_(torch.log(torch.tensor(float(LAMBDA_OVERRIDE), device=agent.device)))

    n_candidates = max(NUM_CANDIDATE_STATES, NUM_STATES_TO_ANALYZE)
    candidate_states = collect_candidate_states(
        env=env,
        agent=agent,
        n_candidates=n_candidates,
        max_steps=max(MAX_ROLLOUT_STEPS, n_candidates),
        eval_policy=USE_DETERMINISTIC_POLICY_FOR_STATE_ROLLOUT,
    )

    selected_states = select_states(
        candidate_states=candidate_states,
        agent=agent,
        mode=STATE_SELECTION_MODE,
        n_select=NUM_STATES_TO_ANALYZE,
    )

    rows = analyze_denoise_geometry(
        agent=agent,
        states_np=selected_states,
        steps_override=DIFFUSION_STEPS_OVERRIDE,
    )

    save_rows_to_csv(rows, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
