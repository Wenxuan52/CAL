# agents/qsm/utils.py
import torch
import math
import numpy as np

# -----------------------------
# Diffusion beta schedules
# -----------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

def vp_beta_schedule(timesteps):
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max, b_min = 10.0, 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


# -----------------------------
# Forward process (q(a_t | a_0))
# -----------------------------
def ddpm_forward_process(actions, t, alpha_hats, noise=None):
    """
    q(a_t | a_0) = sqrt(alpha_hat_t) * a_0 + sqrt(1 - alpha_hat_t) * eps
    """
    if noise is None:
        noise = torch.randn_like(actions)
    sqrt_alpha_hat = torch.sqrt(alpha_hats[t]).unsqueeze(1)
    sqrt_one_minus = torch.sqrt(1 - alpha_hats[t]).unsqueeze(1)
    noisy_actions = sqrt_alpha_hat * actions + sqrt_one_minus * noise
    return noisy_actions, noise


# -----------------------------
# DDPM reverse sampler
# -----------------------------
@torch.no_grad()
def ddpm_sampler(
    score_model,
    state,
    T,
    alphas,
    alpha_hats,
    betas,
    device="cuda"
):
    """
    DDPM reverse sampling for QSM (Torch version)

    Args:
        score_model: trained DiffusionScoreModel
        state: [B, state_dim] tensor
        T: number of diffusion steps
        alphas, alpha_hats, betas: diffusion schedules (numpy or tensor)
        device: torch.device

    Returns:
        actions: [B, action_dim]
    """
    if isinstance(alphas, np.ndarray):
        alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
        alpha_hats = torch.tensor(alpha_hats, dtype=torch.float32, device=device)
        betas = torch.tensor(betas, dtype=torch.float32, device=device)

    B = state.size(0)
    a_t = torch.randn(B, score_model.action_dim, device=device)  # start from Gaussian

    for t in reversed(range(T)):
        t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float32)

        eps_pred = score_model(state, a_t, t_tensor)

        alpha_t = alphas[t]
        alpha_hat_t = alpha_hats[t]
        beta_t = betas[t]

        # DDPM update
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)

        mean = coef1 * (a_t - coef2 * eps_pred)

        if t > 0:
            noise = torch.randn_like(a_t)
            sigma_t = torch.sqrt(beta_t)
            a_t = mean + sigma_t * noise
        else:
            a_t = mean  # final step

    return torch.tanh(a_t)


def safe_ddpm_sampler(
    model,
    state,
    T,
    alphas,
    alpha_hats,
    betas,
    safety_critic=None,
    safe_threshold=0.0,
    step_size=0.1,
    temperature=1.0,
    action_dim=None,
    device="cuda",
):
    """
    Safe-aware DDPM sampler:
    if the generated action is unsafe (Q_h > safe_threshold),
    project it back toward safe manifold using -∇_a Q_h.
    """
    B = state.size(0)
    x = torch.randn(B, action_dim, device=device)
    state_for_grad = state.detach()

    for t in reversed(range(T)):
        # -------------------------------------------------
        # Reverse diffusion (no grad)
        # -------------------------------------------------
        with torch.no_grad():
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float32)
            eps_theta = model(state, x, t_tensor)
            alpha = alphas[t]
            alpha_hat = alpha_hats[t]
            beta = betas[t]

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * eps_theta
            ) + torch.sqrt(beta) * noise * temperature

        # -------------------------------------------------
        # Safety correction (with grad)
        # -------------------------------------------------
        if safety_critic is not None:
            # Make sure gradients are enabled even if the caller wrapped us in
            # a no_grad() context (e.g. agent.sample_action()).
            with torch.enable_grad():
                # clone and enable grad for new autograd graph
                x = x.detach().clone().requires_grad_(True)

                qh = safety_critic(state_for_grad, x).mean(0)
                unsafe_mask = (qh > safe_threshold).float()

                if unsafe_mask.sum() > 0:
                    dq_da = torch.autograd.grad(
                        qh.sum(), x, create_graph=False, retain_graph=False
                    )[0]

                    # Project unsafe actions back
                    x = (x - step_size * dq_da * unsafe_mask).detach()
                    x = torch.clamp(x, -1.0, 1.0)
                else:
                    x = x.detach()

    return torch.tanh(x)


import torch

def _collapse_to_batch_scalar(qh: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    将 critic 输出 qh 折叠为形状 [B, 1] 的张量，自动识别 batch 维。
    兼容形状：
      - [B, 1] / [B] -> [B, 1]
      - [E, B, 1] -> [B, 1]   (例如 QcEnsemble 输出)
      - [B, A] -> [B, 1]
      - [B, A, E] -> [B, 1]
      - [E, B] -> [B, 1]
    """
    if qh.dim() == 0:
        return qh.view(1, 1).expand(batch_size, 1)

    # 找出等于 batch_size 的维度作为 batch 维
    cand = [i for i in range(qh.dim()) if qh.size(i) == batch_size]
    if len(cand) == 0:
        # 没找到 batch 维，说明输出与输入无关，直接求均值广播
        v = qh.mean()
        return v.view(1, 1).expand(batch_size, 1)

    bdim = cand[0]
    # 把 batch 维换到第 0 维
    perm = [bdim] + [i for i in range(qh.dim()) if i != bdim]
    q = qh.permute(*perm)  # [B, ...]

    # 折叠剩余维度
    while q.dim() > 2:
        q = q.mean(dim=-1)
    if q.size(1) != 1:
        q = q.mean(dim=1, keepdim=True)
    return q  # [B, 1]


def safe_langevin_sampler(
    model,
    qh_model,
    state,
    T: int = 20,
    eta: float = 1e-2,
    sigma: float = 1e-3,
    safe_threshold: float = 0.0,
    step_size: float = 0.1,
    schedule_eta: bool = True,
    schedule_sigma: bool = True,
):
    """
    Langevin 动力学采样器（带 Hamilton–Jacobi 安全修正）
    - model: diffusion score 模型
    - qh_model: 安全 critic（Q_h）
    - state: 当前状态 [B, S]
    """
    device = state.device
    B = state.size(0)
    x = torch.randn(B, model.action_dim, device=device)

    for t in reversed(range(T)):
        # 时间步
        t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eta_t = eta * (1 - t / T) if schedule_eta else eta
        sigma_t = sigma * (t / T) if schedule_sigma else sigma

        # --- 保证 state 与 x 的 batch 对齐 ---
        if state.size(0) != x.size(0):
            if state.size(0) == 1:
                # 单状态扩展成整个 batch
                state = state.expand(x.size(0), -1)
            else:
                raise RuntimeError(
                    f"[safe_langevin_sampler] state batch {state.size(0)} != x batch {x.size(0)}"
                )

        # --- Langevin 主更新 ---
        x = x.detach().requires_grad_(True)
        s_theta = model(state, x, t_tensor)
        noise = torch.randn_like(x) if t > 0 else 0.0
        x = x + eta_t * s_theta + sigma_t * noise

        # --- 安全修正阶段 ---
        with torch.enable_grad():
            x.requires_grad_(True)

            # 前向传播安全 critic
            qh_raw = qh_model(state, x)

            # 自动识别 batch 维并折叠到 [B, 1]
            qh = _collapse_to_batch_scalar(qh_raw, x.size(0))

            # 构造安全掩码并广播到动作维
            unsafe_mask = (qh > safe_threshold).float().expand(-1, x.size(1))

            # 计算 ∇_a Q_h(s,a)
            dq_da_h = torch.autograd.grad(
                qh.sum(),
                x,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            # 若 critic 不依赖动作，梯度为 None，则设为 0
            if dq_da_h is None:
                dq_da_h = torch.zeros_like(x)

            # 安全修正更新
            x = x - step_size * dq_da_h * unsafe_mask

        # 断开计算图以节省显存
        x = x.detach()

    return torch.tanh(x)


# -----------------------------
# Soft target update
# -----------------------------
def soft_update(target, source, tau):
    for t_param, param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + param.data * tau)
