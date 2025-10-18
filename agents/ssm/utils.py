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
            # clone and enable grad for new autograd graph
            x = x.detach().clone().requires_grad_(True)

            qh = safety_critic(state, x).mean(0)
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

    return x



# -----------------------------
# Soft target update
# -----------------------------
def soft_update(target, source, tau):
    for t_param, param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + param.data * tau)
