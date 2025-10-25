"""Utility helpers for the diffusion-based Safe Score Matching agent."""

from typing import Callable, Optional, Union

import torch
import math
import numpy as np


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Perform Polyak averaging between ``source`` and ``target`` modules."""

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def soft_gate(qc_value: torch.Tensor, kappa: float = 0.0, alpha: float = 5.0) -> torch.Tensor:
    """Smooth gate that interpolates between reward maximisation and safety guidance."""

    return torch.sigmoid(alpha * (kappa - qc_value))


# ---------------------------------------------------------------------------
# Diffusion utilities
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from https://arxiv.org/abs/2102.09672."""

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def vp_beta_schedule(timesteps: int) -> torch.Tensor:
    t = torch.arange(1, timesteps + 1, dtype=torch.float32)
    T = float(timesteps)
    b_max, b_min = 10.0, 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


def ddpm_forward_process(
    actions: torch.Tensor,
    t: torch.Tensor,
    alpha_hats: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    """Forward diffusion q(a_t | a_0) following the DDPM formulation."""

    if noise is None:
        noise = torch.randn_like(actions)
    sqrt_alpha_hat = torch.sqrt(alpha_hats[t]).unsqueeze(1)
    sqrt_one_minus = torch.sqrt(1 - alpha_hats[t]).unsqueeze(1)
    noisy_actions = sqrt_alpha_hat * actions + sqrt_one_minus * noise
    return noisy_actions, noise


def ddpm_sampler(
    score_model,
    state: torch.Tensor,
    T: int,
    alphas: torch.Tensor,
    alpha_hats: torch.Tensor,
    betas: torch.Tensor,
    *,
    guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    device: Union[torch.device, str] = "cuda",
):
    """DDPM reverse sampling with optional score guidance."""

    if isinstance(alphas, np.ndarray):
        alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
        alpha_hats = torch.tensor(alpha_hats, dtype=torch.float32, device=device)
        betas = torch.tensor(betas, dtype=torch.float32, device=device)

    B = state.size(0)
    action_dim = score_model.action_dim
    a_t = torch.randn(B, action_dim, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((B, 1), float(t), device=device)
        eps_pred = score_model(state, a_t, t_tensor)

        if guidance_fn is not None:
            with torch.enable_grad():
                a_t.requires_grad_(True)
                guidance = guidance_fn(state, a_t, t_tensor)
            a_t = a_t.detach()
            eps_pred = eps_pred - guidance.detach()

        alpha_t = alphas[t]
        alpha_hat_t = alpha_hats[t]
        beta_t = betas[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        mean = coef1 * (a_t - coef2 * eps_pred)

        if t > 0:
            noise = torch.randn_like(a_t)
            sigma_t = torch.sqrt(beta_t)
            a_t = mean + sigma_t * noise
        else:
            a_t = mean

    return torch.tanh(a_t)