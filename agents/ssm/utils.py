"""Utility helpers for the diffusion-based ``ssm_test`` agent."""

import math
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch

from agents.guass_test.model import QNetwork, SafetyValueNetwork


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule introduced in https://arxiv.org/abs/2102.09672."""

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


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
) -> torch.Tensor:
    """DDPM-style ancestral sampler that supports additive safety guidance."""

    if not torch.is_tensor(alphas):
        alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
    else:
        alphas = alphas.to(device)
    if not torch.is_tensor(alpha_hats):
        alpha_hats = torch.tensor(alpha_hats, dtype=torch.float32, device=device)
    else:
        alpha_hats = alpha_hats.to(device)
    if not torch.is_tensor(betas):
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
    else:
        betas = betas.to(device)

    B = state.size(0)
    action_dim = score_model.action_dim
    action = torch.randn(B, action_dim, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((B, 1), float(t), device=device)

        action = action.detach()
        action.requires_grad_(True)
        eps = score_model(state, action, t_tensor)

        if guidance_fn is not None:
            with torch.enable_grad():
                guidance = guidance_fn(state, action, t_tensor)
            eps = eps - guidance.detach()

        eps = eps.detach()
        action = action.detach()

        alpha_t = alphas[t]
        alpha_hat_t = alpha_hats[t]
        beta_t = betas[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        mean = coef1 * (action - coef2 * eps)

        if t > 0:
            noise = torch.randn_like(action)
            sigma_t = torch.sqrt(beta_t)
            action = mean + sigma_t * noise
        else:
            action = mean

    return action


def atanh_clamped(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse tanh used for domain alignment."""

    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def load_and_freeze_critics(
    critic_path: Union[str, Path],
    safety_path: Union[str, Path],
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> Tuple[QNetwork, SafetyValueNetwork]:
    """Instantiate the Gaussian critics and load pretrained weights."""

    critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    safety_q = SafetyValueNetwork(state_dim, action_dim, hidden_dim).to(device)

    critic_state = torch.load(critic_path, map_location=device)
    safety_state = torch.load(safety_path, map_location=device)
    critic.load_state_dict(critic_state)
    safety_q.load_state_dict(safety_state)

    critic.eval()
    safety_q.eval()
    for module in (critic, safety_q):
        for param in module.parameters():
            param.requires_grad_(False)

    return critic, safety_q


def compute_phi(
    state: torch.Tensor,
    action: torch.Tensor,
    critic: torch.nn.Module,
    safety_q: torch.nn.Module,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    safe_margin: float = 0.0,
    grad_clip: float = 10.0,
) -> torch.Tensor:
    """Compute the piecewise guidance field φ(s, a) used for training."""

    state = state.to(action.device)

    # action_var = torch.tanh(action.detach().clone()).requires_grad_(True)
    action = action.clone().detach().requires_grad_(True)

    q_value = critic(state, action)
    if isinstance(q_value, (tuple, list)):
        q_value = q_value[0]
    grad_q = torch.autograd.grad(q_value.mean(), action, retain_graph=False, create_graph=False)[0]

    qh_value = safety_q(state, action)
    if isinstance(qh_value, (tuple, list)):
        qh_value = qh_value[0]
    grad_qh = torch.autograd.grad(qh_value.mean(), action, retain_graph=False, create_graph=False)[0]

    safe_mask = (qh_value <= safe_margin).float()
    unsafe_mask = 1.0 - safe_mask

    phi = alpha * safe_mask * grad_q - beta * unsafe_mask * grad_qh

    norm = phi.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    phi = phi / norm
    if grad_clip > 0:
        phi = torch.clamp(phi, -grad_clip, grad_clip)

    return phi.detach()
