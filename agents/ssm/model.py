from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class QNetwork(nn.Module):
    """Double Q-network used for both reward and safety critics."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], dim=-1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)

        return q1, q2


class DiffusionPolicy(nn.Module):
    """Diffusion-based policy that predicts noise for DDPM denoising."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        T: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T

        self.eps_model = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        alpha_hat_prev = torch.cat([alpha_hats.new_tensor([1.0]), alpha_hats[:-1]])

        posterior_variances = betas * (1.0 - alpha_hat_prev) / (1.0 - alpha_hats)
        posterior_variances[0] = 1e-20
        posterior_mean_coef1 = betas * torch.sqrt(alpha_hat_prev) / (1.0 - alpha_hats)
        posterior_mean_coef2 = (
            (1.0 - alpha_hat_prev) * torch.sqrt(alphas) / (1.0 - alpha_hats)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)
        self.register_buffer("alpha_hat_prev", alpha_hat_prev)
        self.register_buffer("sqrt_one_minus_alpha_hats", torch.sqrt(1.0 - alpha_hats))
        self.register_buffer("posterior_variances", posterior_variances)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def forward(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = t.float() / float(self.T)
        inp = torch.cat([state, x_t, t_embed], dim=-1)
        return self.eps_model(inp)

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_long = t.squeeze(-1).long()
        alpha_hat_t = self.alpha_hats[t_long].view(a0.size(0), *([1] * (a0.dim() - 1)))
        eps = torch.randn_like(a0)
        x_t = torch.sqrt(alpha_hat_t) * a0 + torch.sqrt(1 - alpha_hat_t) * eps
        return x_t, eps

    def loss(
        self,
        state: torch.Tensor,
        a0: torch.Tensor,
        critic,
        safety_critic,
        alpha: float = 1.0,
        beta: float = 3.0,
        safe_threshold: float = 0.0,
        m_q: float = 1.0,
        vh_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = a0.size(0)
        t = torch.randint(0, self.T, (batch_size,), device=a0.device)
        x_t, eps = self.q_sample(a0, t)

        t_embed = t.float().unsqueeze(-1)
        eps_pred = self.forward(state, x_t, t_embed)
        recon_loss = F.mse_loss(eps_pred, eps)

        # Convert noise prediction to a score estimate following the VP formulation.
        t_long = t.long()
        alpha_hat_t = self.alpha_hats[t_long]
        std_t = torch.sqrt(1.0 - alpha_hat_t + 1e-8)
        std_t = std_t.view(batch_size, *([1] * (eps_pred.dim() - 1)))
        score_pred = -eps_pred / std_t

        # Align the learned score with reward/safety gradients at noisy actions.
        state_detached = state.detach()
        action_for_grad = x_t.detach().requires_grad_(True)

        q1, q2 = critic(state_detached, action_for_grad)
        q = torch.min(q1, q2)
        grad_q = torch.autograd.grad(
            q.sum(), action_for_grad, retain_graph=True, create_graph=False
        )[0]

        qh1, qh2 = safety_critic(state_detached, action_for_grad)
        if vh_state is None:
            vh_eval = torch.min(qh1, qh2).squeeze(-1)
        else:
            vh_eval = vh_state.detach()
        safe_mask = (vh_eval <= safe_threshold).float().unsqueeze(-1)
        qh_mean = 0.5 * (qh1 + qh2)
        grad_qh = torch.autograd.grad(
            qh_mean.sum(), action_for_grad, retain_graph=False, create_graph=False
        )[0]

        target_grad = safe_mask * alpha * grad_q - (1.0 - safe_mask) * beta * grad_qh
        target_grad = target_grad.detach()

        score_loss = F.mse_loss(score_pred, target_grad)

        return recon_loss + m_q * score_loss

    def to(self, *args, **kwargs):  # pragma: no cover - passthrough helper
        return super().to(*args, **kwargs)