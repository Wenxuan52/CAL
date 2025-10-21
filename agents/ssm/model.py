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

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)

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

    def loss(self, state: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        batch_size = a0.size(0)
        t = torch.randint(0, self.T, (batch_size,), device=a0.device)
        x_t, eps = self.q_sample(a0, t)
        eps_pred = self.forward(state, x_t, t.float().unsqueeze(-1))
        return F.mse_loss(eps_pred, eps)

    def to(self, *args, **kwargs):  # pragma: no cover - passthrough helper
        return super().to(*args, **kwargs)
