import torch
import torch.nn as nn


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings as used in diffusion models."""
    half_dim = embedding_dim // 2
    device = timesteps.device
    timesteps = timesteps.float()
    if half_dim == 0:
        return timesteps.unsqueeze(-1)
    emb = torch.log(torch.tensor(10000.0, device=device)) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ScoreNetwork(nn.Module):
    """Score network for diffusion-based policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        time_embed_dim: int,
        time_embed_type: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("ScoreNetwork requires at least one hidden layer")

        self.time_embed_dim = time_embed_dim
        self.time_embed_type = time_embed_type

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        layers = []
        input_dim = state_dim + action_dim + time_embed_dim
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)
        elif tau.dim() > 2:
            tau = tau.reshape(tau.size(0), -1)
        if self.time_embed_type == "sinusoidal":
            time_embed = get_timestep_embedding(tau.squeeze(-1), self.time_embed_dim)
        elif self.time_embed_type == "identity":
            time_embed = tau.expand(-1, self.time_embed_dim)
        else:
            raise ValueError(f"Unsupported time embedding type: {self.time_embed_type}")
        time_embed = self.time_mlp(time_embed)
        x = torch.cat([state, action, time_embed], dim=-1)
        return self.net(x)