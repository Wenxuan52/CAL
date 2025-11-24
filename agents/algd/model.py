import math
import torch
import torch.nn as nn
from agents.cal.model import GaussianPolicy


def sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps * emb
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1)
    return emb


class DiffusionScoreNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_hidden_layers, t_embed_dim, time_embedding, action_space=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_embed_dim = t_embed_dim
        self.time_embedding = time_embedding

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        layers = []
        input_dim = state_dim + action_dim + t_embed_dim
        last_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def encode_time(self, times: torch.Tensor) -> torch.Tensor:
        if self.time_embedding == 'identity':
            return times.repeat(1, self.t_embed_dim)
        return sinusoidal_embedding(times, self.t_embed_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, times: torch.Tensor):
        t_emb = self.encode_time(times)
        x = torch.cat([state, action, t_emb], dim=-1)
        return self.net(x)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


__all__ = ["DiffusionScoreNetwork", "GaussianPolicy"]
