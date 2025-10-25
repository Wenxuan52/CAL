"""Model components for the diffusion-based Safe Score Matching agent."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, input_dim: int = 1, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        half_dim = output_size // 2
        if learnable:
            self.kernel = nn.Parameter(torch.randn(half_dim, input_dim) * 0.2)
        else:
            freq = torch.exp(torch.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
            self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            f = 2 * math.pi * x @ self.kernel.T
        else:
            f = x * self.freq
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, output_dim: int, activation=nn.SiLU(), use_layer_norm: bool = False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionScoreModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.time_embed = FourierFeatures(time_dim, input_dim=1)
        self.cond_encoder = MLP(state_dim, [hidden_dim // 2, hidden_dim // 2], hidden_dim // 2, activation=nn.SiLU())
        self.reverse_net = MLP(
            input_dim=action_dim + state_dim + hidden_dim // 2 + time_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=action_dim,
            activation=nn.SiLU(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(time)
        cond = self.cond_encoder(state)
        x = torch.cat([action, state, cond, t_embed], dim=-1)
        return self.reverse_net(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class SafetyQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)