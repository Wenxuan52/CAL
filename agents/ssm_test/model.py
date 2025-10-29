"""Neural building blocks for the diffusion-based ``ssm_test`` agent."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Fourier feature embedding used for diffusion time conditioning."""

    def __init__(self, output_size: int, input_dim: int = 1, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        half_dim = output_size // 2
        self.learnable = learnable
        if learnable:
            self.kernel = nn.Parameter(torch.randn(half_dim, input_dim) * 0.2)
        else:
            freq = torch.exp(torch.arange(half_dim) * -(math.log(10000.0) / (half_dim - 1)))
            self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            proj = 2 * math.pi * x @ self.kernel.T
        else:
            proj = x * self.freq
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MLP(nn.Module):
    """Simple feed-forward network with configurable hidden layers."""

    def __init__(self, input_dim: int, hidden_dims, output_dim: int, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.SiLU()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation.__class__())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionScoreModel(nn.Module):
    """Conditional score network that predicts guidance vectors for diffusion."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.time_embed = FourierFeatures(time_dim, input_dim=1)
        self.state_encoder = MLP(state_dim, [hidden_dim // 2, hidden_dim // 2], hidden_dim // 2)
        self.score_head = MLP(
            input_dim=action_dim + state_dim + hidden_dim // 2 + time_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=action_dim,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_feat = self.time_embed(time)
        cond = self.state_encoder(state)
        features = torch.cat([action, state, cond, time_feat], dim=-1)
        return self.score_head(features)
