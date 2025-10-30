"""Neural building blocks for the diffusion-based SSM agent."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def weights_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0.0)


class QNetwork(nn.Module):
    """Twin-head critic that mirrors the Gaussian SSM baseline."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], dim=1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)
        return q1, q2


class SafetyValueNetwork(nn.Module):
    """Single-head network used for the safety critic Qₕ."""

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([state, action], dim=1)
        x = self.activation(self.linear1(xu))
        x = self.activation(self.linear2(x))
        return self.linear3(x)


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

    def __init__(self, input_dim: int, hidden_dims, output_dim: int, activation: Optional[nn.Module] = None):
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