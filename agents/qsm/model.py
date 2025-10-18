# agents/qsm/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# -----------------------------
# Fourier Feature Embedding
# -----------------------------
class FourierFeatures(nn.Module):
    def __init__(self, output_size, input_dim=1, learnable=True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        half_dim = output_size // 2
        if learnable:
            self.kernel = nn.Parameter(torch.randn(half_dim, input_dim) * 0.2)
        else:
            freq = torch.exp(torch.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
            self.register_buffer("freq", freq)

    def forward(self, x):
        """
        x: (B, 1) time index
        return: (B, output_size)
        """
        if self.learnable:
            f = 2 * math.pi * x @ self.kernel.T
        else:
            f = x * self.freq
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


# -----------------------------
# MLP network
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), use_layer_norm=False):
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

    def forward(self, x):
        return self.net(x)


# -----------------------------
# DDPM-style Diffusion Score Model
# -----------------------------
class DiffusionScoreModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.time_embed = FourierFeatures(time_dim, input_dim=1)
        # conditional encoder for state
        self.cond_encoder = MLP(state_dim, [128, 128], 128, activation=nn.SiLU())
        # reverse process network
        self.reverse_net = MLP(
            input_dim=action_dim + state_dim + 128 + time_dim,
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            output_dim=action_dim,
            activation=nn.SiLU()
        )

    def forward(self, state, action, time):
        """
        state: (B, S)
        action: (B, A)
        time: (B, 1)
        """
        t_embed = self.time_embed(time)
        cond = self.cond_encoder(state)
        x = torch.cat([action, state, cond, t_embed], dim=-1)
        return self.reverse_net(x)


# -----------------------------
# Q network (State-Action Value)
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)

