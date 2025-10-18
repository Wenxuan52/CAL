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


# -----------------------------
# Safe Q network (State-Action Value)
# -----------------------------
    
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Initialize Policy weights for ensemble networks
def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class QcEnsemble(nn.Module):
    def __init__(self, state_size, action_size, ensemble_size, hidden_size=256):
        super(QcEnsemble, self).__init__()
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.00003)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00006)
        self.nn3 = EnsembleFC(hidden_size, 1, ensemble_size, weight_decay=0.0001)
        self.activation = nn.SiLU()
        self.ensemble_size = ensemble_size
        self.apply(init_weights)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        xu = xu.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        nn1_output = self.activation(self.nn1(xu))
        nn2_output = self.activation(self.nn2(nn1_output))
        nn3_output = self.nn3(nn2_output)
        return nn3_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss