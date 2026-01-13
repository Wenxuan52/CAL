import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class SquashedGaussianActor(nn.Module):
    """Squashed (tanh) Gaussian policy for continuous actions.

    This is PPO-friendly: it can
      - sample actions
      - compute log_prob(action|state) for *given* actions
      - compute an (approx) entropy term

    Notes:
      * Entropy of the squashed distribution has no simple closed form; we use the
        base Normal entropy as a common practical approximation.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int, action_space=None, log_sig_max: float = LOG_SIG_MAX):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)

        self.log_sig_max = log_sig_max

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=self.log_sig_max)
        return mean, log_std

    def _dist(self, state: torch.Tensor) -> Normal:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return Normal(mean, std)

    @torch.no_grad()
    def get_a_mean(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        a = torch.tanh(mean) * self.action_scale + self.action_bias
        return a

    def sample(self, state: torch.Tensor):
        """Sample action with reparameterization trick; returns (action, log_prob, mean_action)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        # tanh + rescale correction
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log pi(a|s) for provided *bounded* action."""
        # unscale to [-1, 1]
        y = (action - self.action_bias) / (self.action_scale + EPS)
        y = torch.clamp(y, -1 + 1e-6, 1 - 1e-6)
        # atanh
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
        return log_prob.sum(dim=-1, keepdim=True)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """Return (log_prob, entropy_approx) for PPO."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # log_prob for provided action
        logp = self.log_prob(state, action)

        # entropy approximation (base normal)
        ent = normal.entropy().sum(dim=-1, keepdim=True)
        return logp, ent


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        v = self.linear3(x)
        return v
