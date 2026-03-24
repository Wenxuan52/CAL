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
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int, action_space=None, log_sig_max: float = LOG_SIG_MAX):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

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

    @torch.no_grad()
    def get_a_mean(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        y = (action - self.action_bias) / (self.action_scale + EPS)
        y = torch.clamp(y, -1 + 1e-6, 1 - 1e-6)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
        return log_prob.sum(dim=-1, keepdim=True)


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
        return self.linear3(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
