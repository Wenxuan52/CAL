import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0.0)


class QNetwork(nn.Module):
    """
    Double Q critic used for reward value function Q_r(s, a).
    和原 CAL 中的 QNetwork 结构一致：两个头 Q1/Q2。
    """
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super(QNetwork, self).__init__()

        # Q1
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], dim=1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class SafetyQNetwork(nn.Module):
    """
    单头安全 Q 函数 Q_h(s, a)，近似 RCRL 的 safety value。
    """
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super(SafetyQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        xu = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class GaussianPolicy(nn.Module):
    """
    标准 SAC 风格的高斯策略：
    - 输入 state
    - 输出动作分布 N(mean, std)
    - sample() 返回 (tanh 后的 action, log_prob, mean_action)
    """
    def __init__(self, args, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
            action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0

    def forward(self, state: torch.Tensor):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()          # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log_prob 修正 tanh & action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """
    可选的确定性策略（目前 RCRL 版本没有用到，
    保留是为了和你原仓库风格一致）。
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)

        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
            action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0

    def forward(self, state: torch.Tensor):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state: torch.Tensor):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1).to(mean.device)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0, device=mean.device), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicy, self).to(device)
