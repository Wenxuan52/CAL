import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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
        nn1_output = self.activation(self.nn1(xu[None, :, :].repeat([self.ensemble_size, 1, 1])))
        nn2_output = self.activation(self.nn2(nn1_output))
        nn3_output = self.nn3(nn2_output)

        return nn3_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

# -----------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    一个简单的 DDPM-style diffusion policy：
    - 时间步 T：默认 10
    - 噪声预测网络：MLP，3 层隐藏层，每层 128 单元 + ReLU
    - 输入：state, noisy_action, t
    - 输出：预测噪声 epsilon
    - 采样：从 N(0,I) 开始反向 denoise 得到动作
    """
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim=128,
                 T=10,
                 action_space=None):
        super(DiffusionPolicy, self).__init__()

        self.state_dim = num_inputs
        self.action_dim = num_actions
        self.hidden_dim = hidden_dim
        self.T = T  # diffusion steps

        # ====== VE-style noise schedule (σ_t from small to large) ======
        sigma_min = 0.01
        sigma_max = 1.0
        t_steps = torch.linspace(0.0, 1.0, T)                 # t ∈ [0,1]

        # 几何插值：σ_t = σ_min * (σ_max/σ_min)^t
        sigmas = sigma_min * (sigma_max / sigma_min) ** t_steps   # [T]
        sigma2 = sigmas ** 2                                     # σ_t^2

        # 离散的 Δσ^2_k = σ_k^2 - σ_{k-1}^2，约定 σ_-1 = 0
        sigma2_prev = torch.cat(
            [torch.zeros(1, device=sigma2.device), sigma2[:-1]], dim=0
        )
        delta_sigma2 = sigma2 - sigma2_prev                       # [T]

        # 存起来给 SDE 和 score MC 用
        self.register_buffer("ve_sigmas", sigmas)           # σ_t
        self.register_buffer("ve_sigma2", sigma2)           # σ_t^2
        self.register_buffer("ve_delta_sigma2", delta_sigma2)  # Δσ^2_t


        # ====== time embedding（简单 embedding）======
        self.time_embed_dim = 16
        self.time_embedding = nn.Embedding(T, self.time_embed_dim)

        # ====== 噪声预测 MLP：3 层，hidden=128 ======
        input_dim = self.state_dim + self.action_dim + self.time_embed_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        self.eps_head = nn.Linear(hidden_dim, self.action_dim)
        
        self.score_head = nn.Linear(hidden_dim, self.action_dim)

        self.apply(weights_init_)
        
        # ====== guidance 参数（Step4-A 用）======
        # guidance_scale 决定 φ_theta 对采样的影响强度；默认 0，后面由 ALGDAgent 覆盖
        self.guidance_scale = 0.0
        # 可选：是否在使用前把 φ 归一化（只学方向）
        self.guidance_normalize = True

        # ====== action rescaling ======
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            )

    # ---------- 工具函数 ----------

    def _extract(self, a, t, x_shape):
        """
        根据 batch 的 t，从 precomputed 的 buffer 里取出对应系数，
        并 reshape 成 [B, 1, ...] 的形式
        """
        out = a.gather(-1, t.long())
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0, noise, t):
        """
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x0: [B, action_dim]
        noise: [B, action_dim]
        t: [B] long
        """
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        return sqrt_alphas_cumprod_t * x0 + \
            sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, state, x_t, t):
        """
        预测噪声 epsilon
        state: [B, state_dim]
        x_t: [B, action_dim]
        t: [B] long (0 ~ T-1)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(state.size(0))
        t_embed = self.time_embedding(t)  # [B, time_embed_dim]
        h = torch.cat([state, x_t, t_embed], dim=-1)
        h = self.mlp(h)
        eps = self.eps_head(h)
        return eps
    
    def score(self, state, x, tau=None):
        B = state.size(0)
        if tau is None:
            tau = torch.zeros(B, dtype=torch.long, device=state.device)
        elif tau.dim() == 0:
            tau = tau.unsqueeze(0).repeat(B)

        t_embed = self.time_embedding(tau)          # [B, time_dim]
        h = torch.cat([state, x, t_embed], dim=-1)  # 用 latent x 而不是 action
        h = self.mlp(h)
        phi_theta = self.score_head(h)              # [B, act_dim] = latent 维度

        return phi_theta
    
    def latent_to_action(self, x):
        return torch.tanh(x) * self.action_scale + self.action_bias


    # ---------- 反向采样 ----------

    def p_sample(self, state, x_t, t):
        """
        纯 VE-SDE latent 反向一步：
          x_{t-1} = x_t - Δσ^2(t) * φ_theta(s, x_t, t) + sqrt(Δσ^2(t)) * ε
        其中 Δσ^2(t) = sigma_t^2 - sigma_{t-1}^2，预先存在 self.ve_delta_sigma2 里。
        """
        B = x_t.size(0)
        device = x_t.device
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        # Δσ^2(t)
        delta_sigma2_t = self._extract(self.ve_delta_sigma2, t_batch, x_t.shape)
        delta_sigma2_t = delta_sigma2_t.clamp(min=1e-8)
        step_std = torch.sqrt(delta_sigma2_t)

        # φ_theta(s, x_t, t)
        phi = self.score(state, x_t, tau=t_batch)

        if self.guidance_normalize:
            phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8)

        # 这里不要再乘额外 guidance_scale=“拍脑袋 heuristics”，
        # 真要乘可以当一个 η<1 的步长缩放参数。
        drift = delta_sigma2_t * phi     # 如果想更稳，可以改成 η * delta_sigma2_t * phi

        mean = x_t - drift

        if t == 0:
            return mean

        noise = torch.randn_like(x_t)
        return mean + step_std * noise



    def p_sample_loop(self, state, steps=None):
        if steps is None:
            steps = self.T

        B = state.size(0)
        device = state.device

        # σ_T
        sigma_T = self.ve_sigmas[-1].to(device)
        x_t = torch.randn(B, self.action_dim, device=device) * sigma_T

        for t in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, t)

        action = self.latent_to_action(x_t)
        return action



    # ---------- 对外接口 ----------

    def sample(self, state, steps=None):
        """
        用于训练采样：给定 state，输出一个随机 action
        """
        return self.p_sample_loop(state, steps=steps)
    
    def sample_with_traj(self, state, steps=None):
        if steps is None:
            steps = self.T

        B = state.size(0)
        device = state.device

        sigma_T = self.ve_sigmas[-1].to(device)
        x_t = torch.randn(B, self.action_dim, device=device) * sigma_T

        taus = torch.randint(low=0, high=steps, size=(B,), device=device)
        x_tau = torch.zeros(B, self.action_dim, device=device)

        for t in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, t)
            mask = (taus == t)
            if mask.any():
                x_tau[mask] = x_t[mask]

        a0 = self.latent_to_action(x_t)
        return a0, x_tau, taus


    def sample_deterministic(self, state, steps=None):
        """
        用于评估：VE-SDE 的 deterministic 版本（只保留 drift，不加噪声）
          x_{t-1} = x_t - Δσ^2(t) * φ_theta(s, x_t, t)
        """
        if steps is None:
            steps = self.T

        B = state.size(0)
        device = state.device

        # 从 x_T 开始，这里用零或者 σ_T * 0 都行；如果你想更贴近采样，可以用 σ_T * noise
        x_t = torch.zeros(B, self.action_dim, device=device)

        for t in reversed(range(steps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            # Δσ^2(t)：来自你在 __init__ 里注册的 VE schedule
            delta_sigma2_t = self._extract(self.ve_delta_sigma2, t_batch, x_t.shape)
            delta_sigma2_t = delta_sigma2_t.clamp(min=1e-8)

            # φ_theta(s, x_t, t)：在 latent 上工作的 score
            phi = self.score(state, x_t, tau=t_batch)

            if self.guidance_normalize:
                phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8)

            # 只保留 drift，不加噪声 = deterministic
            x_t = x_t - delta_sigma2_t * phi     # 如果想更稳，可以额外乘个 eta<1

        # 映射到动作空间
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action


    def get_a_mean(self, state):
        """
        为了兼容你之前 eval 时用的 mean 行为，这里用 deterministic 采样代替
        """
        return self.sample_deterministic(state)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DiffusionPolicy, self).to(device)
