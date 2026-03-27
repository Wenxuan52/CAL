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

        # ====== diffusion schedule ======
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )

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
    
    def score(self, state, action, tau=None):
        """
        简化版 score 网络接口：
        输入 (s, a, tau)，输出 φ_theta(s,a,tau)，用来拟合 -∇_a L_A(s,a,λ)

        目前我们还没真正用到扩散时间 tau，
        可以先把 tau 统一设为 0（代表“最终动作”对应的时间步），
        但接口上预留 tau，方便以后扩展成多时间步的 ALGD。
        """
        B = state.size(0)

        if tau is None:
            # 先简单用 t=0，表示“最终动作处的 score”
            tau = torch.zeros(B, dtype=torch.long, device=state.device)
        else:
            # 支持标量或 [B] 的 long tensor
            if tau.dim() == 0:
                tau = tau.unsqueeze(0).repeat(B)

        t_embed = self.time_embedding(tau)  # [B, time_embed_dim]
        h = torch.cat([state, action, t_embed], dim=-1)  # [B, state+action+time]
        h = self.mlp(h)
        phi_theta = self.score_head(h)  # [B, action_dim]
        return phi_theta

    # ---------- 反向采样 ----------

    def p_sample(self, state, x_t, t):
        """
        一个 DDPM 的单步反向采样： x_t -> x_{t-1}
        现在加入一个简单的 ALGD guidance：
        x_{t-1} = x_{t-1}^{DDPM} + guidance_scale * φ_theta(s, a_t, t)
        """
        b = x_t.size(0)
        t_batch = torch.full((b,), t, dtype=torch.long, device=x_t.device)

        beta_t = self._extract(self.betas, t_batch, x_t.shape)
        alpha_t = self._extract(self.alphas, t_batch, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
        alpha_cumprod_prev_t = self._extract(
            self.alphas_cumprod_prev, t_batch, x_t.shape
        )

        # 原来的噪声预测
        eps_theta = self.forward(state, x_t, t_batch)

        # DDPM 公式：预测 x0
        x0_pred = (
            x_t - torch.sqrt(1.0 - alpha_cumprod_t) * eps_theta
        ) / torch.sqrt(alpha_cumprod_t + 1e-8)

        # posterior 均值（DDPM 的 x_{t-1}^{prior}）
        coef1 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1.0 - alpha_cumprod_t + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t + 1e-8)
        mean = coef1 * x0_pred + coef2 * x_t   # [B, act_dim]

        # ====== Step4-A: 用 φ_theta 对 mean 做一个小的 guidance ======
        if self.guidance_scale != 0.0:
            # 把当前 latent x_t 映射成环境动作 a_t（和训练 score 时的尺度一致）
            a_t = torch.tanh(x_t) * self.action_scale + self.action_bias  # [B, act_dim]

            # φ_theta(s, a_t, t)：近似 -∇_a L_A(s,a_t,λ)
            phi = self.score(state, a_t, tau=t_batch)  # [B, act_dim]

            if self.guidance_normalize:
                # 只强调方向，长度统一到 1，避免某些 state 梯度特别大
                phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8)

            # 注意：L_A 的梯度是在 action 空间，我们这里直接把它当作对 latent 的小偏移
            # 也就是假设 near-linear，diffusion 网络可以自己适应这个近似。
            mean = mean + self.guidance_scale * phi

        # ====== 加噪声生成 x_{t-1} ======
        if t == 0:
            # t=0 不再加噪声，直接用 deterministic 均值
            return mean

        noise = torch.randn_like(x_t)
        var = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t + 1e-8)
        std = torch.sqrt(var + 1e-8)

        return mean + std * noise

    def p_sample_loop(self, state, steps=None):
        """
        从 x_T ~ N(0, I) 开始，反向走到 x_0
        state: [B, state_dim]
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        x_t = torch.randn(b, self.action_dim, device=state.device)

        for i in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, i)

        # 映射到环境 action 空间
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action

    # ---------- 对外接口 ----------

    def sample(self, state, steps=None):
        """
        用于训练采样：给定 state，输出一个随机 action
        """
        return self.p_sample_loop(state, steps=steps)
    
    def sample_with_traj(self, state, steps=None):
        """
        Step4-B: 在反向采样时顺便记录一次 a^τ 以及对应的时间步 τ
        - 返回: a0, a_tau, tau
          a0   : 最终动作 a^0, [B, act_dim]
          a_tau: 轨迹中的某个中间动作 a^τ, [B, act_dim]
          tau  : 对应时间步 τ, [B] long
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        device = state.device

        # 起点: x_T ~ N(0, I)
        x_t = torch.randn(b, self.action_dim, device=device)

        # 每个样本随机选一个 τ（你可以之后改成偏向小 t 的分布）
        taus = torch.randint(low=0, high=steps, size=(b,), device=device)  # [B]
        a_tau = torch.zeros(b, self.action_dim, device=device)

        for i in reversed(range(steps)):
            # 先走一步 x_t -> x_{t-1} (这里 i 就是当前使用的 t)
            x_t = self.p_sample(state, x_t, i)

            # 如果某些样本的 τ == 当前 t，就在这里记录对应的 a^τ
            mask = (taus == i)
            if mask.any():
                a_t = torch.tanh(x_t) * self.action_scale + self.action_bias  # [B, act_dim]
                a_tau[mask] = a_t[mask]

        # 最后得到 x_0 -> a^0
        a0 = torch.tanh(x_t) * self.action_scale + self.action_bias
        return a0, a_tau, taus

    def sample_with_full_trajectory(self, state, steps=None, start_noise=None):
        """
        Offline analysis helper:
        return the full reverse-denoising action trajectory.

        Returns:
            a0: [B, act_dim] final action
            actions_by_step: [steps, B, act_dim]
                - index 0 corresponds to the first reverse step (t = steps-1)
                - index steps-1 corresponds to the last reverse step (t = 0)
            timesteps_by_step: [steps] long
                - DDPM timestep used at each denoising step
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        device = state.device

        if start_noise is None:
            x_t = torch.randn(b, self.action_dim, device=device)
        else:
            x_t = start_noise.to(device)
            if x_t.shape != (b, self.action_dim):
                raise ValueError(
                    f"start_noise shape must be {(b, self.action_dim)}, got {tuple(x_t.shape)}"
                )

        traj_actions = []
        traj_timesteps = []
        for i in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, i)
            a_t = torch.tanh(x_t) * self.action_scale + self.action_bias
            traj_actions.append(a_t)
            traj_timesteps.append(i)

        a0 = traj_actions[-1]
        actions_by_step = torch.stack(traj_actions, dim=0)
        timesteps_by_step = torch.tensor(traj_timesteps, device=device, dtype=torch.long)
        return a0, actions_by_step, timesteps_by_step


    def sample_deterministic(self, state, steps=None):
        """
        用于评估：简单做法是固定噪声 = 0（近似 deterministic）
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        x_t = torch.zeros(b, self.action_dim, device=state.device)

        for i in reversed(range(steps)):
            # t=0 时不加噪声，本身就是 deterministic
            x_t = self.p_sample(state, x_t, i)

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
