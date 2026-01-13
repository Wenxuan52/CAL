import numpy as np
import torch
import torch.nn.functional as F
import os

from pathlib import Path

from agents.base_agent import Agent
from agents.algd.utils import soft_update
from agents.algd.model_v3 import QNetwork, DiffusionPolicy, QcEnsemble


class ALGDAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda")
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args

        self.update_counter = 0
        self.update_step = 0
        # parameter debug
        self.actor_update_step = 0

        # Safety params
        self.c = args.c
        self.cost_lr_scale = 1.
        
        self.rho = getattr(args, "rho", 4.0) ##### 1.0
        self.T = getattr(args, "diffusion_T", 5)
       
        self.actor_loss_coef = getattr(args, "actor_loss_coef", 1.0)
        self.score_coef = getattr(args, "score_coef", 0.1)
        
        # ====== Step4-B: score matching 的 MC 超参 ======
        # 每个 (s, a^τ, τ) 周围采多少个 a^{0,(i)}
        self.score_mc_samples = getattr(args, "score_mc_samples", 4) # 4
        # MC 里高斯扰动的尺度系数: σ(τ) = sigma_scale * sqrt(1 - ᾱ_τ)
        self.score_sigma_scale = getattr(args, "score_sigma_scale", 1.0)
        # softmax 温度 β：w_i ∝ exp( -L_A / β )
        self.score_beta = getattr(args, "score_beta", 1.0)
        
        # comparison 实验
        self.use_aug_lag = getattr(args, "use_aug_lag", True)

        # Reward critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety critics
        self.safety_critics = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # ==== Diffusion Policy ====
        self.policy = DiffusionPolicy(
            num_inputs=num_inputs,
            num_actions=action_space.shape[0],
            hidden_dim=128,
            T=self.T,
            action_space=action_space
        ).to(self.device)
        
        self.policy.guidance_scale = getattr(args, "guidance_scale", 0.05)
        # 如需关闭归一化，可以在 args 里加个开关；没有则默认 True
        self.policy.guidance_normalize = getattr(args, "guidance_normalize", True)

        # Lagrange multiplier for safety
        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8))).to(self.device)
        self.log_lam.requires_grad = True

        self.kappa = 0

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Target cost
        if args.safetygym:
            self.target_cost = args.cost_lim * (1 - self.safety_discount ** args.epoch_length) / \
                               (1 - self.safety_discount) / args.epoch_length \
                               if self.safety_discount < 1 else args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)
        
        # ====== profiling: score_mc time ======
        self.profile_score_mc = getattr(args, "profile_score_mc", True)
        self.profile_warmup = getattr(args, "profile_warmup", 50)   # 前 50 次不计（避开cuda热身）
        self.profile_every = getattr(args, "profile_every", 1)      # 每隔几次记录一次
        self._mc_prof_step = 0

        # Welford online stats for ms
        self._mc_time_n = 0
        self._mc_time_mean = 0.0
        self._mc_time_M2 = 0.0
    
    def _welford_update(self, x_ms: float):
        self._mc_time_n += 1
        delta = x_ms - self._mc_time_mean
        self._mc_time_mean += delta / self._mc_time_n
        delta2 = x_ms - self._mc_time_mean
        self._mc_time_M2 += delta * delta2

    def get_score_mc_time_stats(self):
        if self._mc_time_n < 2:
            return {"n": self._mc_time_n, "mean_ms": self._mc_time_mean, "var_ms2": 0.0, "std_ms": 0.0}
        var = self._mc_time_M2 / (self._mc_time_n - 1)
        return {"n": self._mc_time_n, "mean_ms": self._mc_time_mean, "var_ms2": var, "std_ms": var ** 0.5}


    @property
    def lam(self):
        return self.log_lam.exp()
    
    def get_last_log(self):
        return getattr(self, "last_log", {})
    
    def compute_L(self, state, action):
        """
        Standard Lagrangian:
        L(s,a,λ) = -min_j Q_j(s,a) + λ * (Qc_risk(s,a) - h)
        """
        Q1, Q2 = self.critic(state, action)
        Q_min = torch.min(Q1, Q2)

        QCs = self.safety_critics(state, action)
        qc_std, qc_mean = torch.std_mean(QCs, dim=0)
        if self.args.qc_ens_size == 1:
            qc_std = torch.zeros_like(qc_mean).to(self.device)
        qc_risk = qc_mean + self.args.k * qc_std

        lam = self.lam
        h = self.target_cost
        L = -Q_min + lam * (qc_risk - h)
        return L, Q_min, qc_risk
    
    def compute_LA(self, state, action):
        """
        Augmented Lagrangian:
        L_A(s,a,λ) = -min_j Q_j(s,a)
                     + 1/(2ρ) ( [λ + ρ( Qc̄(s,a) - h )]_+^2 - λ^2 )

        这里的 Qc̄ 用的是 ensemble 均值 + k * std，和你原来的风险 cost 定义保持一致。
        """
        # 1) reward 部分: min_j Q_j(s,a)
        Q1, Q2 = self.critic(state, action)
        Q_min = torch.min(Q1, Q2)   # [B, 1]

        # 2) cost 部分: ensemble + risk term
        QCs = self.safety_critics(state, action)   # [E, B, 1]
        qc_std, qc_mean = torch.std_mean(QCs, dim=0)   # [B,1], [B,1]
        if self.args.qc_ens_size == 1:
            qc_std = torch.zeros_like(qc_mean).to(self.device)
        qc_risk = qc_mean + self.args.k * qc_std     # 和原来的 actor_QC 一致

        # 3) Augmented Lagrangian penalty term
        lam = self.lam           # 标量 tensor
        rho = self.rho           # 标量 float
        h = self.target_cost     # 约束预算 h

        # [λ + ρ (Qc - h)]_+
        penalty_arg = lam + rho * (qc_risk - h)
        penalty = 0.5 / rho * (torch.clamp(penalty_arg, min=0.0) ** 2 - lam ** 2)

        LA = -Q_min + penalty
        return LA, Q_min, qc_risk
    
    def compute_energy(self, state, action):
        if self.use_aug_lag:
            return self.compute_LA(state, action)
        else:
            return self.compute_L(state, action)
    
    def compute_score_target(self, state, action):
        """
        简化版 ALGD 目标 score φ*：
        φ*(s,a) ≈ -∇_a L_A(s,a,λ)

        注意：
        - 这里我们只用 1 次采样（不做多重 MC 和 softmax 权重）
        - 输出的 φ* 会被 detach，用作监督信号（target），不反向进 critics
        """
        # 1) 拷贝 action，用于对 a 建图，防止污染主图
        action_for_grad = action.detach().clone()
        action_for_grad.requires_grad_(True)

        # 2) 计算 L_A(s,a)
        LA, Q_min, qc_risk = self.compute_energy(state, action_for_grad)

        # 3) 对 a 求梯度：∇_a sum_i L_A(s_i, a_i)
        grad_a = torch.autograd.grad(
            outputs=LA.sum(),
            inputs=action_for_grad,
            create_graph=False,   # 不要高阶梯度，只做一次性分析/target
            retain_graph=False
        )[0]  # shape: [B, act_dim]

        # 4) φ* = -∇_a L_A
        phi_star = -grad_a.detach()  # 作为监督信号，不让梯度回传到 critics

        # 5) 你也可以在这里顺便打印 norm（可选）
        grad_norm = grad_a.norm(dim=-1).mean().item()
        # print(f"[ALGD] score target norm (|-∇_a L_A|) = {grad_norm:.6f}")

        return phi_star
    
    def compute_score_target_mc(self, state, a_tau, tau):
        """
        Step4-B: 多 τ + Monte Carlo φ* 版本
        对每个样本 (s, a^τ, τ)，在 a^τ 周围采 N 个 a^{0,(i)}，算出:
            φ*(s,a^τ,τ) ≈ - (1/β) * Σ_i w_i ∇_a L_A(s,a^{0,(i)},λ)

        输入:
            state : [B, state_dim]
            a_tau : [B, act_dim]   (轨迹中的中间动作 a^τ)
            tau   : [B] long       (对应时间步 τ)
        输出:
            phi_star: [B, act_dim]
        """
        ##############
        do_profile = self.profile_score_mc and state.is_cuda
        if do_profile:
            self._mc_prof_step += 1
            should_record = (self._mc_prof_step > self.profile_warmup) and (self._mc_prof_step % self.profile_every == 0)
            if should_record:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
        ##############

        
        B, act_dim = a_tau.shape
        N = self.score_mc_samples
        device = state.device

        # --------- 准备 MC 样本 ----------
        # 状态扩展到 [B*N, ...]
        state_exp = state.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)  # [B*N, state_dim]

        # a^τ 扩展 & 加噪得到 a^{0,(i)}
        a_tau_exp = a_tau.unsqueeze(1).expand(B, N, -1)                    # [B, N, act_dim]

        # 从 diffusion 里拿 ᾱ_τ，用来决定噪声尺度 σ(τ)
        alphas_cumprod = self.policy.alphas_cumprod.to(device)             # [T]
        alpha_bar_tau = alphas_cumprod[tau]                                # [B]
        sigma_tau = self.score_sigma_scale * torch.sqrt(1.0 - alpha_bar_tau + 1e-8)  # [B]
        sigma_tau = sigma_tau.view(B, 1, 1)                                # [B,1,1]

        noise = torch.randn(B, N, act_dim, device=device)
        a0_samples = a_tau_exp + sigma_tau * noise                         # [B, N, act_dim]
        a0_samples_flat = a0_samples.reshape(B * N, act_dim)
        a0_samples_flat.requires_grad_(True)

        # --------- 计算 L_A 和梯度 ∇_a L_A ----------
        LA_flat, _, _ = self.compute_energy(state_exp, a0_samples_flat)        # [B*N, 1]
        grad_a_flat = torch.autograd.grad(
            outputs=LA_flat.sum(),
            inputs=a0_samples_flat,
            create_graph=False,
            retain_graph=False
        )[0]                                                               # [B*N, act_dim]

        LA = LA_flat.view(B, N)                                            # [B, N]
        grad_a = grad_a_flat.view(B, N, act_dim)                           # [B, N, act_dim]

        # --------- 计算权重 w_i ----------
        # w_i ∝ exp( - L_A / β )
        beta = self.score_beta
        with torch.no_grad():
            weights = torch.softmax(-LA / beta, dim=1)                     # [B, N]
        weights = weights.unsqueeze(-1)                                    # [B, N, 1]

        # --------- 聚合得到 φ* ----------
        # φ* = - (1/β) * Σ_i w_i ∇_a L_A
        phi_star = - (weights * grad_a).sum(dim=1) / beta                  # [B, act_dim]

        # 不让梯度回流到 critics，只作为监督信号
        phi_star = phi_star.detach()
        
        ##############
        if do_profile and should_record:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            self._welford_update(float(elapsed_ms))
        ##############

        return phi_star


    def train(self, training=True):
        self.training = training
        self.policy.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action = self.policy.sample(state)
        else:
            action = self.policy.sample_deterministic(state)
        return action.detach().cpu().numpy()[0]

    def update_critic(self, state, action, reward, cost, next_state, mask):
        # 用 diffusion policy，目标值里不再减 alpha * log_prob
        next_action = self.policy.sample(next_state)

        # ---------- reward critic ----------
        current_Q1, current_Q2 = self.critic(state, action)
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)
        target_Q = reward + (mask * self.discount * target_V)
        target_Q = target_Q.detach()

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- safety critics ----------
        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
        current_QCs = self.safety_critics(state, action)  # [E, B, 1]
        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)
        next_QC_random_max = next_QCs[qc_idxs].max(dim=0, keepdim=True).values

        if self.args.safetygym:
            mask = torch.ones_like(mask).to(self.device)
        next_QC = next_QC_random_max.repeat(self.args.qc_ens_size, 1, 1) if self.args.intrgt_max else next_QCs
        target_QCs = cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) + \
                     (mask[None, :, :].repeat(self.args.qc_ens_size, 1, 1) * self.safety_discount * next_QC)
        safety_critic_loss = F.mse_loss(current_QCs, target_QCs.detach())

        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()

    def update_actor(self, state, action_taken):
        # 1) 用当前 policy 采样动作，并顺便拿到 on-policy 的 (a^τ, τ)
        action, a_tau, tau = self.policy.sample_with_traj(state)

        # 2) actor loss = E[L_A]，仍然在最终动作 a^0 上计算
        LA, actor_Q, actor_QC = self.compute_energy(state, action)
        actor_loss = LA.mean()

        # 3) Step4-B: 多 τ + MC φ* 的 score matching
        phi_star = self.compute_score_target_mc(state, a_tau, tau)      # [B, act_dim]
        phi_theta = self.policy.score(state, a_tau, tau=tau)            # [B, act_dim]

        score_loss = F.mse_loss(phi_theta, phi_star)

        # 4) 总的 actor loss =  E[L_A] + coeff * L_score
        total_loss = self.actor_loss_coef * actor_loss + self.score_coef * score_loss

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        
        # ====== 计数 + 打印（每 400 次）======
        # self.actor_update_step += 1
        # if self.actor_update_step % 400 == 0:
        #     # 注意这里用 .item()，避免打印一个 tensor
        #     a = actor_loss.item()
        #     s = score_loss.item()
        #     scaled_a = self.actor_loss_coef * a      # 如果你以后加了 mu，就改成 self.mu * a
        #     scaled_s = self.score_coef * s
        #     print(
        #         f"[ALGD] actor_step {self.actor_update_step} | "
        #         f"actor_loss={a:.4f} (scaled={scaled_a:.4f}) | "
        #         f"score_loss={s:.4f} (scaled={scaled_s:.4f}) | "
        #         f"total={scaled_a + scaled_s:.4f}"
        #     )

        # 5) λ 的更新保持原来的写法（用 replay 里的 action_taken）
        with torch.no_grad():
            current_QCs = self.safety_critics(state, action_taken)
            current_std, current_mean = torch.std_mean(current_QCs, dim=0)
            if self.args.qc_ens_size == 1:
                current_std = torch.zeros_like(current_mean).to(self.device)
            current_QC = current_mean + self.args.k * current_std

        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.lam * (self.target_cost - current_QC).detach())
        lam_loss.backward()
        self.log_lam_optimizer.step()
        
        with torch.no_grad():
            stats = self.get_score_mc_time_stats()
            self.last_log = {
                "lambda": float(self.lam.item()),
                "log_lambda": float(self.log_lam.item()),
                "lambda_loss": float(lam_loss.item()),
                "qc_risk_mean": float(current_QC.mean().item()),
                "violation_mean": float(torch.clamp(current_QC - self.target_cost, min=0.0).mean().item()),
                "use_aug_lag": int(self.use_aug_lag),
                
                # profiling
                "score_mc_samples": int(self.score_mc_samples),
                "score_mc_time_ms_mean": float(stats["mean_ms"]),
                "score_mc_time_ms_std": float(stats["std_ms"]),
                "score_mc_time_n": int(stats["n"]),
            }


    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        cost_batch = torch.FloatTensor(reward_batch[:, 1]).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch[:, 0]).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        self.update_critic(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch)
        self.update_actor(state_batch, action_batch)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

    # Save model parameters
    def save_model(self, save_dir, suffix=""):

        actor_path = save_dir / f"actor_{suffix}.pth"
        critics_path = save_dir / f"critics_{suffix}.pth"
        safetycritics_path = save_dir / f"safetycritics_{suffix}.pth"

        print(f"[Model] Saving models to:\n  {actor_path}\n  {critics_path}\n  {safetycritics_path}")

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critics_path)
        torch.save(self.safety_critics.state_dict(), safetycritics_path)


    # Load model parameters
    def load_model(self, actor_path, critics_path, safetycritics_path):
        print('Loading models from {}, {}, and {}'.format(actor_path, critics_path, safetycritics_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critics_path is not None:
            self.critic.load_state_dict(torch.load(critics_path))
        if safetycritics_path is not None:
            self.safety_critics.load_state_dict(torch.load(safetycritics_path))