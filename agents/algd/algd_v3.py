import numpy as np
import torch
import torch.nn.functional as F
import os

from pathlib import Path

from agents.base_agent import Agent
from agents.algd.utils import soft_update
from agents.algd.model_v1 import QNetwork, DiffusionPolicy, QcEnsemble


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

        # Safety params
        self.c = args.c
        self.cost_lr_scale = 1.
        
        self.rho = getattr(args, "rho", 1.0)
        self.T = getattr(args, "diffusion_T", 5)
        self.score_coef = getattr(args, "score_coef", 0.1)

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

    @property
    def lam(self):
        return self.log_lam.exp()
    
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
        LA, Q_min, qc_risk = self.compute_LA(state, action_for_grad)

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
        # 1) 用当前 policy 采样动作
        action = self.policy.sample(state)

        # 2) actor loss = E[L_A]
        LA, actor_Q, actor_QC = self.compute_LA(state, action)
        actor_loss = LA.mean()

        # 3) ALGD score matching：φ_theta ≈ -∇_a L_A
        #    (简化版：只在 a^0 上做，tau 先固定为 0)
        phi_star = self.compute_score_target(state, action)              # [B, act_dim]
        phi_theta = self.policy.score(state, action, tau=None)          # [B, act_dim]

        score_loss = F.mse_loss(phi_theta, phi_star)

        # 4) 总的 actor loss =  E[L_A] + coeff * MSE
        total_loss = actor_loss + self.score_coef * score_loss

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()

        # 5) λ 的更新保持原样（用 action_taken 做）
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