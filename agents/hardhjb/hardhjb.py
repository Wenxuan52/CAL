import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.hardhjb.model import QNetwork, SafetyQNetwork, GaussianPolicy


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class HJBAgent(Agent):

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        # 折扣因子
        self.discount = args.gamma
        # 安全 value 的折扣（通常和 gamma 一样）
        self.safety_discount = getattr(args, "safety_gamma", args.gamma)
        
        self.safety_threshold = getattr(args, "safety_threshold", 1.0)

        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency

        self.update_counter = 0

        # -------------------- Critic: reward Q_r --------------------
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # -------------------- Safety critic: Q_h --------------------
        self.safety_critic = SafetyQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        self.safety_critic_target = SafetyQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())

        # -------------------- Policy --------------------
        self.policy = GaussianPolicy(
            args, num_inputs, action_space.shape[0], args.hidden_size, action_space
        ).to(self.device)

        # temperature α（自动调节熵系数）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # -------------------- λ(s) 网络 --------------------
        # 输出通过 Softplus 保证 λ(s) >= 0
        self.lambda_net = nn.Sequential(
            nn.Linear(num_inputs, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
        ).to(self.device)

        # 目标熵（和原 CAL 一样）
        self.target_entropy = -torch.prod(
            torch.Tensor(action_space.shape).to(self.device)
        ).item()

        # -------------------- Optimizers --------------------
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(
            self.safety_critic.parameters(),
            lr=getattr(args, "qc_lr", args.lr),
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.lambda_optimizer = torch.optim.Adam(self.lambda_net.parameters(), lr=args.lr)

        # 训练模式
        self.train()
        self.critic_target.train()
        self.safety_critic.train()
        self.safety_critic_target.train()

    # ------------------------------------------------------------------
    # 通用接口
    # ------------------------------------------------------------------
    def train(self, training: bool = True):
        self.training = training
        self.policy.train(training)
        self.critic.train(training)
        self.safety_critic.train(training)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state, eval: bool = False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Critic 更新：Q_r 与 Q_h
    # ------------------------------------------------------------------
    def update_critic(self, state, action, reward, cost, next_state, mask):
        """
        state, next_state: (B, state_dim)
        action:           (B, act_dim)
        reward:           (B, 1)
        cost:             (B, 1)   ← 这里当作 h(s)
        mask:             (B, 1)   0 表示终止，1 表示非终止
        """
        # ----------------- Reward Q_r 部分（SAC 风格） -----------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + mask * self.discount * target_V

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------- Safety Q_h 部分（RCRL HJB 风格） -----------------
        # h(s) 直接用 cost
        h_s = cost

        with torch.no_grad():
            next_action, _, _ = self.policy.sample(next_state)
            next_Qh = self.safety_critic_target(next_state, next_action)

            # RCRL 的 Bellman 形式：
            #   V_h(s) = (1-γ) h(s) + γ max(h(s), V_h(s'))
            # 对 episodic：在终止时不再 bootstrap
            max_term = torch.max(h_s, next_Qh)
            bootstrap = (1.0 - self.safety_discount) * h_s + self.safety_discount * max_term
            target_Qh = (1.0 - mask) * h_s + mask * bootstrap

        current_Qh = self.safety_critic(state, action)
        safety_critic_loss = F.mse_loss(current_Qh, target_Qh)

        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()

    # ------------------------------------------------------------------
    # Actor + λ(s) 更新（RCRL 的 Lagrangian）
    # ------------------------------------------------------------------
    def update_actor(self, state, action_taken=None):
        """
        RCRL 的 Lagrangian surrogate：
            J(π, λ) = E_s[ -Q_r(s, π(s)) + λ(s) Q_h(s, π(s)) ]
        我们用梯度下降来最小化 J，对 λ 使用梯度上升
        （实现时用 -E[λ(s) Q_h] 做 loss 即可）。
        """
        # 1. 采样当前策略动作
        action, log_prob, _ = self.policy.sample(state)

        # 2. 计算 Q_r(s, π(s))
        actor_Q1, actor_Q2 = self.critic(state, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # 3. 计算 Q_h(s, π(s))
        actor_Qh = self.safety_critic(state, action)

        # 4. λ(s) = Softplus(f(s))  ≥ 0
        #    —— 注意：这里用于 actor_loss 的 λ 要从图里 detach，
        #       避免在 actor 更新时给 lambda_net 也传梯度。
        safety_violation = actor_Qh - self.safety_threshold

        raw_lambda_for_actor = self.lambda_net(state).detach()
        lambda_s_for_actor = F.softplus(raw_lambda_for_actor)
        
        # 5. Actor loss（只更新 policy & alpha）
        actor_loss = torch.mean(
            self.alpha.detach() * log_prob
            - actor_Q
            + lambda_s_for_actor * safety_violation
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 6. α (entropy temperature) 更新（和 SAC 一样）
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 7. λ(s) 更新：max λ(s) s.t. V_h ≤ 0
        #    实现为最小化 -E[λ(s) Q_h] （相当于对 λ 梯度上升）
        #    —— 这里重新 forward 一次 lambda_net，构建新的计算图。
        raw_lambda = self.lambda_net(state)
        lambda_s = F.softplus(raw_lambda)

        self.lambda_optimizer.zero_grad()
        safety_violation_detached = (actor_Qh.detach() - self.safety_threshold)
        lambda_loss = -torch.mean(lambda_s * safety_violation_detached)
        lambda_loss.backward()
        self.lambda_optimizer.step()

    # ------------------------------------------------------------------
    # 外部调用的统一入口（和原 CAL 接口保持一致）
    # ------------------------------------------------------------------
    def update_parameters(self, memory, updates: int):
        """
        memory 是一个 tuple：
            (state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        其中 reward_batch.shape = (B, 2)，
            reward_batch[:, 0] = 环境 reward
            reward_batch[:, 1] = cost / constraint（这里当作 h(s)）
        """
        self.update_counter += 1

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)

        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        reward = reward_batch[:, 0:1]  # (B,1)
        cost = reward_batch[:, 1:2]    # (B,1) 作为 h(s)

        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # 更新 critic（Q_r 与 Q_h）
        self.update_critic(
            state_batch, action_batch, reward, cost, next_state_batch, mask_batch
        )

        # 更新 actor & λ(s)
        self.update_actor(state_batch, action_batch)

        # 更新 target 网络
        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_target, self.safety_critic, self.critic_tau)

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------
    def save_model(self, save_dir: Path, suffix: str = ""):
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        actor_path = save_dir / f"actor_{suffix}.pth"
        critics_path = save_dir / f"critics_{suffix}.pth"
        safetycritics_path = save_dir / f"safety_critics_{suffix}.pth"
        lambda_path = save_dir / f"lambda_{suffix}.pth"

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critics_path)
        torch.save(self.safety_critic.state_dict(), safetycritics_path)
        torch.save(self.lambda_net.state_dict(), lambda_path)

    def load_model(
        self,
        actor_path: Path,
        critics_path: Path,
        safetycritics_path: Path,
        lambda_path: Path = None,
    ):
        print(
            f"Loading models from {actor_path}, {critics_path}, {safetycritics_path}, {lambda_path}"
        )
        if actor_path is not None and os.path.exists(actor_path):
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critics_path is not None and os.path.exists(critics_path):
            self.critic.load_state_dict(torch.load(critics_path, map_location=self.device))
        if safetycritics_path is not None and os.path.exists(safetycritics_path):
            self.safety_critic.load_state_dict(
                torch.load(safetycritics_path, map_location=self.device)
            )
        if lambda_path is not None and os.path.exists(lambda_path):
            self.lambda_net.load_state_dict(
                torch.load(lambda_path, map_location=self.device)
            )
