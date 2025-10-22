import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.ssm_gauss.model import QNetwork, QcNetwork, GaussianPolicy
from agents.ssm_gauss.utils import soft_gate, soft_update


class SSM_GaussAgent(Agent):
    """Safe Score Matching agent that uses a Gaussian policy."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.tau = args.tau
        self.update_counter = 0

        # Networks
        action_dim = action_space.shape[0]
        self.critic = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.qc = QcNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.qc_target = QcNetwork(num_inputs, action_dim, args.hidden_size).to(self.device)
        self.qc_target.load_state_dict(self.qc.state_dict())

        self.policy = GaussianPolicy(args, num_inputs, action_dim, args.hidden_size, action_space).to(self.device)

        # Entropy and safety parameters
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(np.prod(action_space.shape))

        self.beta = getattr(args, "safe_beta", 1.0)
        self.kappa = getattr(args, "safe_thresh", 0.0)
        self.g_alpha = getattr(args, "gate_alpha", 5.0)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        qc_lr = getattr(args, "qc_lr", args.lr)
        self.qc_optimizer = torch.optim.Adam(self.qc.parameters(), lr=qc_lr)

        print(
            f"[SSM_Gauss] initialized on {self.device} (beta={self.beta}, kappa={self.kappa}, gate_alpha={self.g_alpha})"
        )

        self.train()
        self.critic_target.train()
        self.qc_target.train()

    def train(self, training: bool = True):
        self.training = training
        self.critic.train(training)
        self.qc.train(training)
        self.policy.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, eval=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval:
            _, _, action = self.policy.sample(state_tensor)
        else:
            action, _, _ = self.policy.sample(state_tensor)
        return action.detach().cpu().numpy()[0]

    def update_reward_critic(self, state, action, reward, next_state, mask):
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            q_target = reward + mask * self.discount * target_v
        current_q1, current_q2 = self.critic(state, action)
        loss_q = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()
        return loss_q.item()

    def update_safety_critic(self, state, action, cost, next_state, mask):
        with torch.no_grad():
            next_action, _, _ = self.policy.sample(next_state)
            next_qc = self.qc_target(next_state, next_action)
            blended = torch.maximum(cost, next_qc)
            qc_target = (1 - self.safety_discount) * cost + self.safety_discount * blended
            qc_target = mask * qc_target + (1 - mask) * cost
        current_qc = self.qc(state, action)
        loss_qc = F.mse_loss(current_qc, qc_target)

        self.qc_optimizer.zero_grad()
        loss_qc.backward()
        self.qc_optimizer.step()
        return loss_qc.item()

    def update_actor(self, state):
        action, log_prob, _ = self.policy.sample(state)
        q1, q2 = self.critic(state, action)
        qc_value = self.qc(state, action)

        gate = soft_gate(qc_value, kappa=self.kappa, alpha=self.g_alpha)
        q_min = torch.min(q1, q2)

        loss_pi = (
            self.alpha.detach() * log_prob
            - gate * q_min
            + (1 - gate) * self.beta * F.softplus(qc_value)
        ).mean()

        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return loss_pi.item()

    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state = torch.FloatTensor(state_batch).to(self.device)
        next_state = torch.FloatTensor(next_state_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch[:, 0]).unsqueeze(1).to(self.device)
        cost = torch.FloatTensor(reward_batch[:, 1]).unsqueeze(1).to(self.device)
        mask = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        q_loss = self.update_reward_critic(state, action, reward, next_state, mask)
        qc_loss = self.update_safety_critic(state, action, cost, next_state, mask)
        pi_loss = self.update_actor(state)

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.qc_target, self.qc, self.tau)

        return {"loss_q": q_loss, "loss_qc": qc_loss, "loss_pi": pi_loss}

    def act(self, obs, sample=False):
        return self.select_action(obs, eval=not sample)

    def save_model(self, save_dir, suffix=""):
        actor_path = save_dir / f"ssm_gauss_actor_{suffix}.pth"
        critic_path = save_dir / f"ssm_gauss_critic_{suffix}.pth"
        safety_path = save_dir / f"ssm_gauss_safety_{suffix}.pth"

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.qc.state_dict(), safety_path)

    def load_model(self, actor_path=None, critic_path=None, safety_path=None):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
        if safety_path is not None:
            self.qc.load_state_dict(torch.load(safety_path, map_location=self.device))
            self.qc_target.load_state_dict(self.qc.state_dict())
