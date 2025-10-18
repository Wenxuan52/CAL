import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import Agent
from agents.ssm.model import QcEnsemble          # safety critic ensemble
from agents.ssm.model import QNetwork, DiffusionScoreModel
from agents.ssm.utils import (
    cosine_beta_schedule,
    vp_beta_schedule,
    soft_update,
    safe_ddpm_sampler,
)

# ============================================================
# Safe Score Matching (SSM) Agent
# ============================================================
class SSMAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        # ---------- basic hyperparameters ----------
        self.discount = args.gamma
        self.tau = args.tau
        self.T = args.T
        self.M_q = args.M_q
        self.actor_lr = args.lr
        self.critic_lr = args.lr
        self.ddpm_temperature = args.ddpm_temperature
        self.beta_schedule = args.beta_schedule

        # ---------- main critics ----------
        self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # ---------- diffusion score model ----------
        self.score_model = DiffusionScoreModel(
            state_dim=num_inputs,
            action_dim=action_space.shape[0],
            hidden_dim=args.hidden_size,
            time_dim=args.time_dim,
        ).to(self.device)

        # ---------- optimizers ----------
        self.actor_optim = torch.optim.Adam(self.score_model.parameters(), lr=self.actor_lr)
        self.critic_optim_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_optim_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # ---------- diffusion schedule ----------
        if self.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(self.T)
        elif self.beta_schedule == "vp":
            self.betas = vp_beta_schedule(self.T)
        else:
            self.betas = torch.linspace(1e-4, 2e-2, self.T)
        self.betas = self.betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(self.device)

        # ============================================================
        # Safety-related components (new for SSM)
        # ============================================================
        # Q_h(s,a) ensemble critic to approximate constraint value
        self.safety_critic = QcEnsemble(num_inputs, action_space.shape[0], args.ensemble_size, hidden_size=args.hidden_size).to(self.device)
        self.safety_target = QcEnsemble(num_inputs, action_space.shape[0], args.ensemble_size, hidden_size=args.hidden_size).to(self.device)
        self.safety_target.load_state_dict(self.safety_critic.state_dict())
        self.safety_optim = torch.optim.Adam(self.safety_critic.parameters(), lr=args.qc_lr)

        # safety control parameters
        self.safety_discount = getattr(args, "safety_gamma", 0.99)     # γ_h
        self.safe_threshold = getattr(args, "safe_threshold", 0.0)     # Q_h <= 0 safe
        self.alpha_sm = getattr(args, "alpha_sm", 1.0)                 # safe gradient scaling
        self.beta_sm = getattr(args, "beta_sm", 1.5)                   # unsafe gradient scaling

    # ============================================================
    # critic update (reward + safety critic)
    # ============================================================
    def update_critic(self, state, action, reward, cost, next_state, mask):
        # ---------- reward critics (Q) ----------
        with torch.no_grad():
            next_action = self.sample_action(next_state)
            q1_next, _ = self.target_critic_1(next_state, next_action)
            q2_next, _ = self.target_critic_2(next_state, next_action)
            target_v = torch.min(q1_next, q2_next)
            target_q = reward + mask * self.discount * target_v

        q1, _ = self.critic_1(state, action)
        q2, _ = self.critic_2(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim_1.zero_grad()
        self.critic_optim_2.zero_grad()
        critic_loss.backward()
        self.critic_optim_1.step()
        self.critic_optim_2.step()

        # ---------- safety critic (Q_h) ----------
        with torch.no_grad():
            next_action = self.sample_action(next_state)
            qh_next = self.safety_target(next_state, next_action).mean(0)  # ensemble mean
            target_qh = cost + mask * self.safety_discount * qh_next

        qh = self.safety_critic(state, action).mean(0)
        safety_loss = F.mse_loss(qh, target_qh)

        self.safety_optim.zero_grad()
        safety_loss.backward()
        self.safety_optim.step()

        return critic_loss.item(), safety_loss.item()

    # ============================================================
    # actor update (core of SSM)
    # ============================================================
    def update_actor(self, state, action):
        B = state.size(0)
        device = self.device

        # ---------- sample time t ----------
        t = torch.randint(0, self.T, (B,), device=device).float()

        # ---------- forward diffusion ----------
        noise = torch.randn_like(action)
        alpha_hat = self.alpha_hats[t.long()].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_hat) * action + torch.sqrt(1 - alpha_hat) * noise
        noisy_action.requires_grad_(True)

        # ---------- compute gradients ----------
        q1, _ = self.critic_1(state, noisy_action)
        q2, _ = self.critic_2(state, noisy_action)
        dq_da_1 = torch.autograd.grad(q1.sum(), noisy_action, create_graph=True)[0]
        dq_da_2 = torch.autograd.grad(q2.sum(), noisy_action, create_graph=True)[0]
        dq_da = 0.5 * (dq_da_1 + dq_da_2).detach()  # ∇a Q field

        qh = self.safety_critic(state, noisy_action).mean(0)
        dq_da_h = torch.autograd.grad(qh.sum(), noisy_action, create_graph=True)[0].detach()  # ∇a Q_h field

        # ---------- region masks ----------
        safe_mask = (qh <= self.safe_threshold).float()   # inside safe set
        unsafe_mask = 1.0 - safe_mask

        # ---------- target score field ----------
        target_field = self.alpha_sm * dq_da * safe_mask + (-self.beta_sm) * dq_da_h * unsafe_mask

        # ---------- DDPM score prediction ----------
        eps_pred = self.score_model(state, noisy_action, t.unsqueeze(1))

        # ---------- SSM loss ----------
        actor_loss = ((target_field - eps_pred) ** 2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item()

    # ============================================================
    # sample action using learned diffusion model
    # ============================================================
    def sample_action(self, state):
        with torch.no_grad():
            return safe_ddpm_sampler(
                model=self.score_model,
                state=state,
                T=self.T,
                alphas=self.alphas,
                alpha_hats=self.alpha_hats,
                betas=self.betas,
                safety_critic=self.safety_critic,
                safe_threshold=self.safe_threshold,
                step_size=0.1,
                temperature=self.ddpm_temperature,
                action_dim=self.score_model.action_dim,
                device=self.device,
            )
    
    # ============================================================
    # action selection for environment interaction
    # ============================================================
    def select_action(self, state, eval=False):
        """
        Environment-facing API: choose an action given a single state.
        If eval=False, sample a stochastic action using diffusion policy.
        If eval=True, use a deterministic action (e.g., last step of denoising).
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.sample_action(state_tensor)  # DDPM-style denoising
        return action.detach().cpu().numpy()[0]


    # ============================================================
    # training update per iteration
    # ============================================================
    def update(self, replay_buffer, logger=None, step=0):
        # sample from buffer: expect (s,a,r,c,s',done)
        state, action, reward, cost, next_state, done = replay_buffer.sample(self.args.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        cost = torch.FloatTensor(cost).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        mask = 1 - torch.FloatTensor(done).unsqueeze(1).to(self.device)

        critic_loss, safety_loss = self.update_critic(state, action, reward, cost, next_state, mask)
        actor_loss = self.update_actor(state, action)

        # target updates
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        soft_update(self.safety_target, self.safety_critic, self.tau)

        # logging
        if logger is not None:
            logger.log("train/critic_loss", critic_loss, step)
            logger.log("train/safety_loss", safety_loss, step)
            logger.log("train/actor_loss", actor_loss, step)

        return {
            "critic_loss": critic_loss,
            "safety_loss": safety_loss,
            "actor_loss": actor_loss,
        }
