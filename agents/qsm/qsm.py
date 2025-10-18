# agents/qsm/qsm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import Agent
from agents.qsm.model import QNetwork, DiffusionScoreModel
from agents.qsm.utils import ddpm_forward_process, cosine_beta_schedule, vp_beta_schedule, soft_update, ddpm_sampler



class QSMAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.discount = args.gamma
        self.tau = args.tau
        self.T = args.T
        self.M_q = args.M_q
        self.actor_lr = args.lr
        self.critic_lr = args.lr
        self.ddpm_temperature = args.ddpm_temperature
        self.beta_schedule = args.beta_schedule

        # ====== Networks ======
        self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # ====== Diffusion Score Model ======
        self.score_model = DiffusionScoreModel(
            state_dim=num_inputs,
            action_dim=action_space.shape[0],
            hidden_dim=args.hidden_size,
            time_dim=args.time_dim,
        ).to(self.device)

        # ====== Optimizers ======
        self.actor_optim = torch.optim.Adam(self.score_model.parameters(), lr=self.actor_lr)
        self.critic_optim_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_optim_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # ====== Diffusion Params ======
        if self.beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(self.T)
        elif self.beta_schedule == 'vp':
            self.betas = vp_beta_schedule(self.T)
        else:
            self.betas = torch.linspace(1e-4, 2e-2, self.T)

        self.betas = self.betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(self.device)

    # =====================
    #     Q Network update
    # =====================
    def update_critic(self, state, action, reward, next_state, mask):
        with torch.no_grad():
            # Sample next action
            next_action = self.sample_action(next_state)

            # Calculate the Q-values of target critics
            next_q1 = self.target_critic_1(next_state, next_action)
            next_q2 = self.target_critic_2(next_state, next_action)

            # Take the minimum Q to reduce overestimation
            target_v = torch.min(next_q1, next_q2)
            target_q = reward + mask * self.discount * target_v

        # The Q value of the current critic
        cur_q1 = self.critic_1(state, action)
        cur_q2 = self.critic_2(state, action)

        # critic loss
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

        # Optimizer Updates
        self.critic_optim_1.zero_grad()
        self.critic_optim_2.zero_grad()
        critic_loss.backward()
        self.critic_optim_1.step()
        self.critic_optim_2.step()

        return critic_loss.item()


    # =====================
    #     Actor update
    # =====================
    def update_actor(self, state, action):
        B = state.size(0)
        A = action.size(1)
        device = self.device

        # sample time step
        t = torch.randint(0, self.T, (B,), device=device).float()

        # forward diffusion (add noise)
        noise = torch.randn_like(action, device=device)
        alpha_hat = torch.tensor(self.alpha_hats, device=device)[t.long()].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_hat) * action + torch.sqrt(1 - alpha_hat) * noise

        # compute ∇_a Q
        noisy_action.requires_grad_(True)
        q1 = self.critic_1(state, noisy_action).sum()
        q2 = self.critic_2(state, noisy_action).sum()
        dq_da_1 = torch.autograd.grad(q1, noisy_action, create_graph=True)[0]
        dq_da_2 = torch.autograd.grad(q2, noisy_action, create_graph=True)[0]
        dq_da = 0.5 * (dq_da_1 + dq_da_2).detach()

        # DDPM score prediction
        eps_pred = self.score_model(state, noisy_action, t.unsqueeze(1))

        # Q-Score Matching loss
        actor_loss = ((-self.M_q * dq_da - eps_pred) ** 2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item()

    def sample_action(self, state):
        with torch.no_grad():
            return ddpm_sampler(
                self.score_model,
                state,
                self.T,
                self.alphas,
                self.alpha_hats,
                self.betas,
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
        with torch.no_grad():
            action = self.sample_action(state_tensor)  # DDPM-style denoising
        return action.cpu().detach().numpy()[0]

    def update(self, replay_buffer, logger=None, step=0):
        state, action, reward, next_state, done = replay_buffer.sample(self.args.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        mask = 1 - torch.FloatTensor(done).unsqueeze(1).to(self.device)

        critic_loss = self.update_critic(state, action, reward, next_state, mask)
        actor_loss = self.update_actor(state, action)

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

        # if logger is not None:
        #     logger.log("train/critic_loss", critic_loss, step)
        #     logger.log("train/actor_loss", actor_loss, step)
