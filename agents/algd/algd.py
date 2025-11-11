import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from agents.base_agent import Agent
from agents.cal.model import QNetwork, QcEnsemble
from agents.cal.utils import soft_update
from .model import ScoreNetwork


class ALGDAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args

        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float32)

        self.update_counter = 0

        # Safety related params
        self.c = args.c
        self.cost_lr_scale = 1.0

        # Reward critic
        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety critics
        self.safety_critics = QcEnsemble(num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(num_inputs, self.action_dim, args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # Diffusion-based score network
        self.score_network = ScoreNetwork(num_inputs, self.action_dim, args.hidden_size, args.t_dim).to(self.device)

        # Diffusion hyperparameters
        self.K = max(int(args.diffusion_K), 1)
        self.sigmas = torch.linspace(args.sigma_min, args.sigma_max, self.K + 1, device=self.device)
        self.sigma_sq = self.sigmas ** 2
        self.sigma_sq_diff = self.sigma_sq[1:] - self.sigma_sq[:-1]
        self.diffusion_step_scale = args.diffusion_step_scale

        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)
        self.score_optimizer = torch.optim.Adam(self.score_network.parameters(), lr=args.lr)

        # Lagrange multiplier
        self.lambda_param = torch.tensor(0.0, device=self.device)
        self.lambda_lr = args.lr

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Set target cost
        if args.safetygym:
            if self.safety_discount < 1:
                factor = (1 - self.safety_discount ** args.epoch_length) / (1 - self.safety_discount)
                self.target_cost = args.cost_lim * factor / args.epoch_length
            else:
                self.target_cost = args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)

    def train(self, training=True):
        self.training = training
        self.critic.train(training)
        self.safety_critics.train(training)
        self.score_network.train(training)

    def select_action(self, state, eval=False):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.randn((1, self.action_dim), device=self.device) * self.sigmas[-1]

            for step in range(self.K, 0, -1):
                tau_norm = torch.full((1,), step / self.K, device=self.device)
                score = self.score_network(state_tensor, action, tau_norm)
                step_size = self.sigma_sq_diff[step - 1] * self.diffusion_step_scale
                noise = torch.zeros_like(action) if (eval and self.args.deterministic_eval) else torch.randn_like(action)
                action = action - step_size * score + torch.sqrt(torch.clamp(step_size, min=1e-12)) * noise

            action = action.clamp(self.action_low, self.action_high)
        return action.detach().cpu().numpy()[0]

    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action = self.sample_diffusion_action(next_state)

        next_Q1, next_Q2 = self.critic_target(next_state, next_action)
        next_V = torch.min(next_Q1, next_Q2)
        target_Q = reward + (mask * self.discount * next_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
        current_QCs = self.safety_critics(state, action)
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

    def sample_diffusion_action(self, state_batch):
        with torch.no_grad():
            batch_size = state_batch.size(0)
            action = torch.randn((batch_size, self.action_dim), device=self.device) * self.sigmas[-1]
            state = state_batch
            for step in range(self.K, 0, -1):
                tau_norm = torch.full((batch_size,), step / self.K, device=self.device)
                score = self.score_network(state, action, tau_norm)
                step_size = self.sigma_sq_diff[step - 1] * self.diffusion_step_scale
                noise = torch.randn_like(action)
                action = action - step_size * score + torch.sqrt(torch.clamp(step_size, min=1e-12)) * noise
            return action.clamp(self.action_low, self.action_high)

    def update_score_network(self, state_batch, action_batch):
        batch_size = state_batch.size(0)
        device = self.device

        tau_indices = torch.randint(1, self.K + 1, (batch_size,), device=device, dtype=torch.long)
        tau_norm = tau_indices.float() / self.K
        sigma_tau = self.sigmas[tau_indices]

        noise = torch.randn_like(action_batch)
        a_tau = action_batch + sigma_tau.unsqueeze(-1) * noise

        mc_noise = torch.randn(self.args.n_mc, batch_size, self.action_dim, device=device)
        a0_samples = a_tau.unsqueeze(0) + sigma_tau.view(1, -1, 1) * mc_noise
        a0_samples = a0_samples.requires_grad_(True)

        repeated_states = state_batch.unsqueeze(0).repeat(self.args.n_mc, 1, 1)
        flat_states = repeated_states.reshape(-1, state_batch.size(1))
        flat_actions = a0_samples.reshape(-1, self.action_dim)

        q1, q2 = self.critic(flat_states, flat_actions)
        min_q = torch.min(q1, q2).view(self.args.n_mc, batch_size, 1)
        qc_values = self.safety_critics(flat_states, flat_actions)
        qc_mean = qc_values.mean(dim=0).view(self.args.n_mc, batch_size, 1)

        lambda_value = self.lambda_param.detach()
        lagrangian = -min_q + lambda_value * (qc_mean - self.target_cost)

        weights = torch.softmax((-lagrangian.detach().squeeze(-1)) / self.args.beta, dim=0)

        grads = torch.autograd.grad(lagrangian.sum(), a0_samples, retain_graph=False, create_graph=False)[0]
        score_target = (-weights.unsqueeze(-1) / self.args.beta * grads).sum(dim=0).detach()

        score_pred = self.score_network(state_batch, a_tau, tau_norm)
        score_loss = F.mse_loss(score_pred, score_target)

        self.score_optimizer.zero_grad()
        score_loss.backward()
        self.score_optimizer.step()

    def update_lambda(self, state_batch, action_batch):
        with torch.no_grad():
            qc_values = self.safety_critics(state_batch, action_batch)
            bar_qc = qc_values.mean(dim=0)
            violation = bar_qc - self.target_cost
            lambda_update = violation.mean() * self.lambda_lr
            self.lambda_param = torch.clamp(self.lambda_param + lambda_update, min=0.0)

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
        self.update_score_network(state_batch, action_batch)
        self.update_lambda(state_batch, action_batch)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

    def save_model(self, save_dir: Path, suffix: str = ""):
        actor_path = save_dir / f"score_{suffix}.pth"
        critics_path = save_dir / f"critics_{suffix}.pth"
        safetycritics_path = save_dir / f"safetycritics_{suffix}.pth"

        print(f"[Model] Saving models to:\n  {actor_path}\n  {critics_path}\n  {safetycritics_path}")

        torch.save(self.score_network.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critics_path)
        torch.save(self.safety_critics.state_dict(), safetycritics_path)

    def load_model(self, score_path, critics_path, safetycritics_path):
        print('Loading models from {}, {}, and {}'.format(score_path, critics_path, safetycritics_path))
        if score_path is not None:
            self.score_network.load_state_dict(torch.load(score_path, map_location=self.device))
        if critics_path is not None:
            self.critic.load_state_dict(torch.load(critics_path, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
        if safetycritics_path is not None:
            self.safety_critics.load_state_dict(torch.load(safetycritics_path, map_location=self.device))
            self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())
