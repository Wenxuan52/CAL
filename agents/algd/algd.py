import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.cal.utils import soft_update
from agents.cal.model import QNetwork, QcEnsemble, GaussianPolicy
from agents.algd.model import DiffusionScoreNetwork


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
        self.total_env_steps = 0

        # Safety related params
        self.c = args.c
        self.cost_lr_scale = 1.

        # Reward critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety critics
        self.safety_critics = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # Diffusion score policy
        self.score_model = DiffusionScoreNetwork(
            state_dim=num_inputs,
            action_dim=action_space.shape[0],
            hidden_size=args.hidden_size,
            num_hidden_layers=args.score_hidden_layers,
            t_embed_dim=args.t_dim,
            time_embedding=args.time_embedding,
            action_space=action_space,
        ).to(self.device)

        # Warmup Gaussian policy for initial stability
        self.gaussian_policy = GaussianPolicy(args, num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8))).to(self.device)
        self.log_lam.requires_grad = True

        self.kappa = 0

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()

        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)
        self.score_optimizer = torch.optim.Adam(self.score_model.parameters(), lr=args.lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

        # Diffusion / score matching params
        self.diffusion_K = args.diffusion_K
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.n_mc = args.n_mc
        self.beta = args.beta
        self.diffusion_step_scale = args.diffusion_step_scale
        self.deterministic_eval = args.deterministic_eval
        self.warmup_steps = args.algd_warmup_steps

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Set target cost
        if args.safetygym:
            self.target_cost = args.cost_lim * (1 - self.safety_discount**args.epoch_length) / (
                1 - self.safety_discount) / args.epoch_length if self.safety_discount < 1 else args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)

    def train(self, training=True):
        self.training = training
        self.gaussian_policy.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)
        self.score_model.train(training)

    @property
    def lam(self):
        return self.log_lam.exp()

    def _sigma_schedule(self):
        return torch.linspace(self.sigma_max, self.sigma_min, self.diffusion_K, device=self.device)

    def _time_for_step(self, step_idx: int):
        if self.diffusion_K <= 1:
            return torch.zeros(1, 1, device=self.device)
        return torch.full((1, 1), float(step_idx) / float(self.diffusion_K - 1), device=self.device)

    def _reverse_diffusion(self, state_tensor: torch.Tensor, eval: bool = False):
        sigmas = self._sigma_schedule()
        action = torch.randn((state_tensor.size(0), self.score_model.action_dim), device=self.device) * self.sigma_max

        for idx, sigma in enumerate(sigmas):
            t = self._time_for_step(idx).expand(state_tensor.size(0), 1)
            score = self.score_model(state_tensor, action, t)
            action = action + self.diffusion_step_scale * (sigma ** 2) * score
            if (idx < len(sigmas) - 1) and not (eval and self.deterministic_eval):
                noise = torch.randn_like(action) * sigma * np.sqrt(self.diffusion_step_scale)
                action = action + noise

        action = torch.tanh(action) * self.score_model.action_scale + self.score_model.action_bias
        return action

    def select_action(self, state, eval=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            self.total_env_steps += 1

        if (not eval) and self.total_env_steps <= self.warmup_steps:
            action, _, _ = self.gaussian_policy.sample(state_tensor)
        else:
            action = self._reverse_diffusion(state_tensor, eval=eval)
        return action.detach().cpu().numpy()[0]

    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action = self._reverse_diffusion(next_state, eval=False).detach()
        next_log_prob = torch.zeros_like(mask)  # diffusion policy has implicit density

        # Reward critics update
        current_Q1, current_Q2 = self.critic(state, action)
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - next_log_prob
        target_Q = reward + (mask * self.discount * target_V)
        target_Q = target_Q.detach()

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Safety critics update
        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
        current_QCs = self.safety_critics(state, action)  # shape(E, B, 1)
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
        batch_size = state.size(0)

        # Reward critic
        actor_Q1, actor_Q2 = self.critic(state, action_taken)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # Safety critic
        actor_QCs = self.safety_critics(state, action_taken)
        actor_std, actor_mean = torch.std_mean(actor_QCs, dim=0)
        if self.args.qc_ens_size == 1:
            actor_std = torch.zeros_like(actor_std).to(self.device)
        actor_QC = actor_mean + self.args.k * actor_std

        advantage = actor_Q - self.lam.detach() * actor_QC

        # Diffusion score matching
        repeated_states = state.repeat_interleave(self.n_mc, dim=0)
        repeated_actions = action_taken.repeat_interleave(self.n_mc, dim=0)
        times = torch.rand((batch_size * self.n_mc, 1), device=self.device)
        sigmas = self.sigma_min + times * (self.sigma_max - self.sigma_min)
        noise = torch.randn_like(repeated_actions) * sigmas
        noisy_actions = repeated_actions + noise

        target_score = -(noisy_actions - repeated_actions) / (sigmas ** 2)
        predicted_score = self.score_model(repeated_states, noisy_actions, times)

        score_loss = (predicted_score - target_score).pow(2).mean(dim=-1)
        score_loss = score_loss.view(batch_size, self.n_mc)

        weights = torch.softmax(self.beta * advantage.detach().squeeze(-1), dim=0).clamp(min=1e-4)
        weights = weights / weights.sum()
        weighted_loss = (score_loss.mean(dim=1) * weights).sum()

        self.score_optimizer.zero_grad()
        weighted_loss.backward()
        self.score_optimizer.step()

        # Update Lagrange multiplier
        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.lam * (self.target_cost - actor_QC).detach())
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
