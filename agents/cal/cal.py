import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

from pathlib import Path

from agents.base_agent import Agent

# model.py
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

class GaussianPolicy(nn.Module):
    # MEAN_CLAMP_MIN = -5
    # MEAN_CLAMP_MAX = 5
    # COV_CLAMP_MIN = -5
    # COV_CLAMP_MAX = 20
    def __init__(self, args, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.) # [1]*a_dim
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.) # [0]*a_dim
        self.log_sig_max = LOG_SIG_MAX

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=self.log_sig_max)
        return mean, log_std

    def get_a_mean(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def sample_multiple_actions(self, state, n_pars):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.sample((n_pars, ))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        log_std = torch.log(log_std.exp() * self.action_scale)
        return actions, mean, log_std, x_t
    
    def calibrate_log_prob(self, normal, x_t):
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        calibrated_log_prob = log_prob - torch.log(self.action_scale * (1 - torch.tanh(x_t).pow(2)) + epsilon)
        return calibrated_log_prob.sum(-1)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

# utils.py
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Agent
class CALAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda")
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args

        self.update_counter = 0

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

        # policy
        self.policy = GaussianPolicy(args, num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8))).to(self.device)
        self.log_lam.requires_grad = True

        self.kappa = 0

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

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
        self.policy.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)


    @property
    def alpha(self):
        return self.log_alpha.exp()


    @property
    def lam(self):
        return self.log_lam.exp()


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action, next_log_prob, _ = self.policy.sample(next_state)

        # Reward critics update
        current_Q1, current_Q2 = self.critic(state, action)
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
        target_Q = reward + (mask * self.discount * target_V)
        target_Q = target_Q.detach()

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Safety critics update
        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)
        current_QCs = self.safety_critics(state, action) # shape(E, B, 1)
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
        action, log_prob, _ = self.policy.sample(state)

        # Reward critic
        actor_Q1, actor_Q2 = self.critic(state, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # Safety critic
        actor_QCs = self.safety_critics(state, action)
        with torch.no_grad():
            current_QCs = self.safety_critics(state, action_taken)
            current_std, current_mean = torch.std_mean(current_QCs, dim=0)
            if self.args.qc_ens_size == 1:
                current_std = torch.zeros_like(current_mean).to(self.device)
            current_QC = current_mean + self.args.k * current_std
        actor_std, actor_mean = torch.std_mean(actor_QCs, dim=0)
        if self.args.qc_ens_size == 1:
            actor_std = torch.zeros_like(actor_std).to(self.device)
        actor_QC = actor_mean + self.args.k * actor_std

        # Compute gradient rectification
        self.rect = self.c * torch.mean(self.target_cost - current_QC)
        self.rect = torch.clamp(self.rect.detach(), max=self.lam.item())

        # Policy loss
        lam = self.lam.detach()
        actor_loss = torch.mean(
            self.alpha.detach() * log_prob
            - actor_Q
            + (lam - self.rect) * actor_QC
        )

        # Optimize the policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = torch.mean(self.alpha * (-log_prob - self.target_entropy).detach())
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

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
