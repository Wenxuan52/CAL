import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.ssm.model import DiffusionPolicy, QNetwork
from agents.ssm.utils import safe_ddpm_sample, soft_update


class SSMAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.0
        ).to(self.device)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.0
        ).to(self.device)
        self._norm_scale = torch.where(
            self.action_scale.abs() > 1e-6,
            self.action_scale,
            torch.ones_like(self.action_scale),
        )

        self.action_dim = action_space.shape[0]
        self.guidance_alpha = args.alpha_sm
        self.guidance_beta = args.beta_sm if hasattr(args, "beta_sm") else 3.0
        self.guidance_step = 0.05
        self.noise_scale = getattr(args, "ddpm_temperature", 1.0)
        self.safe_threshold = getattr(args, "safe_threshold", 0.0)
        self.target_sampling_steps = getattr(args, "target_ddpm_steps", max(1, args.T // 2))
        self.score_weight = getattr(args, "M_q", 1.0)
        self.vh_samples = getattr(args, "vh_samples", 4)

        # Critics
        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.safety_critic = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.safety_critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())

        # Diffusion policy
        self.policy = DiffusionPolicy(num_inputs, self.action_dim, args.hidden_size, T=args.T).to(self.device)
        self.policy_target = DiffusionPolicy(num_inputs, self.action_dim, args.hidden_size, T=args.T).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=args.qc_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        self.update_counter = 0

        self.train()
        self.critic_target.train()
        self.safety_critic_target.train()
        self.policy_target.train()

    def train(self, training: bool = True):
        self.training = training
        self.critic.train(training)
        self.safety_critic.train(training)
        self.policy.train(training)

    def select_action(self, state, eval=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = safe_ddpm_sample(
            self.policy,
            self.critic,
            self.safety_critic,
            state_tensor,
            alpha=self.guidance_alpha,
            beta=self.guidance_beta,
            step_size=self.guidance_step,
            noise_scale=self.noise_scale if not eval else 0.0,
            safe_threshold=self.safe_threshold,
            guidance=True,
        )
        action = torch.clamp(action, -1.0, 1.0)
        action = action * self.action_scale + self.action_bias
        return action.squeeze(0).cpu().numpy()

    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_cost = torch.FloatTensor(reward_batch).to(self.device)
        reward_batch = reward_cost[:, 0:1]
        cost_batch = reward_cost[:, 1:2]
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(-1)

        action_batch = (action_batch - self.action_bias) / self._norm_scale
        action_batch = torch.clamp(action_batch, -1.0, 1.0)

        # Critic update (reward)
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        with torch.no_grad():
            next_action = safe_ddpm_sample(
                self.policy_target,
                self.critic_target,
                self.safety_critic_target,
                next_state_batch,
                alpha=self.guidance_alpha,
                beta=self.guidance_beta,
                step_size=self.guidance_step,
                noise_scale=self.noise_scale,
                safe_threshold=self.safe_threshold,
                guidance=False,
                num_steps=self.target_sampling_steps,
            )
            next_action = torch.clamp(next_action, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + mask_batch * self.discount * target_q
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Safety critic update
        current_qh1, current_qh2 = self.safety_critic(state_batch, action_batch)
        with torch.no_grad():
            next_vh = self._estimate_vh(
                next_state_batch,
                policy=self.policy_target,
                critic=self.safety_critic_target,
                critic_for_guidance=self.critic_target,
            )
            target_qh = (1 - self.safety_discount) * cost_batch \
             + self.safety_discount * torch.maximum(cost_batch, next_vh)
        safety_loss = F.mse_loss(current_qh1, target_qh) + F.mse_loss(current_qh2, target_qh)
        self.safety_critic_optimizer.zero_grad()
        safety_loss.backward()
        self.safety_critic_optimizer.step()

        # Diffusion policy update
        with torch.no_grad():
            vh_state = self._estimate_vh(
                state_batch,
                policy=self.policy,
                critic=self.safety_critic,
                critic_for_guidance=self.critic,
            ).squeeze(-1)
        policy_loss = self.policy.loss(
            state_batch,
            action_batch,
            self.critic,
            self.safety_critic,
            alpha=self.guidance_alpha,
            beta=self.guidance_beta,
            safe_threshold=self.safe_threshold,
            m_q=self.score_weight,
            vh_state=vh_state,
        )
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.safety_critic_target, self.safety_critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

    def _estimate_vh(
        self,
        state_batch: torch.Tensor,
        policy,
        critic,
        critic_for_guidance,
    ) -> torch.Tensor:
        num_samples = max(1, self.vh_samples)
        if num_samples == 1:
            expanded_states = state_batch
        else:
            expanded_states = state_batch.repeat_interleave(num_samples, dim=0)

        actions = safe_ddpm_sample(
            policy,
            critic_for_guidance,
            critic,
            expanded_states,
            alpha=self.guidance_alpha,
            beta=self.guidance_beta,
            step_size=self.guidance_step,
            noise_scale=self.noise_scale,
            safe_threshold=self.safe_threshold,
            guidance=False,
            num_steps=self.target_sampling_steps,
        )

        qh1, qh2 = critic(expanded_states, actions)
        qh_min = torch.min(qh1, qh2)

        if num_samples == 1:
            return qh_min

        vh = qh_min.view(num_samples, state_batch.size(0), 1).mean(dim=0)
        return vh

    def act(self, obs, sample=False):
        return self.select_action(obs, eval=not sample)

    def save_model(self, save_dir, suffix=""):
        actor_path = save_dir / f"ssm_actor_{suffix}.pth"
        critic_path = save_dir / f"ssm_critic_{suffix}.pth"
        safety_path = save_dir / f"ssm_safety_critic_{suffix}.pth"
        policy_path = save_dir / f"ssm_diffusion_{suffix}.pth"

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.safety_critic.state_dict(), safety_path)
        torch.save(self.policy_target.state_dict(), policy_path)

    def load_model(self, actor_path=None, critic_path=None, safety_path=None, policy_path=None):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if safety_path is not None:
            self.safety_critic.load_state_dict(torch.load(safety_path))
        if policy_path is not None:
            self.policy_target.load_state_dict(torch.load(policy_path))