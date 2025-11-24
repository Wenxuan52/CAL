import torch

from agents.cal.cal import CALAgent


class ALGDAgent(CALAgent):
    """CAL variant with score matching regularization on the policy."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.score_matching_coef = args.algd_score_coef
        self.score_sigma_min = torch.tensor(args.algd_score_sigma_min, device=self.device)
        self.score_sigma_max = torch.tensor(args.algd_score_sigma_max, device=self.device)
        self.diffusion_steps = args.algd_diffusion_steps
        self.langevin_scale = args.algd_langevin_scale

    def _diffusion_policy_sample(self, state, eval=False):
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        sigma_schedule = torch.exp(
            torch.linspace(
                torch.log(self.score_sigma_max),
                torch.log(self.score_sigma_min),
                self.diffusion_steps,
                device=self.device,
            )
        )

        x = x_t
        for sigma in sigma_schedule:
            x.requires_grad_(True)
            log_prob_x = normal.log_prob(x).sum(1)
            score = torch.autograd.grad(log_prob_x.sum(), x, create_graph=True)[0]

            step_size = self.langevin_scale * (sigma ** 2)
            noise = torch.zeros_like(x) if eval else torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        y = torch.tanh(x)
        action = y * self.policy.action_scale + self.policy.action_bias
        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(self.policy.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, normal

    def update_actor(self, state, action_taken):
        # Diffusion policy action and log-prob
        action, log_prob, x_t, normal = self._diffusion_policy_sample(state)

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

        # Diffusion-based score matching on latent action
        eps = torch.randn_like(x_t)
        log_sigma = torch.rand_like(x_t) * (
            torch.log(self.score_sigma_max) - torch.log(self.score_sigma_min)
        ) + torch.log(self.score_sigma_min)
        sigma = torch.exp(log_sigma)
        noisy_x = (x_t + sigma * eps).detach()
        noisy_x.requires_grad_(True)
        noisy_log_prob = normal.log_prob(noisy_x)
        noisy_log_prob = noisy_log_prob.sum(1, keepdim=True)
        policy_score = torch.autograd.grad(
            noisy_log_prob.sum(), noisy_x, create_graph=True, retain_graph=True
        )[0]
        target_score = -eps / sigma
        score_matching_loss = 0.5 * torch.mean((policy_score - target_score) ** 2)

        # Policy loss with constraint adjustment
        lam = self.lam.detach()
        actor_loss = torch.mean(
            self.alpha.detach() * log_prob
            - actor_Q
            + (lam - self.rect) * actor_QC
        )
        actor_loss = actor_loss + self.score_matching_coef * score_matching_loss

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

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _, _ = self._diffusion_policy_sample(state, eval=eval)
        return action.detach().cpu().numpy()[0]
