import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from agents.base_agent import Agent
from agents.algd.model import QNetwork, QcEnsemble

# ======================================================
# Diffusion schedule (VE-SDE) according to ALGD paper
# ======================================================

class VESDESchedule:
    """
    σ(τ) = σ_min * (σ_max / σ_min)^(τ), τ ∈ [0,1]
    K-step discretization converts τ = k/K.
    """
    def __init__(self, sigma_min=0.02, sigma_max=50.0, K=10):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.K = K

    def sigma(self, k):
        # k ∈ [0, K] integer step
        t = k / self.K
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sigma_sq(self, k):
        s = self.sigma(k)
        return s * s

    def dsigma_sq(self, k):
        # approximate derivative of σ² wrt τ using finite differences
        # dsigma²/dτ ~ (σ²(k+1) - σ²(k)) / (1/K)
        if k == self.K:
            k_prev = k - 1
            return (self.sigma_sq(k) - self.sigma_sq(k_prev)) * self.K
        else:
            return (self.sigma_sq(k+1) - self.sigma_sq(k)) * self.K


# ======================================================
# Fourier-style time embedding (simple but effective)
# ======================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.freqs = nn.Parameter(torch.exp(torch.linspace(np.log(1e-4), np.log(1e4), half)), requires_grad=False)

    def forward(self, t):
        """
        t: float tensor [batch, 1]
        returns [batch, dim]
        """
        t = t.unsqueeze(-1)  # [B, 1, 1]
        freqs = self.freqs.unsqueeze(0).unsqueeze(0)  # [1,1,H]
        angles = t * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.squeeze(1)  # [B, dim]


# ======================================================
# Score Network φθ(s, aτ, τ)
# (Simple MLP + time embedding)
# ======================================================

class ScoreNetwork(nn.Module):
    """
    φθ(s, aτ, t_emb(τ)) → score of shape = action_dim
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, time_emb_dim=32):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)

        input_dim = state_dim + action_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Weight init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, action, t_scalar):
        """
        state: [B, state_dim]
        action: [B, action_dim]
        t_scalar: float tensor [B,1] representing τ
        """
        t_emb = self.time_embedding(t_scalar)  # [B, time_emb_dim]
        x = torch.cat([state, action, t_emb], dim=-1)
        return self.net(x)


# ======================================================
# Augmented Lagrangian LA(s,a,λ)
# (cost_weight acts ONLY on cost term)
# ======================================================

def augmented_lagrangian(Q, Qc, h, lam, rho, cost_weight=1.0):
    """
    Implements Eq.(9) in the ALGD paper:

    LA = -Q + 1/(2ρ) ( [λ + ρ(cost_weight*(Qc - h))]_+^2 - λ^2 )
    """
    # cost_scaled = cost_weight * (Qc - h)
    violation = lam + rho * (cost_weight * (Qc - h))
    hinge = torch.relu(violation)

    # 1/(2ρ) ( hinge^2 - lam^2 )
    penalty = (hinge * hinge - lam * lam) / (2 * rho)

    LA = -Q + penalty
    return LA


# ======================================================
# Diffusion Policy (reverse VE-SDE guided by LA)
# ======================================================

class DiffusionPolicy(nn.Module):
    """
    Diffusion-based policy used by ALGD.
    Replaces GaussianPolicy in CAL.
    """

    def __init__(self, args, state_dim, action_dim, action_space,
                 score_network: ScoreNetwork, schedule: VESDESchedule,
                 rho=1.0, beta=1.0, cost_weight=1.0, K=10):

        super().__init__()

        self.args = args
        self.device = torch.device("cuda")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K
        self.schedule = schedule
        self.score_net = score_network.to(self.device)

        # Action scaling (same as in GaussianPolicy)
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.
            ).to(self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            ).to(self.device)

        # ALGD hyperparameters
        self.rho = rho
        self.beta = beta
        self.cost_weight = cost_weight

    # -----------------------------------------------------
    # Reverse-time sampling (Algorithm 1)
    # -----------------------------------------------------
    def sample(self, state, critic, safety_critics, lam, target_cost):

        device = state.device
        B = state.size(0)

        # 1. 起始噪声 a_K ~ N(0, I) （不需要梯度）
        with torch.no_grad():
            a_k = torch.randn(B, self.action_dim, device=device)

        # 2. K → 0 逆扩散
        for k in reversed(range(self.K)):
            # 标准化时间 τ = k / K
            t_scalar = torch.full((B, 1), float(k) / self.K, device=device)

            # 2.1 先用 score_net 预测得分（不需要对参数求导）
            with torch.no_grad():
                score = self.score_net(state, a_k, t_scalar)  # [B, action_dim]

            # 2.2 对 action 求 ∇_a LA(s,a,λ)
            with torch.enable_grad():
                a_req = a_k.detach().clone().requires_grad_(True)

                grad_energy = self.compute_grad_energy(
                    state=state,
                    action=a_req,
                    critic=critic,
                    safety_critics=safety_critics,
                    lam=lam,
                    target_cost=target_cost
                )   # [B, action_dim]

            # 2.3 利用 VE-SDE 的 dsigma²/dt 更新（在 no_grad 下做数值更新）
            with torch.no_grad():
                dsigma2_dt = self.schedule.dsigma_sq(k)   # 标量 float / tensor

                # drift = - (score + grad_E / β) * dσ²/dt
                drift = -(score + grad_energy / self.beta) * dsigma2_dt

                # Euler-Maruyama：a_{k-1} = a_k + drift + sqrt(dσ²/dt) * ξ
                noise = torch.randn_like(a_k)
                step_std = torch.sqrt(a_k.new_tensor(max(dsigma2_dt, 1e-12)))
                a_k = a_req + drift + step_std * noise
                # 上面用了 a_req（有梯度的版本）做漂移基准，但更新完再 detach，下次循环重新启用 grad

        # 3. 把最后的 a_0 映射回环境动作空间
        with torch.no_grad():
            a0 = torch.tanh(a_k) * self.action_scale + self.action_bias

        return a0.detach()

    # -----------------------------------------------------
    # Compute ∇a L_A(s,a,λ)
    # -----------------------------------------------------
    def compute_grad_energy(self, state, action, critic, safety_critics, lam, target_cost):
        """
        Computes gradient of augmented Lagrangian wrt action:
            ∇_a LA(s, a, λ)

        要求:
        - action: [B, action_dim] 且 requires_grad=True
        - critic / safety_critics: 正常的 nn.Module
        - lam, target_cost: 标量或 [1] Tensor（会自动 broadcast）
        """

        # 确保 action 可导（但不要 detach，否则会丢图）
        if not action.requires_grad:
            action.requires_grad_(True)

        # Q(s,a)
        Q1, Q2 = critic(state, action)
        Q = torch.min(Q1, Q2)      # [B,1]

        # Qc(s,a) — ensemble → mean
        QCs = safety_critics(state, action)   # [E,B,1]
        Qc_mean = QCs.mean(dim=0)             # [B,1]

        # Augmented Lagrangian LA(s,a,λ)
        LA = augmented_lagrangian(
            Q, Qc_mean,
            h=target_cost,
            lam=lam,
            rho=self.rho,
            cost_weight=self.cost_weight
        )   # [B,1]

        # ∇_a LA(s,a,λ)
        grad = torch.autograd.grad(
            LA.sum(),          # 标量
            action,            # 对 action 求导
            create_graph=False,
            retain_graph=False
        )[0]                   # [B, action_dim]

        return grad

    # -----------------------------------------------------
    # For training (not used during evaluation)
    # -----------------------------------------------------
    def sample_no_grad(self, state):
        # convenience alias for deterministic action selection
        return self.sample(state, None, None, None, None)

    
# ======================================================
# ALGDAgent (based on CALAgent structure)
# ======================================================

class ALGDAgent(Agent):
    """
    ALGD Agent: identical critic updates as CAL,
    but actor replaced with diffusion + score matching.
    """

    def __init__(self, num_inputs, action_space, args, cost_weight=1.0):
        super().__init__()
        self.device = torch.device("cuda")

        # =====================
        # Basic hyperparameters
        # =====================
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args
        self.update_counter = 0

        # =====================
        # Safety / cost config
        # =====================
        self.c = args.c
        self.cost_weight = cost_weight  # NEW
        self.rho = args.rho if hasattr(args, "rho") else 1.0
        self.beta = args.beta if hasattr(args, "beta") else 1.0
        self.diffusion_K = args.diffusion_K if hasattr(args, "diffusion_K") else 4

        # -------------------------------------
        # Reward critic (kept identical to CAL)
        # -------------------------------------
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # -------------------------------------
        # Safety critics: ensemble (same as CAL)
        # -------------------------------------
        self.safety_critics = QcEnsemble(num_inputs, action_space.shape[0],
                                         args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(num_inputs, action_space.shape[0],
                                                args.qc_ens_size, args.hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # =======================================================
        # Diffusion policy replaces GaussianPolicy (ALGD core)
        # =======================================================
        self.schedule = VESDESchedule(K=self.diffusion_K)
        self.score_net = ScoreNetwork(num_inputs,
                                      action_space.shape[0],
                                      hidden_dim=args.hidden_size)

        self.policy = DiffusionPolicy(
            args=args,
            state_dim=num_inputs,
            action_dim=action_space.shape[0],
            action_space=action_space,
            score_network=self.score_net,
            schedule=self.schedule,
            rho=self.rho,
            beta=self.beta,
            cost_weight=self.cost_weight,
            K=self.diffusion_K
        ).to(self.device)

        # Optimizer for score network (actor in ALGD)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.score_net.parameters(), lr=args.lr
        )

        # Temperature α (same as CAL)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr)

        # Lagrange multiplier λ (same as CAL)
        self.log_lam = torch.tensor(np.log(0.6931), device=self.device, requires_grad=True)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

        # Critic optimizers (unchanged)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)

        # =====================
        # Target cost definition
        # =====================
        if args.safetygym:
            self.target_cost = args.cost_lim * (1 - self.safety_discount ** args.epoch_length) / \
                               (1 - self.safety_discount) / args.epoch_length \
                               if self.safety_discount < 1 else args.cost_lim
        else:
            self.target_cost = args.cost_lim

        print("[ALGD] Constraint Budget =", self.target_cost)


    # ======================================================
    # Properties
    # ======================================================
    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lam(self):
        return self.log_lam.exp()

    def train(self, training=True):
        self.training = training
        self.critic.train(training)
        self.safety_critics.train(training)
        self.policy.score_net.train(training)

    # ======================================================
    # Action selection (use diffusion sampling)
    # ======================================================
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if eval:
            # No randomness → direct reverse pass
            action = self.policy.sample(
                state,
                critic=self.critic,
                safety_critics=self.safety_critics,
                lam=self.lam.detach(),
                target_cost=self.target_cost
            )
        else:
            # Same procedure but slightly noisy in diffusion steps
            action = self.policy.sample(
                state,
                critic=self.critic,
                safety_critics=self.safety_critics,
                lam=self.lam.detach(),
                target_cost=self.target_cost
            )

        return action.detach().cpu().numpy()[0]


    # ======================================================
    # Critic update = SAME AS CAL
    # ======================================================
    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action = self.policy.sample(
            next_state,
            critic=self.critic_target,
            safety_critics=self.safety_critic_targets,
            lam=self.lam.detach(),
            target_cost=self.target_cost
        )

        # ------------------------------
        # Reward critic update (same CAL)
        # ------------------------------
        current_Q1, current_Q2 = self.critic(state, action)

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)

        target_Q = reward + mask * self.discount * target_V
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------
        # Safety critic update (same CAL)
        # ------------------------------
        qc_idxs = np.random.choice(self.args.qc_ens_size, self.args.M)

        current_QCs = self.safety_critics(state, action)
        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)

        next_QC_random_max = next_QCs[qc_idxs].max(dim=0, keepdim=True).values
        next_QC = next_QC_random_max.repeat(self.args.qc_ens_size, 1, 1)

        target_QCs = cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) + \
                      mask[None, :, :].repeat(self.args.qc_ens_size, 1, 1) * \
                      self.safety_discount * next_QC

        safety_critic_loss = F.mse_loss(current_QCs, target_QCs.detach())
        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()


    # ======================================================
    # ALGD Actor update = Score Matching (Eq.20)
    # ======================================================
    def update_actor(self, state_batch):
        B = state_batch.size(0)
        device = self.device

        # ---------
        # Sample τ
        # ---------
        tau = torch.rand(B, 1, device=device)  # τ ∈ [0,1]
        k = (tau * self.diffusion_K).long().clamp(0, self.diffusion_K)

        # ---------
        # Sample aτ from forward diffusion
        # ---------
        noise = torch.randn(B, self.policy.action_dim, device=device)
        sigma_k = torch.tensor([self.schedule.sigma(k_i.item()) for k_i in k],
                               device=device).unsqueeze(1)
        a_tau = noise * sigma_k

        # ---------
        # Compute φ*(Eq.20): energy-guided score target
        # ---------
        # need gradients w.r.t. the diffused action to form grad_energy; using
        # torch.enable_grad() keeps autograd active even if surrounding code
        # switches it off elsewhere.
        with torch.enable_grad():
            grad_energy = self.policy.compute_grad_energy(
                state_batch, a_tau,
                critic=self.critic,
                safety_critics=self.safety_critics,
                lam=self.lam.detach(),
                target_cost=self.target_cost
            )

            # phi_star serves as a target only; detach to avoid leaking
            # gradients back into critics or lambda parameters during actor
            # optimisation.
            phi_star = (- a_tau / sigma_k**2 - grad_energy / self.beta).detach()

        # ---------
        # Predicted score φθ
        # ---------
        score_pred = self.policy.score_net(
            state_batch, a_tau, tau
        )

        # ---------
        # Score matching loss
        # ---------
        actor_loss = F.mse_loss(score_pred, phi_star)

        # ---------
        # Optimize score network
        # ---------
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ======================
        # α update (same CAL)
        # ======================
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -self.log_alpha.mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ======================
        # λ update (same CAL)
        # ======================
        Q1, Q2 = self.critic(state_batch,
                             self.select_action(state_batch.cpu().numpy()))
        Q = torch.min(Q1, Q2)

        QCs = self.safety_critics(state_batch,
                                  self.select_action(state_batch.cpu().numpy()))
        Qc_mean = QCs.mean(dim=0)[:, 0:1]

        lam_loss = torch.mean(self.lam * (self.target_cost - Qc_mean.detach()))

        self.log_lam_optimizer.zero_grad()
        lam_loss.backward()
        self.log_lam_optimizer.step()


    # ======================================================
    # Main update_parameters
    # ======================================================
    def update_parameters(self, memory, updates):
        self.update_counter += 1

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        cost_batch = torch.FloatTensor(reward_batch[:, 1]).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch[:, 0]).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # critic update
        self.update_critic(state_batch, action_batch, reward_batch,
                           cost_batch, next_state_batch, mask_batch)

        # actor update (score matching)
        self.update_actor(state_batch)

        # target network update
        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

            
    # ======================================================
    # Save / Load
    # ======================================================
    def save_model(self, save_dir, suffix=""):
        actor_path = save_dir / f"algd_actor_{suffix}.pth"
        critics_path = save_dir / f"algd_critics_{suffix}.pth"
        safetycritics_path = save_dir / f"algd_safetycritics_{suffix}.pth"

        print(f"[ALGD] Saving models to:\n  {actor_path}\n  {critics_path}\n  {safetycritics_path}")

        torch.save(self.policy.score_net.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critics_path)
        torch.save(self.safety_critics.state_dict(), safetycritics_path)

    def load_model(self, actor_path, critics_path, safetycritics_path):
        print("[ALGD] Loading models...")
        if actor_path is not None:
            self.policy.score_net.load_state_dict(torch.load(actor_path))
        if critics_path is not None:
            self.critic.load_state_dict(torch.load(critics_path))
        if safetycritics_path is not None:
            self.safety_critics.load_state_dict(torch.load(safetycritics_path))
