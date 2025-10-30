"""Safe Score Matching agent with jointly trained critics and diffusion policy."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import Agent
from agents.guass_test.utils import soft_update
from agents.ssm.model import DiffusionScoreModel, QNetwork, SafetyValueNetwork
from agents.ssm.utils import (
    atanh_clamped,
    compute_phi,
    cosine_beta_schedule,
    ddpm_sampler,
)


class SSMAgent(Agent):
    """Diffusion-based SSM agent with unfrozen reward and safety critics."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.device = device
        self.args = args

        # ---------- Hyper-parameters ----------
        self.state_dim = num_inputs
        self.action_dim = action_space.shape[0]
        self.hidden_size = getattr(args, "hidden_size", 256)
        self.discount = getattr(args, "gamma", 0.99)
        self.safety_discount = getattr(args, "safety_gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        self.batch_size = getattr(args, "policy_train_batch_size", 256)
        self.T = getattr(args, "T", 50)
        self.alpha_coef = getattr(args, "alpha_coef", 1.0)
        self.beta_coef = getattr(args, "beta_coef", 1.0)
        self.safe_margin = getattr(args, "safe_margin", 0.0)
        self.grad_clip = getattr(args, "grad_clip", 10.0)
        self.vh_samples = getattr(args, "vh_samples", 16)

        # SSM specific scheduling
        self.update_q_every = getattr(args, "update_q_every", 1)
        self.update_qh_every = getattr(args, "update_qh_every", 1)
        self.update_score_every = getattr(args, "update_score_every", 1)
        self.warmup_epochs = getattr(args, "warmup_epochs", 0)

        # ---------- Networks ----------
        self.Q = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.Qh = SafetyValueNetwork(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.Qh_target = SafetyValueNetwork(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.Qh_target.load_state_dict(self.Qh.state_dict())

        self.score_model = DiffusionScoreModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_size,
            time_dim=getattr(args, "time_dim", 64),
        ).to(device)

        # ---------- Optimisers ----------
        self.q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=getattr(args, "q_lr", 3e-4))
        self.qh_optimizer = torch.optim.Adam(self.Qh.parameters(), lr=getattr(args, "qh_lr", 3e-4))
        self.score_optimizer = torch.optim.Adam(self.score_model.parameters(), lr=getattr(args, "lr", 1e-4))

        # ---------- Diffusion buffers ----------
        betas = cosine_beta_schedule(self.T).to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", (1.0 - betas))
        self.register_buffer("alpha_hats", torch.cumprod(self.alphas, dim=0))

        self.total_updates = 0
        self.current_epoch = 0
        self._score_frozen = False

        self.train()

    # ------------------------------------------------------------------
    # Torch helpers
    # ------------------------------------------------------------------
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def train(self, training: bool = True):
        self.training = training
        self.Q.train(training)
        self.Qh.train(training)
        self.score_model.train(training)

    # ------------------------------------------------------------------
    # Public API helpers
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def select_action(self, state, eval: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.sample_action(state_tensor, guidance=True)
            action = torch.tanh(raw_action)
        return action.cpu().numpy()[0]

    def act(self, obs, sample: bool = False):
        return self.select_action(obs, eval=not sample)

    def sample_action(self, state: torch.Tensor, guidance: bool = False) -> torch.Tensor:
        guidance_fn = self.guidance_fn if guidance else None
        return ddpm_sampler(
            self.score_model,
            state,
            self.T,
            self.alphas,
            self.alpha_hats,
            self.betas,
            guidance_fn=guidance_fn,
            device=self.device,
        )

    def guidance_fn(self, state: torch.Tensor, action: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        action_tanh = torch.tanh(action)
        return compute_phi(
            state,
            action_tanh,
            self.Q,
            self.Qh,
            alpha=self.alpha_coef,
            beta=self.beta_coef,
            safe_margin=self.safe_margin,
            grad_clip=self.grad_clip,
        )

    def estimate_vh(
        self, state: torch.Tensor, num_samples: Optional[int] = None
    ) -> torch.Tensor:
        if num_samples is None:
            num_samples = self.vh_samples
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch = state.size(0)
        samples = torch.randn(num_samples, batch, self.action_dim, device=self.device)
        samples = torch.tanh(samples)
        q_values = []
        for idx in range(num_samples):
            action_sample = samples[idx]
            q_val = self.Qh(state, action_sample)
            q_values.append(q_val)
        return torch.stack(q_values).mean(dim=0)

    # ------------------------------------------------------------------
    # Update routines
    # ------------------------------------------------------------------
    def update(self, replay_buffer, logger=None) -> Dict[str, float]:
        if len(replay_buffer) < self.batch_size:
            return {}

        state_np, action_np, reward_np, next_state_np, done_np = replay_buffer.sample(self.batch_size)
        reward_arr = torch.as_tensor(reward_np, dtype=torch.float32, device=self.device)
        if reward_arr.dim() == 1:
            reward_tensor = reward_arr.unsqueeze(1)
            cost_tensor = torch.zeros_like(reward_tensor)
        else:
            reward_tensor = reward_arr[:, :1]
            if reward_arr.size(1) > 1:
                cost_tensor = reward_arr[:, 1:2]
            else:
                cost_tensor = torch.zeros_like(reward_tensor)
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        not_done = 1.0 - done

        self.total_updates += 1
        metrics: Dict[str, float] = {}

        if self.current_epoch < self.warmup_epochs:
            if self.total_updates % self.update_score_every == 0:
                metrics["loss_score"] = float(
                    self.update_score_model(state, action)
                )
            return metrics

        self._freeze_score_model()
        if self.total_updates % self.update_q_every == 0:
            metrics["loss_q"] = float(
                self.update_reward_critic(state, action, reward_tensor, next_state, not_done)
            )
            soft_update(self.Q_target, self.Q, self.tau)

        if self.total_updates % self.update_qh_every == 0:
            metrics["loss_qh"] = float(
                self.update_safety_critic(state, action, cost_tensor, next_state, not_done)
            )
            soft_update(self.Qh_target, self.Qh, self.tau)

        self._unfreeze_score_model()

        if self.total_updates % self.update_score_every == 0:
            metrics["loss_score"] = float(
                self.update_score_model(state, action)
            )

        return metrics

    def update_reward_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        not_done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action_raw = self.sample_action(next_state, guidance=True)
            next_action = torch.tanh(next_action_raw)
            target_q1, target_q2 = self.Q_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_v

        current_q1, current_q2 = self.Q(state, action)
        loss_q = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        return loss_q.detach()

    def _alpha_function(self, value: torch.Tensor) -> torch.Tensor:
        return self.alpha_coef * torch.relu(value)

    def _h_from_cost(self, cost: torch.Tensor) -> torch.Tensor:
        return cost - 10.0

    def update_safety_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        next_state: torch.Tensor,
        not_done: torch.Tensor,
    ) -> torch.Tensor:
        current_vh = self.Qh(state, action)

        current_vh_detached = current_vh.detach()

        with torch.no_grad():
            next_action_raw = self.sample_action(next_state, guidance=True)
            next_action = torch.tanh(next_action_raw)
            next_vh = self.Qh_target(next_state, next_action)
            next_vh = not_done * self.safety_discount * next_vh

            stage_violation = self._h_from_cost(cost)
            alpha_term = self._alpha_function(current_vh_detached)
            delta_v = next_vh - current_vh_detached
            hat_qh = torch.maximum(
                stage_violation,
                torch.relu(delta_v + alpha_term),
            )

        qh_loss = F.mse_loss(current_vh, hat_qh)

        self.qh_optimizer.zero_grad()
        qh_loss.backward()
        self.qh_optimizer.step()
        return qh_loss.detach()

    def update_score_model(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_unbounded = atanh_clamped(action)
        batch = state.size(0)
        t_indices = torch.randint(0, self.T, (batch,), device=self.device)
        noise = torch.randn_like(action_unbounded)
        alpha_hat = self.alpha_hats[t_indices].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_hat) * action_unbounded + torch.sqrt(1 - alpha_hat) * noise
        noisy_action_teacher = torch.tanh(noisy_action)

        phi_target = compute_phi(
            state,
            noisy_action_teacher,
            self.Q,
            self.Qh,
            alpha=self.alpha_coef,
            beta=self.beta_coef,
            safe_margin=self.safe_margin,
            grad_clip=self.grad_clip,
        )

        eps_pred = self.score_model(state, noisy_action.detach(), t_indices.float().unsqueeze(1))
        loss = F.mse_loss(eps_pred + phi_target, torch.zeros_like(eps_pred))

        self.score_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.score_model.parameters(), self.grad_clip)
        self.score_optimizer.step()
        return loss.detach()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _freeze_score_model(self):
        if self._score_frozen:
            return
        for param in self.score_model.parameters():
            param.requires_grad_(False)
        self._score_frozen = True

    def _unfreeze_score_model(self):
        if not self._score_frozen:
            return
        for param in self.score_model.parameters():
            param.requires_grad_(True)
        self._score_frozen = False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_model(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.Q.state_dict(), path / "Q.pth")
        torch.save(self.Q_target.state_dict(), path / "Q_target.pth")
        torch.save(self.Qh.state_dict(), path / "Qh.pth")
        torch.save(self.Qh_target.state_dict(), path / "Qh_target.pth")
        torch.save(self.score_model.state_dict(), path / "Score.pth")

