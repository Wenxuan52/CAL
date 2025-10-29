"""Diffusion policy agent that reuses pretrained Gaussian critics for safety guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.base_agent import Agent
from agents.ssm_test.model import DiffusionScoreModel
from agents.ssm_test.utils import (
    atanh_clamped,
    compute_phi,
    cosine_beta_schedule,
    ddpm_sampler,
    load_and_freeze_critics,
)


class SSMDiffusionAgent(Agent):
    """Migration of the Gaussian SSM agent to a diffusion policy variant."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.T = getattr(args, "T", 1000)
        self.lr = getattr(args, "lr", 1e-4)
        self.alpha_coef = getattr(args, "alpha_coef", 1.0)
        self.beta_coef = getattr(args, "beta_coef", 1.0)
        self.safe_margin = getattr(args, "safe_margin", 0.0)
        self.grad_clip = getattr(args, "grad_clip", 10.0)
        self.vh_samples = getattr(args, "vh_samples", 16)
        self.time_dim = getattr(args, "time_dim", 64)
        self.hidden_size = getattr(args, "hidden_size", 256)

        self.state_dim = num_inputs
        self.action_dim = action_space.shape[0]

        # Diffusion score model (policy)
        self.score_model = DiffusionScoreModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_size,
            time_dim=self.time_dim,
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(self.score_model.parameters(), lr=self.lr)

        # Diffusion schedule buffers
        betas = cosine_beta_schedule(self.T).to(self.device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", (1.0 - betas))
        self.register_buffer("alpha_hats", torch.cumprod(self.alphas, dim=0))

        # Load pretrained Gaussian critics
        critic_path = Path(getattr(args, "pretrained_critic_path"))
        safety_path = Path(getattr(args, "pretrained_safety_path"))
        self.critic, self.safety_q = load_and_freeze_critics(
            critic_path,
            safety_path,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_size,
            device=self.device,
        )

        self.train()

    # ------------------------------------------------------------------
    # Torch helpers
    # ------------------------------------------------------------------
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def train(self, training: bool = True):
        self.training = training
        self.score_model.train(training)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
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
        return compute_phi(
            state,
            action,
            self.critic,
            self.safety_q,
            alpha=self.alpha_coef,
            beta=self.beta_coef,
            safe_margin=self.safe_margin,
            grad_clip=self.grad_clip,
        )

    def estimate_vh(self, state: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
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
            q_val = self.safety_q(state, action_sample)
            if isinstance(q_val, (tuple, list)):
                q_val = q_val[0]
            q_values.append(q_val)
        return torch.stack(q_values).mean(dim=0)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def update_actor(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = state.size(0)
        t_indices = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(action)
        alpha_hat = self.alpha_hats[t_indices].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_hat) * action + torch.sqrt(1 - alpha_hat) * noise

        phi_target = compute_phi(
            state,
            noisy_action,
            self.critic,
            self.safety_q,
            alpha=self.alpha_coef,
            beta=self.beta_coef,
            safe_margin=self.safe_margin,
            grad_clip=self.grad_clip,
        )

        eps_pred = self.score_model(state, noisy_action.detach(), t_indices.float().unsqueeze(1))
        loss = F.mse_loss(eps_pred, phi_target)

        self.actor_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.score_model.parameters(), self.grad_clip)
        self.actor_optim.step()
        return loss.detach()

    def update_parameters(self, memory, updates) -> Dict[str, float]:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory
        state = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        action = atanh_clamped(action)

        loss_actor = self.update_actor(state, action)
        loss_value = loss_actor.item() if isinstance(loss_actor, torch.Tensor) else float(loss_actor)
        return {"loss_actor": float(loss_value)}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_model(self, path: Path, suffix: str = "") -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.score_model.state_dict(), path / f"ssm_test_score_{suffix}.pth")

