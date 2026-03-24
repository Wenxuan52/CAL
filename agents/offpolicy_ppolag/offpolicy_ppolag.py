import copy
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.base_agent import Agent
from agents.offpolicy_ppolag.model import SquashedGaussianActor, ValueNetwork, soft_update


class OffPolicyPPOLagAgent(Agent):
    """Replay-based off-policy PPO-Lag variant with truncated IS + V-trace style targets."""

    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        use_cuda = bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "safety_gamma", self.gamma)
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)

        self.clip_eps = getattr(args, "clip_ratio", 0.2)
        self.ppo_epochs = getattr(args, "ppo_epochs", 2)
        self.minibatch_size = getattr(args, "policy_train_batch_size", 256)
        self.seq_len = getattr(args, "trace_len", 16)

        self.rho_bar = getattr(args, "rho_bar", 1.5)
        self.c_bar = getattr(args, "c_bar", 1.0)
        self.entropy_coef = getattr(args, "ent_coef", 0.0)
        self.vf_coef = getattr(args, "vf_coef", 0.5)
        self.cost_vf_coef = getattr(args, "cost_vf_coef", 0.5)
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)

        self.tau = getattr(args, "target_tau", 0.01)
        self.target_update_every = getattr(args, "target_update_every", 1)

        hidden_dim = getattr(args, "hidden_size", 256)
        self.policy = SquashedGaussianActor(num_inputs, action_space.shape[0], hidden_dim, action_space=action_space).to(self.device)
        self.value_r = ValueNetwork(num_inputs, hidden_dim).to(self.device)
        self.value_c = ValueNetwork(num_inputs, hidden_dim).to(self.device)
        self.value_r_targ = copy.deepcopy(self.value_r).to(self.device)
        self.value_c_targ = copy.deepcopy(self.value_c).to(self.device)

        lr = getattr(args, "lr", 3e-4)
        vf_lr = getattr(args, "vf_lr", lr)
        dual_lr = getattr(args, "lambda_lr", 5e-4)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vr_optimizer = torch.optim.Adam(self.value_r.parameters(), lr=vf_lr)
        self.vc_optimizer = torch.optim.Adam(self.value_c.parameters(), lr=vf_lr)

        lam_init = getattr(args, "lambda_init", 1.0)
        self.log_lam = torch.tensor(np.log(np.clip(lam_init, 1e-8, 1e8)), dtype=torch.float32, device=self.device, requires_grad=True)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=dual_lr)
        self.lambda_max = getattr(args, "lambda_max", 80.0)

        self.recent_replay_size = int(getattr(args, "recent_replay_size", 50000))
        self.replay = deque(maxlen=self.recent_replay_size)

        if args.safetygym:
            if self.cost_gamma < 1:
                self.target_cost = args.cost_lim * (1 - self.cost_gamma ** args.epoch_length) / (1 - self.cost_gamma) / args.epoch_length
            else:
                self.target_cost = args.cost_lim
        else:
            self.target_cost = args.cost_lim

        self.update_counter = 0
        self.last_log = {}
        self.train()

        print("[OffPolicyPPOLag] Constraint Budget:", self.target_cost)

    def train(self, training=True):
        self.training = training
        self.policy.train(training)
        self.value_r.train(training)
        self.value_c.train(training)

    @property
    def lam(self):
        return self.log_lam.exp()

    def get_last_log(self):
        return self.last_log

    def select_action(self, state, eval=False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state_t)
        else:
            action = self.policy.get_a_mean(state_t)
        return action.detach().cpu().numpy()[0]

    @torch.no_grad()
    def store_transition(self, state, action, reward, next_state, done):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        log_mu = self.policy.log_prob(s, a).item()

        self.replay.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward[0]),
            "cost": float(reward[1]),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "done": float(done),
            "log_mu": float(log_mu),
        })

    def _sample_sequence_batch(self, batch_size, seq_len):
        n = len(self.replay)
        if n <= seq_len + 1:
            return None

        starts = np.random.randint(0, n - seq_len, size=batch_size)
        states, actions, rewards, costs, next_states, dones, log_mu = [], [], [], [], [], [], []

        for st in starts:
            seq = [self.replay[st + t] for t in range(seq_len)]
            states.append(np.stack([x["state"] for x in seq], axis=0))
            actions.append(np.stack([x["action"] for x in seq], axis=0))
            rewards.append(np.asarray([x["reward"] for x in seq], dtype=np.float32))
            costs.append(np.asarray([x["cost"] for x in seq], dtype=np.float32))
            next_states.append(np.stack([x["next_state"] for x in seq], axis=0))
            dones.append(np.asarray([x["done"] for x in seq], dtype=np.float32))
            log_mu.append(np.asarray([x["log_mu"] for x in seq], dtype=np.float32))

        # [T, B, ...]
        return {
            "state": torch.as_tensor(np.stack(states, axis=1), device=self.device),
            "action": torch.as_tensor(np.stack(actions, axis=1), device=self.device),
            "reward": torch.as_tensor(np.stack(rewards, axis=1), device=self.device).unsqueeze(-1),
            "cost": torch.as_tensor(np.stack(costs, axis=1), device=self.device).unsqueeze(-1),
            "next_state": torch.as_tensor(np.stack(next_states, axis=1), device=self.device),
            "done": torch.as_tensor(np.stack(dones, axis=1), device=self.device).unsqueeze(-1),
            "log_mu": torch.as_tensor(np.stack(log_mu, axis=1), device=self.device).unsqueeze(-1),
        }

    def _vtrace_adv_ret(self, rewards, values, next_values, dones, ratios):
        rho = torch.clamp(ratios, max=self.rho_bar)
        c = torch.clamp(ratios, max=self.c_bar)
        not_done = 1.0 - dones

        deltas = rho * (rewards + self.gamma * not_done * next_values - values)

        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        acc = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            acc = deltas[t] + self.gamma * self.gae_lambda * not_done[t] * c[t] * acc
            adv[t] = acc
        returns = adv + values
        return adv, returns

    def _vtrace_adv_ret_cost(self, costs, values, next_values, dones, ratios):
        rho = torch.clamp(ratios, max=self.rho_bar)
        c = torch.clamp(ratios, max=self.c_bar)
        not_done = 1.0 - dones

        deltas = rho * (costs + self.cost_gamma * not_done * next_values - values)

        T = costs.shape[0]
        adv = torch.zeros_like(costs)
        acc = torch.zeros_like(costs[0])
        for t in reversed(range(T)):
            acc = deltas[t] + self.cost_gamma * self.gae_lambda * not_done[t] * c[t] * acc
            adv[t] = acc
        returns = adv + values
        return adv, returns

    def update_parameters(self, memory, updates):
        self.update_counter += 1
        if len(self.replay) < max(self.minibatch_size, self.seq_len * 2):
            return

        actor_loss_value, vr_loss_value, vc_loss_value, jc_hat_value = None, None, None, None

        for _ in range(self.ppo_epochs):
            batch = self._sample_sequence_batch(self.minibatch_size, self.seq_len)
            if batch is None:
                return

            states = batch["state"]
            actions = batch["action"]
            rewards = batch["reward"]
            costs = batch["cost"]
            next_states = batch["next_state"]
            dones = batch["done"]
            old_log_mu = batch["log_mu"]

            T, B = rewards.shape[0], rewards.shape[1]
            s_flat = states.reshape(T * B, -1)
            a_flat = actions.reshape(T * B, -1)

            logp_flat = self.policy.log_prob(s_flat, a_flat)
            logp = logp_flat.view(T, B, 1)
            ratios = torch.exp(torch.clamp(logp - old_log_mu, -10.0, 10.0))

            with torch.no_grad():
                v_r = self.value_r_targ(s_flat).view(T, B, 1)
                nv_r = self.value_r_targ(next_states.reshape(T * B, -1)).view(T, B, 1)
                v_c = self.value_c_targ(s_flat).view(T, B, 1)
                nv_c = self.value_c_targ(next_states.reshape(T * B, -1)).view(T, B, 1)

                adv_r, ret_r = self._vtrace_adv_ret(rewards, v_r, nv_r, dones, ratios)
                adv_c, ret_c = self._vtrace_adv_ret_cost(costs, v_c, nv_c, dones, ratios)

                adv = adv_r - self.lam.detach() * adv_c
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio_actor = torch.exp(torch.clamp(logp - old_log_mu, -10.0, 10.0))
            surr1 = ratio_actor * adv
            surr2 = torch.clamp(ratio_actor, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            entropy_proxy = -logp.mean()
            actor_loss = -(torch.min(surr1, surr2)).mean() - self.entropy_coef * entropy_proxy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            v_pred_r = self.value_r(s_flat).view(T, B, 1)
            v_pred_c = self.value_c(s_flat).view(T, B, 1)
            vr_loss = F.mse_loss(v_pred_r, ret_r)
            vc_loss = F.mse_loss(v_pred_c, ret_c)

            self.vr_optimizer.zero_grad()
            (self.vf_coef * vr_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.value_r.parameters(), self.max_grad_norm)
            self.vr_optimizer.step()

            self.vc_optimizer.zero_grad()
            (self.cost_vf_coef * vc_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.value_c.parameters(), self.max_grad_norm)
            self.vc_optimizer.step()

            # dual update with off-policy corrected discounted cost estimate
            with torch.no_grad():
                gamma_t = torch.pow(self.cost_gamma * torch.ones_like(costs), torch.arange(T, device=self.device, dtype=costs.dtype).view(T, 1, 1))
                corrected_cost = torch.clamp(ratios, max=self.rho_bar) * costs
                jc_hat = (gamma_t * corrected_cost).sum(dim=0).mean()

            self.log_lam_optimizer.zero_grad()
            lam_loss = self.lam * (self.target_cost - jc_hat).detach()
            lam_loss.backward()
            self.log_lam_optimizer.step()
            with torch.no_grad():
                self.log_lam.clamp_(np.log(1e-8), np.log(self.lambda_max))

            actor_loss_value = float(actor_loss.detach().cpu().item())
            vr_loss_value = float(vr_loss.detach().cpu().item())
            vc_loss_value = float(vc_loss.detach().cpu().item())
            jc_hat_value = float(jc_hat.detach().cpu().item())

            if self.update_counter % self.target_update_every == 0:
                soft_update(self.value_r_targ, self.value_r, self.tau)
                soft_update(self.value_c_targ, self.value_c, self.tau)

        self.last_log = {
            "lambda": float(self.lam.detach().cpu().item()),
            "target_cost": float(self.target_cost),
            "jc_hat": jc_hat_value,
            "actor_loss": actor_loss_value,
            "v_loss": vr_loss_value,
            "vc_loss": vc_loss_value,
            "replay_size": len(self.replay),
        }

    def save_model(self, save_dir, suffix=""):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        actor_path = save_dir / f"actor_{suffix}.pth"
        vr_path = save_dir / f"value_r_{suffix}.pth"
        vc_path = save_dir / f"value_c_{suffix}.pth"

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.value_r.state_dict(), vr_path)
        torch.save(self.value_c.state_dict(), vc_path)

    def load_model(self, actor_path, vr_path, vc_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if vr_path is not None:
            self.value_r.load_state_dict(torch.load(vr_path, map_location=self.device))
            self.value_r_targ.load_state_dict(self.value_r.state_dict())
        if vc_path is not None:
            self.value_c.load_state_dict(torch.load(vc_path, map_location=self.device))
            self.value_c_targ.load_state_dict(self.value_c.state_dict())
