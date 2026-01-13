import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path

from agents.base_agent import Agent
from agents.ppolag.model import SquashedGaussianActor, ValueNetwork


class PPOLagAgent(Agent):
    """PPO + Lagrangian baseline (CMDP-style).

    This implementation is written to mirror the structure of CALAgent (select_action,
    update_parameters, save_model/load_model) in cal.py, but uses an on-policy PPO loop.

    Expected memory format (same top-level tuple shape as CAL):
        (state, action, reward_cost, next_state, mask)
    where reward_cost is an array shaped (T, 2) with [:,0]=reward, [:,1]=cost.

    Notes:
      - PPO needs rollout-ordered data for GAE. If your data loader shuffles transitions,
        pass rollouts in order (or extend this class to accept episode/trajectory indices).
    """

    def __init__(self, num_inputs, action_space, args):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.gamma = getattr(args, "gamma", 0.99)
        self.cost_gamma = getattr(args, "safety_gamma", getattr(args, "gamma", 0.99))

        # PPO hyperparams (safe defaults via getattr)
        self.clip_eps = getattr(args, "clip_eps", getattr(args, "clip_param", 0.2))
        self.ppo_epochs = getattr(args, "ppo_epochs", getattr(args, "update_epochs", 10))
        self.minibatch_size = getattr(args, "minibatch_size", getattr(args, "batch_size", 256))
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)
        self.cost_gae_lambda = getattr(args, "cost_gae_lambda", self.gae_lambda)

        self.vf_coef = getattr(args, "vf_coef", 0.5)
        self.cost_vf_coef = getattr(args, "cost_vf_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 0.0)

        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        self.target_kl = getattr(args, "target_kl", None)

        hidden_dim = getattr(args, "hidden_size", 256)

        # Actor + critics
        self.policy = SquashedGaussianActor(num_inputs, action_space.shape[0], hidden_dim, action_space=action_space).to(self.device)
        self.value_r = ValueNetwork(num_inputs, hidden_dim).to(self.device)
        self.value_c = ValueNetwork(num_inputs, hidden_dim).to(self.device)

        # Lagrange multiplier (log-param)
        lam_init = getattr(args, "lam_init", 0.6931)
        self.log_lam = torch.tensor(np.log(np.clip(lam_init, 1e-8, 1e8)), device=self.device, requires_grad=True)

        lr = getattr(args, "lr", 3e-4)
        vf_lr = getattr(args, "vf_lr", lr)
        dual_lr = getattr(args, "dual_lr", lr)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vr_optimizer = torch.optim.Adam(self.value_r.parameters(), lr=vf_lr)
        self.vc_optimizer = torch.optim.Adam(self.value_c.parameters(), lr=vf_lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=dual_lr)

        # Target cost (match CAL's convention)
        safetygym = getattr(args, "safetygym", False)
        epoch_length = getattr(args, "epoch_length", None)
        cost_lim = getattr(args, "cost_lim", 0.0)
        if safetygym and epoch_length is not None:
            if self.cost_gamma < 1:
                self.target_cost = cost_lim * (1 - self.cost_gamma ** epoch_length) / (1 - self.cost_gamma) / epoch_length
            else:
                self.target_cost = cost_lim
        else:
            self.target_cost = cost_lim

        self.update_counter = 0
        self.train()

        print("[PPOLag] Constraint Budget:", self.target_cost)

    def train(self, training=True):
        self.training = training
        self.policy.train(training)
        self.value_r.train(training)
        self.value_c.train(training)

    @property
    def lam(self):
        return self.log_lam.exp()

    def select_action(self, state, eval=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, logp, _ = self.policy.sample(state_t)
        else:
            action = self.policy.get_a_mean(state_t)
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def _gae(rewards, values, next_values, masks, gamma, lam):
        """Generalized Advantage Estimation (vectorized backward pass)."""
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def _unpack_memory(self, memory):
        """Accept either tuple (as CAL) or dict."""
        if isinstance(memory, dict):
            s = memory["state"]
            a = memory["action"]
            rc = memory["reward"]  # expecting (T,2)
            ns = memory.get("next_state", None)
            m = memory.get("mask", None)
            if ns is None:
                # if next_state not provided, shift states by one with last repeated
                ns = np.concatenate([s[1:], s[-1:]], axis=0)
            if m is None:
                m = np.ones((len(s),), dtype=np.float32)
            return s, a, rc, ns, m
        return memory

    def update_parameters(self, memory, updates):
        """Run one PPO-Lagrangian update over an on-policy rollout.

        memory: (state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
          - reward_batch shape: (T, 2) where [:,0]=reward, [:,1]=cost
          - mask_batch: 1.0 if not done else 0.0 (for bootstrap)
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        self.update_counter += 1

        # ------------------- unpack -------------------
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self._unpack_memory(memory)

        states = torch.FloatTensor(state_batch).to(self.device)
        actions = torch.FloatTensor(action_batch).to(self.device)
        rewards = torch.FloatTensor(reward_batch[:, 0]).to(self.device).unsqueeze(1)
        costs = torch.FloatTensor(reward_batch[:, 1]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)  # 1=not done, 0=done

        debug_nan = getattr(self.args, "debug_nan", False)

        def ck(x, name):
            if debug_nan and (not torch.isfinite(x).all()):
                x_det = x.detach()
                print(f"[NaN] {name}: min={x_det.min().item()} max={x_det.max().item()} mean={x_det.mean().item()}")
                raise RuntimeError(f"NaN/Inf detected in {name}")

        ck(states, "states")
        ck(actions, "actions")
        ck(rewards, "rewards")
        ck(costs, "costs")
        ck(masks, "masks")

        # ------------------- compute old stats + GAE -------------------
        with torch.no_grad():
            old_logp = self.policy.log_prob(states, actions)  # shape (T,1) or (T,)
            v_r = self.value_r(states)
            v_c = self.value_c(states)
            nv_r = self.value_r(next_states)
            nv_c = self.value_c(next_states)

            ck(old_logp, "old_logp")
            ck(v_r, "v_r")
            ck(v_c, "v_c")

            adv_r, ret_r = self._gae(rewards, v_r, nv_r, masks, self.gamma, self.gae_lambda)
            adv_c, ret_c = self._gae(costs, v_c, nv_c, masks, self.cost_gamma, self.cost_gae_lambda)

            ck(adv_r, "adv_r")
            ck(adv_c, "adv_c")
            ck(ret_r, "ret_r")
            ck(ret_c, "ret_c")

            # Lagrangian-combined advantage
            adv = adv_r - self.lam.detach() * adv_c

            # Normalize advantage (avoid division by 0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ck(adv, "adv_normed")

        T = states.shape[0]
        if T < 2:
            return {"lam": float(self.lam.detach().cpu().item()), "target_cost": float(getattr(self, "target_cost", getattr(self.args, "cost_limit", 0.0)))}

        mb = int(min(self.minibatch_size, T))

        # ------------------- dual update (lambda) -------------------
        # Use episodic cumulative cost estimator (more consistent with "budget=10" per episode)
        # Identify episode boundaries by done where mask==0
        with torch.no_grad():
            done = (masks.squeeze(1) < 0.5).detach().cpu().numpy().astype(bool)
            costs_np = costs.squeeze(1).detach().cpu().numpy()

            ep_costs = []
            acc = 0.0
            for t in range(T):
                acc += float(costs_np[t])
                if done[t]:
                    ep_costs.append(acc)
                    acc = 0.0

            if len(ep_costs) > 0:
                jc_hat = torch.tensor(np.mean(ep_costs), device=self.device, dtype=torch.float32).view(1, 1)
            else:
                # fallback: use rollout average * rollout length as a rough sum
                jc_hat = costs.mean() * float(T)

            ck(jc_hat, "jc_hat")

        # target cost (prefer self.target_cost; else args.cost_limit; else args.cost_lim)
        target_cost = getattr(self, "target_cost", None)
        if target_cost is None:
            target_cost = getattr(self.args, "cost_limit", None)
        if target_cost is None:
            target_cost = getattr(self.args, "cost_lim", 0.0)
        target_cost = float(target_cost)

        # Update lambda every N updates
        dual_every = int(getattr(self, "dual_update_every", getattr(self.args, "dual_update_every", 1)))
        if dual_every < 1:
            dual_every = 1

        if (self.update_counter % dual_every) == 0:
            self.log_lam_optimizer.zero_grad()

            # Minimize lam*(target - jc)  => if jc>target, grad negative => lam increases (dual ascent)
            lam_loss = torch.mean(self.lam * (target_cost - jc_hat).detach())
            ck(lam_loss, "lam_loss")

            lam_loss.backward()
            self.log_lam_optimizer.step()

            # Clamp lambda to keep training stable
            lam_max = float(getattr(self, "lambda_max", getattr(self.args, "lambda_max", 10.0)))
            with torch.no_grad():
                self.lam.clamp_(0.0, lam_max)

        # ------------------- PPO epochs -------------------
        idx = torch.arange(T, device=self.device)

        # For debugging, record a few stats
        last_actor_loss = None
        last_v_loss = None
        last_vc_loss = None

        for _ in range(self.ppo_epochs):
            perm = idx[torch.randperm(T)]

            for start in range(0, T, mb):
                mb_idx = perm[start:start + mb]

                s_mb = states[mb_idx]
                a_mb = actions[mb_idx]
                oldlogp_mb = old_logp[mb_idx]
                adv_mb = adv[mb_idx]
                ret_r_mb = ret_r[mb_idx]
                ret_c_mb = ret_c[mb_idx]

                # --- actor ---
                logp, ent = self.policy.evaluate_actions(s_mb, a_mb)
                ck(logp, "logp")
                ck(ent, "ent")

                # ratio stabilization
                log_ratio = logp - oldlogp_mb
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                ck(ratio, "ratio")

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent.mean()
                ck(actor_loss, "actor_loss")

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                last_actor_loss = float(actor_loss.detach().cpu().item())

                # --- reward value ---
                v_pred = self.value_r(s_mb)
                v_loss = F.mse_loss(v_pred, ret_r_mb)
                ck(v_loss, "v_loss")

                self.vr_optimizer.zero_grad()
                (self.vf_coef * v_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_r.parameters(), self.max_grad_norm)
                self.vr_optimizer.step()
                last_v_loss = float(v_loss.detach().cpu().item())

                # --- cost value ---
                vc_pred = self.value_c(s_mb)
                vc_loss = F.mse_loss(vc_pred, ret_c_mb)
                ck(vc_loss, "vc_loss")

                self.vc_optimizer.zero_grad()
                (self.cost_vf_coef * vc_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_c.parameters(), self.max_grad_norm)
                self.vc_optimizer.step()
                last_vc_loss = float(vc_loss.detach().cpu().item())

                # Optional early stopping on KL
                if self.target_kl is not None:
                    with torch.no_grad():
                        # typical approx KL estimator
                        approx_kl = (oldlogp_mb - logp).mean().item()
                    if approx_kl > float(self.target_kl):
                        break

        return {
            "lam": float(self.lam.detach().cpu().item()),
            "target_cost": float(target_cost),
            "jc_hat": float(jc_hat.detach().cpu().item()),
            "actor_loss": last_actor_loss,
            "v_loss": last_v_loss,
            "vc_loss": last_vc_loss,
        }


    # Save model parameters
    def save_model(self, save_dir, suffix=""):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        actor_path = save_dir / f"actor_{suffix}.pth"
        vr_path = save_dir / f"value_r_{suffix}.pth"
        vc_path = save_dir / f"value_c_{suffix}.pth"
        dual_path = save_dir / f"dual_{suffix}.pth"

        print(f"[Model] Saving models to:\n  {actor_path}\n  {vr_path}\n  {vc_path}\n  {dual_path}")

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.value_r.state_dict(), vr_path)
        torch.save(self.value_c.state_dict(), vc_path)
        torch.save({"log_lam": self.log_lam.detach().cpu()}, dual_path)

    # Load model parameters
    def load_model(self, actor_path=None, vr_path=None, vc_path=None, dual_path=None):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        if vr_path is not None:
            self.value_r.load_state_dict(torch.load(vr_path, map_location=self.device))
        if vc_path is not None:
            self.value_c.load_state_dict(torch.load(vc_path, map_location=self.device))
        if dual_path is not None:
            dual = torch.load(dual_path, map_location=self.device)
            if isinstance(dual, dict) and "log_lam" in dual:
                self.log_lam.data.copy_(dual["log_lam"].to(self.device))
