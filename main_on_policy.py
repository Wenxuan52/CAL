# -*- coding: utf-8 -*-
"""
main_on_policy.py
A dedicated on-policy training script for PPOLag (PPO + Lagrangian) in this repo.

Usage example:
  python main_on_policy.py --env_name Safexp-PointButton1-v0 --experiment_name pointbutton_ppolag --num_epoch 250 --cuda

Notes:
- This script intentionally does NOT use ReplayMemory.
- It collects on-policy rollouts and updates PPOLag with PPO-style epochs/minibatches.
"""

import os
import time
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import setproctitle

# Env samplers (same as main.py)
from sampler.mujoco_env_sampler import MuJoCoEnvSampler
from sampler.safetygym_env_sampler import SafetygymEnvSampler

# Agent
from agents.ppolag.ppolag import PPOLagAgent


def evaluate(args, env_sampler, agent, num_eval_epo=1):
    """Match main.py's evaluation behavior."""
    env_sampler.current_state = None
    avg_return, avg_cost_return = 0.0, 0.0
    epo_len = args.epoch_length

    for _ in range(num_eval_epo):
        sum_reward, sum_cost = 0.0, 0.0
        eval_step = 0
        done = False
        while (not done) and (eval_step < epo_len):
            _, _, _, reward, done, _ = env_sampler.sample(agent, eval_step, eval_t=True)
            sum_reward += float(reward[0])
            if args.safetygym:
                sum_cost += float(reward[1])
            else:
                sum_cost += float((args.gamma ** eval_step) * reward[1])
            eval_step += 1

        avg_return += sum_reward
        avg_cost_return += sum_cost

    avg_return /= float(num_eval_epo)
    avg_cost_return /= float(num_eval_epo)
    return avg_return, avg_cost_return


def train_on_policy(args, env_sampler, agent, writer=None):
    total_step = 0
    update_steps = 0
    epoch = 0
    history = []

    # Rollout buffer: list of tuples (s, a, r2, s2, mask)
    rollout = []

    for _ in range(args.num_epoch):
        sta = time.time()
        epo_len = args.epoch_length

        for t in range(epo_len):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, t)

            # Safety checks (helps catch env/state NaNs early)
            if args.debug_nan:
                if not np.isfinite(cur_state).all():
                    raise RuntimeError("NaN/Inf detected in cur_state from env.")
                if not np.isfinite(next_state).all():
                    raise RuntimeError("NaN/Inf detected in next_state from env.")
                if not np.isfinite(np.asarray(reward)).all():
                    raise RuntimeError("NaN/Inf detected in reward from env.")

            mask = 0.0 if done else 1.0
            rollout.append((cur_state, action, reward, next_state, mask))
            total_step += 1

            # Update whenever rollout is full
            if len(rollout) >= args.steps_per_rollout:
                batch_state = np.stack([x[0] for x in rollout], axis=0)
                batch_action = np.stack([x[1] for x in rollout], axis=0)
                batch_reward = np.stack([x[2] for x in rollout], axis=0)  # (T,2): reward,cost
                batch_next_state = np.stack([x[3] for x in rollout], axis=0)
                batch_mask = np.asarray([x[4] for x in rollout], dtype=np.float32)  # (T,)

                agent.update_parameters(
                    (batch_state, batch_action, batch_reward, batch_next_state, batch_mask),
                    update_steps
                )
                update_steps += 1
                rollout = []

            # Periodic evaluation (keep same cadence as main.py)
            if (total_step % epo_len == 0) or (total_step == 1):
                test_reward, test_cost = evaluate(args, env_sampler, agent, args.num_eval_epochs)
                print(
                    "env: {}, exp: {}, step: {}, test_return: {}, test_cost: {}, budget: {}, seed: {}, cuda_num: {}, time: {}s".format(
                        args.env_name,
                        args.experiment_name,
                        total_step,
                        np.around(test_reward, 2),
                        np.around(test_cost, 2),
                        args.cost_lim,
                        args.seed,
                        args.cuda_num,
                        int(time.time() - sta),
                    )
                )
                if args.use_tensorboard and writer is not None:
                    writer.add_scalar("Eval/return", test_reward, total_step)
                    writer.add_scalar("Eval/cost", test_cost, total_step)

                if args.save_history:
                    history.append({"step": int(total_step), "return": float(test_reward), "cost": float(test_cost)})

        # Flush leftover rollout at epoch end (optional but recommended)
        if len(rollout) > 0:
            batch_state = np.stack([x[0] for x in rollout], axis=0)
            batch_action = np.stack([x[1] for x in rollout], axis=0)
            batch_reward = np.stack([x[2] for x in rollout], axis=0)
            batch_next_state = np.stack([x[3] for x in rollout], axis=0)
            batch_mask = np.asarray([x[4] for x in rollout], dtype=np.float32)

            agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_mask), update_steps)
            update_steps += 1
            rollout = []

        epoch += 1

    # Save history
    save_dir = None
    if args.save_history and len(history) > 0:
        import datetime

        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_dir = Path.cwd() / "results" / args.env_name / args.experiment_name / "{}_seed{}".format(date_str, args.seed)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "history.csv"
        pd.DataFrame(history).to_csv(save_path, index=False)
        print("[History] Saved training history to {}".format(save_path))

    # Save model parameters
    if args.save_parameters:
        if save_dir is None:
            import datetime

            date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            save_dir = Path.cwd() / "results" / args.env_name / args.experiment_name / "{}_seed{}".format(date_str, args.seed)
            save_dir.mkdir(parents=True, exist_ok=True)
        agent.save_model(save_dir)


def main(args):
    torch.set_num_threads(args.n_training_threads)

    run_dir = Path.cwd() / "results" / args.env_name / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_name)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.safetygym:
        # safety-gym classic API
        env.seed(args.seed)
    else:
        # newer gym API (best-effort, compatible with older envs)
        try:
            env.reset(seed=args.seed)
        except TypeError:
            pass
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(args.seed)

    # obs dim adjustments (match main.py)
    s_dim = env.observation_space.shape[0]
    if args.env_name == "Ant-v3":
        s_dim = int(27)
    elif args.env_name == "Humanoid-v3":
        s_dim = int(45)

    # Tensorboard
    writer = None
    if args.use_tensorboard:
        import datetime

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = "{}_seed{}".format(time_str, args.seed)
        log_dir = Path("/root/tf-logs") / args.env_name / args.experiment_name / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    setproctitle.setproctitle(str(args.env_name) + "-ppolag-" + str(args.seed))

    # Sampler
    if args.safetygym:
        env_sampler = SafetygymEnvSampler(args, env)
    else:
        env_sampler = MuJoCoEnvSampler(args, env)

    # Agent
    agent = PPOLagAgent(s_dim, env.action_space, args)

    # Train
    train_on_policy(args, env_sampler, agent, writer=writer)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # Keep consistent with your existing entry behavior
    import safety_gym  # noqa: F401
    from env.constraints import get_threshold
    from configs.ppolag import PPOLagParser as readParser

    args = readParser()
    args.agent = "ppolag"

    # Auto-detect safety-gym envs (match main.py)
    if "Safe" in args.env_name:
        args.constraint_type = "safetygym"
        args.safetygym = True
        args.epoch_length = 400

    # Cost limit / budget
    args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)

    # CUDA selection (match main.py behavior)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)

    # Keep main.py's behavior: randomize seed each run
    if args.random_seed_each_run:
        args.seed = torch.randint(0, 10000, (1,)).item()

    # If user didn't specify steps_per_rollout, default to epoch_length
    if (not hasattr(args, "steps_per_rollout")) or (args.steps_per_rollout is None) or (args.steps_per_rollout <= 0):
        args.steps_per_rollout = int(args.epoch_length)

    main(args)
