"""Utility script to quickly sanity-check environment interaction.

This script mirrors the seeding and reset logic used in ``main.py`` so that
Safety-Gym environments are registered correctly and produce observations that
match the training setup.
"""

import argparse
import random

import gym
import numpy as np
import torch
import safety_gym  # noqa: F401  # ensures Safety-Gym environments are registered


def parse_args():
    parser = argparse.ArgumentParser(description="Environment smoke test")
    parser.add_argument(
        "--env_name",
        default="Safexp-PointButton1-v0",
        help="Name of the Gym environment to create",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for numpy, torch and the environment",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of interaction steps to run before finishing",
    )
    parser.add_argument(
        "--render_mode",
        default=None,
        help="Optional render mode passed to gym.make",
    )
    return parser.parse_args()


def _reset_env(env, seed):
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if isinstance(reset_result, tuple):
        observation, info = reset_result
    else:
        observation, info = reset_result, {}
    return observation, info


def main():
    args = parse_args()

    print(f"\n🚀 Testing environment: {args.env_name}")
    env = gym.make(args.env_name, render_mode=args.render_mode)

    # Match main.py seeding behaviour
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    observation, info = _reset_env(env, args.seed)

    print("✅ Environment created successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nInitial observation sample:")
    obs_preview = np.array(observation).flatten()
    print(obs_preview[:5], "..." if obs_preview.size > 5 else "")

    step_in_episode = 0
    episode_index = 0

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        result = env.step(action)

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            cost = info.get("cost") if isinstance(info, dict) else None
        elif len(result) == 6:
            observation, reward, cost, terminated, truncated, info = result
        else:
            raise RuntimeError(
                "Unexpected number of return values from env.step(): "
                f"{len(result)}"
            )

        step_in_episode += 1

        if terminated or truncated:
            episode_index += 1
            print(f"\n🚩 Episode {episode_index} ended at step {step_in_episode}")
            print(f"  terminated={terminated}, truncated={truncated}")
            print(f"  reward={reward:.3f}, cost={cost}")
            print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else info}")
            print("-" * 60)

            observation, info = _reset_env(env, args.seed)
            step_in_episode = 0

    env.close()
    print("\n✅ Finished environment test successfully.")


if __name__ == "__main__":
    main()
