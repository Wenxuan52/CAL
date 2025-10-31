import gym
import numpy as np

def main():
    # 可更换为任意环境名，例如 'Ant-v3', 'Humanoid-v3', 'HalfCheetah-v3', 'SafetyCarButton1-v0'
    env_name = 'Safexp-PointButton1-v0'
    seed = 42

    print(f"\n🚀 Testing environment: {env_name}")
    env = gym.make(env_name, render_mode=None)

    # 设置随机种子
    np.random.seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    print(f"✅ Environment created successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print("\nInitial observation sample:")
    print(obs[:5], "...")  # 仅展示前几个元素

    step_in_ep, ep_idx = 0, 0
    max_steps = 3000

    for step in range(max_steps):
        action = env.action_space.sample()
        result = env.step(action)
        # Gymnasium 统一返回六个元素 (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            cost = info.get("cost", None)
        elif len(result) == 6:  # Safety-Gymnasium 风格
            obs, reward, cost, terminated, truncated, info = result
        else:
            raise RuntimeError("Unexpected number of return values from env.step().")

        step_in_ep += 1

        if terminated or truncated:
            ep_idx += 1
            print(f"\n🚩 Episode {ep_idx} ended at step {step_in_ep}")
            print(f"  terminated={terminated}, truncated={truncated}")
            print(f"  reward={reward:.3f}, cost={cost}")
            print(f"  Info: {info}")
            print("-" * 60)

            obs, info = env.reset()
            step_in_ep = 0

    env.close()
    print("\n✅ Finished environment test successfully.")


if __name__ == "__main__":
    main()
