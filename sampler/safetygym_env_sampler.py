import gym
import numpy as np

class SafetygymEnvSampler():
    def __init__(self, args, env, max_path_length=400):
        self.env = env
        self.args = args

        self.path_length = 0
        self.total_path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length

#     def sample(self, agent, i, eval_t=False):
#         self.total_path_length += 1
#         if i % self.args.epoch_length == 0:
#             self.current_state = self.env.reset()
#         cur_state = self.current_state
#         action = agent.select_action(cur_state, eval_t)
#         next_state, reward, done, info = self.env.step(action)

#         if not eval_t:
#             done = False if i == self.args.epoch_length - 1 or "TimeLimit.truncated" in info else done
#             done = True if "goal_met" in info and info["goal_met"] else done

#         cost = info['cost']
#         self.path_length += 1
#         reward = np.array([reward, cost])
#         self.current_state = next_state
#         return cur_state, action, next_state, reward, done, info

    def sample(self, agent, step_idx: int, eval_mode: bool = False):
        """
        单步环境采样函数（兼容 Gym / Gymnasium / Safety-Gymnasium）

        Args:
            agent: 具有 select_action(state, eval_mode) 方法的智能体
            step_idx (int): 当前全局训练步数，用于统计或调试
            eval_mode (bool): 若为 True，则不修改环境状态（评估模式）

        Returns:
            tuple: (cur_state, action, next_state, reward_vec, done, info)
                   reward_vec = np.array([reward, cost])
        """
        # 1️⃣ 确保环境已重置
        if self.current_state is None:
            reset_out = self.env.reset()
            # Gymnasium 风格 (obs, info)
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                self.current_state, _ = reset_out
            else:  # 旧 Gym 风格
                self.current_state = reset_out

        cur_state = self.current_state

        # 2️⃣ 策略采样动作
        action = agent.select_action(cur_state, eval_mode)

        # 3️⃣ 环境执行一步
        step_out = self.env.step(action)

        # Gymnasium 6元组: (obs, reward, cost, terminated, truncated, info)
        # Gym 5元组: (obs, reward, done, info)
        if len(step_out) == 6:
            next_state, reward, cost, terminated, truncated, info = step_out
            done = terminated or truncated
        elif len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            cost = info.get("cost", 0.0)
            done = terminated or truncated
        elif len(step_out) == 4:
            next_state, reward, done, info = step_out
            cost = info.get("cost", 0.0)
        else:
            raise RuntimeError(f"Unexpected number of outputs from env.step(): {len(step_out)}")

        # 4️⃣ 统计步数
        self.path_length += 1
        self.total_path_length += 1

        # 5️⃣ 环境截断 / 成功终止条件
        if not eval_mode:
            truncated = (
                "TimeLimit.truncated" in info
                or self.path_length >= self.max_path_length
            )
            done = done or truncated
            if info.get("goal_met", False):
                done = True

        # 6️⃣ 构建 reward 向量 [reward, cost]
        reward_vec = np.array([reward, cost], dtype=np.float32)

        # 7️⃣ 若 episode 结束，则重置环境
        if done or self.path_length >= self.max_path_length:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                next_state, _ = reset_out
            else:
                next_state = reset_out
            self.path_length = 0

        # 8️⃣ 更新当前状态缓存
        self.current_state = next_state

        return cur_state, action, next_state, reward_vec, done, info



    def get_ter_action(self, agent):
        action = agent.select_action(self.cur_s_for_RLmodel, eval=False)
        return action
