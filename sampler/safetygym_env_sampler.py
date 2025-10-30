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

    def sample(self, agent, i, eval_t=False):
        # 初始状态重置
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        action = agent.select_action(cur_state, eval_t)
        next_state, reward, done, info = self.env.step(action)

        cost = info.get('cost', 0.0)

        self.path_length += 1
        self.total_path_length += 1

        # 检查截断条件
        if not eval_t:
            # 兼容 gym 和 gymnasium 的 done 逻辑
            truncated = "TimeLimit.truncated" in info or self.path_length >= self.max_path_length
            done = done or truncated
            done = True if ("goal_met" in info and info["goal_met"]) else done

        # 构建 reward 向量
        reward = np.array([reward, cost])

        if done or self.path_length >= self.max_path_length:
            next_state = self.env.reset()
            self.path_length = 0

        self.current_state = next_state

        return cur_state, action, next_state, reward, done, info


    def get_ter_action(self, agent):
        action = agent.select_action(self.cur_s_for_RLmodel, eval=False)
        return action
