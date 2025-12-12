# src/puzzlegame/agents/easy_agent.py

# easy_agent is a simple agent that let computer compare two numbers.
# for example, if current_pos < target_pos, return action "smaller" (0)
# if current_pos > target_pos, return action "bigger" (1)

import numpy as np
import random

class EasyAgent:
    def __init__(self):
        print(f"🤖 Easy Agent 已加载。")

    def get_action(self, obs):
        current_pos = obs['current_pos']
        target_pos = obs['target_pos']

        if current_pos < target_pos:
            return 0  # smaller
        elif current_pos > target_pos:
            return 1  # bigger
        else:
            return 2  # equal
    
    def generate_demonstrations(self, num_episodes=100):
        """
        生成演示数据。
        :param num_episodes: 生成多少局
        :return: 数据字典
        """
        # 用于存储所有状态和动作
        all_states = []
        all_actions = []

        for episode in range(num_episodes):
            # 随机生成当前状态
            current_pos = random.randint(0, 9)
            target_pos = random.randint(0, 9)

            done = False

            while not done:
                obs = {
                    'current_pos': current_pos,
                    'target_pos': target_pos
                }
                action = self.get_action(obs)

                # 存储状态和动作
                state_vec = np.array([current_pos, target_pos])
                all_states.append(state_vec)
                all_actions.append(action)

                # 更新状态 (简单模拟)
                if action == 0:  # smaller
                    current_pos += 1
                elif action == 1:  # bigger
                    current_pos -= 1
                else:  # equal
                    done = True

        data = {
            'states': np.array(all_states),
            'actions': np.array(all_actions)
        }
        return data