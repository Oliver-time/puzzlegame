# src/puzzlegame/agents/expert_agent.py

import numpy as np
import random
from puzzlegame.core.environment import PuzzleGame

class ExpertAgent:
    def __init__(self, env):
        self.env = env
        print(f"🎮 专家教师已加载。")

    def get_action(self, obs):
        current_pos = obs['current_pos']
        target_pos = obs['target_pos']

        if current_pos < target_pos:
            return 1  # 右移
        elif current_pos > target_pos:
            return 0  # 左移
        else:
            return 2  # 确认

    def generate_demonstrations(self, num_episodes=100, save_path=None):
        """
        生成演示数据。
        :param num_episodes: 生成多少局
        :param save_path: 保存路径 (例如: "../data/raw/expert_data.npz")
        :return: 数据字典
        """
        # 用于存储所有状态和动作
        all_states = []
        all_actions = []
        
        # 动作计数器
        action_counts = {0: 0, 1: 0, 2: 0}  # 0:左移, 1:右移, 2:确认

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                action = self.get_action(obs)
                
                # 更新动作计数
                action_counts[action] += 1
                
                # 获取下一个状态 (为了构建状态向量)
                next_obs, reward, done, _ = self.env.step(action)
                
                # --- 构建状态向量 (与之前保持一致) ---
                background = obs['background']
                puzzle = obs['puzzle']
                pos_feature = np.array([obs['current_pos']])
                
                state_vec = np.concatenate([
                    background / 100.0,
                    puzzle / 100.0,
                    pos_feature / self.env.n
                ])
                
                # 存入列表
                all_states.append(state_vec)
                all_actions.append(action)
                
                obs = next_obs

        # 转换为 NumPy 数组
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        # --- 显示动作统计 ---
        total_actions = len(all_actions)
        print(f"\n📊 动作统计:")
        print(f"  左移 (动作0): {action_counts[0]} 次")
        print(f"  右移 (动作1): {action_counts[1]} 次")
        print(f"  确认 (动作2): {action_counts[2]} 次")
        print(f"  总计: {total_actions} 次")

        # --- 保存数据 ---
        if save_path:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            np.savez(save_path, states=all_states, actions=all_actions)
            print(f"\n💾 数据已保存至: {save_path}")
            print(f"📊 数据形状: 状态 {all_states.shape}, 动作 {all_actions.shape}")

        return all_states, all_actions