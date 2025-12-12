# src/puzzlegame/collect_data.py

import os
import numpy as np
# ✅ 严格使用绝对导入 (按照你的要求)
from puzzlegame.core.environment import PuzzleGame
from puzzlegame.agents.expert_agent import ExpertAgent
from puzzlegame.agents.easy_agent import EasyAgent

def collect_expert():
    # 1. 初始化环境
    env = PuzzleGame(n=20, m=3) 
    
    # 2. 初始化专家
    expert = ExpertAgent(env)
    
    # --- 修正路径逻辑 ---
    # 获取当前文件(__file__)的绝对路径，并定位到同级目录下的 data 文件夹
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(CURRENT_DIR, "data", "raw", "expert_demos.npz")
    
    print("🚀 开始收集专家数据...")
    states, actions = expert.generate_demonstrations(
        num_episodes=1000,      # 生成10000局游戏的数据
        save_path=save_path
    )
    
    print(f"\n🎉 完成！总共收集了 {len(states)} 帧数据。")
    print(f"📄 数据文件已保存在: {os.path.abspath(save_path)}")

def collect_easy():
    easy_agent = EasyAgent()
    
    # --- 修正路径逻辑 ---
    # 获取当前文件(__file__)的绝对路径，并定位到同级目录下的 data 文件夹
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(CURRENT_DIR, "data", "raw", "easy_demos.npz")
    
    print("🚀 开始收集简单代理数据...")
    data = easy_agent.generate_demonstrations(
        num_episodes=1000      # 生成1000局游戏的数据
    )

    # 保存数据
    np.savez_compressed(save_path, states=np.array(data['states']), actions=np.array(data['actions']))
    
    print(f"\n🎉 完成！总共收集了 {len(data['states'])} 帧数据。")
    print(f"📄 数据文件已保存在: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    collect_easy()