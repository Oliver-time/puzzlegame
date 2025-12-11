# src/puzzlegame/test_expert.py

from puzzlegame.core.environment import PuzzleGame
from puzzlegame.agents.expert_agent import ExpertAgent

def main():
    # 1. 初始化环境
    env = PuzzleGame(n=20, m=3)
    
    # 2. 初始化专家
    expert = ExpertAgent(env)
    
    # 3. 生成少量数据 (1条轨迹) 用于测试
    print("开始测试专家...")
    data = expert.generate_demonstrations(num_episodes=1)
    
    # 4. 打印前几步看看
    print("\n--- 查看生成的数据样本 ---")
    for i, step in enumerate(data[:5]): # 只看前5步
        print(f"步骤 {i}: 位置 {step['current_pos']} -> 动作 {step['action']} (目标位置: {step['target_pos']})")

if __name__ == "__main__":
    main()