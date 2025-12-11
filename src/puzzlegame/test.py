# src/puzzlegame/test.py

import os
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from puzzlegame.core.environment import PuzzleGame
from puzzlegame.algorithms.behavioral_cloning import PuzzleNet

def process_state(raw_state):
    # ... (保持之前的处理逻辑不变，确保维度为24) ...
    processed = []
    if isinstance(raw_state, dict):
        for value in raw_state.values():
            if isinstance(value, (list, np.ndarray)):
                processed.extend(value)
            else:
                processed.append(value)
    elif isinstance(raw_state, (list, np.ndarray)):
        processed = list(raw_state)
    else:
        processed = [raw_state]

    processed = np.array(processed)
    flat_state = processed.ravel()
    
    # 假设训练维度是24
    expected_dim = 24
    if flat_state.size > expected_dim:
        return flat_state[:expected_dim]
    elif flat_state.size < expected_dim:
        padded = np.zeros(expected_dim)
        padded[:flat_state.size] = flat_state
        return padded
    return flat_state

def main():
    # --- ✅ 修改：指向新训练的加权模型 ---
    model_path = os.path.join(CURRENT_DIR, "data", "models", "bc_model_weighted.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件: {model_path}")
        print("请先运行 train_bc.py 生成模型")
        return

    model = PuzzleNet(input_dim=24, hidden_dim=128, output_dim=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ 加载加权模型成功: {model_path}")

    env = PuzzleGame(n=20, m=3)
    raw_state = env.reset() 
    print(f"🎮 开始游戏测试... 目标: 移动 {env.m} 个方块到右侧")

    # 连续动作计数器 (辅助策略，双重保险)
    consecutive_same_action = 0
    last_action = -1
    done = False
    step = 0
    
    while not done:
        step += 1
        processed_state = process_state(raw_state)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(state_tensor)
            action_idx = torch.argmax(logits, dim=1).item()
        
        # --- 辅助逻辑：防止物理死循环 ---
        if action_idx == last_action:
            consecutive_same_action += 1
        else:
            consecutive_same_action = 0
            last_action = action_idx

        # 如果连续推荐同一动作超过阈值，强制停止 (假设0是停止或左移)
        if consecutive_same_action >= 5:
            print(f"🛑 触发物理刹车！")
            action_to_take = 0 # 假设0是安全动作
        else:
            action_to_take = action_idx

        # 执行环境步进
        result = env.step(action_to_take)
        if len(result) == 4:
            raw_state, reward, done, info = result
        elif len(result) == 5:
            raw_state, reward, done, _, info = result

        print(f"Step {step}: 动作={action_to_take} (模型: {action_idx}), 奖励={reward}, 完成={done}")
        
        if step > 100:
            print("⚠️  超过最大步数")
            break

    if done and reward > 0:
        print("🎉 模型成功完成任务！")
    else:
        print("❌ 模型未能完成任务")

if __name__ == "__main__":
    main()