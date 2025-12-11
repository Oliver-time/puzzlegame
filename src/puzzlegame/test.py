# src/puzzlegame/test.py

import os
import torch
import numpy as np

# 获取当前文件所在的目录 (即 src/puzzlegame/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from puzzlegame.core.environment import PuzzleGame
from puzzlegame.algorithms.behavioral_cloning import PuzzleNet

def process_state(raw_state):
    """
    将环境返回的状态（可能是字典或数组）转换为一维 numpy 数组
    并强制调整维度为 24 (与训练时保持一致)
    """
    processed = []
    
    if isinstance(raw_state, dict):
        # 如果状态是字典，提取所有值并展平
        for value in raw_state.values():
            if isinstance(value, (list, np.ndarray)):
                processed.extend(value)
            else:
                processed.append(value)
    elif isinstance(raw_state, (list, np.ndarray)):
        # 如果本身就是列表或数组
        processed = list(raw_state)
    else:
        # 兜底
        processed = [raw_state]

    # 转换为 numpy 数阵
    processed = np.array(processed)
    
    # --- 获取实际的特征数量 ---
    flat_state = processed.ravel()
    current_dim = flat_state.size 
    expected_dim = 24

    # --- 强制维度对齐 ---
    if current_dim == expected_dim:
        return flat_state
    elif current_dim > expected_dim:
        print(f"⚠️  状态维度过多 ({current_dim})，已自动截断为 {expected_dim}")
        return flat_state[:expected_dim]
    else:
        print(f"⚠️  状态维度不足 ({current_dim})，已自动填充0至 {expected_dim}")
        padded = np.zeros(expected_dim)
        padded[:current_dim] = flat_state
        return padded

def main():
    # 1. 构建模型路径并加载
    model_path = os.path.join(CURRENT_DIR, "data", "models", "bc_model.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件: {model_path}")
        return

    model = PuzzleNet(input_dim=24, hidden_dim=128, output_dim=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ 加载模型成功: {model_path}")

    # 2. 创建环境
    env = PuzzleGame(n=20, m=3)
    
    # 重置环境
    raw_state = env.reset() 
    print(f"🎮 开始游戏测试... 目标: 移动 {env.m} 个方块到右侧")

    # 3. 运行游戏
    done = False
    step = 0
    
    while not done:
        step += 1
        
        # --- 预处理状态 ---
        processed_state = process_state(raw_state)
        
        # 转换为 Tensor
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
        
        # 模型预测
        with torch.no_grad():
            logits = model(state_tensor)
            action_idx = torch.argmax(logits, dim=1).item()
        
        # --- 修复：环境只返回了 4 个值 ---
        # 常见的返回格式: (next_state, reward, done, info)
        result = env.step(action_idx)
        
        # 根据返回值的数量进行解包
        if len(result) == 4:
            raw_state, reward, done, info = result
        elif len(result) == 5:
            # 兼容新版 Gym 格式 (next_state, reward, terminated, truncated, info)
            raw_state, reward, done, _, info = result
        else:
            # 如果格式异常，直接报错
            raise ValueError(f"env.step() 返回了 {len(result)} 个值，无法解析: {result}")

        print(f"Step {step}: 动作={action_idx}, 奖励={reward}, 完成={done}")
        
        if step > 100:
            print("⚠️  超过最大步数，游戏结束")
            break

    if done and reward > 0:
        print("🎉 模型成功完成了任务！")
    else:
        print("❌ 模型未能完成任务")

if __name__ == "__main__":
    main()