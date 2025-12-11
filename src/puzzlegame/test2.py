"""
test2.py: 在真实 PuzzleGame 环境中可视化模型决策概率（适配 environment.py 实际参数）
"""
import torch
import numpy as np
import sys
import os

# 根据实际项目路径调整（假设在 puzzlegame 根目录下运行）
sys.path.append(".")

from puzzlegame.core.environment import PuzzleGame
from puzzlegame.algorithms.behavioral_cloning import PuzzleNetFeatureBased, PuzzleNetSimple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(CURRENT_DIR), 'puzzlegame')

def load_model(model_path, input_dim=24, use_feature_based=True):
    """加载训练好的模型（适配新版模型类）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_feature_based:
        model = PuzzleNetFeatureBased(
            bg_dim=20, puzzle_dim=3, pos_dim=1,  # 与环境参数严格对应
            feature_dim=8, hidden_dim=128, output_dim=3
        )
    else:
        model = PuzzleNetSimple(input_dim=input_dim, hidden_dim=128, output_dim=3)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # ==============================
    # 1. 配置参数（与训练时保持一致！）
    # ==============================
    MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "bc_model_feature_based.pth")  # 替换为你的模型路径
    USE_FEATURE_BASED = True  # 根据训练时选择的模型类型调整
    INPUT_DIM = 24            # 20(bg) + 3(piece) + 1(pos) 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==============================
    # 2. 初始化环境和模型
    # ==============================
    print("正在初始化环境...")
    env = PuzzleGame(n=20, m=3)  # 严格按 __init__(self, n=20, m=3) 传参
    print("环境初始化成功！")

    print("正在加载模型...")
    try:
        model = load_model(MODEL_PATH, INPUT_DIM, USE_FEATURE_BASED)
        print(f"模型加载成功！运行设备: {DEVICE}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # ==============================
    # 3. 游戏主循环 + 概率推理
    # ==============================
    print("\n" + "="*60)
    print("🎮 拼图游戏演示：模型将实时显示每一步的动作概率")
    print("="*60)
    
    obs = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        print(f"\n--- 步骤 {step} ---")
        
        # --- 构造模型输入（关键：按实际环境状态拼接）---
        bg_vec = obs['background']                    # 长度 50
        piece_vec = obs['puzzle']                    # 长度 5
        pos_vec = np.array([obs['current_pos']])     # 标量转为向量
        
        input_vec = np.concatenate([bg_vec, piece_vec, pos_vec])  # 总长度 56
        assert len(input_vec) == INPUT_DIM, f"输入维度错误！期望 {INPUT_DIM}，实际 {len(input_vec)}"
        
        input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(DEVICE)

        # --- 模型推理 ---
        with torch.no_grad():
            if USE_FEATURE_BASED:
                logits, _ = model(input_tensor)  # 忽略注意力权重
            else:
                logits = model(input_tensor)
        
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()  # 转为一维概率数组

        # --- 打印概率 ---
        print(f"动作概率分布:")
        action_names = ["⬅️ 左移", "➡️ 右移", "✅ 确认"]
        for i, prob in enumerate(probs):
            print(f"  {action_names[i]}: {prob:.4f} ({prob*100:.2f}%)")
        
        # --- 选择动作（贪婪策略）---
        action = np.argmax(probs)
        print(f"\n--> 模型选择动作: {action_names[action]}")

        # --- 执行动作 ---
        obs, reward, done, info = env.step(action)
        
        # --- 渲染环境 ---
        print(f"\n环境渲染:")
        env.render()

        # --- 交互控制 ---
        if not done:
            user_input = input("\n按回车继续，输入 'q' 结束测试: ")
            if user_input.lower() == 'q':
                break

    print("\n" + "="*60)
    print("游戏结束。")
    print("="*60)

if __name__ == '__main__':
    main()