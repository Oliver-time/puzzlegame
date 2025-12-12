# src/puzzlegame/test_easy.py

import torch
import numpy as np
import sys
import os

sys.path.append(".")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(CURRENT_DIR), 'puzzlegame')

from puzzlegame.algorithms.behavioral_cloning import SimpleNet

def load_easy_model(model_path, input_dim=2):
    """加载简单代理模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleNet(input_dim=input_dim, middle_dim=32, output_dim=3)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # ==============================
    # 1. 配置参数
    # ==============================
    MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "easy_model.pth")  # 替换为你的模型路径
    INPUT_DIM = 2  # 简单代理的输入维度

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==============================
    # 2. 加载模型
    # ==============================
    print("正在加载简单代理模型...")
    try:
        model = load_easy_model(MODEL_PATH, INPUT_DIM)
        print(f"模型加载成功！运行设备: {DEVICE}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    # ==============================
    # 3. 模拟输入并进行推理
    # ==============================
    print("开始推理测试...")
    for i in range(10):
        left_number = np.random.randint(0, 9)
        right_number = np.random.randint(0, 9)
        state_vec = np.array([left_number, right_number], dtype=np.float32)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            action_logits = model(state_tensor)
        print(f"输入状态: 左边={left_number}, 右边={right_number} -> 模型输出 logits: {action_logits.cpu().numpy().round(3)}")

if __name__ == "__main__":
    main()
