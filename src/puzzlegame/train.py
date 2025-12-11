"""
训练脚本：专门用于解决 AI 学不会停下的问题
使用方法: python src/puzzlegame/train.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- 🔧 路径修复：动态获取项目根目录 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(CURRENT_DIR), 'puzzlegame')

from puzzlegame.algorithms.behavioral_cloning import train_bc_model

def main():
    # --- ✅ 修改：指向新训练的加权模型 ---
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "expert_demos.npz")
    model_save_path = os.path.join(PROJECT_ROOT, "data", "models", "bc_model_feature_based.pth")
    
    # 训练参数
    n_epochs = 2
    batch_size = 64
    learning_rate = 1e-3
    
    # 调用训练函数
    train_bc_model(
        data_path=data_path,
        model_save_path=model_save_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=learning_rate
    )

if __name__ == "__main__":
    main()