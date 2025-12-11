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
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from puzzlegame.algorithms.behavioral_cloning import PuzzleNet

def train_bc_model():
    # ================= 1. 路径配置与数据加载 =================
    data_path = os.path.join(CURRENT_DIR, "data", "raw", "expert_demos.npz")
    
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件: {data_path}")
        print("请先运行数据收集脚本生成 expert_demos.npz")
        return

    try:
        data = np.load(data_path)
        states = data['states']
        actions = data['actions']
        print(f"✅ 成功加载数据，样本总数: {len(states)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # ================= 2. 计算类别权重（修复维度问题）=================
    # --- ✅ 核心修复：明确指定有3个类别（0,1,2），即使某些类别未出现 ---
    num_classes = 3  # 必须与模型输出维度一致
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    print(f"📊 原始动作分布: {dict(zip(unique_actions, action_counts))}")
    
    # 初始化所有类别的权重为1.0
    class_weights = np.ones(num_classes)
    total_samples = len(actions)
    
    for idx, act in enumerate(unique_actions):
        # 转换为整数索引
        act = int(act)
        # 反频率权重
        weight = total_samples / (num_classes * action_counts[idx])
        class_weights[act] = weight
        
        # --- ✅ 放大停止动作（假设动作2是停止）的权重 ---
        if act == 2:
            class_weights[act] *= 10.0
            print(f"🔥 动作 {act} ('停止') 的权重被放大至: {class_weights[act]:.2f}")
    
    # 转换为 Tensor
    class_weights = torch.FloatTensor(class_weights)
    print(f"⚖️  最终类别权重（所有3类）: {class_weights.numpy()}")  # 应输出3个值

    # ================= 3. 数据集准备 =================
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ================= 4. 模型与优化器 =================
    model = PuzzleNet(input_dim=24, hidden_dim=128, output_dim=3)
    # --- ✅ 关键：权重维度现在与输出层匹配 ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ================= 5. 训练循环 =================
    model.train()
    epochs = 100
    
    print("🚀 开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], 平均 Loss: {avg_loss:.4f}")
    
    # ================= 6. 保存模型 =================
    model_dir = os.path.join(CURRENT_DIR, "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "bc_model_weighted.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型训练完成！已保存至: {model_path}")

if __name__ == "__main__":
    train_bc_model()