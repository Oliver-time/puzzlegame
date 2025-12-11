# src/puzzlegame/algorithms/train_bc.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from puzzlegame.algorithms.behavioral_cloning import PuzzleNet

def train_bc_betterstop():
    # ================= 1. 加载数据 =================
    data_path = "data/raw/expert_demos.npz"
    data = np.load(data_path)
    states = data['states']  # 形状: (N, 24)
    actions = data['actions'] # 形状: (N,)
    
    print(f"✅ 数据加载完成，总样本数: {len(states)}")
    
    # ================= 2. 计算类别权重 (核心修改点) =================
    # 统计每个动作出现的次数
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    print(f"📊 动作分布: {dict(zip(unique_actions, action_counts))}")
    
    # 方法：权重与频率成反比，并对“停止”动作（假设是2）进行额外放大
    # 公式: weight = total_samples / (n_classes * samples_per_class)
    # 但我们手动干预，给停止动作更大的权重
    class_weights = np.ones(len(unique_actions))
    
    total_samples = len(actions)
    for idx, act in enumerate(unique_actions):
        # 基础权重：频率越低，权重越高
        base_weight = total_samples / (len(unique_actions) * action_counts[idx])
        class_weights[idx] = base_weight
        
        # --- ✅ 重点：针对“停止”动作（动作2）进行暴力放大 ---
        if act == 2: # 假设 2 是停止/确认动作
            class_weights[idx] *= 10.0 # 放大10倍！让模型极度害怕预测错停止帧
            print(f"🔥 动作 {act} (停止) 的权重被放大至: {class_weights[idx]:.2f}")
    
    # 转换为 Tensor
    class_weights = torch.FloatTensor(class_weights)
    print(f"⚖️  最终类别权重: {class_weights.numpy()}")

    # ================= 3. 构建 DataLoader =================
    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.LongTensor(actions)
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ================= 4. 初始化模型与优化器 =================
    model = PuzzleNet(input_dim=24, hidden_dim=128, output_dim=3)
    # --- ✅ 关键：将权重传入 CrossEntropyLoss ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ================= 5. 训练循环 =================
    model.train()
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 打印损失
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    # ================= 6. 保存模型 =================
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/bc_model_weighted.pth")
    print(f"✅ 模型训练完成并保存: {model_dir}/bc_model_weighted.pth")

if __name__ == "__main__":
    train_bc_betterstop()
