# src/puzzlegame/algorithms/behavioral_cloning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class PuzzleNet(nn.Module):
    """简单的全连接网络用于拼图动作预测"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(PuzzleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def train_bc_model(data_path, model_save_path, n_epochs=100, batch_size=32, lr=1e-3):
    """训练行为克隆模型"""
    print("🚀 开始训练 BC 模型...")
    print(f"📁 读取数据: {os.path.abspath(data_path)}")
    
    # --- 数据加载 ---
    data = np.load(data_path)
    states = data['states']  # shape: (N, 24)
    actions = data['actions']
    
    print(f"📊 数据加载完成，共 {len(states)} 条数据。")
    
    # --- ✅ 核心修复：获取特征维度（第1维）---
    # states.shape[0] = 样本数 (895)
    # states.shape[1] = 特征数 (24) - 这是我们需要的输入维度
    input_dim = int(states.shape[1])  # 修复：使用 shape[1] 而不是整个 shape
    output_dim = 3  # 动作空间维度 (左/上/右)
    
    print(f"🧠 构建模型: 输入维度 = {input_dim} (类型: {type(input_dim)})")
    
    # --- 模型初始化 ---
    model = PuzzleNet(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- 数据预处理 ---
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions).squeeze() # 确保标签形状正确
    
    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 训练循环 ---
    print("⏳ 正在训练...")
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    # --- 保存模型 ---
    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 模型已保存至: {os.path.abspath(model_save_path)}")