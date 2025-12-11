# src/puzzlegame/algorithms/behavioral_cloning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class FeatureExtractionNet(nn.Module):
    """特征提取网络：从环境背景中提取关键特征"""
    def __init__(self, input_dim=20, feature_dim=8):
        super(FeatureExtractionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim)  # 提炼为8个关键特征
        )
    
    def forward(self, x):
        return self.net(x)

class PuzzleNetFeatureBased(nn.Module):
    """基于特征提炼的拼图网络"""
    def __init__(self, bg_dim=20, puzzle_dim=3, pos_dim=1, 
                 feature_dim=8, hidden_dim=128, output_dim=3):
        super(PuzzleNetFeatureBased, self).__init__()
        
        # 1. 环境特征提取器
        self.bg_feature_extractor = FeatureExtractionNet(bg_dim, feature_dim)
        
        # 2. 拼图特征提取器（可选的，可以直接用）
        self.puzzle_net = nn.Sequential(
            nn.Linear(puzzle_dim, 8),
            nn.ReLU()
        )
        
        # 3. 位置编码器
        self.pos_net = nn.Sequential(
            nn.Linear(pos_dim, 4),
            nn.ReLU()
        )
        
        # 4. 特征融合和比较层
        # 总输入：环境特征(8) + 拼图特征(8) + 位置特征(4) = 20
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 8 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            # 特别注意：这里添加一个"匹配度"输出层
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # 5. 注意力机制（可选）：让网络关注环境中的关键位置
        self.attention = nn.Sequential(
            nn.Linear(bg_dim + puzzle_dim + pos_dim, 16),
            nn.ReLU(),
            nn.Linear(16, bg_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 分割输入
        bg = x[:, :20]          # 背景部分
        puzzle = x[:, 20:23]    # 拼图部分
        pos = x[:, 23:]         # 位置部分
        
        # 方法A：直接特征提取
        bg_features = self.bg_feature_extractor(bg)  # 环境特征 (batch, 8)
        puzzle_features = self.puzzle_net(puzzle)    # 拼图特征 (batch, 8)
        pos_features = self.pos_net(pos)             # 位置特征 (batch, 4)
        
        # 方法B：使用注意力机制（增强版本）
        combined_input = torch.cat([bg, puzzle, pos], dim=1)
        attention_weights = self.attention(combined_input)  # (batch, 20)
        
        # 应用注意力到背景特征
        attended_bg = bg * attention_weights
        bg_features_attended = self.bg_feature_extractor(attended_bg)
        
        # 融合所有特征
        combined_features = torch.cat([
            bg_features_attended,  # 使用注意力版本
            puzzle_features,
            pos_features
        ], dim=1)
        
        # 最终决策
        output = self.fusion(combined_features)
        
        return output, attention_weights  # 返回注意力和便于分析


class PuzzleNetSimple(nn.Module):
    """简单的全连接网络用于拼图动作预测（保持向后兼容）"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(PuzzleNetSimple, self).__init__()
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
    
def train_bc_model(data_path, model_save_path, n_epochs=100, 
                   batch_size=32, lr=1e-3, use_feature_based=True):
    """训练行为克隆模型"""
    print("🚀 开始训练 BC 模型...")
    print(f"📁 读取数据: {os.path.abspath(data_path)}")
    
    # --- 数据加载 ---
    data = np.load(data_path)
    states = data['states']
    actions = data['actions']
    
    print(f"📊 数据加载完成，共 {len(states)} 条数据。")
    
    # 获取特征维度
    input_dim = int(states.shape[1])
    output_dim = 3
    
    print(f"🧠 输入维度: {input_dim}")
    
    # --- 模型初始化 ---
    if use_feature_based and input_dim == 24:
        print("🔧 使用特征提炼网络")
        model = PuzzleNetFeatureBased(
            bg_dim=20, 
            puzzle_dim=3, 
            pos_dim=1,
            feature_dim=8,
            hidden_dim=128, 
            output_dim=output_dim
        )
    else:
        print("🔧 使用简单网络")
        model = PuzzleNetSimple(
            input_dim=input_dim, 
            hidden_dim=128, 
            output_dim=output_dim
        )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- 数据预处理 ---
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions).squeeze()
    
    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 训练循环 ---
    print("⏳ 正在训练...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            
            if use_feature_based:
                outputs, attention = model(batch_states)
            else:
                outputs = model(batch_states)
            
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += batch_actions.size(0)
            correct += (predicted == batch_actions).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.1f}%")
            
            # 可视化注意力（每隔一段时间）
            if use_feature_based and (epoch+1) % 30 == 0:
                visualize_attention(model, batch_states[:3])
    
    # --- 保存模型 ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 模型已保存至: {os.path.abspath(model_save_path)}")
    
    return model

def visualize_attention(model, sample_batch):
    """可视化注意力权重"""
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(sample_batch)
    
    print("\n🔍 注意力可视化（前3个样本）:")
    for i in range(min(3, len(sample_batch))):
        attention = attention_weights[i].numpy()
        print(f"样本{i}:")
        print(f"  背景值: {sample_batch[i, :20].numpy().round(2)}")
        print(f"  注意力: {attention.round(3)}")
        print(f"  重点关注位置: {np.where(attention > 0.1)[0]}")