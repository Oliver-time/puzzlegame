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

def display_game_info(env, current_pos, target_pos):
    """显示游戏任务的具体情况"""
    print(f"\n{'='*60}")
    print(f"📊 任务详情:")
    print(f"  拼图总长度 (n): {env.n}")
    print(f"  拼图块长度 (m): {env.m}")
    print(f"  当前拼图位置: {current_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  距离目标: {abs(current_pos - target_pos)} 步")
    print(f"  拼图块值: {env.puzzle_piece}")
    
    # 显示简化的游戏状态
    display = []
    for i in range(env.n):
        # 检查是否是目标区域
        is_target = target_pos <= i < target_pos + env.m
        # 检查当前是否有拼图块
        has_puzzle = current_pos <= i < current_pos + env.m
        
        if has_puzzle and is_target:
            display.append('[🎯]')  # 正确位置
        elif has_puzzle:
            display.append('[🧩]')  # 拼图块
        elif is_target:
            display.append('[⬜]')  # 目标缺口
        else:
            display.append(' . ')   # 空位置
            
    print(f"\n  游戏状态:")
    print(f"  {' '.join(display[:min(30, len(display))])}")
    if env.n > 30:
        print(f"  ... (共{env.n}个位置)")
    print(f"{'='*60}\n")

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
    
    # 获取初始状态信息
    if isinstance(raw_state, dict):
        current_pos = raw_state.get('current_pos', 0)
        target_pos = raw_state.get('target_pos', 0)
    else:
        current_pos = 0
        target_pos = env.target_pos if hasattr(env, 'target_pos') else 0
    
    print(f"🎮 开始游戏测试...")
    print(f"🔧 环境设置: n={env.n}, m={env.m}")
    print(f"🎯 任务目标: 将拼图块移动到目标位置 {target_pos}")
    
    # 显示初始任务情况
    display_game_info(env, current_pos, target_pos)

    done = False
    step = 0
    total_reward = 0
    
    while not done:
        step += 1
        processed_state = process_state(raw_state)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(state_tensor)
            action_idx = torch.argmax(logits, dim=1).item()
            action_probs = torch.softmax(logits, dim=1)[0].numpy()
        
        # 动作映射
        action_map = {0: "← 左移", 1: "→ 右移", 2: "✓ 确认放置"}
        action_name = action_map.get(action_idx, f"未知动作 {action_idx}")
        
        # 执行环境步进
        result = env.step(action_idx)
        if len(result) == 4:
            raw_state, reward, done, info = result
        elif len(result) == 5:
            raw_state, reward, done, _, info = result
        
        total_reward += reward
        
        # 获取当前位置
        if isinstance(raw_state, dict):
            current_pos = raw_state.get('current_pos', current_pos)
            target_pos = raw_state.get('target_pos', target_pos)
        
        print(f"\n📋 Step {step}:")
        print(f"  🤖 模型决策: {action_name} (置信度: {action_probs[action_idx]:.3f})")
        print(f"  🏆 即时奖励: {reward:.1f}")
        print(f"  📍 当前位置: {current_pos}")
        print(f"  🎯 目标位置: {target_pos}")
        print(f"  📏 剩余距离: {abs(current_pos - target_pos)}")
        
        # 每5步显示一次详细状态
        if step % 5 == 0 or done:
            display_game_info(env, current_pos, target_pos)
        
        # 步数上限设为50
        if step >= 50:
            print(f"\n⚠️  超过最大步数限制（50步）")
            print(f"📊 统计: 总步数={step}, 总奖励={total_reward}")
            done = True
            break

    # 最终结果
    print(f"\n{'='*60}")
    print(f"🎯 任务完成情况:")
    if reward > 0:
        print(f"  ✅ 成功！拼图块已正确放置到目标位置")
        print(f"  🎉 最终奖励: {reward}")
    else:
        print(f"  ❌ 失败！未能在目标位置放置拼图块")
        print(f"  📍 当前位置: {current_pos}, 目标位置: {target_pos}")
    
    print(f"  📊 统计:")
    print(f"    总步数: {step}")
    print(f"    总奖励: {total_reward}")
    print(f"    最终位置: {current_pos}")
    print(f"    目标位置: {target_pos}")
    print(f"    准确度: {'正确' if current_pos == target_pos else '错误'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()