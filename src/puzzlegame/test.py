# src/puzzlegame/test.py

import os
import torch
import numpy as np
import random
import time

# 添加项目路径
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from puzzlegame.core.environment import PuzzleGame
from puzzlegame.algorithms.behavioral_cloning import PuzzleNetFeatureBased

def load_model():
    """加载模型"""
    model_path = os.path.join(CURRENT_DIR, "data", "models", "bc_model_feature_based.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件: {model_path}")
        return None
    
    print(f"✅ 加载模型: {model_path}")
    
    model = PuzzleNetFeatureBased(
        bg_dim=20, 
        puzzle_dim=3, 
        pos_dim=1,
        feature_dim=8,
        hidden_dim=128, 
        output_dim=3
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def state_to_tensor(obs):
    """将观测转换为模型输入张量"""
    if isinstance(obs, dict):
        # 构建与训练时相同的状态向量
        background = obs['background'] / 100.0
        puzzle = obs['puzzle'] / 100.0
        current_pos = np.array([obs['current_pos'] / 20.0])  # n=20
        
        state_vec = np.concatenate([background, puzzle, current_pos])
        return torch.FloatTensor(state_vec).unsqueeze(0)
    else:
        return torch.FloatTensor(obs).unsqueeze(0)

def get_model_prediction(model, state_tensor):
    """获取模型预测，处理返回元组的情况"""
    with torch.no_grad():
        result = model(state_tensor)
        
        # 检查返回类型
        if isinstance(result, tuple):
            outputs, _ = result  # 特征提炼网络返回 (outputs, attention)
        else:
            outputs = result  # 简单网络只返回 outputs
        
        action_idx = torch.argmax(outputs, dim=1).item()
        return action_idx

def test_complete_games(model, num_games=50, max_steps=50):
    """测试完整游戏"""
    print(f"\n🎮 开始完整游戏测试 ({num_games}局)")
    
    env = PuzzleGame(n=20, m=3)
    success_count = 0
    failed_games = []
    total_steps_list = []
    
    for game_idx in range(num_games):
        obs = env.reset()
        done = False
        steps = 0
        game_history = []
        
        while not done and steps < max_steps:
            # 模型预测
            state_tensor = state_to_tensor(obs)
            action_idx = get_model_prediction(model, state_tensor)
            
            # 执行动作
            obs, reward, done, _ = env.step(action_idx)
            steps += 1
            
            # 记录游戏过程
            game_history.append({
                'step': steps,
                'action': action_idx,
                'current_pos': obs['current_pos'],
                'target_pos': obs['target_pos'],
                'reward': reward
            })
        
        # 检查结果
        success = (reward > 0)
        if success:
            success_count += 1
            total_steps_list.append(steps)
        else:
            failed_games.append({
                'game_idx': game_idx,
                'steps': steps,
                'final_pos': obs['current_pos'],
                'target_pos': obs['target_pos'],
                'history': game_history
            })
    
    # 输出统计
    success_rate = success_count / num_games * 100
    print(f"📊 完整游戏测试结果:")
    print(f"  成功局数: {success_count}/{num_games}")
    print(f"  成功率: {success_rate:.1f}%")
    
    if success_count > 0:
        avg_steps = np.mean(total_steps_list)
        print(f"  平均成功步数: {avg_steps:.1f}")
    
    return success_rate, failed_games

def display_failed_game(failed_games):
    """展示一局失败的游戏过程"""
    if not failed_games:
        print("\n🎉 没有失败的游戏！")
        return
    
    game = random.choice(failed_games)  # 随机选择一局失败游戏
    
    print(f"\n🔍 随机展示失败游戏 #{game['game_idx']}:")
    print(f"  最终位置: {game['final_pos']}")
    print(f"  目标位置: {game['target_pos']}")
    print(f"  总步数: {game['steps']}")
    
    # 展示关键步骤
    print(f"\n📋 游戏过程关键步骤:")
    
    # 只展示开始、中间和结束的步骤
    history = game['history']
    if len(history) > 0:
        display_indices = [0, len(history)//4, len(history)//2, 3*len(history)//4, -1]
        display_indices = [i for i in display_indices if 0 <= i < len(history)]
        
        for idx in display_indices:
            step_info = history[idx]
            action_names = ["←左移", "→右移", "✓确认"]
            
            print(f"  步骤{step_info['step']:2d}: {action_names[step_info['action']]} "
                  f"| 位置:{step_info['current_pos']:2d} "
                  f"| 目标:{step_info['target_pos']:2d} "
                  f"| 距离:{abs(step_info['current_pos'] - step_info['target_pos']):2d}")
    
    # 显示最终状态可视化
    print(f"\n🎯 最终状态:")
    display = []
    for i in range(20):
        is_target = game['target_pos'] <= i < game['target_pos'] + 3
        is_current = game['final_pos'] <= i < game['final_pos'] + 3
        
        if is_current and is_target:
            display.append('[🎯]')
        elif is_current:
            display.append('[🧩]')
        elif is_target:
            display.append('[⬜]')
        else:
            display.append(' . ')
    
    print(f"  {' '.join(display)}")
    print(f"  当前位置: {game['final_pos']}, 目标位置: {game['target_pos']}")

def test_on_expert_data(model, num_samples=50):
    """在专家数据上测试模型准确率"""
    print(f"\n📚 加载专家数据进行测试 ({num_samples}个样本)")
    
    # 加载专家数据
    data_path = os.path.join(CURRENT_DIR, "data", "raw", "expert_demos.npz")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到专家数据文件: {data_path}")
        return 0
    
    data = np.load(data_path)
    states = data['states']
    actions = data['actions']
    
    print(f"  找到专家数据: {len(states)} 个样本")
    
    # 随机选择样本
    if len(states) > num_samples:
        indices = random.sample(range(len(states)), num_samples)
        test_states = states[indices]
        test_actions = actions[indices]
    else:
        test_states = states
        test_actions = actions
    
    # 测试
    correct = 0
    total = len(test_states)
    
    model.eval()
    for i in range(total):
        state_tensor = torch.FloatTensor(test_states[i]).unsqueeze(0)
        
        # 使用统一的预测函数
        predicted = get_model_prediction(model, state_tensor)
        
        if predicted == int(test_actions[i]):
            correct += 1
    
    accuracy = correct / total * 100
    print(f"📊 专家数据测试结果:")
    print(f"  测试样本数: {total}")
    print(f"  正确预测数: {correct}")
    print(f"  准确率: {accuracy:.1f}%")
    
    # 显示一些错误样本
    if accuracy < 100 and correct < total:
        print(f"\n🔍 错误样本分析 (显示3个):")
        error_count = 0
        for i in range(total):
            if error_count >= 3:
                break
                
            state_tensor = torch.FloatTensor(test_states[i]).unsqueeze(0)
            predicted = get_model_prediction(model, state_tensor)
            
            if predicted != int(test_actions[i]):
                # 解析状态
                bg_values = test_states[i][:20] * 100
                puzzle_values = test_states[i][20:23] * 100
                current_pos = int(test_states[i][23] * 20)
                
                # 找到缺口位置
                gap_positions = []
                for pos in range(20):
                    if bg_values[pos] < 100:
                        gap_positions.append(pos)
                
                action_names = ["左移", "右移", "确认"]
                print(f"  样本{i}:")
                print(f"    当前位置: {current_pos}")
                print(f"    缺口位置: {gap_positions}")
                print(f"    专家动作: {action_names[int(test_actions[i])]}")
                print(f"    模型预测: {action_names[predicted]}")
                error_count += 1
    
    return accuracy

def main():
    print("🧪 开始模型测试")
    print("=" * 50)
    
    # 加载模型
    model = load_model()
    if model is None:
        return
    
    start_time = time.time()
    
    try:
        # 测试1: 完整游戏测试
        success_rate, failed_games = test_complete_games(model, num_games=50)
        
        # 展示一局失败的游戏
        display_failed_game(failed_games)
        
        # 测试2: 专家数据测试
        expert_accuracy = test_on_expert_data(model, num_samples=50)
        
        # 总结
        total_time = time.time() - start_time
        
        print(f"\n" + "=" * 50)
        print(f"📈 测试总结:")
        print(f"  完整游戏成功率: {success_rate:.1f}%")
        print(f"  专家数据准确率: {expert_accuracy:.1f}%")
        print(f"  测试总用时: {total_time:.1f}秒")
        
        # 性能评估
        if success_rate >= 80 and expert_accuracy >= 80:
            print(f"  ✅ 模型性能优秀")
        elif success_rate >= 60 and expert_accuracy >= 60:
            print(f"  ⚠️  模型性能良好")
        elif success_rate >= 40 or expert_accuracy >= 40:
            print(f"  ⚠️  模型性能一般")
        else:
            print(f"  ❌ 模型性能较差，需要改进")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()