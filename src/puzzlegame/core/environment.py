# src/puzzlegame/core/environment.py

import random
import numpy as np

class PuzzleGame:
    def __init__(self, n=50, m=5): #n是拼图总长度，m是拼图块长度
        self.n = n
        self.m = m
        self.full_pattern = None
        self.puzzle_piece = None
        self.gap_pattern = None
        self.target_pos = None
        self.current_pos = 0
        self.done = False
        self.reset()

    def reset(self):
        # 生成完整图案
        self.full_pattern = np.ones(self.n) * 100 #创建长度为n的数组，初始值为100
        
        # 随机生成正确位置
        self.target_pos = random.randint(self.m, self.n - self.m)
        
        # 生成拼图块 (随机高度)
        self.puzzle_piece = np.random.randint(0, 100, size=self.m)
        
        # 生成缺口背景
        self.gap_pattern = self.full_pattern.copy()
        self.gap_pattern[self.target_pos:self.target_pos+self.m] = 100 - self.puzzle_piece
        
        # 重置玩家位置
        self.current_pos = random.randint(0, self.n)
        #self.current_pos = self.n - 1
        self.done = False
        
        # 返回初始观察 (为了通用性，返回一个包含必要信息的字典)
        return self._get_obs()

    def _get_obs(self):
        # 在真实训练中，这个方法会把状态整理成神经网络需要的格式
        # 现在为了简单，我们只返回必要的数据
        return {
            'background': self.gap_pattern.copy(),
            'puzzle': self.puzzle_piece.copy(),
            'current_pos': self.current_pos,
            'target_pos': self.target_pos  # 注意：在真实AI训练中，通常不会把target_pos给AI看，这里为了方便调试和人类游玩
        }

    def step(self, action):
        # Action: 0=左移, 1=右移, 2=确认
        if self.done:
            return self._get_obs(), 0, True, {}

        # 执行移动逻辑
        if action == 0: # 左移
            self.current_pos = max(0, self.current_pos - 1)
        elif action == 1: # 右移
            self.current_pos = min(self.n - self.m, self.current_pos + 1)
        elif action == 2: # 确认放置
            self.done = True
            # 判断结果
            if self.current_pos == self.target_pos:
                reward = 100
                print(f"\n🎉 恭喜！完美拼合！正确位置: {self.target_pos}")
            else:
                reward = -10
                print(f"\n💥 失败！拼图错位。正确位置: {self.target_pos}, 你的位置: {self.current_pos}")
            return self._get_obs(), reward, True, {}

        # 中间步骤的奖励 (暂时设为0，或者可以根据距离给一点小奖励)
        reward = 0
        return self._get_obs(), reward, False, {}

    def render(self):
        # 直接以数字输出，拼图覆盖处显示为背景数字 + 拼图片数字，并用 [] 包裹
        parts = []
        for i in range(self.n):
            rel = i - self.current_pos
            if 0 <= rel < self.m and rel < len(self.puzzle_piece):
                piece_val = int(self.puzzle_piece[rel])
                bg_val = int(self.gap_pattern[i])
                parts.append(f'[{piece_val + bg_val}]')
            else:
                parts.append(str(int(self.gap_pattern[i])))
        print(' '.join(parts))
