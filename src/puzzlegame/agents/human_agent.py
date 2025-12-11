import sys
from puzzlegame.core.environment import PuzzleGame

class HumanAgent:
    def __init__(self, env):
        self.env = env

    def run(self):
        print("\n=== 拼图游戏开始 (人类玩家版) ===")
        print("🎮 操作说明: A=左移, D=右移, S=确认放置, Q=退出")
        
        # 重置环境
        obs = self.env.reset()
        self.env.render()
        
        # 游戏主循环
        while True:
            try:
                # 获取用户输入
                action_input = input("请输入操作 (a/d/s/q): ").strip().lower()
                
                if action_input == 'q':
                    print("👋 游戏结束。")
                    break
                elif action_input == 'a':
                    obs, reward, done, _ = self.env.step(0) # 左移
                elif action_input == 'd':
                    obs, reward, done, _ = self.env.step(1) # 右移
                elif action_input == 's':
                    obs, reward, done, _ = self.env.step(2) # 确认
                else:
                    print("⚠️ 无效输入，请输入 a, d, s 或 q")
                    continue
                
                # 渲染新状态
                self.env.render()
                
                # 如果游戏结束，询问是否重玩
                if done:
                    play_again = input("\n是否再来一局？(y/n): ").strip().lower()
                    if play_again == 'y':
                        obs = self.env.reset()
                        self.env.render()
                    else:
                        break
                        
            except KeyboardInterrupt:
                print("\n👋 强制退出。")
                break