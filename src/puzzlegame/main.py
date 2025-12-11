"""
拼图游戏主入口
用于启动人类玩家模式或后续的AI模式
"""
from puzzlegame.agents.human_agent import HumanAgent
from puzzlegame.core.environment import PuzzleGame

def main():
    # 1. 初始化环境 (可以在这里调整参数，比如 n=20, m=3)
    env = PuzzleGame(n=20, m=3)
    
    # 2. 初始化人类玩家代理
    agent = HumanAgent(env)
    
    # 3. 启动游戏循环
    agent.run()

if __name__ == "__main__":
    main()