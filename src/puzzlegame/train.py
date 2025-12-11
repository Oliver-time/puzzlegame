# src/puzzlegame/train.py

import os
import sys
# 注意：这里先不导入算法，先确保路径没问题
from puzzlegame.algorithms.behavioral_cloning import train_bc_model

def main():
    # --- 核心修复：获取当前 Python 文件所在的目录 ---
    # __file__ 是 Python 的内置变量，代表当前文件的路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --- 基于 CURRENT_DIR 构建数据路径 ---
    # 这样无论你在哪个目录下运行脚本，路径都是相对于这个文件的位置
    data_path = os.path.join(CURRENT_DIR, "data", "raw", "expert_demos.npz")
    model_save_path = os.path.join(CURRENT_DIR, "data", "models", "bc_model.pth")
    
    print(f"🔍 正在查找数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到数据文件！")
        print(f"💡 请确认文件是否存在。")
        return

    train_bc_model(
        data_path=data_path,
        model_save_path=model_save_path,
        n_epochs=100,
        batch_size=32
    )

if __name__ == "__main__":
    main()