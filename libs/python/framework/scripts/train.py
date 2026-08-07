#!/usr/bin/env python3
"""
================================================================================
 训练脚本 (train.py)
 ──────────────────────────────────────────────────────────────────────────────
 启动模仿学习或强化学习训练。

 用法:
   # 行为克隆训练
   python scripts/train.py experiment=bc_manipulation

   # PPO 强化学习训练 (从示范数据启动)
   python scripts/train.py experiment=ppo_pick_place

   # 使用特定配置文件
   python scripts/train.py --config config/experiment/bc_manipulation.yaml

   # 覆盖配置参数 (Hydra 风格)
   python scripts/train.py experiment=bc_manipulation training.learning_rate=0.001

 训练流程:
   1. 加载实验配置文件
   2. 初始化数据加载器和仿真环境
   3. 创建算法训练器
   4. 执行训练循环
   5. 保存模型检查点
   6. 记录实验指标到 MLflow
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import sys
import argparse
from pathlib import Path

# ─── 添加项目根目录到系统路径 ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.utils import setup_logging
from src.orchestration.pipeline import TrainingPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="具身智能训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/experiment/bc_manipulation.yaml",
        help="实验配置文件路径"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="配置覆盖参数 (如 training.learning_rate=0.001)"
    )
    return parser.parse_args()


def main():
    """主函数: 启动训练流水线"""
    args = parse_args()
    logger = setup_logging("train")

    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    logger.info(f"加载配置: {config_path}")

    # ── 创建并运行训练流水线 ────────────────────────────────────────────────
    pipeline = TrainingPipeline(config_path)
    results = pipeline.run()

    logger.info(f"训练完成!")
    logger.info(f"结果: {results}")


if __name__ == "__main__":
    main()
