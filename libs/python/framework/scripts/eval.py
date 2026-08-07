#!/usr/bin/env python3
"""
================================================================================
 评估脚本 (eval.py)
 ──────────────────────────────────────────────────────────────────────────────
 在仿真环境中评估训练好的策略。

 用法:
   # 评估 BC 策略
   python scripts/eval.py --checkpoint=checkpoints/bc/best.pt --sim=mujoco

   # 评估并渲染视频
   python scripts/eval.py --checkpoint=checkpoints/ppo/best.pt --render --video=videos/

   # 比较多个策略
   python scripts/eval.py --checkpoint=checkpoints/bc/best.pt,checkpoints/ppo/best.pt

 评估流程:
   1. 加载策略模型权重
   2. 初始化仿真环境 (指定后端)
   3. 运行多个评估 episode
   4. 计算统计指标 (成功率、平均奖励等)
   5. (可选)渲染并保存视频
   6. 输出评估报告
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

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.utils import setup_logging, load_config
from src.algorithm.policy import ActorPolicy, GaussianPolicy
from src.simulation.backend import SimBackendFactory, RobotEnv
from src.orchestration.evaluator import Evaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="具身智能策略评估工具",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径 (逗号分隔多个)"
    )
    parser.add_argument(
        "--config", type=str, default="config/experiment/bc_manipulation.yaml",
        help="实验配置文件 (用于环境配置)"
    )
    parser.add_argument(
        "--sim", type=str, default="mujoco",
        help="仿真后端 (mujoco / isaac_sim / pybullet)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50,
        help="评估回合数"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="是否渲染视频"
    )
    parser.add_argument(
        "--video", type=str, default="videos/eval",
        help="视频输出目录"
    )
    return parser.parse_args()


def main():
    """主函数: 评估策略"""
    args = parse_args()
    logger = setup_logging("eval")

    # ── 加载配置 ────────────────────────────────────────────────────────────
    config = load_config(args.config)
    sim_config = config.get("simulation", {})
    sim_config["backend"] = args.sim

    # ── 创建仿真环境 ────────────────────────────────────────────────────────
    def make_env():
        factory = SimBackendFactory()
        backend = factory.create(args.sim, sim_config)
        return RobotEnv(backend, sim_config)

    # ── 创建评估器 ──────────────────────────────────────────────────────────
    evaluator = Evaluator(
        env_fn=make_env,
        render=args.render,
        video_dir=args.video if args.render else None,
    )

    # ── 遍历检查点 ──────────────────────────────────────────────────────────
    checkpoints = [c.strip() for c in args.checkpoint.split(",")]
    for cp_path in checkpoints:
        logger.info(f"=" * 50)
        logger.info(f"评估策略: {cp_path}")

        if not os.path.exists(cp_path):
            logger.error(f"检查点不存在: {cp_path}")
            continue

        # ── 加载策略 ──────────────────────────────────────────────────────
        obs_dim = config.get("training", {}).get("obs_dim", 128)
        action_dim = config.get("training", {}).get("action_dim", 7)
        policy_net = GaussianPolicy(obs_dim, action_dim)
        policy = ActorPolicy(policy_net)
        policy.load(cp_path)

        # ── 运行评估 ──────────────────────────────────────────────────────
        results = evaluator.evaluate(
            policy=policy,
            num_episodes=args.num_episodes,
        )

        # ── 输出结果 ──────────────────────────────────────────────────────
        logger.info(f"评估结果:")
        logger.info(f"  成功率:       {results.success_rate:.2%}")
        logger.info(f"  平均奖励:     {results.avg_reward:.3f}")
        logger.info(f"  平均轨迹长度: {results.avg_episode_length:.1f} 步")
        logger.info(f"  测试回合数:   {results.total_episodes}")

    logger.info("评估完成!")


if __name__ == "__main__":
    main()
