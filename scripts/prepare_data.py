#!/usr/bin/env python3
"""
================================================================================
 数据准备脚本 (prepare_data.py)
 ──────────────────────────────────────────────────────────────────────────────
 从开源数据集下载、转换和标准化数据。

 用法:
   # 下载并处理 BridgeData V2
   python scripts/prepare_data.py --datasets=bridge_v2 --output=/data/unified

   # 下载多个数据集
   python scripts/prepare_data.py --datasets=bridge_v2,roboturk --max_episodes=1000

   # 查看可用数据集
   python scripts/prepare_data.py --list

 工作流程:
   1. 加载数据集注册表
   2. 根据名称选择数据集和适配器
   3. 下载原始数据 (如需要)
   4. 通过适配器转换为 UnifiedSample
   5. 标准化为 Arrow IPC 格式并存入输出目录
   6. (可选)应用数据增强
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
from src.common.utils import setup_logging, ensure_dir
from src.data.registry import DatasetRegistry
from src.data.adapters.base_adapter import AdapterFactory
from src.data.adapters.bridge_v2_adapter import BridgeV2Adapter
from src.data.adapters.roboturk_adapter import RoboTurkAdapter
from src.data.standardization import TrajectoryWriter


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="具身智能数据集准备工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/prepare_data.py --datasets=bridge_v2 --output=/data/unified
  python scripts/prepare_data.py --list
  python scripts/prepare_data.py --datasets=roboturk --max_episodes=500 --augment
        """
    )
    parser.add_argument(
        "--datasets", type=str, default="bridge_v2",
        help="数据集名称 (逗号分隔多个)"
    )
    parser.add_argument(
        "--output", type=str, default="data/unified",
        help="标准化数据输出目录"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=-1,
        help="每个数据集最大处理轨迹数 (-1 表示全部)"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="是否应用数据增强"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用数据集"
    )
    return parser.parse_args()


def main():
    """主函数: 准备数据集"""
    args = parse_args()
    logger = setup_logging("prepare_data")

    # ── 创建适配器工厂并注册内置适配器 ──────────────────────────────────────
    factory = AdapterFactory()
    factory.register("bridge_v2", BridgeV2Adapter)
    factory.register("roboturk", RoboTurkAdapter)

    # ── 列出可用数据集 ──────────────────────────────────────────────────────
    if args.list:
        registry = DatasetRegistry()
        print("\n可用数据集:")
        print("-" * 60)
        for meta in registry.list_all():
            print(f"  {meta.name:20s} | {meta.source.name:20s} | "
                  f"{meta.robot_type.name:15s} | {meta.num_trajs:>6d} trajs")
        print("-" * 60)
        return

    # ── 准备输出目录 ────────────────────────────────────────────────────────
    output_dir = ensure_dir(args.output)
    logger.info(f"输出目录: {output_dir}")

    # ── 处理每个数据集 ──────────────────────────────────────────────────────
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    for ds_name in dataset_names:
        logger.info(f"=" * 50)
        logger.info(f"开始处理数据集: {ds_name}")

        # ── 创建适配器 ──────────────────────────────────────────────────────
        adapter = factory.create(ds_name)
        if adapter is None:
            logger.error(f"未找到数据集 '{ds_name}' 的适配器")
            continue

        # ── 加载数据并转换为统一格式 ─────────────────────────────────────────
        logger.info(f"正在加载 {ds_name}...")
        kwargs = {}
        if args.max_episodes > 0:
            kwargs["max_episodes"] = args.max_episodes

        samples = adapter.load(ds_name, **kwargs)

        # ── 写入标准化格式 (Arrow IPC) ──────────────────────────────────────
        writer = TrajectoryWriter(str(output_dir / ds_name))
        num_trajs = writer.write_samples(samples, ds_name)
        logger.info(f"完成: {ds_name} -> {num_trajs} 条轨迹")

    logger.info("所有数据集准备完成!")


if __name__ == "__main__":
    main()
