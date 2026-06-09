#!/usr/bin/env python3
"""
================================================================================
 数据准备脚本 (prepare_data.py)
 ──────────────────────────────────────────────────────────────────────────────
 从本地原始数据文件或远程 URL 加载数据集，转换为统一的 Arrow IPC 格式。

 ═══════════════════════════════════════════════════════════════════════════════
 核心设计: 本地优先 (Local-First)
 ═══════════════════════════════════════════════════════════════════════════════
 
 数据路径优先级 (从高到低):
   1. 命令行 --local_path 参数           # 临时覆盖，优先级最高
   2. 实验配置文件 data.data_dir          # 实验级覆盖
   3. 数据集注册表 registry.yaml 中配置   # 数据集默认路径
   4. 脚本内置默认值 data/unified/       # 兜底默认值

 数据加载流程:
   ┌─────────────────────────────────────────────────────────────────┐
   │ 步骤 1: 确定本地路径                                            │
   │   path = --local_path ?? registry.local_path ?? "data/unified/" │
   │                                                                 │
   │ 步骤 2: 检查本地是否已有标准化数据                               │
   │   if os.path.exists(path) and *.arrow files exist:              │
   │       print("本地已存在，跳过准备")                              │
   │       return                                                    │
   │                                                                 │
   │ 步骤 3: 从本地原始数据或远程下载 → 适配器转换                     │
   │   raw_path = --raw_path ?? registry.raw_path ?? path + "/raw"   │
   │   adapter.load(source=raw_path)  →  UnifiedSample 流            │
   │                                                                 │
   │ 步骤 4: 写入标准化 Arrow IPC 格式                                │
   │   TrajectoryWriter(path).write_samples(samples)                 │
   └─────────────────────────────────────────────────────────────────┘

 用法:
   # === 场景 1: 本地已有标准化数据 (最常用) ===
   # 训练时 dataloader 直接从 local_path 读取，无需运行本脚本
   # train.py 中: UnifiedDataset(config.data.data_dir)
   
   # === 场景 2: 本地有原始数据，需要转换为标准化格式 ===
   python scripts/prepare_data.py --datasets=bridge_v2 \\
       --raw_path=data/raw/bridge_v2          \\  # 原始数据位置
       --local_path=data/unified/bridge_v2     \\  # 输出位置
       --max_episodes=1000

   # === 场景 3: 从远程下载并转换 (首次使用) ===
   python scripts/prepare_data.py --datasets=bridge_v2 \\
       --download                           \\  # 允许从远程下载
       --local_path=data/unified/bridge_v2

   # === 场景 4: 查看可用数据集 ===
   python scripts/prepare_data.py --list

   # === 场景 5: 测试模式 (生成模拟数据) ===
   python scripts/prepare_data.py --datasets=bridge_v2 --demo
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import sys
import argparse
from pathlib import Path

# ─── 可选: 配置 HuggingFace 国内镜像加速下载 ───────────────────────────────
# 如果在中国大陆且下载 HuggingFace 数据集速度慢，取消下面这行的注释:
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ─── 添加项目根目录到系统路径 ──────────────────────────────────────────────
# 确保 Python 可以找到 src/ 包
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
    """
    解析命令行参数
    ────────────────────────────────────────────────────────────────────────────
    所有数据路径均可在命令行中覆盖，优先级高于配置文件。
    如果某个路径未指定，则从 datasets registry.yaml 中读取默认值。
    """
    parser = argparse.ArgumentParser(
        description="具身智能数据集准备工具 — 本地优先数据加载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
路径优先级: 命令行 > 实验配置 > registry.yaml > 内置默认值

示例:
  # 从本地原始数据转换
  python scripts/prepare_data.py --datasets=bridge_v2 --raw_path=data/raw/bridge_v2

  # 指定输出路径
  python scripts/prepare_data.py --datasets=roboturk --local_path=/mnt/nvme/data/roboturk

  # 从远程下载 (首次)
  python scripts/prepare_data.py --datasets=bridge_v2 --download --max_episodes=500
        """
    )
    # ── 数据集选择 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--datasets", type=str, default="bridge_v2",
        help="数据集名称 (逗号分隔多个，如 bridge_v2,roboturk)"
    )
    # ── 本地路径配置 (核心参数) ─────────────────────────────────────────────
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="【关键】标准化数据的本地输出/读取目录。"
             "默认从 registry.yaml 的 local_path 字段读取。"
             "示例: --local_path=data/unified/bridge_v2"
    )
    parser.add_argument(
        "--raw_path", type=str, default=None,
        help="原始数据本地缓存路径。适配器从此目录读取原始文件。"
             "默认从 registry.yaml 的 raw_path 字段读取。"
             "示例: --raw_path=data/raw/bridge_v2"
    )
    # ── 下载控制 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--download", action="store_true",
        help="允许从远程下载原始数据。默认 false —— 仅从本地加载。"
             "首次使用某个数据集时需要此参数。"
    )
    # ── 处理控制 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_episodes", type=int, default=-1,
        help="每个数据集最大处理轨迹数。"
             "-1 表示处理全部，设置小值 (如 100) 可用于快速测试。"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="是否在转换过程中应用数据增强"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo 模式: 当本地和远程数据都不可用时，"
             "自动生成模拟数据进行流水线测试。"
    )
    # ── 信息查询 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有已注册的数据集及其本地路径"
    )
    return parser.parse_args()


def main():
    """
    主函数: 准备数据集 (本地优先)
    ────────────────────────────────────────────────────────────────────────────
    执行流程:
      1. 解析命令行参数 → 获取路径配置
      2. 加载数据集注册表 → 获取数据集默认路径
      3. 确定最终路径 (命令行 > registry.yaml > 默认值)
      4. 检查本地是否已有标准化数据 → 有则跳过
      5. 从本地原始数据或远程下载 → 适配器转换 → 写入 Arrow 文件

    路径确定逻辑:
      final_path = args.local_path or registry.local_path or "data/unified/{name}"
    """
    args = parse_args()
    logger = setup_logging("prepare_data")

    # ── 加载数据集注册表 (从 YAML) ─────────────────────────────────────────
    # DatasetRegistry 会自动读取 config/dataset/registry.yaml
    registry = DatasetRegistry()

    # ── 列出可用数据集 (含本地路径信息) ────────────────────────────────────
    if args.list:
        print("\n" + "=" * 70)
        print(f"{'数据集名称':20s} | {'本地路径':30s} | {'轨迹数':>8s}")
        print("=" * 70)
        for meta in registry.list_all():
            # 从 registry.yaml 中读取 local_path 配置
            local_path = getattr(meta, 'local_path', '') or f"data/unified/{meta.name}"
            print(f"  {meta.name:20s} | {local_path:30s} | {meta.num_trajs:>8d}")
        print("=" * 70)
        print("提示: 使用 --local_path 参数覆盖数据路径")
        return

    # ── 创建适配器工厂并注册内置适配器 ──────────────────────────────────────
    factory = AdapterFactory()
    factory.register("bridge_v2", BridgeV2Adapter)
    factory.register("roboturk", RoboTurkAdapter)

    # ── 处理每个数据集 ──────────────────────────────────────────────────────
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    for ds_name in dataset_names:
        logger.info("=" * 60)
        logger.info(f"▶ 开始处理数据集: {ds_name}")

        # ── 步骤 1: 确定本地输出路径 (优先级: 命令行 > registry.yaml) ─────
        # ────────────────────────────────────────────────────────────────────
        # 从注册表获取该数据集的元信息
        ds_meta = registry.get(ds_name)

        # 确定标准化数据输出路径
        if args.local_path:
            # 优先级 1: 命令行显式指定
            output_dir = args.local_path
            logger.info(f"[路径来源] 命令行 --local_path 参数")
        elif ds_meta and hasattr(ds_meta, 'local_path') and ds_meta.local_path:
            # 优先级 2: 数据集注册表配置
            output_dir = ds_meta.local_path
            logger.info(f"[路径来源] registry.yaml 配置")
        else:
            # 优先级 3: 内置默认值
            output_dir = os.path.join("data", "unified", ds_name)
            logger.info(f"[路径来源] 内置默认路径")

        # 确保输出目录存在
        output_dir = ensure_dir(output_dir)
        logger.info(f"[本地路径] {output_dir}")

        # ── 步骤 2: 检查本地是否已有标准化数据 ─────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        # 扫描输出目录下是否有 .arrow 文件
        existing_arrow_files = list(Path(output_dir).glob("*.arrow"))
        if existing_arrow_files:
            logger.info(f"[跳过] 本地路径 '{output_dir}' 已存在 "
                        f"{len(existing_arrow_files)} 个 Arrow 文件")
            logger.info(f"  如需重新处理，请删除或移动该目录下的 .arrow 文件")
            logger.info(f"  或使用不同的 --local_path 指定新路径")
            continue

        # ── 步骤 3: 确定原始数据路径 ───────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        if args.raw_path:
            raw_path = args.raw_path
            logger.info(f"[原始数据] 命令行指定: {raw_path}")
        elif ds_meta and hasattr(ds_meta, 'raw_path') and ds_meta.raw_path:
            raw_path = ds_meta.raw_path
            logger.info(f"[原始数据] registry.yaml 配置: {raw_path}")
        else:
            raw_path = os.path.join("data", "raw", ds_name)
            logger.info(f"[原始数据] 使用默认缓存路径: {raw_path}")

        # ── 步骤 4: 创建适配器并加载数据 ───────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        adapter = factory.create(ds_name)
        if adapter is None:
            logger.error(f"✗ 未找到数据集 '{ds_name}' 的适配器")
            logger.error(f"  请在 AdapterFactory 中注册对应的适配器类")
            continue

        # 构造适配器参数
        kwargs = {"demo_mode": args.demo}
        if args.max_episodes > 0:
            kwargs["max_episodes"] = args.max_episodes

        # 根据 --download 标志决定是否允许远程下载
        if not args.download:
            logger.info(f"[加载模式] 仅本地加载 (使用 --download 允许远程下载)")
            kwargs["local_only"] = True
        else:
            logger.info(f"[加载模式] 允许远程下载")

        # ── 步骤 5: 执行数据加载与转换 ─────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        logger.info(f"正在加载并转换 '{ds_name}'...")
        try:
            # adapter.load() 从 raw_path 读取原始数据，
            # 返回 UnifiedSample 迭代器 (逐个样本产出)
            samples = adapter.load(raw_path, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"✗ 本地原始数据未找到: {e}")
            logger.error(f"  请检查 --raw_path 参数是否正确")
            logger.error(f"  或使用 --download 参数从远程下载")
            if args.demo:
                logger.info(f"  启用 --demo 模式，使用模拟数据继续...")
                kwargs["demo_mode"] = True
                samples = adapter.load(raw_path, **kwargs)
            else:
                logger.error(f"  或使用 --demo 参数生成模拟数据测试流水线")
                continue
        except ConnectionError as e:
            logger.error(f"✗ 远程下载失败: {e}")
            logger.error(f"  请检查网络连接或使用 --demo 模式")
            continue
        except Exception as e:
            logger.error(f"✗ 加载数据集 '{ds_name}' 失败: {e}")
            logger.error(f"  提示: 使用 --demo 参数生成模拟数据进行流水线测试")
            continue

        # ── 步骤 6: 写入标准化格式 (Arrow IPC) ─────────────────────────────
        # ────────────────────────────────────────────────────────────────────
        logger.info(f"正在写入标准化 Arrow 文件到: {output_dir}")
        writer = TrajectoryWriter(output_dir)
        num_trajs = writer.write_samples(samples, ds_name)
        logger.info(f"✓ 完成: {ds_name} → {num_trajs} 条轨迹 → {output_dir}")

    logger.info("=" * 60)
    logger.info("所有数据集准备完成!")


if __name__ == "__main__":
    main()
