#!/usr/bin/env python3
"""
================================================================================
 数据集查看脚本 (inspect_data.py)
 ──────────────────────────────────────────────────────────────────────────────
 查看本地数据集的内容、格式、类型、统计信息等，为后续训练和调试提供参考。

 ═══════════════════════════════════════════════════════════════════════════════
 功能概览
 ═══════════════════════════════════════════════════════════════════════════════
   1. --list / -l             列出所有已注册数据集及其本地状态
   2. --dataset / -d NAME     指定要查看的数据集
   3. --info                  查看数据集注册元信息
   4. --tree / -t             查看数据集目录结构
   5. --schema                查看 Arrow 数据文件的列模式 (Schema)
   6. --stats                 查看数据统计信息 (轨迹数、样本数、大小等)
   7. --samples / -n N        查看前 N 条样本内容
   8. --summary / -s          综合摘要 (默认选项，包含 info + tree + stats)

 用法:
   # 列出所有可用数据集
   python scripts/inspect_data.py --list

   # 查看 bridge_v2 数据集综合信息
   python scripts/inspect_data.py --dataset=bridge_v2

   # 查看特定数据集的目录结构
   python scripts/inspect_data.py --dataset=roboturk --tree

   # 查看数据集的 Schema
   python scripts/inspect_data.py --dataset=bridge_v2 --schema

   # 查看数据集统计信息
   python scripts/inspect_data.py --dataset=roboturk --stats

   # 查看前 5 条样本内容
   python scripts/inspect_data.py --dataset=bridge_v2 --samples=5

   # 查看所有数据集的综合摘要
   python scripts/inspect_data.py --all-datasets

   # 指定数据路径查看
   python scripts/inspect_data.py --dataset=bridge_v2 --local_path=/mnt/nvme/data/bridge_v2

 ═══════════════════════════════════════════════════════════════════════════════
 支持的查看格式
 ═══════════════════════════════════════════════════════════════════════════════
   - Arrow IPC 文件 (.arrow):    标准化数据格式，查看 schema、内容、统计
   - TFRecord 文件 (.tfrecord):  原始 RLDS 格式，查看文件大小、轨迹分布
   - HDF5 文件 (.h5/.hdf5):      原始格式，查看数据集和属性
   - 目录结构:                    查看文件分布和占用空间
   - 注册表元信息:                从 registry.yaml 读取数据集配置
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

# ─── 添加项目根目录到系统路径 ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.utils import setup_logging
from src.data.registry import DatasetRegistry
from src.data.standardization import TrajectoryReader


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _format_bytes(size_bytes: int) -> str:
    """将字节数转换为人类可读的文件大小字符串"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"


def _format_timestamp(ts: float) -> str:
    """将时间戳转换为可读日期字符串"""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _pretty_dict(d: Dict[str, Any], indent: int = 4) -> str:
    """将字典格式化为漂亮的缩进字符串"""
    return json.dumps(d, indent=indent, ensure_ascii=False, default=str)


def _print_header(title: str, width: int = 72) -> None:
    """打印带边框的分节标题"""
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {title: <{width - 4}}║")
    print("╚" + "═" * (width - 2) + "╝")
    print()


def _print_key_value(key: str, value: Any, key_width: int = 24) -> None:
    """打印键值对，带对齐"""
    print(f"  {key:<{key_width}} : {value}")


def _print_separator(char: str = "─", width: int = 70) -> None:
    """打印分隔线"""
    print(f"  {char * width}")


# ═══════════════════════════════════════════════════════════════════════════════
# 核心查看函数
# ═══════════════════════════════════════════════════════════════════════════════

def list_datasets(registry: DatasetRegistry) -> None:
    """
    列出所有已注册的数据集及其状态
    ────────────────────────────────────────────────────────────────────────────
    显示:
      - 数据集名称和来源
      - 标准化数据路径 (local_path) 是否存在
      - 原始数据路径 (raw_path) 是否存在
      - 文件数量和总大小
      - 轨迹数量和机器人类型
    """
    datasets = registry.list_all()

    _print_header(f"📋 已注册数据集一览 (共 {len(datasets)} 个)")

    print(f"  {'名称':<20s} {'来源':<22s} {'本地状态':<10s} {'轨迹数':>8s} {'数据大小':>10s}")
    print(f"  {'─' * 20} {'─' * 22} {'─' * 10} {'─' * 8} {'─' * 10}")

    for meta in datasets:
        # 检查本地路径状态
        local_path = meta.local_path or f"data/unified/{meta.name}"
        local_abs = Path(project_root) / local_path if not os.path.isabs(local_path) else Path(local_path)

        if local_abs.exists():
            # 统计文件和大小
            arrow_files = list(local_abs.rglob("*.arrow"))
            tfrecord_files = list(local_abs.rglob("*.tfrecord*"))
            h5_files = list(local_abs.rglob("*.h5")) + list(local_abs.rglob("*.hdf5"))
            all_files = list(local_abs.rglob("*")) if local_abs.is_dir() else [local_abs]

            if arrow_files:
                status = "✅ Arrow"
                total_size = sum(f.stat().st_size for f in arrow_files)
            elif tfrecord_files:
                status = "📦 TFRecord"
                total_size = sum(f.stat().st_size for f in tfrecord_files)
            elif h5_files:
                status = "💾 HDF5"
                total_size = sum(f.stat().st_size for f in h5_files)
            else:
                status = "📁 目录"
                total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        else:
            status = "❌ 不存在"
            total_size = 0

        print(f"  {meta.name:<20s} {meta.source.name:<22s} {status:<10s} {meta.num_trajs:>8d} {_format_bytes(total_size):>10s}")

    print()
    print(f"  💡 提示: 使用 --dataset=<名称> 查看数据集详情")


def _inspect_directory(path: Path, max_depth: int = 3) -> None:
    """递归查看目录结构（树形）"""
    if not path.exists():
        print(f"  ❌ 路径不存在: {path}")
        return

    def _walk(dir_path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            print(f"{prefix}  └── ...")
            return

        entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└──" if is_last else "├──"

            if entry.is_dir():
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                print(f"{prefix}{connector} 📁 {entry.name}/  ({_format_bytes(size)})")
                _walk(entry, prefix + ("    " if is_last else "│   "), depth + 1)
            else:
                stat = entry.stat()
                print(f"{prefix}{connector} 📄 {entry.name}  ({_format_bytes(stat.st_size)})")

    _walk(path)


def show_info(registry: DatasetRegistry, dataset_name: str) -> None:
    """显示数据集注册元信息"""
    meta = registry.get(dataset_name)
    if meta is None:
        print(f"  ❌ 数据集 '{dataset_name}' 未在注册表中找到。使用 --list 查看可用数据集。")
        return

    _print_header(f"📋 数据集元信息: {meta.name}")

    _print_key_value("名称", meta.name)
    _print_key_value("来源", f"{meta.source.name} ({meta.source.value})")
    _print_key_value("描述", meta.description)
    _print_key_value("标准化路径", meta.local_path or "(未配置)")
    _print_key_value("原始数据路径", meta.raw_path or "(未配置)")
    _print_key_value("下载地址", meta.url or "(未配置)")
    _print_key_value("原始格式", meta.format)
    _print_key_value("机器人类型", meta.robot_type.name)
    _print_key_value("传感器模态", ", ".join(m.name for m in meta.modalities))
    _print_key_value("轨迹数", f"{meta.num_trajs:,}")
    _print_key_value("版本", meta.version)
    _print_key_value("适配器", meta.adapter_name)
    _print_key_value("许可证", meta.license)
    if meta.citation:
        _print_key_value("引用", meta.citation[:60] + ("..." if len(meta.citation) > 60 else ""))


def show_tree(registry: DatasetRegistry, dataset_name: str, local_path: Optional[str] = None) -> None:
    """显示数据集目录树"""
    meta = registry.get(dataset_name)

    # 确定路径
    if local_path:
        target_path = Path(local_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path
    elif meta and meta.local_path:
        target_path = Path(meta.local_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path
    else:
        target_path = project_root / "data" / "unified" / dataset_name

    _print_header(f"📁 目录结构: {dataset_name}")
    print(f"  路径: {target_path.resolve()}")
    print()

    if not target_path.exists():
        print(f"  ❌ 路径不存在: {target_path.resolve()}")
        print(f"  💡 请先运行 scripts/prepare_data.py 准备数据，或使用 --local_path 指定其他路径。")
        return

    _inspect_directory(target_path)


def show_schema(dataset_name: str, local_path: Optional[str] = None) -> None:
    """显示 Arrow 数据文件的 Schema（列名、类型、样本形状）"""
    # 确定路径
    if local_path:
        data_dir = Path(local_path)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
    else:
        data_dir = project_root / "data" / "unified" / dataset_name

    _print_header(f"📊 数据 Schema: {dataset_name}")
    print(f"  数据目录: {data_dir.resolve()}")
    print()

    if not data_dir.exists():
        # 尝试 TFRecord 等其他格式
        raw_dir = project_root / "data" / "unified" / dataset_name
        tfrecord_files = list(raw_dir.rglob("*.tfrecord*"))
        if tfrecord_files:
            print(f"  数据格式: TFRecord (原始 RLDS 格式)")
            print(f"  文件数: {len(tfrecord_files)}")
            sample_file = tfrecord_files[0]
            print(f"  示例文件: {sample_file.name} ({_format_bytes(sample_file.stat().st_size)})")
            print()
            print(f"  ⚠️  TFRecord 格式的 Schema 需要加载样本后才能显示。")
            print(f"     请使用 --samples 参数查看具体内容。")
            return
        else:
            print(f"  ❌ 数据目录不存在，无法读取 Schema。")
            print(f"  💡 请先运行 scripts/prepare_data.py 准备数据。")
            return

    try:
        reader = TrajectoryReader(str(data_dir))
        if len(reader) == 0:
            print(f"  ⚠️  目录中没有 .arrow 文件。")
            return

        print(f"  📄 轨迹文件数: {len(reader)}")
        print()

        # 读取第一个轨迹来推断 Schema
        traj = reader.read_trajectory(0)

        _print_key_value("trajectory.length", traj.length)

        # 观测 Schema
        print()
        print(f"  ┌─ observations (观测):")
        for obs_name, obs_tensor in traj.observations.items():
            dtype_str = str(obs_tensor.dtype)
            shape_str = str(list(obs_tensor.shape))
            # 估算每个观测的大小
            elem_size = obs_tensor.element_size()
            total_bytes = obs_tensor.numel() * elem_size
            print(f"  │   ├── {obs_name:<20s}  dtype={dtype_str:<12s}  shape={shape_str:<24s}  ≈ {_format_bytes(total_bytes)}/帧")
        print(f"  │   └── 合计观测: {sum(v.numel() * v.element_size() for v in traj.observations.values())} 字节/帧")

        # 动作 Schema
        print()
        print(f"  ├─ actions (动作):")
        print(f"  │    dtype={str(traj.actions.dtype):<12s}  shape={list(traj.actions.shape)}  "
              f"范围≈[{traj.actions.min().item():.3f}, {traj.actions.max().item():.3f}]")

        # 奖励 Schema
        print(f"  ├─ rewards (奖励):")
        print(f"  │    dtype={str(traj.rewards.dtype):<12s}  shape={list(traj.rewards.shape)}  "
              f"范围≈[{traj.rewards.min().item():.3f}, {traj.rewards.max().item():.3f}]")

        # done Schema
        print(f"  └─ dones (终止标志):")
        print(f"       dtype={str(traj.dones.dtype):<12s}  shape={list(traj.dones.shape)}  "
              f"True计数={(traj.dones.sum()).item()}")

        # 语言嵌入 (如果有)
        if traj.language_embeds is not None:
            print()
            print(f"  ┌─ language_embeds (语言嵌入):")
            print(f"  │    dtype={str(traj.language_embeds.dtype):<12s}  shape={list(traj.language_embeds.shape)}")

        print()
        print(f"  💡 提示: 使用 --samples=N 查看具体样本数据")

    except ImportError:
        print("  ❌ 缺少 pyarrow 库。请安装: pip install pyarrow")
    except Exception as e:
        print(f"  ❌ 读取 Schema 失败: {e}")


def show_stats(dataset_name: str, local_path: Optional[str] = None) -> None:
    """显示数据集统计信息"""
    # 确定路径
    if local_path:
        data_dir = Path(local_path)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
    else:
        data_dir = project_root / "data" / "unified" / dataset_name

    _print_header(f"📈 数据统计: {dataset_name}")
    print(f"  数据目录: {data_dir.resolve()}")
    print()

    if not data_dir.exists():
        print(f"  ❌ 数据目录不存在。")
        return

    # 统计文件信息
    arrow_files = sorted(data_dir.glob("*.arrow"))
    tfrecord_files = sorted(data_dir.glob("*.tfrecord*"))
    h5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("*.hdf5"))
    all_files = sorted(data_dir.rglob("*"))

    total_files = 0
    total_size = 0
    file_types = defaultdict(lambda: {"count": 0, "size": 0})

    for f in all_files:
        if f.is_file():
            total_files += 1
            size = f.stat().st_size
            total_size += size
            ext = f.suffix if f.suffix else f.name.split(".")[-1] if "." in f.name else "unknown"
            file_types[ext]["count"] += 1
            file_types[ext]["size"] += size

    print(f"  总文件数: {total_files}")
    print(f"  总大小:   {_format_bytes(total_size)}")
    print()

    # 按扩展名分类
    _print_key_value("文件类型分布", "")
    for ext, info in sorted(file_types.items(), key=lambda x: -x[1]["size"]):
        pct = info["size"] / total_size * 100 if total_size > 0 else 0
        print(f"    ├── .{ext:<12s} {info['count']:>5d} 个文件  {_format_bytes(info['size']):>10s}  ({pct:.1f}%)")
    print()

    # Arrow 文件统计
    if arrow_files:
        try:
            reader = TrajectoryReader(str(data_dir))
            num_trajs = len(reader)
            print(f"  ┌─ Arrow 文件统计:")
            print(f"  │   轨迹文件数: {num_trajs}")
            print(f"  │   文件大小区间: {_format_bytes(min(f.stat().st_size for f in arrow_files))} ~ "
                  f"{_format_bytes(max(f.stat().st_size for f in arrow_files))}")

            # 采样一些轨迹获取长度分布
            traj_lengths = []
            sample_indices = min(num_trajs, 50)  # 最多采样 50 条
            step_indices = list(range(0, num_trajs, max(1, num_trajs // sample_indices)))[:sample_indices]

            for idx in step_indices:
                traj = reader.read_trajectory(idx)
                traj_lengths.append(traj.length)

            if traj_lengths:
                lengths_tensor = __import__("torch").tensor(traj_lengths, dtype=torch.float32)
                print(f"  │   采样轨迹数: {len(traj_lengths)} (共 {num_trajs} 条)")
                print(f"  │   轨迹长度范围: {min(traj_lengths)} ~ {max(traj_lengths)} 帧")
                print(f"  │   平均轨迹长度: {sum(traj_lengths) / len(traj_lengths):.1f} ± {lengths_tensor.std().item():.1f} 帧")
                print(f"  │   总样本帧数: {sum(traj_lengths)} (采样估算)")
                print(f"  └──")

        except ImportError:
            print(f"  ⚠️  缺少 pyarrow 库，无法读取 Arrow 统计。")
        except Exception as e:
            print(f"  ⚠️  读取 Arrow 统计失败: {e}")

    # TFRecord 文件统计
    if tfrecord_files:
        print(f"  ┌─ TFRecord 文件统计:")
        sizes = [f.stat().st_size for f in tfrecord_files]
        print(f"  │   文件数: {len(tfrecord_files)}")
        print(f"  │   总大小: {_format_bytes(sum(sizes))}")
        print(f"  │   文件大小区间: {_format_bytes(min(sizes))} ~ {_format_bytes(max(sizes))}")
        print(f"  │   平均大小: {_format_bytes(sum(sizes) / len(sizes))}")
        print(f"  └──")

    # HDF5 文件统计
    if h5_files:
        print(f"  ┌─ HDF5 文件统计:")
        sizes = [f.stat().st_size for f in h5_files]
        print(f"  │   文件数: {len(h5_files)}")
        print(f"  │   总大小: {_format_bytes(sum(sizes))}")
        print(f"  │   文件大小区间: {_format_bytes(min(sizes))} ~ {_format_bytes(max(sizes))}")
        print(f"  └──")


def _show_file_samples(data_dir: Path, dataset_name: str, num_samples: int) -> bool:
    """尝试以非 Arrow 格式显示文件样本（TFRecord / HDF5）
    Returns:
        True 如果找到了可显示的文件
    """
    tfrecord_files = sorted(data_dir.glob("*.tfrecord*"))
    h5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("*.hdf5"))

    if tfrecord_files:
        print(f"  数据格式: TFRecord (原始 RLDS 格式)")
        print(f"  文件数: {len(tfrecord_files)}")
        print()
        for i, f in enumerate(tfrecord_files[:num_samples]):
            stat = f.stat()
            print(f"  ─── 文件 [{i + 1}/{min(len(tfrecord_files), num_samples)}] ───")
            print(f"    文件名: {f.name}")
            print(f"    大小:   {_format_bytes(stat.st_size)}")
            print(f"    修改时间: {_format_timestamp(stat.st_mtime)}")
        if len(tfrecord_files) > num_samples:
            print(f"    ... 还有 {len(tfrecord_files) - num_samples} 个文件未显示")
        print()
        print(f"  💡 提示: TFRecord 是 RLDS 格式的序列化文件，")
        print(f"     需使用 tensorflow_datasets 或 datasets 库解析具体内容。")
        return True

    if h5_files:
        print(f"  数据格式: HDF5")
        print(f"  文件数: {len(h5_files)}")
        print()
        for i, f in enumerate(h5_files[:num_samples]):
            stat = f.stat()
            print(f"  ─── 文件 [{i + 1}/{min(len(h5_files), num_samples)}] ───")
            print(f"    文件名: {f.name}")
            print(f"    大小:   {_format_bytes(stat.st_size)}")
            print(f"    修改时间: {_format_timestamp(stat.st_mtime)}")
        if len(h5_files) > num_samples:
            print(f"    ... 还有 {len(h5_files) - num_samples} 个文件未显示")
        print()
        print(f"  💡 提示: HDF5 文件的详细内容可通过 h5py / hdf5 工具查看。")
        return True

    return False


def show_samples(dataset_name: str, local_path: Optional[str] = None, num_samples: int = 3) -> None:
    """显示数据集中的样本内容"""
    # 确定路径
    if local_path:
        data_dir = Path(local_path)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
    else:
        data_dir = project_root / "data" / "unified" / dataset_name

    _print_header(f"🔍 样本预览: {dataset_name} (前 {num_samples} 条)")

    if not data_dir.exists():
        print(f"  ❌ 数据目录不存在: {data_dir.resolve()}")
        print(f"  💡 请先运行 scripts/prepare_data.py 准备数据，或使用 --local_path 指定其他路径。")
        return

    # ── 优先尝试 Arrow 格式 ────────────────────────────────────────────────
    arrow_files = sorted(data_dir.glob("*.arrow"))
    if not arrow_files:
        # 没有 Arrow 文件，回退到显示其他格式的文件信息
        if _show_file_samples(data_dir, dataset_name, num_samples):
            return
        print(f"  ⚠️  目录中无可识别的数据文件 (.arrow / .tfrecord / .h5)。")
        return

    try:
        reader = TrajectoryReader(str(data_dir))
        if len(reader) == 0:
            print(f"  ⚠️  目录中没有有效的 .arrow 文件。")
            return

        print(f"  轨迹总数: {len(reader)}")
        print(f"  显示前 {min(num_samples, len(reader))} 条轨迹的前 3 帧:")
        print()

        for traj_idx in range(min(num_samples, len(reader))):
            traj = reader.read_trajectory(traj_idx)
            print(f"  ─── 轨迹 [{traj_idx + 1}/{len(reader)}] ──────────────────────")
            print(f"    长度: {traj.length} 帧")
            if traj.task_id:
                print(f"    任务: {traj.task_id}")
            if traj.success is not None:
                print(f"    成功: {traj.success}")

            # 显示前 3 帧
            max_frames = min(3, traj.length)
            for t in range(max_frames):
                print(f"    ┌─ 帧 [{t}] ───────────────────────────")
                for obs_name, obs_tensor in traj.observations.items():
                    val = obs_tensor[t]
                    shape_str = str(list(val.shape))
                    if val.dtype in (torch.float32, torch.float64) and val.numel() <= 16:
                        # 小向量显示数值
                        print(f"    │  {obs_name:<16s}  shape={shape_str:<20s}  value={val.tolist()}")
                    else:
                        # 大张量显示统计
                        print(f"    │  {obs_name:<16s}  shape={shape_str:<20s}  "
                              f"mean={val.mean().item():.4f}  std={val.std().item():.4f}  "
                              f"min={val.min().item():.4f}  max={val.max().item():.4f}")

                # 动作
                action = traj.actions[t]
                print(f"    │  {'action':<16s}  shape={str(list(action.shape)):<20s}  "
                      f"value={action.tolist() if action.numel() <= 16 else f'mean={action.mean():.4f} std={action.std():.4f}'}")

                # 奖励
                print(f"    │  {'reward':<16s}  {traj.rewards[t].item():.4f}")

                # 终止标志
                print(f"    └─ {'done':<16s}  {bool(traj.dones[t].item())}")

            if traj.length > 3:
                print(f"      ... 还有 {traj.length - 3} 帧未显示")
            print()

    except ImportError:
        print("  ❌ 缺少 pyarrow 库。请安装: pip install pyarrow")
    except Exception as e:
        print(f"  ❌ 读取样本失败: {e}")


def show_summary(registry: DatasetRegistry, dataset_name: str, local_path: Optional[str] = None) -> None:
    """显示数据集的综合摘要信息"""
    meta = registry.get(dataset_name)

    _print_header(f"📊 数据集综合摘要: {dataset_name}")

    # ── 1. 元信息 ──────────────────────────────────────────────────────────
    if meta:
        print("  ▸ 注册信息")
        _print_key_value("名称", meta.name, key_width=18)
        _print_key_value("来源", meta.source.name, key_width=18)
        _print_key_value("描述", meta.description[:60] + ("..." if len(meta.description) > 60 else ""), key_width=18)
        _print_key_value("机器人", meta.robot_type.name, key_width=18)
        _print_key_value("传感器", ", ".join(m.name for m in meta.modalities), key_width=18)
        _print_key_value("格式", meta.format, key_width=18)
        _print_key_value("适配器", meta.adapter_name, key_width=18)
        _print_key_value("注册轨迹数", f"{meta.num_trajs:,}", key_width=18)
        print()
    else:
        print(f"  ⚠️  数据集 '{dataset_name}' 未在注册表中找到。")
        print()

    # ── 2. 本地路径状态 ────────────────────────────────────────────────────
    if local_path:
        target_path = Path(local_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path
    elif meta and meta.local_path:
        target_path = Path(meta.local_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path
    else:
        target_path = project_root / "data" / "unified" / dataset_name

    print("  ▸ 本地存储状态")
    _print_key_value("期望路径", str(target_path), key_width=18)

    if target_path.exists():
        # 统计
        all_files = [f for f in target_path.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in all_files)
        arrow_files = list(target_path.rglob("*.arrow"))
        tfrecord_files = list(target_path.rglob("*.tfrecord*"))

        _print_key_value("状态", "✅ 存在", key_width=18)
        _print_key_value("总文件数", len(all_files), key_width=18)
        _print_key_value("总大小", _format_bytes(total_size), key_width=18)
        _print_key_value("Arrow 文件", f"{len(arrow_files)} 个" if arrow_files else "无", key_width=18)
        _print_key_value("TFRecord 文件", f"{len(tfrecord_files)} 个" if tfrecord_files else "无", key_width=18)
        if all_files:
            _print_key_value("最后修改", _format_timestamp(max(f.stat().st_mtime for f in all_files)), key_width=18)
    else:
        _print_key_value("状态", "❌ 不存在", key_width=18)
        _print_key_value("建议", f"运行: python scripts/prepare_data.py --datasets={dataset_name}", key_width=18)
    print()

    # ── 3. 轨迹统计 (如果有 Arrow 文件) ─────────────────────────────────────
    arrow_files = sorted(target_path.glob("*.arrow")) if target_path.exists() else []
    if arrow_files:
        print("  ▸ 轨迹统计信息")
        try:
            reader = TrajectoryReader(str(target_path))
            num_trajs = len(reader)
            _print_key_value("轨迹数", num_trajs, key_width=18)

            # 采样轨迹长度
            sample_size = min(num_trajs, 100)
            step = max(1, num_trajs // sample_size)
            lengths = []

            import torch
            for idx in range(0, num_trajs, step):
                traj = reader.read_trajectory(idx)
                lengths.append(traj.length)

            if lengths:
                _print_key_value("平均长度", f"{sum(lengths) / len(lengths):.1f} 帧", key_width=18)
                _print_key_value("长度范围", f"{min(lengths)} ~ {max(lengths)} 帧", key_width=18)

                # 观测和动作维度
                sample_traj = reader.read_trajectory(0)
                obs_dims = {k: list(v.shape[1:]) for k, v in sample_traj.observations.items()}
                _print_key_value("观测维度", str(obs_dims), key_width=18)
                _print_key_value("动作维度", list(sample_traj.actions.shape[1:]), key_width=18)

        except Exception as e:
            _print_key_value("读取失败", str(e), key_width=18)
    print()

    print(f"  💡 查看更多详情:")
    print(f"     --tree       查看目录结构")
    print(f"     --schema     查看数据 Schema")
    print(f"     --stats      查看详细统计")
    print(f"     --samples=N  查看 N 条样本内容")


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="🧐 具身智能数据集查看工具 — 查看本地数据集的格式、类型、内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用数据集
  python scripts/inspect_data.py --list

  # 查看 bridge_v2 综合信息
  python scripts/inspect_data.py --dataset=bridge_v2

  # 查看 roboturk 的目录结构
  python scripts/inspect_data.py --dataset=roboturk --tree

  # 查看数据 Schema
  python scripts/inspect_data.py --dataset=bridge_v2 --schema

  # 查看样本内容
  python scripts/inspect_data.py --dataset=roboturk --samples=5

  # 查看所有数据集
  python scripts/inspect_data.py --all-datasets
        """
    )

    # ── 主要模式 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="列出所有已注册的数据集及其本地状态"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="要查看的数据集名称 (如 bridge_v2, roboturk)"
    )
    parser.add_argument(
        "--all-datasets", "-a", action="store_true",
        help="查看所有已注册数据集的综合摘要"
    )

    # ── 查看模式 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--info", action="store_true",
        help="查看数据集注册元信息"
    )
    parser.add_argument(
        "--tree", "-t", action="store_true",
        help="查看数据集目录结构树"
    )
    parser.add_argument(
        "--schema", action="store_true",
        help="查看 Arrow 数据文件的 Schema (列名、类型、形状)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="查看数据统计信息 (文件数、大小、轨迹长度分布等)"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=0, const=3, nargs="?",
        help="查看前 N 条样本内容 (默认 3 条)"
    )
    parser.add_argument(
        "--summary", "-s", action="store_true",
        help="查看数据集综合摘要 (默认项)"
    )

    # ── 路径覆盖 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="覆盖数据路径 (用于查看非默认位置的数据)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging("inspect_data")

    # ── 加载数据集注册表 ───────────────────────────────────────────────────
    registry = DatasetRegistry()

    # ── 模式 1: 列出所有数据集 ─────────────────────────────────────────────
    if args.list:
        list_datasets(registry)
        return

    # ── 模式 2: 查看所有数据集 ─────────────────────────────────────────────
    if args.all_datasets:
        datasets = registry.list_all()
        if not datasets:
            print("  ⚠️  注册表中没有数据集。")
            return
        for meta in datasets:
            show_summary(registry, meta.name, args.local_path)
        return

    # ── 模式 3: 查看特定数据集 ─────────────────────────────────────────────
    dataset_name = args.dataset
    if dataset_name is None:
        print("  ❌ 请指定数据集名称 (--dataset=NAME) 或使用 --list 查看可用数据集。")
        print()
        print("  快速入门:")
        print("    # 列出所有数据集")
        print("    python scripts/inspect_data.py --list")
        print()
        print("    # 查看 bridge_v2 数据集")
        print("    python scripts/inspect_data.py --dataset=bridge_v2")
        return

    # 检查数据集是否存在 (即使未注册也允许查看本地文件)
    meta = registry.get(dataset_name)
    if meta is None:
        print(f"  ⚠️  数据集 '{dataset_name}' 未在注册表中找到，但仍可尝试查看本地路径。")
        print()

    # ── 确定查看模式 ───────────────────────────────────────────────────────
    # 如果指定了具体的查看选项，按选项执行；否则执行综合摘要
    has_specific_mode = any([
        args.info, args.tree, args.schema,
        args.stats, args.samples > 0, args.summary
    ])

    if not has_specific_mode:
        # 默认模式：综合摘要
        show_summary(registry, dataset_name, args.local_path)
    else:
        if args.info:
            show_info(registry, dataset_name)
        if args.tree:
            show_tree(registry, dataset_name, args.local_path)
        if args.schema:
            show_schema(dataset_name, args.local_path)
        if args.stats:
            show_stats(dataset_name, args.local_path)
        if args.samples > 0:
            show_samples(dataset_name, args.local_path, args.samples)
        if args.summary:
            show_summary(registry, dataset_name, args.local_path)


if __name__ == "__main__":
    main()
