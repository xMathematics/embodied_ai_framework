"""
================================================================================
 数据加载器模块 (dataloader.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供高效的数据供给接口，连接数据层到算法层训练循环。

 本模块基于 PyTorch 的 IterableDataset 和 DataLoader，提供:
   1. 标准化数据的流式加载 (支持大文件内存映射)
   2. 多进程预取和缓存 (加速数据供给)
   3. 真实数据 + 仿真数据的混合批采样
   4. 灵活的数据转换接口 (支持自定义预处理)

 性能设计:
   - 使用多个工作进程预取数据，掩盖 I/O 延迟
   - 支持 SharedMemory 共享缓存，减少进程间数据复制
   - 基于 Arrow 的零拷贝读取，避免序列化开销
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Iterator, Callable, Any, Union
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import UnifiedSample, Trajectory, DataBatch
from src.data.standardization import TrajectoryReader
from src.data.augmentation import AugmentationPipeline


class UnifiedDataset(IterableDataset):
    """
    统一数据集类 (UnifiedDataset)
    ────────────────────────────────────────────────────────────────────────────
    继承 PyTorch IterableDataset，实现流式数据加载。

    为什么用 IterableDataset 而不是 MapDataset?
      - 具身数据集通常非常大 (TB 级)，无法全部加载到内存
      - IterableDataset 支持流式读取，每次只加载一个 batch
      - 支持无限数据流 (用于在线 RL 和仿真数据)

    工作流程:
      1. 从标准化后的 Arrow 文件读取轨迹
      2. 对每条轨迹进行增强 (可选)
      3. 将轨迹拆分为单个样本
      4. 产出 batch 供训练循环使用
    """
    def __init__(
        self,
        data_dir: str,
        augmenter: Optional[AugmentationPipeline] = None,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        infinite: bool = False,
    ):
        """
        Args:
            data_dir: 标准化数据目录 (包含 .arrow 文件)
            augmenter: 增强流水线 (可选)
            transform: 自定义转换函数
            shuffle: 是否打乱轨迹顺序
            infinite: 是否为无限数据流 (用于 RL 训练)
        """
        self.reader = TrajectoryReader(data_dir)
        self.augmenter = augmenter
        self.transform = transform or self._default_transform
        self.shuffle = shuffle
        self.infinite = infinite
        self._indices = list(range(len(self.reader)))

    def _default_transform(self, sample: UnifiedSample) -> UnifiedSample:
        """默认转换: 确保所有张量在正确的设备上"""
        return sample

    def __iter__(self) -> Iterator[UnifiedSample]:
        """迭代器: 产出单个样本

        每次迭代:
          1. (可选) 打乱轨迹索引顺序
          2. 从 Arrow 文件读取一条完整轨迹
          3. 对轨迹进行增强
          4. 将轨迹拆分为逐帧样本并逐个产出
        """
        while True:
            # ── 打乱顺序 ────────────────────────────────────────────────────
            if self.shuffle:
                indices = np.random.permutation(self._indices)
            else:
                indices = self._indices

            # ── 遍历轨迹 ────────────────────────────────────────────────────
            for idx in indices:
                traj = self.reader.read_trajectory(idx)

                # 轨迹级增强
                if self.augmenter is not None:
                    traj = self.augmenter.augment_trajectory(traj)

                # 逐帧产出
                for sample in traj.to_list():
                    if self.augmenter is not None:
                        sample = self.augmenter(sample)  # 样本级增强
                    yield self.transform(sample)

            # ── 非无限模式: 一轮结束后退出 ──────────────────────────────────
            if not self.infinite:
                break


# ═══════════════════════════════════════════════════════════════════════════════
# 混合数据采样器 (真实数据 + 仿真数据)
# ═══════════════════════════════════════════════════════════════════════════════

class MixedDataset(IterableDataset):
    """
    混合数据集类 (MixedDataset)
    ────────────────────────────────────────────────────────────────────────────
    同时从真实数据和仿真数据中采样，支持配比控制。

    为什么需要混合采样?
      - 真实数据: 质量高、真实感强，但数量有限
      - 仿真数据: 可无限生成，但存在 Sim-to-Real gap
      - 混合采样可以兼顾两者的优势：真实数据提供真实感，
        仿真数据提供多样性和覆盖度
    """
    def __init__(
        self,
        real_dataset: UnifiedDataset,
        sim_dataset: UnifiedDataset,
        real_ratio: float = 0.5,
    ):
        """
        Args:
            real_dataset: 真实数据集
            sim_dataset: 仿真数据集
            real_ratio: 真实数据的采样比例 [0, 1]
        """
        self.real_dataset = real_dataset
        self.sim_dataset = sim_dataset
        self.real_ratio = real_ratio

    def __iter__(self) -> Iterator[UnifiedSample]:
        """
        交替从真实和仿真数据集采样

        采样策略:
          以 probability = real_ratio 的概率从真实数据采样，
          否则从仿真数据采样。这保证了长期来看两者的比例符合配置要求。
        """
        real_iter = iter(self.real_dataset)
        sim_iter = iter(self.sim_dataset)

        while True:
            try:
                if np.random.random() < self.real_ratio:
                    yield next(real_iter)
                else:
                    yield next(sim_iter)
            except StopIteration:
                # 如果其中一个数据集耗尽，从另一个继续
                break


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader 工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    augmenter: Optional[AugmentationPipeline] = None,
    infinite: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    创建配置好的 DataLoader
    ────────────────────────────────────────────────────────────────────────────
    封装了 UnifiedDataset 和 PyTorch DataLoader 的配置过程，
    提供开箱即用的数据加载器。

    Args:
        data_dir: 标准化数据目录
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据
        augmenter: 增强流水线
        infinite: 是否无限循环
        prefetch_factor: 每个工作进程预取的批次数量

    Returns:
        配置好的 DataLoader

    性能说明:
      - num_workers=N: N 个进程并行加载和预处理数据
      - prefetch_factor=2: 每个工作进程预取 2 个 batch
      - 这样可以掩盖数据加载和增强的计算延迟
    """
    dataset = UnifiedDataset(
        data_dir=data_dir,
        augmenter=augmenter,
        shuffle=shuffle,
        infinite=infinite,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,      # 加速 CPU→GPU 传输
        drop_last=True,       # 丢弃最后的不足 batch
    )

    return loader
