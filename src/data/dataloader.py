"""
================================================================================
 数据加载器模块 (dataloader.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供高效的数据供给接口，连接数据层到算法层训练循环。

 ═══════════════════════════════════════════════════════════════════════════════
 本地数据加载策略
 ═══════════════════════════════════════════════════════════════════════════════
 
 本模块全部从**本地磁盘**加载数据，路径由配置文件统一管理。
 绝不直接访问远程 URL —— 所有远程数据需提前通过 prepare_data.py 下载到本地。

 本地路径配置方式 (优先级从高到低):
   1. Hydra 实验配置: experiment.yaml → data.data_dir
   2. 数据集注册表:   config/dataset/registry.yaml → datasets[].local_path
   3. 内置默认值:     data/unified/{dataset_name}

 典型配置示例 (config/experiment/bc_manipulation.yaml):
 ```yaml
 data:
   data_dir: "data/unified/bridge_v2"    # ← 本地路径，指向 Arrow 文件目录
   batch_size: 256                        #    该目录下需要有 .arrow 文件
   num_workers: 4                         #    由 prepare_data.py 生成
 ```

 目录结构约定:
   data/unified/
   ├── bridge_v2/          # BridgeData V2 标准化数据
   │   ├── bridge_v2_traj_0000.arrow     # 轨迹 0
   │   ├── bridge_v2_traj_0001.arrow     # 轨迹 1
   │   └── ...                           # 更多轨迹文件
   ├── roboturk/           # RoboTurk 标准化数据
   │   └── ...
   └── maniskill/          # ManiSkill 标准化数据
       └── ...

 性能设计:
   - 使用多个工作进程预取数据，掩盖 I/O 延迟
   - 支持 SharedMemory 共享缓存，减少进程间数据复制
   - 基于 Arrow IPC 的零拷贝内存映射读取，避免序列化开销
   - NVMe SSD 推荐: 4 个工作进程即可打满 PCIe 带宽
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
    统一数据集类 (UnifiedDataset) — 本地数据加载器
    ────────────────────────────────────────────────────────────────────────────
    继承 PyTorch IterableDataset，从**本地磁盘**的 Arrow IPC 文件流式加载数据。

    ═══════════════════════════════════════════════════════════════════════════
    为什么用 IterableDataset 而不是 MapDataset?
    ═══════════════════════════════════════════════════════════════════════════
      - 具身数据集通常非常大 (TB 级)，无法全部加载到内存
      - IterableDataset 支持流式读取，每次只加载一个 batch
      - 支持无限数据流 (用于在线 RL 和仿真数据)
      - Arrow IPC 支持内存映射 (mmap)，文件无需完全读入 RAM

    ═══════════════════════════════════════════════════════════════════════════
    数据路径说明
    ═══════════════════════════════════════════════════════════════════════════
      data_dir 参数由上层传入，来源为:
        - 实验配置文件: config.data.data_dir
        - 或由训练脚本直接从 Hydra 配置读取
        
      data_dir 指向**本地**目录，该目录下应包含 .arrow 文件。
      这些 .arrow 文件由 prepare_data.py 预处理生成。

    ═══════════════════════════════════════════════════════════════════════════
    工作流程
    ═══════════════════════════════════════════════════════════════════════════
      1. 扫描 data_dir 目录下的所有 .arrow 文件 → 建立文件索引列表
      2. TrajectoryReader 通过内存映射 (mmap) 打开每个文件
      3. 迭代时:
         a. 随机选择一个轨迹文件 (或按顺序)
         b. 从 Arrow 文件读取完整轨迹
         c. (可选) 对轨迹进行数据增强
         d. 将轨迹拆分为逐帧 UnifiedSample
         e. 产出单个 sample
      4. DataLoader 的多个工作进程并行执行上述流程
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
            data_dir: 【本地路径】标准化数据目录 (包含 .arrow 文件)
                      - 相对路径: 相对于项目根目录，如 "data/unified/bridge_v2"
                      - 绝对路径: 如 "/mnt/nvme/data/bridge_v2"
                      路径来源: Hydra 配置 → data.data_dir
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
    创建配置好的 DataLoader — 本地数据加载入口
    ────────────────────────────────────────────────────────────────────────────
    这是算法层训练循环调用数据层的**唯一入口**。
    封装了 UnifiedDataset 和 PyTorch DataLoader 的完整配置过程。

    ═══════════════════════════════════════════════════════════════════════════
    典型调用方式 (在训练脚本中):
    ═══════════════════════════════════════════════════════════════════════════
    
    ```python
    # 方式 1: 从 Hydra 配置读取本地路径 (推荐)
    from omegaconf import DictConfig
    loader = create_dataloader(
        data_dir=cfg.data.data_dir,      # ← 从配置文件读取本地路径
        batch_size=cfg.data.batch_size,   #     "data/unified/bridge_v2"
        num_workers=cfg.data.num_workers,
    )
    
    # 方式 2: 直接指定路径 (快速测试)
    loader = create_dataloader(
        data_dir="data/unified/bridge_v2",  # 硬编码路径，仅用于调试
        batch_size=256,
    )
    
    # 方式 3: 使用环境变量 (多用户共享)
    import os
    loader = create_dataloader(
        data_dir=os.environ.get("DATA_DIR", "data/unified/bridge_v2"),
    )
    ```

    ═══════════════════════════════════════════════════════════════════════════
    参数说明:
    ═══════════════════════════════════════════════════════════════════════════
    Args:
        data_dir:      【本地路径】标准化 Arrow 数据目录
                       优先从实验配置读取，支持相对/绝对路径
        batch_size:    批次大小。VLA 大模型建议 8-16，BC 建议 128-256
        num_workers:   数据加载工作进程数。推荐:
                       - HDD: 2-4 (IO 瓶颈)
                       - SSD: 4-8 (IO 较快)
                       - NVMe: 8-16 (可打满 PCIe)
        shuffle:       是否打乱轨迹顺序。训练=True，评估=False
        augmenter:     增强流水线。传入即启用数据增强
        infinite:      是否无限循环。RL 训练=True，IL 训练=False
        prefetch_factor: 每个工作进程预取的批次数量。默认 2

    Returns:
        配置好的 PyTorch DataLoader

    ═══════════════════════════════════════════════════════════════════════════
    性能调优建议:
    ═══════════════════════════════════════════════════════════════════════════
      - num_workers=N: N 个进程并行加载和预处理数据
      - prefetch_factor=2: 每个工作进程预取 2 个 batch
      - pin_memory=True: 加速 CPU→GPU 的 Tensor 传输
      - 这样可以掩盖数据加载和增强的计算延迟
      - 如果 GPU 利用率不足，尝试增大 num_workers 和 prefetch_factor
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
        pin_memory=True,      # 加速 CPU→GPU 的 Tensor 传输
        drop_last=True,       # 丢弃最后不足 batch 大小的数据
    )

    return loader
