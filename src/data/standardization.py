"""
================================================================================
 数据标准化模块 (standardization.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 定义统一的中间数据格式和转换管线，将不同源的数据统一为标准格式。

 核心概念:
   统一中间格式 (UnifiedTrajectory)
     基于 Apache Arrow 列式内存格式，支持零拷贝读取和高性能序列化。
     所有数据集适配器的输出都会转换为 UnifiedTrajectory 格式。

 为什么要使用 Arrow?
   1. 列式格式: 按列存储，适合批量处理 (DataLoader 按列读取效率高)
   2. 零拷贝: 不需要序列化/反序列化，跨语言共享数据
   3. 内存映射: 大文件不需要完全加载到内存
   4. 与 PyTorch 无缝集成: 支持 ZeroCopy 转换为 Tensor
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import UnifiedSample, Trajectory


class TrajectoryWriter:
    """
    轨迹写入器 (TrajectoryWriter)
    ────────────────────────────────────────────────────────────────────────────
    职责: 将 UnifiedSample 流聚合为完整轨迹，并写入磁盘 (Arrow/IPC 格式)。

    工作原理:
      1. 接收逐帧的 UnifiedSample (来自 DatasetAdapter)
      2. 按 episode 分界 (done=True) 或固定长度切分为轨迹
      3. 将轨迹序列化为 Arrow IPC 格式存储

    为什么使用 IPC (Inter-Process Communication) 格式?
      - IPC 是 Apache Arrow 的二进制序列化格式
      - 写入速度快，可随机读取
      - 支持内存映射，无需完全加载到 RAM
    """
    def __init__(self, output_dir: str, chunk_size: int = 100):
        """
        Args:
            output_dir: 输出目录
            chunk_size: 每个 Arrow 文件包含的最大轨迹数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

    def write_samples(self, samples: Iterator[UnifiedSample],
                      dataset_name: str) -> int:
        """
        将 UnifiedSample 流写入 Arrow 文件

        Args:
            samples: UnifiedSample 迭代器
            dataset_name: 数据集名称 (用于文件名)

        Returns:
            写入的轨迹总数

        工作流程:
          1. 从迭代器逐个读取样本
          2. 遇到 done=True 时结束当前轨迹并写入
          3. 攒够 chunk_size 个轨迹后写入一个新文件
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        current_trajectory = []
        trajectory_count = 0
        file_count = 0

        for sample in samples:
            # ── 将样本转为可序列化的字典 ────────────────────────────────────
            record = {
                "rgb": sample.obs.get("rgb", torch.zeros(3, 128, 128)).numpy().tobytes(),
                "robot_state": sample.obs.get("robot_state", torch.zeros(7)).numpy().tobytes(),
                "action": sample.action.numpy().tobytes(),
                "reward": sample.reward,
                "done": sample.done,
            }
            current_trajectory.append(record)

            # ── 遇到终止标志或达到最大长度时 ────────────────────────────────
            if sample.done or len(current_trajectory) >= 1000:
                # 写入单个轨迹
                traj_table = pa.Table.from_pydict({
                    k: [r[k] for r in current_trajectory]
                    for k in current_trajectory[0].keys()
                })

                file_path = self.output_dir / f"{dataset_name}_traj_{trajectory_count}.arrow"
                with pa.OSFile(str(file_path), "wb") as f:
                    writer = ipc.new_file(f, traj_table.schema)
                    writer.write_table(traj_table)
                    writer.close()

                trajectory_count += 1
                current_trajectory = []

                # ── 达到 chunk 大小后开始新文件 ─────────────────────────────
                if trajectory_count % self.chunk_size == 0:
                    file_count += 1
                    print(f"[标准化] 已写入 {trajectory_count} 条轨迹 "
                          f"(文件 {file_count})")

        return trajectory_count


class TrajectoryReader:
    """
    轨迹读取器 (TrajectoryReader)
    ────────────────────────────────────────────────────────────────────────────
    职责: 从 Arrow IPC 文件读取标准化轨迹数据。

    支持:
      - 按索引随机读取单条轨迹
      - 按批次读取多条轨迹
      - 内存映射模式 (大文件不全部加载)
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._file_list = sorted(self.data_dir.glob("*.arrow"))

    def __len__(self) -> int:
        """返回轨迹总数"""
        return len(self._file_list)

    def read_trajectory(self, index: int) -> Trajectory:
        """
        读取指定索引的轨迹
        ────────────────────────────────────────────────────────────────────────
        原理:
          使用 Arrow IPC 的零拷贝读取机制——文件被内存映射到地址空间，
          读取时无需复制数据，直接引用文件内容。

        Args:
            index: 轨迹索引

        Returns:
            Trajectory 对象
        """
        import pyarrow.ipc as ipc

        file_path = self._file_list[index]
        with pa.OSFile(str(file_path), "rb") as f:
            reader = ipc.open_file(f)
            table = reader.read_all()

        # 从 Arrow table 重建 Trajectory
        actions = []
        rewards = []
        dones = []
        rgbs = []
        states = []

        for i in range(len(table)):
            actions.append(
                torch.frombuffer(table["action"][i].as_buffer(), dtype=torch.float32)
            )
            rewards.append(table["reward"][i].as_py())
            dones.append(table["done"][i].as_py())
            rgbs.append(
                torch.frombuffer(table["rgb"][i].as_buffer(), dtype=torch.float32)
                .reshape(3, 128, 128)
            )
            states.append(
                torch.frombuffer(table["robot_state"][i].as_buffer(), dtype=torch.float32)
            )

        return Trajectory(
            observations={
                "rgb": torch.stack(rgbs),
                "robot_state": torch.stack(states),
            },
            actions=torch.stack(actions),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            dones=torch.tensor(dones, dtype=torch.bool),
        )
