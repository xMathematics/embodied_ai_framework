"""
================================================================================
 RoboTurk 数据集适配器 (roboturk_adapter.py)
 ──────────────────────────────────────────────────────────────────────────────
 功能: 将 RoboTurk 数据集转换为统一格式。

 RoboTurk 简介:
   - 来源: Stanford 大学 (远程操控采集)
   - 机器人: Franka Emika Panda
   - 任务: 桌面操控 (pick-place, pushing, opening drawers)
   - 格式: HDF5 (每个 trajectory 一个 .hdf5 文件)
   - 传感器: RGB (256x256), 深度图, 关节角度/力矩, 末端位姿

 数据格式 (HDF5):
   /observations/images/top  : (T, 256, 256, 3) uint8
   /observations/qpos        : (T, 7) float32
   /observations/qvel        : (T, 7) float32
   /action                   : (T, 14) float32 [关节位置(7) + 夹爪宽度(1)] × 2
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Iterator, Dict, Any, Optional
from pathlib import Path
import glob
import os

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import DatasetAdapter
from src.common.types import UnifiedSample


class RoboTurkAdapter(DatasetAdapter):
    """
    RoboTurk 数据集适配器
    ────────────────────────────────────────────────────────────────────────────
    将 HDF5 格式的 RoboTurk 数据转换为 UnifiedSample 格式。
    """
    def __init__(self, image_size: tuple = (256, 256)):
        self.image_size = image_size
        self._metadata = {
            "name": "roboturk",
            "robot": "Franka Emika Panda",
            "total_episodes": 8000,
            "sensors": ["rgb", "depth", "robot_state"],
            "action_dim": 14,
        }

    def can_handle(self, source: str) -> bool:
        """判断是否能处理 RoboTurk 数据源"""
        source_lower = source.lower()
        return "roboturk" in source_lower or "roborturk" in source_lower

    def load(self, source: str, **kwargs) -> Iterator[UnifiedSample]:
        """
        加载 RoboTurk HDF5 数据

        加载策略:
          1. 扫描 source 目录下所有 .hdf5 文件
          2. 对每个文件，读取所有时间步的数据
          3. 逐帧产出 UnifiedSample

        Args:
            source: 数据集根目录路径
            **kwargs:
                max_episodes: 最大加载轨迹数 (-1 表示全部)

        Yields:
            UnifiedSample
        """
        # ── 扫描 HDF5 文件 ──────────────────────────────────────────────────
        hdf5_files = sorted(glob.glob(os.path.join(source, "**/*.hdf5"), recursive=True))
        if not hdf5_files:
            print(f"[RoboTurk] 在 {source} 中未找到 .hdf5 文件")
            return

        max_episodes = kwargs.get("max_episodes", -1)
        if max_episodes > 0:
            hdf5_files = hdf5_files[:max_episodes]

        # ── 逐文件加载 ──────────────────────────────────────────────────────
        for file_idx, hdf5_path in enumerate(hdf5_files):
            try:
                import h5py
                with h5py.File(hdf5_path, "r") as f:
                    # ── 提取数据 ─────────────────────────────────────────────
                    # 观测: 图像 (T, H, W, 3), 关节位置 (T, 7)
                    images = f["observations/images/top"][()]  # (T, 256, 256, 3)
                    qpos = f["observations/qpos"][()]          # (T, 7)
                    actions = f["action"][()]                  # (T, 14)

                    # ── 逐时间步产出样本 ─────────────────────────────────────
                    for t in range(len(images)):
                        # 处理图像
                        rgb = self._process_image(images[t])

                        obs = {
                            "rgb": rgb,
                            "robot_state": torch.tensor(qpos[t], dtype=torch.float32),
                        }

                        action = torch.tensor(actions[t], dtype=torch.float32)
                        # RoboTurk 没有奖励信号，设默认值 0
                        # 最后的 step 设为 done
                        done = (t == len(images) - 1)

                        yield UnifiedSample(
                            obs=obs,
                            action=action,
                            reward=0.0,
                            done=done,
                            metadata={
                                "file": Path(hdf5_path).name,
                                "timestep": t,
                                "dataset": "roboturk",
                            }
                        )
            except Exception as e:
                print(f"[RoboTurk] 加载文件失败 {hdf5_path}: {e}")
                continue

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理: uint8 → float32 [0,1], HWC → CHW
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.fromarray(image)
        img = img.resize(self.image_size, Image.BILINEAR)
        return TF.to_tensor(img)

    def metadata(self) -> Dict[str, Any]:
        return self._metadata
