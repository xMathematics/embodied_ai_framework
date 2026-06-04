"""
================================================================================
 BridgeData V2 数据集适配器 (bridge_v2_adapter.py)
 ──────────────────────────────────────────────────────────────────────────────
 功能: 将 BridgeData V2 数据集 (Open X-Embodiment 子集) 转换为统一格式。

 BridgeData V2 简介:
   - 来源: Open X-Embodiment 项目 (Google/UC Berkeley)
   - 机器人: WidowX 250 机械臂
   - 任务: 桌面操控 (抓取、推、放置等)
   - 格式: RLDS (Reinforcement Learning Dataset Standard) → TFRecord
   - 传感器: RGB 图像 (128x128), 关节位置, 夹爪状态

 本适配器利用 HuggingFace datasets 库加载 RLDS 格式数据，
 并转换为 UnifiedSample 格式供后续处理。

 实现思路:
   1. 使用 tensorflow_datasets 或 HuggingFace datasets 加载原始 RLDS
   2. 遍历 episode 中的每个 step，提取观测和动作
   3. 将 numpy 数组转为 torch.Tensor 并包装为 UnifiedSample
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Iterator, Dict, Any, Optional
import os

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import DatasetAdapter
from src.common.types import UnifiedSample


class BridgeV2Adapter(DatasetAdapter):
    """
    BridgeData V2 适配器
    ────────────────────────────────────────────────────────────────────────────
    将 BridgeData V2 的 RLDS 格式转换为 UnifiedSample。
    """
    def __init__(self, image_size: tuple = (128, 128)):
        """
        Args:
            image_size: 图像调整尺寸 (H, W)，默认 128x128
        """
        self.image_size = image_size
        self._metadata = {
            "name": "bridge_v2",
            "robot": "WidowX 250",
            "total_episodes": 60000,  # 约 6 万条轨迹
            "sensors": ["rgb", "robot_state"],
            "action_dim": 7,          # 末端位姿 (x,y,z,roll,pitch,yaw) + 夹爪
        }

    def can_handle(self, source: str) -> bool:
        """
        判断是否能处理指定数据源
        原理: 检查数据源路径是否包含 BridgeData V2 的特征标记
        """
        source_lower = source.lower()
        return "bridge" in source_lower or "bridge_v2" in source_lower

    def load(self, source: str, **kwargs) -> Iterator[UnifiedSample]:
        """
        加载 BridgeData V2 数据并逐帧产出 UnifiedSample

        Args:
            source: 数据源路径或 HuggingFace 数据集 ID
                    (如 "bridge_v2" 或 "/path/to/local/bridge_v2")

        Yields:
            UnifiedSample: 统一格式的样本

        数据格式说明:
          RLDS 中每个 episode 包含多个 step，每个 step 包含:
            - observation['image']:     (128,128,3) uint8
            - observation['state']:     (7,) float32 (关节位置)
            - action:                   (7,) float32 (末端增量位姿+夹爪)
            - reward:                   float32
            - is_terminal:              bool
        """
        # ── 尝试加载数据集 ──────────────────────────────────────────────────
        try:
            # 优先使用 HuggingFace datasets
            from datasets import load_dataset
            dataset = load_dataset(
                source if "/" in source else "physion/bridge_data_v2",
                split="train",
                streaming=True,  # 流式加载，节省内存
            )
        except ImportError:
            # 回退: 使用 tensorflow_datasets
            try:
                import tensorflow_datasets as tfds
                ds = tfds.load("bridge_dataset", split="train")
                dataset = iter(ds)
            except ImportError:
                raise ImportError(
                    "需要安装 datasets 或 tensorflow_datasets 库:\n"
                    "  pip install datasets  # 推荐\n"
                    "  # 或\n"
                    "  pip install tensorflow_datasets"
                )

        # ── 逐 episode 遍历 ─────────────────────────────────────────────────
        for episode_idx, episode in enumerate(dataset):
            # 如果指定了最大 episode 数，提前停止
            max_episodes = kwargs.get("max_episodes", -1)
            if max_episodes > 0 and episode_idx >= max_episodes:
                break

            # ── 遍历 episode 中的每个 step ──────────────────────────────────
            steps = episode["steps"] if "steps" in episode else episode
            for step in steps:
                # 提取观测
                obs = {
                    "rgb": self._process_image(step["observation"]["image"]),
                    "robot_state": torch.tensor(
                        step["observation"]["state"], dtype=torch.float32
                    ),
                }

                # 提取动作
                action = torch.tensor(step["action"], dtype=torch.float32)

                # 提取奖励和终止信号
                reward = float(step.get("reward", 0.0))
                done = bool(step.get("is_terminal", False))

                # 创建统一样本
                sample = UnifiedSample(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    metadata={
                        "episode": episode_idx,
                        "dataset": "bridge_v2",
                    }
                )
                yield sample

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理流水线
        ────────────────────────────────────────────────────────────────────────
        步骤:
          1. 从 uint8 [0,255] 转为 float32 [0,1]
          2. 从 HWC 转为 CHW 格式 (PyTorch 标准)
          3. 缩放至目标尺寸

        Args:
            image: 原始图像 numpy 数组, shape=(H,W,3), uint8

        Returns:
            处理后的图像张量, shape=(3,H,W), float32
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        # numpy → PIL → torch (自动处理 HWC→CHW 和归一化)
        img = Image.fromarray(image)
        img = img.resize(self.image_size, Image.BILINEAR)
        tensor = TF.to_tensor(img)  # 转为 [0,1] float32, shape=(3,H,W)
        return tensor

    def metadata(self) -> Dict[str, Any]:
        """返回数据集元信息"""
        return self._metadata
