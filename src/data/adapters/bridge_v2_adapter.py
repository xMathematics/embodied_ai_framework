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
            **kwargs:
                max_episodes: 最大加载轨迹数 (-1 表示全部)
                demo_mode:    如果 True 且 HF 数据集不可用，生成模拟数据

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
        demo_mode = kwargs.get("demo_mode", False)
        dataset = None

        # ── 优先从 HuggingFace Hub 加载 ────────────────────────────────────
        try:
            from datasets import load_dataset
            from datasets.exceptions import DatasetNotFoundError

            hf_dataset_id = source if "/" in source else "rail-berkeley/bridge_data_v2"
            dataset = load_dataset(
                hf_dataset_id,
                split="train",
                streaming=True,
            )
            print(f"[BridgeV2] 成功加载 HuggingFace 数据集: {hf_dataset_id}")

        except (DatasetNotFoundError, Exception) as e:
            print(f"[BridgeV2] HuggingFace 数据集不可用: {e}")
            print(f"[BridgeV2] 尝试从本地路径加载...")

            # ── 尝试从本地 RLDS/TFRecord 路径加载 ──────────────────────────
            if os.path.isdir(source):
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(
                        source,
                        split="train",
                        streaming=True,
                    )
                    print(f"[BridgeV2] 成功从本地路径加载: {source}")
                except Exception:
                    dataset = None

        # ── 如果 HF 和本地都加载失败，使用模拟数据 ─────────────────────────
        if dataset is None:
            if demo_mode:
                print(f"[BridgeV2] 使用模拟数据模式 (demo_mode=True)")
                return self._generate_demo_data(**kwargs)
            else:
                print(f"[BridgeV2] 数据集 '{source}' 无法访问。")
                print(f"[BridgeV2] 可用选项:")
                print(f"  1. 设置 demo_mode=True 使用模拟数据进行测试")
                print(f"  2. 从 HuggingFace 下载:")
                print(f"     pip install datasets")
                print(f"     python -c \"from datasets import load_dataset;\"")
                print(f"     load_dataset('rail-berkeley/bridge_data_v2').save_to_disk('./data/bridge_v2')")
                print(f"  3. 使用本地 RLDS/TFRecord 路径")
                raise FileNotFoundError(
                    f"数据集 '{source}' 无法加载。"
                    " 请先下载数据集或设置 demo_mode=True 使用模拟数据。"
                )

        # ── 逐 episode 遍历 (HF Dataset 格式) ──────────────────────────────
        for episode_idx, episode in enumerate(dataset):
            max_episodes = kwargs.get("max_episodes", -1)
            if max_episodes > 0 and episode_idx >= max_episodes:
                break

            # ── 遍历 episode 中的每个 step ──────────────────────────────────
            steps = episode["steps"] if "steps" in episode else episode
            for step in steps:
                obs = {
                    "rgb": self._process_image(step["observation"]["image"]),
                    "robot_state": torch.tensor(
                        step["observation"]["state"], dtype=torch.float32
                    ),
                }
                action = torch.tensor(step["action"], dtype=torch.float32)
                reward = float(step.get("reward", 0.0))
                done = bool(step.get("is_terminal", False))

                yield UnifiedSample(
                    obs=obs, action=action, reward=reward, done=done,
                    metadata={"episode": episode_idx, "dataset": "bridge_v2"},
                )

    def _generate_demo_data(self, **kwargs) -> Iterator[UnifiedSample]:
        """
        生成模拟 BridgeData V2 数据
        ────────────────────────────────────────────────────────────────────────
        当 HuggingFace Hub 和本地路径都不可用时，生成模拟数据供测试使用。
        模拟数据包含随机生成的 RGB 图像和机器人状态，格式与真实数据一致。

        Args:
            **kwargs:
                max_episodes: 最大生成轨迹数 (默认 3)
                steps_per_episode: 每条轨迹的步数 (默认 50)
        """
        import random as _random

        num_episodes = kwargs.get("max_episodes", 3) if kwargs.get("max_episodes", -1) > 0 else 3
        steps_per_episode = kwargs.get("steps_per_episode", 50)

        print(f"[BridgeV2] 生成 {num_episodes} 条模拟轨迹 (每条 {steps_per_episode} 步)")
        print(f"[BridgeV2] 注意: 模拟数据仅用于测试流水线，请用真实数据训练模型")

        for ep_idx in range(num_episodes):
            for step in range(steps_per_episode):
                # 生成随机 RGB 图像 (128,128,3) uint8 → 模拟真实相机输出
                fake_image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
                # 生成随机机器人状态 (7 维关节位置)
                fake_state = np.random.randn(7).astype(np.float32)
                # 生成随机动作 (7 维增量位姿 + 夹爪)
                fake_action = np.random.randn(7).astype(np.float32)

                obs = {
                    "rgb": self._process_image(fake_image),
                    "robot_state": torch.tensor(fake_state, dtype=torch.float32),
                }
                action = torch.tensor(fake_action, dtype=torch.float32)
                done = (step == steps_per_episode - 1)

                yield UnifiedSample(
                    obs=obs, action=action, reward=0.0, done=done,
                    language_embed=None,
                    metadata={
                        "episode": ep_idx,
                        "dataset": "bridge_v2",
                        "demo_mode": True,
                    },
                )

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
