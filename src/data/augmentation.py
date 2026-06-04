"""
================================================================================
 数据增强模块 (augmentation.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供多样化数据增强策略，提升策略的泛化能力和鲁棒性。

 增强策略分为两个层次:
   1. 图像级增强 (ImageAugmenter)
      - 作用于单个样本的观测图像
      - 模拟真实世界的视觉变化 (光照、遮挡、噪声)
      - 防止策略过拟合于特定视觉特征

   2. 轨迹级增强 (TrajectoryAugmenter)
      - 作用于整条轨迹的时序数据
      - 增加轨迹多样性 (时间缩放、动作噪声)
      - 提升策略对执行不完美的容忍度

 设计模式:
   组合模式 (Composite Pattern) —— 多个增强器可以组合成流水线，
   按顺序依次处理数据，形成完整的增强流程。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import List, Optional, Tuple, Callable
import random

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import UnifiedSample, Trajectory
from src.common.interfaces import DataAugmenter


# ═══════════════════════════════════════════════════════════════════════════════
# 图像级增强
# ═══════════════════════════════════════════════════════════════════════════════

class ImageAugmenter(DataAugmenter):
    """
    图像数据增强器 (ImageAugmenter)
    ────────────────────────────────────────────────────────────────────────────
    对观测中的 RGB 和深度图像进行数据增强，模拟真实环境中的视觉变化。

    增强策略 (可通过配置灵活组合):
      1. RandomCrop:   随机裁剪后缩放回原尺寸 (模拟视角偏移)
      2. ColorJitter:  随机调整亮度/对比度/饱和度/色调 (模拟光照变化)
      3. GaussianBlur: 高斯模糊 (模拟运动模糊)
      4. RandomErasing: 随机遮挡 (模拟物体遮挡)
      5. Normalize:    标准化 (使用 ImageNet 统计数据)

    核心原则:
      - 增强必须在训练时随机应用，评估/推理时关闭
      - 不改变语义信息 (不能翻转图像，因为左右方向在机器人任务中很重要)
    """
    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: 增强配置字典
                example:
                {
                    "crop_scale": (0.8, 1.0),
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "blur_kernel": 5,
                    "erase_scale": 0.02,
                }
        """
        if config is None:
            config = {}

        # ── 构建 torchvision 增强流水线 ─────────────────────────────────────
        transforms_list = []

        # 随机裁剪: 模拟相机视角微小偏移
        crop_scale = config.get("crop_scale", (0.9, 1.0))
        transforms_list.append(T.RandomResizedCrop(
            size=(128, 128), scale=crop_scale, ratio=(1.0, 1.0)
        ))

        # 色彩抖动: 模拟光照变化
        transforms_list.append(T.ColorJitter(
            brightness=config.get("brightness", 0.1),
            contrast=config.get("contrast", 0.1),
            saturation=config.get("saturation", 0.1),
            hue=config.get("hue", 0.05),
        ))

        # 高斯模糊: 模拟运动模糊或对焦不准
        if config.get("blur_kernel", 0) > 0:
            kernel = config["blur_kernel"]
            transforms_list.append(T.GaussianBlur(
                kernel_size=kernel,
                sigma=config.get("blur_sigma", (0.1, 2.0))
            ))

        # 随机遮挡: 模拟物体遮挡或传感器故障
        if config.get("erase_prob", 0.0) > 0:
            transforms_list.append(T.RandomErasing(
                p=config["erase_prob"],
                scale=(config.get("erase_scale", 0.02), 0.1),
                ratio=(0.3, 3.3),
                value=0.0,  # 用黑色遮挡
            ))

        self.augment_pipeline = T.Compose(transforms_list)

    def augment_sample(self, sample: UnifiedSample) -> UnifiedSample:
        """增强单个样本的图像观测"""
        aug_obs = {}
        for key, tensor in sample.obs.items():
            if key in ("rgb", "depth") and tensor.ndim == 3:
                # 应用图像增强 (要求输入为 [C, H, W] float32)
                aug_obs[key] = self.augment_pipeline(tensor)
            else:
                aug_obs[key] = tensor

        return UnifiedSample(
            obs=aug_obs,
            action=sample.action,
            reward=sample.reward,
            done=sample.done,
            language_embed=sample.language_embed,
        )

    def augment_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        增强轨迹中的所有图像帧
        确保同一轨迹内使用相同的随机种子，保持时序一致性。
        """
        # 固定随机种子，保证整条轨迹使用相同的增强参数
        seed = random.randint(0, 2**32)
        aug_obs = {}
        for key, tensor in traj.observations.items():
            if key in ("rgb", "depth") and tensor.ndim == 4:  # (T, C, H, W)
                aug_frames = []
                for t in range(tensor.shape[0]):
                    torch.manual_seed(seed + t)
                    aug_frames.append(self.augment_pipeline(tensor[t]))
                aug_obs[key] = torch.stack(aug_frames)
            else:
                aug_obs[key] = tensor

        return Trajectory(
            observations=aug_obs,
            actions=traj.actions,
            rewards=traj.rewards,
            dones=traj.dones,
            language_embeds=traj.language_embeds,
            task_id=traj.task_id,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 轨迹级增强
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryAugmenter(DataAugmenter):
    """
    轨迹数据增强器 (TrajectoryAugmenter)
    ────────────────────────────────────────────────────────────────────────────
    对整条轨迹进行时序维度的增强操作。

    增强策略:
      1. 时间子采样: 随机采样轨迹的子序列 (适应不同控制频率)
      2. 动作噪声注入: 在专家动作上添加高斯噪声 (提升鲁棒性)
      3. 时间缩放: 改变轨迹播放速度 (模拟不同执行速度)
    """
    def __init__(self, noise_std: float = 0.01, subsample_ratio: float = 0.8):
        """
        Args:
            noise_std: 动作噪声标准差 (相对于动作范围的比例)
            subsample_ratio: 子采样比例 (0.8 = 保留 80% 的时间步)
        """
        self.noise_std = noise_std
        self.subsample_ratio = subsample_ratio

    def augment_sample(self, sample: UnifiedSample) -> UnifiedSample:
        """增强单个样本 (注入动作噪声)"""
        noise = torch.randn_like(sample.action) * self.noise_std
        return UnifiedSample(
            obs=sample.obs,
            action=sample.action + noise,
            reward=sample.reward,
            done=sample.done,
            language_embed=sample.language_embed,
        )

    def augment_trajectory(self, traj: Trajectory) -> Trajectory:
        """
        增强完整轨迹:
          1. 时间子采样 (随机选择子序列)
          2. 动作噪声注入
        """
        T = traj.length

        # ── 1. 时间子采样 ──────────────────────────────────────────────────
        target_len = max(2, int(T * self.subsample_ratio))
        # 随机选择起始点 (均匀采样)
        if T > target_len:
            start = random.randint(0, T - target_len)
            indices = list(range(start, start + target_len))
        else:
            indices = list(range(T))

        # ── 2. 应用子采样索引 ───────────────────────────────────────────────
        sampled_obs = {
            k: v[indices] for k, v in traj.observations.items()
        }
        sampled_actions = traj.actions[indices]

        # ── 3. 注入动作噪声 ─────────────────────────────────────────────────
        noise = torch.randn_like(sampled_actions) * self.noise_std
        noisy_actions = sampled_actions + noise

        return Trajectory(
            observations=sampled_obs,
            actions=noisy_actions,
            rewards=traj.rewards[indices] if traj.rewards.numel() > 1 else traj.rewards,
            dones=traj.dones[indices] if traj.dones.numel() > 1 else traj.dones,
            language_embeds=traj.language_embeds,
            task_id=traj.task_id,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 增强流水线 (组合多个增强器)
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentationPipeline:
    """
    增强流水线 (AugmentationPipeline)
    ────────────────────────────────────────────────────────────────────────────
    组合多个增强器，按顺序执行，形成完整的增强流程。

    用法:
        pipeline = AugmentationPipeline([
            ImageAugmenter(config),
            TrajectoryAugmenter(),
        ])
        augmented_sample = pipeline(sample)

    设计模式:
      责任链模式 (Chain of Responsibility) —— 每个增强器独立处理数据
      后传递给下一个，便于灵活组合和插拔。
    """
    def __init__(self, augmenters: List[DataAugmenter]):
        """
        Args:
            augmenters: 增强器列表，按顺序执行
        """
        self.augmenters = augmenters

    def __call__(self, sample: UnifiedSample) -> UnifiedSample:
        """执行完整增强流水线"""
        for augmenter in self.augmenters:
            sample = augmenter.augment_sample(sample)
        return sample

    def augment_trajectory(self, traj: Trajectory) -> Trajectory:
        """对轨迹执行完整增强流水线"""
        for augmenter in self.augmenters:
            traj = augmenter.augment_trajectory(traj)
        return traj
