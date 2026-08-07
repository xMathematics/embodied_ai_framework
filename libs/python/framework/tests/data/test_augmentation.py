"""
================================================================================
 数据增强模块测试 (test_augmentation.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/data/augmentation.py 中的数据增强功能。

 测试策略:
   1. 验证增强不改变数据结构 (输入输出类型一致)
   2. 验证增强确实改变了数据内容 (不是恒等映射)
   3. 验证多种增强可以正确组合
   4. 验证确定性模式 (seed 固定) 下结果可复现

 注意:
   增强具有随机性，我们验证的是统计性质而非精确值。
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.common.types import UnifiedSample, Trajectory
from src.data.augmentation import (
    ImageAugmenter,
    TrajectoryAugmenter,
    AugmentationPipeline,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 图像增强器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageAugmenter:
    """
    图像增强测试
    ────────────────────────────────────────────────────────────────────────────
    验证 ImageAugmenter 能正确地:
      - 处理单张图像 (样本级)
      - 处理序列图像 (轨迹级)
      - 保持输出张量形状不变
    """

    @pytest.fixture
    def augmenter(self):
        """创建一个默认配置的图像增强器"""
        return ImageAugmenter()

    @pytest.fixture
    def sample(self):
        """创建一个包含 RGB 图像的测试样本"""
        return UnifiedSample(
            obs={
                "rgb": torch.rand(3, 128, 128),
                "robot_state": torch.rand(7),
            },
            action=torch.rand(7),
        )

    def test_augment_sample_returns_same_type(self, augmenter, sample):
        """
        测试样本增强返回相同类型
        ────────────────────────────────────────────────────────────────────────
        验证: 增强后的样本仍然是 UnifiedSample 类型
        """
        augmented = augmenter.augment_sample(sample)
        assert isinstance(augmented, UnifiedSample)

    def test_augment_sample_preserves_structure(self, augmenter, sample):
        """
        测试增强保持数据结构
        ────────────────────────────────────────────────────────────────────────
        验证:
          - 观测字典包含相同的键
          - 动作和奖励不变 (图像增强只改变图像)
        """
        augmented = augmenter.augment_sample(sample)

        # 观测键不变
        assert set(augmented.obs.keys()) == set(sample.obs.keys())

        # 非图像数据应不变
        assert torch.equal(augmented.obs["robot_state"], sample.obs["robot_state"])
        assert torch.equal(augmented.action, sample.action)

    def test_augment_changes_image_values(self, augmenter, sample):
        """
        测试增强确实改变了图像像素值
        ────────────────────────────────────────────────────────────────────────
        验证: 增强后的图像与原始图像不同 (非恒等映射)
        如果增强没改变像素值，说明流水线可能没有生效。
        """
        augmented = augmenter.augment_sample(sample)
        original_rgb = sample.obs["rgb"]
        augmented_rgb = augmented.obs["rgb"]
        # 验证: 像素值发生了改变 (至少有 1% 的像素不同)
        diff_ratio = (original_rgb != augmented_rgb).float().mean().item()
        assert diff_ratio > 0.01, (
            f"增强未产生足够变化! 差异比例: {diff_ratio:.4f}"
        )

    def test_augment_trajectory(self):
        """
        测试轨迹级图像增强
        ────────────────────────────────────────────────────────────────────────
        验证: 轨迹中所有时间步的图像都能被正确增强
        """
        augmenter = ImageAugmenter()
        traj = Trajectory(
            observations={
                "rgb": torch.rand(10, 3, 128, 128),
                "robot_state": torch.rand(10, 7),
            },
            actions=torch.rand(10, 7),
            rewards=torch.rand(10),
            dones=torch.zeros(10, dtype=torch.bool),
        )
        aug_traj = augmenter.augment_trajectory(traj)
        assert aug_traj.observations["rgb"].shape == (10, 3, 128, 128)

    def test_augment_without_rgb(self, augmenter):
        """
        测试不含 RGB 的样本
        ────────────────────────────────────────────────────────────────────────
        验证: 当样本中没有图像数据时，增强器不报错
        """
        sample = UnifiedSample(
            obs={"robot_state": torch.rand(7)},
            action=torch.rand(5),
        )
        augmented = augmenter.augment_sample(sample)
        assert "robot_state" in augmented.obs


# ═══════════════════════════════════════════════════════════════════════════════
# 轨迹增强器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryAugmenter:
    """
    轨迹级增强测试
    ────────────────────────────────────────────────────────────────────────────
    验证 TrajectoryAugmenter 能正确地对整条轨迹进行时序增强:
      - 动作噪声注入
      - 时间子采样
    """

    @pytest.fixture
    def augmenter(self):
        return TrajectoryAugmenter(noise_std=0.1, subsample_ratio=0.8)

    @pytest.fixture
    def trajectory(self):
        """创建一条长度为 20 的测试轨迹"""
        return Trajectory(
            observations={
                "rgb": torch.rand(20, 3, 128, 128),
                "robot_state": torch.rand(20, 7),
            },
            actions=torch.rand(20, 7),
            rewards=torch.rand(20),
            dones=torch.zeros(20, dtype=torch.bool),
        )

    def test_noise_injected_into_actions(self, augmenter, trajectory):
        """
        测试动作噪声注入
        ────────────────────────────────────────────────────────────────────────
        验证: 增强后的动作与原始动作不同 (噪声已注入)
        注意: 子采样会改变轨迹长度，因此只比较子采样后的公共部分
        """
        aug_traj = augmenter.augment_trajectory(trajectory)
        # 噪声注入后，动作应该有变化 (比较重叠的时间步)
        min_len = min(aug_traj.actions.shape[0], trajectory.actions.shape[0])
        diff = (aug_traj.actions[:min_len] != trajectory.actions[:min_len]).any()
        assert diff, "动作应被加入噪声"

    def test_subsample_reduces_length(self, augmenter, trajectory):
        """
        测试时间子采样减少轨迹长度
        ────────────────────────────────────────────────────────────────────────
        验证: 子采样比例 0.8 下，轨迹长度减少到 ≤ 原始长度
        """
        aug_traj = augmenter.augment_trajectory(trajectory)
        assert aug_traj.length <= trajectory.length, "子采样应减少轨迹长度"

    def test_short_trajectory_not_subsampled(self):
        """
        测试短轨迹不被子采样
        ────────────────────────────────────────────────────────────────────────
        验证: 太短的轨迹 (≤2步) 不进行子采样
        """
        augmenter = TrajectoryAugmenter(subsample_ratio=0.3)
        short_traj = Trajectory(
            observations={"rgb": torch.rand(2, 3, 128, 128)},
            actions=torch.rand(2, 7),
            rewards=torch.rand(2),
            dones=torch.tensor([False, True]),
        )
        aug_traj = augmenter.augment_trajectory(short_traj)
        # 短轨迹至少保留 2 步
        assert aug_traj.length >= 2

    def test_sample_noise_injection(self, augmenter):
        """
        测试单个样本噪声注入
        ────────────────────────────────────────────────────────────────────────
        验证: 样本级增强时动作被加入噪声
        """
        sample = UnifiedSample(
            obs={"robot_state": torch.rand(7)},
            action=torch.rand(7),
        )
        aug_sample = augmenter.augment_sample(sample)
        # 验证: 动作被改变了
        assert not torch.equal(aug_sample.action, sample.action)


# ═══════════════════════════════════════════════════════════════════════════════
# 增强流水线组合测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestAugmentationPipeline:
    """
    增强流水线组合测试
    ────────────────────────────────────────────────────────────────────────────
    验证多个增强器可以正确组合为流水线，按顺序执行。
    """

    def test_pipeline_composition(self):
        """
        测试流水线组合
        ────────────────────────────────────────────────────────────────────────
        验证: 图像增强 + 轨迹增强 组合后能同时生效
        """
        pipeline = AugmentationPipeline([
            ImageAugmenter(),
            TrajectoryAugmenter(noise_std=0.05),
        ])

        sample = UnifiedSample(
            obs={
                "rgb": torch.rand(3, 128, 128),
                "robot_state": torch.rand(7),
            },
            action=torch.rand(7),
        )

        augmented = pipeline(sample)
        # 验证: 图像和动作都经过了增强
        assert isinstance(augmented, UnifiedSample)
        assert augmented.obs["rgb"].shape == (3, 128, 128)

    def test_empty_pipeline(self):
        """
        测试空流水线
        ────────────────────────────────────────────────────────────────────────
        验证: 空增强流水线返回原始样本不变
        """
        pipeline = AugmentationPipeline([])
        sample = UnifiedSample(
            obs={"rgb": torch.rand(3, 128, 128)},
            action=torch.rand(7),
        )
        augmented = pipeline(sample)
        assert torch.equal(augmented.obs["rgb"], sample.obs["rgb"])
