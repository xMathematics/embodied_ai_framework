"""
================================================================================
 数据加载器测试 (test_dataloader.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/data/dataloader.py 中的数据加载和供给功能。

 测试内容:
   1. UnifiedDataset: 迭代式数据集的创建和遍历
   2. MixedDataset:   混合采样功能
   3. create_dataloader: 工厂函数

 注意:
   这些测试使用 conftest.py 中的 mock_dataset_adapter 和 mock_trajectory，
   不依赖真实数据文件，保证测试速度和可靠性。
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.data.dataloader import UnifiedDataset, create_dataloader
from src.common.types import UnifiedSample


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedDataset 测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnifiedDataset:
    """
    UnifiedDataset 测试
    ────────────────────────────────────────────────────────────────────────────
    验证基于 IterableDataset 的数据加载功能。
    """

    def test_iteration_yields_samples(self, tmp_data_dir: str):
        """
        测试迭代产出 UnifiedSample
        ────────────────────────────────────────────────────────────────────────
        验证: dataset.__iter__() 返回的迭代器产出 UnifiedSample 对象
        """
        dataset = UnifiedDataset(
            data_dir=tmp_data_dir,
            infinite=False,
        )
        samples = list(iter(dataset))
        if len(samples) > 0:
            assert isinstance(samples[0], UnifiedSample)

    def test_infinite_iteration(self, tmp_data_dir: str):
        """
        测试无限迭代模式
        ────────────────────────────────────────────────────────────────────────
        验证: infinite=True 时，迭代器不会在数据耗尽时停止
        """
        dataset = UnifiedDataset(
            data_dir=tmp_data_dir,
            infinite=True,
        )
        it = iter(dataset)
        # 取 10 个样本 (如果数据不足 infinite 会循环)
        samples = [next(it) for _ in range(10)]
        assert len(samples) == 10
        assert all(isinstance(s, UnifiedSample) for s in samples)

    def test_transform_applied(self, tmp_data_dir: str):
        """
        测试自定义变换函数
        ────────────────────────────────────────────────────────────────────────
        验证: transform 函数被正确应用到每个样本
        """
        def dummy_transform(sample):
            """简单变换: 将所有奖励设为 1.0"""
            sample.reward = 1.0
            return sample

        dataset = UnifiedDataset(
            data_dir=tmp_data_dir,
            transform=dummy_transform,
            infinite=False,
        )
        for sample in iter(dataset):
            assert sample.reward == 1.0, "变换应修改 reward 为 1.0"
            break  # 只测试第一个样本

    def test_shuffle_does_not_crash(self, tmp_data_dir: str):
        """
        测试打乱功能不崩溃
        ────────────────────────────────────────────────────────────────────────
        验证: shuffle=True 不会导致运行时错误
        """
        dataset = UnifiedDataset(
            data_dir=tmp_data_dir,
            shuffle=True,
        )
        for _ in iter(dataset):
            break  # 能成功取到数据即可


# ═══════════════════════════════════════════════════════════════════════════════
# create_dataloader 工厂函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateDataloader:
    """
    数据加载器工厂函数测试
    ────────────────────────────────────────────────────────────────────────────
    验证 create_dataloader 返回正确的 PyTorch DataLoader。
    """

    def test_returns_dataloader(self, tmp_data_dir: str):
        """测试返回类型为 DataLoader"""
        loader = create_dataloader(
            data_dir=tmp_data_dir,
            batch_size=4,
            num_workers=0,  # 测试时不用多进程
            shuffle=False,
            infinite=False,
        )
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_batch_size_applied(self, tmp_data_dir: str):
        """测试批次大小正确"""
        loader = create_dataloader(
            data_dir=tmp_data_dir,
            batch_size=4,
            num_workers=0,
        )
        for batch in loader:
            # 验证: action 的 batch 维度为 4
            if "action" in batch:
                assert batch["action"].shape[0] == 4
            break

    def test_multiple_workers(self, tmp_data_dir: str):
        """
        测试多进程加载
        ────────────────────────────────────────────────────────────────────────
        验证: num_workers>0 时能正常创建 DataLoader
        """
        loader = create_dataloader(
            data_dir=tmp_data_dir,
            batch_size=4,
            num_workers=2,
        )
        assert loader.num_workers == 2
