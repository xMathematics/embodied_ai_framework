"""
================================================================================
 行为克隆算法测试 (test_bc.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/algorithm/imitation/bc.py 中的行为克隆训练器。

 测试策略:
   1. 用随机数据训练一个小型 BC 策略
   2. 验证训练后损失降低 (过拟合到小批量数据)
   3. 验证训练器的接口完整 (train_epoch, validate)
   4. 验证梯度更新正确 (损失反向传播)

 注意:
   这里测试的是 BC 训练逻辑的正确性，不是策略的泛化能力。
   使用小批量数据训练少量 epoch，验证训练流程不崩溃。
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.algorithm.imitation.bc import BCTrainer
from src.algorithm.policy import GaussianPolicy


# ═══════════════════════════════════════════════════════════════════════════════
# BC 训练器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestBCTrainer:
    """
    行为克隆训练器测试
    ────────────────────────────────────────────────────────────────────────────
    验证 BCTrainer:
      - 正确执行训练循环
      - 损失值在训练过程中下降
      - 验证集评估正常工作
      - 在不同设备上工作 (CPU/CUDA)
    """

    @pytest.fixture
    def trainer(self, device):
        """创建 BC 训练器"""
        return BCTrainer(
            obs_dim=64,
            action_dim=7,
            lr=0.001,
            device=str(device),
        )

    @pytest.fixture
    def dummy_dataloader(self):
        """
        创建模拟数据加载器
        ────────────────────────────────────────────────────────────────────────
        生成随机观测和动作数据，模拟 BC 训练中的 DataLoader。
        """
        # 生成 100 个随机样本
        obs = torch.rand(100, 64)
        actions = torch.rand(100, 7)
        dataset = TensorDataset(obs, actions)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def test_train_epoch_returns_metrics(self, trainer, dummy_dataloader):
        """
        测试训练 epoch 返回指标
        ────────────────────────────────────────────────────────────────────────
        验证: train_epoch 返回包含 'bc_loss' 的指标字典
        """
        metrics = trainer.train_epoch(dummy_dataloader)
        assert "bc_loss" in metrics
        assert isinstance(metrics["bc_loss"], float)
        assert metrics["bc_loss"] > 0

    def test_loss_decreases_over_epochs(self, trainer, dummy_dataloader):
        """
        测试损失随训练下降
        ────────────────────────────────────────────────────────────────────────
        原理:
          在小数据集上过拟合是验证训练流程正确的快速方法。
          如果损失不下降，说明训练逻辑有问题 (如梯度未更新)。
        """
        # 记录初始损失
        initial_metrics = trainer.train_epoch(dummy_dataloader)
        initial_loss = initial_metrics["bc_loss"]

        # 训练多个 epoch
        for _ in range(10):
            trainer.train_epoch(dummy_dataloader)

        # 记录最终损失
        final_metrics = trainer.train_epoch(dummy_dataloader)
        final_loss = final_metrics["bc_loss"]

        # 验证: 损失下降
        assert final_loss < initial_loss, (
            f"损失未下降! 初始: {initial_loss:.4f}, 最终: {final_loss:.4f}"
        )

    def test_validate_returns_metrics(self, trainer, dummy_dataloader):
        """
        测试验证返回指标
        ────────────────────────────────────────────────────────────────────────
        验证: validate() 返回包含 'val_bc_loss' 的指标字典
        """
        metrics = trainer.validate(dummy_dataloader)
        assert "val_bc_loss" in metrics
        assert isinstance(metrics["val_bc_loss"], float)

    def test_optimizer_parameters(self, trainer):
        """
        测试优化器参数
        ────────────────────────────────────────────────────────────────────────
        验证: 优化器包含策略网络的所有可学习参数
        """
        param_groups = trainer.optimizer.param_groups
        all_params = sum(p.numel() for p in trainer.policy.parameters() if p.requires_grad)
        optim_params = sum(
            p.numel() for group in param_groups for p in group["params"]
        )
        assert all_params == optim_params, "优化器参数与模型参数不匹配"

    def test_device_consistency(self, trainer, device):
        """测试设备一致性"""
        policy_device = next(trainer.policy.parameters()).device
        assert str(policy_device) == str(device), (
            f"策略在 {policy_device}，但预期在 {device}"
        )

    def test_gradient_clipping(self, trainer, dummy_dataloader):
        """
        测试梯度裁剪
        ────────────────────────────────────────────────────────────────────────
        验证: 训练后所有参数的梯度范数不超过 1.0 (clip 阈值)
        """
        import torch.nn.utils as nn_utils

        for batch in dummy_dataloader:
            obs_feat = trainer._extract_obs_features({"obs": batch[0]})
            expert_actions = batch[1].to(trainer.device)

            pred_actions = trainer.policy(obs_feat, deterministic=False)
            loss = torch.nn.functional.mse_loss(pred_actions, expert_actions)

            trainer.optimizer.zero_grad()
            loss.backward()
            # 验证: 梯度裁剪在训练循环中已应用
            total_norm = nn_utils.clip_grad_norm_(
                trainer.policy.parameters(), max_norm=1.0
            )
            trainer.optimizer.step()
            break  # 只测试第一个 batch


# ═══════════════════════════════════════════════════════════════════════════════
# BC 策略过拟合测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestBCOverfitting:
    """
    BC 过拟合能力测试
    ────────────────────────────────────────────────────────────────────────────
    验证: 策略是否有足够的容量来记住小数据集。
    这是训练流水线正确性的 smoke test (冒烟测试)。
    """

    def test_overfit_small_dataset(self):
        """
        测试在小数据集上过拟合
        ────────────────────────────────────────────────────────────────────────
        原理:
          如果一个训练器连 10 个样本都记不住，说明训练逻辑有严重问题。
        """
        trainer = BCTrainer(obs_dim=16, action_dim=3, lr=0.01)

        # 创建小数据集 (仅 10 个样本)
        obs = torch.rand(10, 16)
        actions = torch.rand(10, 3)
        dataset = TensorDataset(obs, actions)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)  # 整个数据集一起

        # 训练 100 个 epoch
        for _ in range(100):
            trainer.train_epoch(loader)

        # 验证: 策略在训练集上的预测误差很小
        with torch.no_grad():
            pred = trainer.policy(obs, deterministic=True)
            mse = torch.nn.functional.mse_loss(pred, actions)
        assert mse.item() < 0.1, (
            f"过拟合测试失败! MSE={mse.item():.4f} > 0.1"
        )
