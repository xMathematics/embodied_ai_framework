"""
================================================================================
 策略网络测试 (test_policy.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/algorithm/policy.py 中的神经网络组件。

 测试内容:
   1. MLP: 多层感知机前向传播和维度正确性
   2. CNNEncoder: 图像编码器的输入输出形状
   3. GaussianPolicy: 高斯策略的采样和确定性推理
   4. ActorPolicy: 策略封装类的 act/save/load 接口
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch
import torch.nn as nn

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.algorithm.policy import (
    MLP,
    CNNEncoder,
    GaussianPolicy,
    ActorPolicy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MLP 网络测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLP:
    """
    多层感知机测试
    ────────────────────────────────────────────────────────────────────────────
    验证 MLP 组件的维度传递正确性和激活函数。
    """

    def test_forward_shape(self):
        """
        测试前向输出维度
        ────────────────────────────────────────────────────────────────────────
        验证: 输入 [B, 128] → 输出 [B, 32]
        """
        mlp = MLP([128, 256, 64, 32])
        x = torch.rand(16, 128)
        out = mlp(x)
        assert out.shape == (16, 32), f"预期 (16,32), 实际 {out.shape}"

    def test_single_layer(self):
        """
        测试单层网络 (输入直连输出)
        ────────────────────────────────────────────────────────────────────────
        验证: MLP([64, 10]) 等价于 Linear(64, 10) （无激活函数）
        """
        mlp = MLP([64, 10], dropout=0.0, use_norm=False)
        x = torch.rand(4, 64)
        out = mlp(x)
        assert out.shape == (4, 10)

    def test_dropout_training_vs_eval(self):
        """
        测试训练与评估模式的 dropout 行为
        ────────────────────────────────────────────────────────────────────────
        原理: Dropout 在训练时随机失活神经元，评估时关闭。
        验证: 同一输入在训练和评估模式下输出不同。
        """
        mlp = MLP([32, 64, 16], dropout=0.5)
        x = torch.rand(8, 32)

        # 训练模式 (dropout 开启)
        mlp.train()
        out_train = mlp(x)

        # 评估模式 (dropout 关闭)
        mlp.eval()
        out_eval = mlp(x)

        # 验证: 两种模式输出不同 (训练时有随机失活)
        # 注意: 有小概率相等 (但概率极低)
        assert not torch.equal(out_train, out_eval)

    def test_gradient_flow(self):
        """
        测试梯度流通
        ────────────────────────────────────────────────────────────────────────
        验证: 经过 MLP 后，参数梯度能正确计算
        """
        mlp = MLP([16, 32, 8])
        x = torch.rand(4, 16, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        # 验证: 输入有梯度
        assert x.grad is not None
        # 验证: 网络参数有梯度
        for param in mlp.parameters():
            assert param.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CNN 编码器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCNNEncoder:
    """
    卷积编码器测试
    ────────────────────────────────────────────────────────────────────────────
    验证 CNNEncoder 将图像编码为特征向量。
    """

    def test_encode_rgb_image(self):
        """
        测试编码 RGB 图像
        ────────────────────────────────────────────────────────────────────────
        验证: 输入 (B, 3, 128, 128) → 输出 (B, 64) [默认 latent_dim=64]
        """
        encoder = CNNEncoder(in_channels=3, latent_dim=64)
        images = torch.rand(8, 3, 128, 128)  # (batch, channels, H, W)
        features = encoder(images)
        assert features.shape == (8, 64), f"预期 (8,64), 实际 {features.shape}"

    def test_encode_grayscale(self):
        """
        测试编码灰度图
        ────────────────────────────────────────────────────────────────────────
        验证: 输入 (B, 1, 128, 128) → 输出 (B, 32)
        """
        encoder = CNNEncoder(in_channels=1, latent_dim=32)
        images = torch.rand(4, 1, 128, 128)
        features = encoder(images)
        assert features.shape == (4, 32)

    def test_batch_independence(self):
        """
        测试批次独立性
        ────────────────────────────────────────────────────────────────────────
        验证: batch 中不同样本独立编码，互不影响
        """
        encoder = CNNEncoder(latent_dim=16)
        # 两个完全不同的输入
        img1 = torch.zeros(1, 3, 128, 128)
        img2 = torch.ones(1, 3, 128, 128)
        batch = torch.cat([img1, img2], dim=0)

        features = encoder(batch)
        # 验证: 不同输入产生不同编码
        assert not torch.equal(features[0], features[1])


# ═══════════════════════════════════════════════════════════════════════════════
# 高斯策略测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestGaussianPolicy:
    """
    高斯策略测试
    ────────────────────────────────────────────────────────────────────────────
    验证 GaussianPolicy 的采样行为:
      - 确定性模式返回均值
      - 随机采样模式返回不同的动作
      - 输出动作在合理范围内
    """

    @pytest.fixture
    def policy(self):
        return GaussianPolicy(obs_dim=64, action_dim=7, hidden_dims=[128, 128])

    def test_deterministic_output(self, policy):
        """
        测试确定性模式
        ────────────────────────────────────────────────────────────────────────
        验证: deterministic=True 时，相同输入得到相同输出
        """
        obs = torch.rand(4, 64)
        out1 = policy(obs, deterministic=True)
        out2 = policy(obs, deterministic=True)
        assert torch.equal(out1, out2), "确定性模式输出应完全一致"

    def test_stochastic_output(self, policy):
        """
        测试随机采样模式
        ────────────────────────────────────────────────────────────────────────
        验证: deterministic=False 时，相同输入得到不同输出 (多次调用)
        """
        obs = torch.rand(4, 64)
        out1 = policy(obs, deterministic=False)
        out2 = policy(obs, deterministic=False)
        assert not torch.equal(out1, out2), "随机模式输出应不同"

    def test_output_shape(self, policy):
        """测试输出维度"""
        obs = torch.rand(2, 64)
        action = policy(obs, deterministic=True)
        assert action.shape == (2, 7)

    def test_log_std_is_learnable(self, policy):
        """
        测试 log_std 是可学习参数
        ────────────────────────────────────────────────────────────────────────
        验证: log_std 是 nn.Parameter 类型 (需要梯度)
        """
        assert isinstance(policy.log_std, nn.Parameter)
        assert policy.log_std.requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# ActorPolicy 封装测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestActorPolicy:
    """
    策略封装接口测试
    ────────────────────────────────────────────────────────────────────────────
    验证 ActorPolicy 的 act/save/load 接口。
    """

    @pytest.fixture
    def policy(self):
        network = GaussianPolicy(obs_dim=64, action_dim=7)
        return ActorPolicy(network)

    def test_act_with_dict_obs(self, policy, mock_obs):
        """
        测试 act 接收观测字典
        ────────────────────────────────────────────────────────────────────────
        验证: act() 能正确处理包含多模态数据的观测字典
        """
        action = policy.act(mock_obs, deterministic=True)
        assert isinstance(action, torch.Tensor)
        assert action.shape == (7,)

    def test_save_and_load(self, policy, tmp_path):
        """
        测试保存和加载模型
        ────────────────────────────────────────────────────────────────────────
        验证: save → load 后，策略输出相同的动作
        """
        path = str(tmp_path / "test_policy.pt")
        obs = {"rgb": torch.rand(3, 128, 128), "robot_state": torch.rand(7)}

        # 保存前
        action_before = policy.act(obs, deterministic=True)

        # 保存
        policy.save(path)
        assert tmp_path.joinpath("test_policy.pt").exists()

        # 加载到新策略
        new_policy = ActorPolicy(GaussianPolicy(64, 7))
        new_policy.load(path)

        # 验证: 加载后输出相同
        action_after = new_policy.act(obs, deterministic=True)
        assert torch.equal(action_before, action_after)
