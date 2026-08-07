"""
================================================================================
 策略网络模块 (policy.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 定义策略网络的神经网络架构基类和通用组件。

 包含:
   1. Policy 抽象类 (已在 interfaces.py 中定义核心接口)
   2. 策略网络工厂 (根据配置创建不同架构的策略)
   3. 通用神经网络组件 (MLP, CNN, Transformer 编码器)
   4. 策略封装: 将网络 + 推理逻辑封装为策略对象

 设计思路:
   策略网络负责将观测映射为动作。不同算法使用的网络架构不同:
     - BC: 简单的前馈网络 (MLP)
     - ACT: Transformer 编码器-解码器
     - Diffusion Policy: U-Net + 噪声预测网络
   本模块提供这些架构的 PyTorch 实现。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple, Type, Any
from abc import ABC, abstractmethod

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import Policy as PolicyInterface


# ═══════════════════════════════════════════════════════════════════════════════
# 通用神经网络组件
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    多层感知机 (MLP) 组件
    ────────────────────────────────────────────────────────────────────────────
    通用的前馈神经网络模块，用于策略网络中的特征提取和动作预测。

    结构:
      输入 → Linear → LayerNorm → ReLU → Dropout → ... → Linear → 输出

    参数:
      - layer_dims: [输入维度, 隐藏层1, ..., 输出维度]
      - activation: 激活函数
      - dropout: Dropout 概率 (防止过拟合)
    """
    def __init__(
        self,
        layer_dims: List[int],
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        """
        Args:
            layer_dims: 每层维度列表，如 [128, 256, 256, 32]
            activation: 激活函数类型
            dropout: Dropout 概率 (0 表示不启用)
            use_norm: 是否使用 LayerNorm
        """
        super().__init__()
        layers = []

        for i in range(len(layer_dims) - 1):
            # ── 线性层: Wx + b ──────────────────────────────────────────────
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # ── LayerNorm: 稳定训练 ─────────────────────────────────────────
            if use_norm and i < len(layer_dims) - 2:  # 输出层前不接 norm
                layers.append(nn.LayerNorm(layer_dims[i + 1]))

            # ── 激活函数 ────────────────────────────────────────────────────
            if i < len(layer_dims) - 2:  # 输出层前不接激活
                layers.append(activation())

            # ── Dropout: 随机失活，防止过拟合 ────────────────────────────────
            if dropout > 0 and i < len(layer_dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


class CNNEncoder(nn.Module):
    """
    卷积神经网络编码器 (CNNEncoder)
    ────────────────────────────────────────────────────────────────────────────
    将 RGB 图像观测编码为特征向量。

    结构 (类似 ResNet 的简化版):
      Conv2D → BatchNorm → ReLU → MaxPool → ... → AdaptiveAvgPool → Linear

    为什么需要 CNN?
      机器人任务中，图像是高维输入 (128x128x3 = 49152 维)，
      直接使用 MLP 参数量太大且没有空间不变性。
      CNN 通过局部连接和权值共享高效提取视觉特征。
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        """
        Args:
            in_channels: 输入图像通道数 (RGB=3, 灰度=1)
            latent_dim: 输出特征维度
        """
        super().__init__()

        # ── 特征提取器 ──────────────────────────────────────────────────────
        # 4 层卷积 + 池化，逐步降低空间维度、增加通道数
        self.features = nn.Sequential(
            # 输入: (3, 128, 128)
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 输出: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输出: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 输出: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 输出: (256, 8, 8)
        )

        # ── 自适应池化 → 固定大小 → 线性映射到 latent_dim ─────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: (256, 1, 1)
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像为特征向量
        Args:
            x: 图像张量, shape=(B, C, H, W)
        Returns:
            特征向量, shape=(B, latent_dim)
        """
        features = self.features(x)          # (B, 256, 8, 8)
        pooled = self.pool(features)         # (B, 256, 1, 1)
        flattened = pooled.flatten(1)        # (B, 256)
        latent = self.fc(flattened)          # (B, latent_dim)
        return latent


class GaussianPolicy(nn.Module):
    """
    高斯策略网络 (GaussianPolicy)
    ────────────────────────────────────────────────────────────────────────────
    输出动作的均值和标准差，适用于连续动作空间的 RL 和 IL。

    原理:
      策略输出一个高斯分布 N(μ, σ)，训练时从分布采样以探索，
      评估时直接取均值 μ (确定性模式)。

    公式:
      μ = mean_net(obs)
      σ = log_std_net(obs) → clamp → exp
      action = μ + σ * noise  (noise ~ N(0,1))
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # ── 均值网络 ─────────────────────────────────────────────────────────
        self.mean_net = MLP(
            [obs_dim] + hidden_dims + [action_dim],
            dropout=0.0,  # 策略网络通常不用 dropout
        )

        # ── 对数标准差 (可学习参数) ─────────────────────────────────────────
        # 与均值网络输出相同维度，代表每个动作维度的独立标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Args:
            obs: 观测特征, shape=(B, obs_dim)
            deterministic: True=取均值, False=采样
        Returns:
            动作, shape=(B, action_dim)
        """
        mean = self.mean_net(obs)                             # (B, action_dim)
        log_std = self.log_std.clamp(-20, 2)                  # 限制范围
        std = log_std.exp()                                    # (action_dim,)

        if deterministic:
            return mean

        # 重参数化采样: action = μ + σ * ε, ε ~ N(0,1)
        noise = torch.randn_like(mean)
        return mean + std * noise


# ═══════════════════════════════════════════════════════════════════════════════
# 策略封装 (Policy Wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class ActorPolicy(PolicyInterface):
    """
    策略封装类 (ActorPolicy)
    ────────────────────────────────────────────────────────────────────────────
    将 PyTorch nn.Module 包装为 Policy 接口，提供统一的 act/save/load API。

    设计考虑:
      底层的神经网络 (GaussianPolicy, CNNEncoder 等) 是 nn.Module，
      但上层算法需要的是 Policy 接口 (有 act/save/load 方法)。
      这个封装类起到"适配器"的作用。
    """
    def __init__(self, actor_network: nn.Module, device: str = "cpu"):
        self.network = actor_network
        self.device = torch.device(device)

    def act(self, obs: Dict[str, torch.Tensor],
            deterministic: bool = False) -> torch.Tensor:
        """
        根据观测输出动作

        流程:
          1. 将观测中的各模态数据编码为统一特征
          2. (可选) 融合多模态特征
          3. 通过策略网络输出动作

        Args:
            obs: 观测字典 (可能包含 "rgb", "robot_state" 等)
            deterministic: 是否确定性推理

        Returns:
            动作张量
        """
        # ── 提取特征 ────────────────────────────────────────────────────────
        features = []
        rgb = obs.get("rgb")
        if rgb is not None:
            # 通过 CNN 编码器提取视觉特征
            features.append(self._encode_image(rgb.to(self.device)))

        state = obs.get("robot_state")
        if state is not None:
            features.append(state.to(self.device).float())

        # 拼接所有特征
        if features:
            obs_feat = torch.cat(features, dim=-1)
        else:
            obs_feat = torch.tensor([], device=self.device)

        # ── 策略前向 ────────────────────────────────────────────────────────
        return self.network(obs_feat, deterministic=deterministic)

    def _encode_image(self, rgb: torch.Tensor) -> torch.Tensor:
        """编码图像特征 (如果网络中包含 CNN 编码器)"""
        # 检查网络是否有 image_encoder 属性
        if hasattr(self.network, "image_encoder"):
            return self.network.image_encoder(rgb)
        return rgb.flatten(1)  # 退化为全连接

    def save(self, path: str) -> None:
        """保存模型权重"""
        torch.save(self.network.state_dict(), path)
        print(f"[Policy] 模型已保存至: {path}")

    def load(self, path: str) -> None:
        """加载模型权重"""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()  # 切换到评估模式
        print(f"[Policy] 模型已加载: {path}")
