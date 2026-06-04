"""
================================================================================
 行为克隆算法 (bc.py) — Behavior Cloning
 ──────────────────────────────────────────────────────────────────────────────
 行为克隆是最基础的模仿学习方法，核心思想非常简单:
 将策略学习视为监督学习问题: 观测 → 动作的映射。

 公式:
   min_θ E_{(o,a)~D} [ L( π_θ(o), a ) ]
   其中:
     - D: 专家数据集
     - π_θ: 参数为 θ 的策略网络
     - L: 损失函数 (MSE for 连续动作, CrossEntropy for 离散动作)

 局限:
   - 协变量偏移 (Covariate Shift): 训练时观测来自专家分布，
     但推理时策略会访问到偏离分布的状态，导致误差累积
   - 不能处理多峰动作分布 (对同一个观测，专家可能有多种合理动作)

 改进方向:
   使用 DAgger (Dataset Aggregation) 算法，在策略执行时重新采集
   专家纠正数据，缓解协变量偏移问题。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import ImitationTrainer
from src.algorithm.policy import MLP, GaussianPolicy


class BCTrainer(ImitationTrainer):
    """
    行为克隆训练器 (BCTrainer)
    ────────────────────────────────────────────────────────────────────────────
    实现最基础的行为克隆算法训练循环。

    训练流程:
      1. 从 DataLoader 获取 batch (观测, 专家动作)
      2. 策略网络前向推理，得到预测动作
      3. 计算预测与专家动作之间的损失 (MSE)
      4. 反向传播更新策略网络参数
      5. 记录训练指标
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy: Optional[GaussianPolicy] = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        """
        Args:
            obs_dim: 观测特征维度
            action_dim: 动作维度
            policy: 策略网络 (None 时自动创建)
            lr: 学习率
            device: 训练设备
        """
        self.device = torch.device(device)

        # ── 策略网络 ────────────────────────────────────────────────────────
        if policy is None:
            self.policy = GaussianPolicy(obs_dim, action_dim).to(self.device)
        else:
            self.policy = policy.to(self.device)

        # ── 优化器 ──────────────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-5,  # L2 正则化，防止过拟合
        )

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 提供专家轨迹的 DataLoader
                batch 格式: {"obs": {"rgb": ..., "robot_state": ...}, "action": ...}

        Returns:
            训练指标字典
        """
        self.policy.train()  # 切换到训练模式
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # ── 从 batch 中提取数据 ─────────────────────────────────────────
            # 组装观测特征
            obs_feat = self._extract_obs_features(batch)
            expert_actions = batch["action"].to(self.device)

            # ── 前向传播 ────────────────────────────────────────────────────
            # 训练时使用采样模式 (deterministic=False)，引入探索
            pred_actions = self.policy(obs_feat, deterministic=False)

            # ── 损失计算: Mean Squared Error ────────────────────────────────
            # 对于连续动作空间，MSE 是最常用的损失函数
            loss = F.mse_loss(pred_actions, expert_actions)

            # ── 反向传播 ────────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪: 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            "bc_loss": total_loss / max(num_batches, 1),
            "num_batches": num_batches,
        }

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        验证集评估
        ────────────────────────────────────────────────────────────────────────
        与训练的区别:
          - 不计算梯度 (torch.no_grad)
          - 使用确定性模式推理 (取均值)
          - 不更新网络参数
        """
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                obs_feat = self._extract_obs_features(batch)
                expert_actions = batch["action"].to(self.device)

                # 验证时使用确定性模式
                pred_actions = self.policy(obs_feat, deterministic=True)
                loss = F.mse_loss(pred_actions, expert_actions)

                total_loss += loss.item()
                num_batches += 1

        return {
            "val_bc_loss": total_loss / max(num_batches, 1),
        }

    def _extract_obs_features(self, batch: Dict) -> torch.Tensor:
        """
        从 batch 中提取并拼接观测特征
        ────────────────────────────────────────────────────────────────────────
        将多模态观测 (图像、本体感知) 展平并拼接为特征向量。

        Args:
            batch: 数据 batch

        Returns:
            特征张量, shape=(B, obs_dim)
        """
        features = []

        # 提取 RGB 特征 (展平)
        if "rgb" in batch.get("obs", {}):
            rgb = batch["obs"]["rgb"].to(self.device)
            features.append(rgb.flatten(1))

        # 提取机器人状态
        if "robot_state" in batch.get("obs", {}):
            state = batch["obs"]["robot_state"].to(self.device).float()
            features.append(state)

        if features:
            return torch.cat(features, dim=-1)

        # 如果 batch 已经是展平后的特征
        if "obs" in batch and isinstance(batch["obs"], torch.Tensor):
            return batch["obs"].to(self.device)

        raise ValueError("Batch 中未找到观测数据")
