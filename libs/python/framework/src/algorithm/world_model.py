"""
================================================================================
 世界模型模块 (world_model.py)
 ──────────────────────────────────────────────────────────────────────────────
 世界模型是一种基于模型的强化学习方法，通过学习环境的动力学模型，
 在"想象"中进行策略训练和规划。

 核心思想:
   世界模型 = 观测编码器 + 动力学预测器 + 奖励预测器
     - 观测编码器: 将高维观测 (图像) 压缩为隐状态 z_t
     - 动力学预测器: 预测 z_{t+1} = f(z_t, a_t)
     - 奖励预测器: 预测 r_t = R(z_t, a_t)

 优势:
   - 样本效率高: 在模型想象的轨迹中学习，不需要大量真实交互
   - 规划能力: 训练后可以在想象中进行规划 (如 Dreamer 的 CEM 规划)
   - 可解释性: 隐状态可以可视化理解

 代表性算法:
   - DreamerV3: 纯基于世界模型的强化学习 (Hafner et al., 2023)
   - TD-MPC2: 时差模型预测控制 (Hansen et al., 2023)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Tuple, Any

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.algorithm.policy import MLP, CNNEncoder


class RSSM(nn.Module):
    """
    循环状态空间模型 (Recurrent State-Space Model)
    ────────────────────────────────────────────────────────────────────────────
    Dreamer 系列算法的核心组件，将观测序列编码为隐状态序列。

    结构:
      确定状态: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
      随机状态: z_t ~ N(μ_t, σ_t), 其中 μ_t, σ_t = f(h_t)

    为什么用 RSSM?
      RSSM 结合了确定性和随机性两种状态:
      - 确定性状态 (h_t): 通过 GRU 捕获长期的时序依赖
      - 随机状态 (z_t): 通过高斯分布建模观测的随机性
      这种混合设计比纯随机状态更能捕获复杂时序模式。
    """
    def __init__(self, state_dim: int = 256, action_dim: int = 8,
                 hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── GRU 单元: 维护确定性状态 h_t ──────────────────────────────────
        # 输入: [z_{t-1}, a_{t-1}], 隐状态: h_{t-1}
        self.gru = nn.GRUCell(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
        )

        # ── 随机状态先验: p(z_t | h_t) ─────────────────────────────────────
        self.prior_net = nn.Linear(hidden_dim, state_dim * 2)  # μ, log_σ

        # ── 随机状态后验: q(z_t | h_t, x_t) ────────────────────────────────
        # 后验在编码观测 x_t 后修正先验
        self.posterior_net = nn.Linear(hidden_dim + state_dim, state_dim * 2)

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor],
                action: torch.Tensor) -> Tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        状态转移: (h_{t-1}, z_{t-1}, a_{t-1}) → (h_t, z_t)

        Args:
            state: (h, z) 上一时刻的确定性状态和随机状态
            action: 动作 a_{t-1}

        Returns:
            (h, z, prior_mean, prior_std): 新状态和先验分布
        """
        h_prev, z_prev = state

        # ── GRU 更新 ────────────────────────────────────────────────────────
        gru_input = torch.cat([z_prev, action], dim=-1)
        h = self.gru(gru_input, h_prev)

        # ── 先验分布 ────────────────────────────────────────────────────────
        prior_params = self.prior_net(h)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = prior_log_std.clamp(-5, 2).exp()

        # ── 采样 z_t ────────────────────────────────────────────────────────
        z = prior_mean + prior_std * torch.randn_like(prior_mean)

        return h, z, prior_mean, prior_std


class WorldModel(nn.Module):
    """
    世界模型 (WorldModel)
    ────────────────────────────────────────────────────────────────────────────
    完整的世界模型，包含三个子模块:
      1. 观测编码器: Encoder(x_t) → z_t
      2. RSSM 动力学: RSSM(z_t, a_t) → z_{t+1}
      3. 观测解码器/奖励预测器: 从隐状态重建观测和预测奖励
    """
    def __init__(self, obs_dim: int, action_dim: int,
                 state_dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        # ── 观测编码器 ──────────────────────────────────────────────────────
        self.encoder = nn.Linear(obs_dim, state_dim)

        # ── 循环状态空间模型 ────────────────────────────────────────────────
        self.rssm = RSSM(state_dim, action_dim, hidden_dim)

        # ── 观测解码器 (重建观测) ──────────────────────────────────────────
        self.decoder = nn.Linear(state_dim, obs_dim)

        # ── 奖励预测器 ──────────────────────────────────────────────────────
        self.reward_pred = MLP([state_dim, hidden_dim, 1])

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> Dict:
        """
        前向传播: 在观测序列上编码、预测、重建

        Args:
            obs_seq: 观测序列, (B, T, obs_dim)
            action_seq: 动作序列, (B, T, action_dim)

        Returns:
            包含所有预测结果的字典
        """
        B, T = obs_seq.shape[:2]

        # 初始化状态
        h = torch.zeros(B, self.rssm.gru.hidden_size, device=obs_seq.device)
        z = torch.zeros(B, self.rssm.state_dim, device=obs_seq.device)

        obs_preds = []
        reward_preds = []
        prior_means = []
        prior_stds = []

        for t in range(T):
            # 编码当前观测 (后验)
            z_encoded = self.encoder(obs_seq[:, t])

            # 状态转移 (先验)
            h, z_prior, p_mean, p_std = self.rssm((h, z), action_seq[:, t])
            prior_means.append(p_mean)
            prior_stds.append(p_std)

            # 解码
            obs_pred = self.decoder(z_prior)
            rew_pred = self.reward_pred(z_prior)
            obs_preds.append(obs_pred)
            reward_preds.append(rew_pred)

            # 使用编码的 z 作为下一时刻输入 (Teacher Forcing)
            z = z_encoded

        return {
            "obs_pred": torch.stack(obs_preds, dim=1),
            "reward_pred": torch.stack(reward_preds, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
        }
