"""
================================================================================
 Action Chunking Transformer (act.py)
 ──────────────────────────────────────────────────────────────────────────────
 ACT (Action Chunking Transformer) 是由 Stanford 提出的一种模仿学习算法，
 旨在解决"协变量偏移"和"多峰动作分布"两个 BC 的核心问题。

 核心创新:
   1. Action Chunking (动作分块):
      传统 BC 预测单个时间步的动作。ACT 预测未来 T 个时间步的动作序列
      (chunk)。这使得策略具有时间上的连贯性，减少了抖动。

   2. CVAE (Conditional Variational Autoencoder):
      使用 CVAE 建模多峰动作分布。对于同一个观测，编码器从动作轨迹中
      提取风格变量 z，解码器结合 z 和观测生成多样化的动作。

   3. Transformer 架构:
      使用编码器-解码器 Transformer 处理时序依赖。

 网络结构:
   Encoder: 观测 → 特征 → CVAE 编码 (得到 z 分布)
   Decoder: 观测 + z → 动作序列 (chunk)

 参考文献:
   "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
   (Zhao et al., 2023)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Tuple

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import ImitationTrainer
from src.algorithm.policy import MLP


class ACTPolicy(nn.Module):
    """
    ACT 策略网络
    ────────────────────────────────────────────────────────────────────────────
    实现 Action Chunking Transformer 的核心网络。

    架构:
      ┌─────────┐   ┌──────────┐   ┌──────────┐
      │ 观测    │ → │ Encoder  │ → │  CVAE    │
      └─────────┘   └──────────┘   └────┬─────┘
                                         ↓ z
      ┌─────────┐   ┌──────────┐   ┌────┴─────┐
      │ 观测    │ → │ Decoder  │ ← │ z + obs   │
      └─────────┘   └──────────┘   └──────────┘
                         ↓
                    动作 chunk (T_steps × action_dim)
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int = 10,     # 预测未来多少步动作
        hidden_dim: int = 256,
        latent_dim: int = 32,     # CVAE 隐变量维度
        n_heads: int = 4,         # Transformer 注意力头数
    ):
        """
        Args:
            obs_dim: 观测特征维度
            action_dim: 单步动作维度
            chunk_size: 动作块大小 (预测步数)
            hidden_dim: 隐藏层维度
            latent_dim: CVAE 隐变量 z 的维度
            n_heads: 注意力头数
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # ── 观测编码器 ──────────────────────────────────────────────────────
        self.obs_encoder = MLP(
            [obs_dim, hidden_dim, hidden_dim],
        )

        # ── CVAE 编码器: 从 (观测, 动作轨迹) 提取隐变量 z ─────────────────
        # 输入: 观测特征 + 动作 chunk (展平)
        self.cvae_encoder = MLP(
            [hidden_dim + chunk_size * action_dim, hidden_dim, latent_dim * 2],
            # 输出 μ 和 log_std, 各 latent_dim 维
        )

        # ── 解码器 (Transformer): 观测 + z → 动作序列 ──────────────────────
        # 使用 Transformer 解码器处理时序
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2
        )

        # 动作头: 将解码器输出映射为动作
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # 位置编码 (为 Transformer 提供时序信息)
        self.pos_embedding = nn.Embedding(chunk_size, hidden_dim)

    def encode_latent(self, obs_feat: torch.Tensor,
                      action_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CVAE 编码: 从 (观测, 动作块) 提取隐变量 z 的分布参数

        Args:
            obs_feat: 观测特征, (B, hidden_dim)
            action_chunk: 专家动作块, (B, chunk_size * action_dim)

        Returns:
            (μ, log_std): 隐变量 z 的高斯分布参数
        """
        encoder_input = torch.cat([obs_feat, action_chunk], dim=-1)
        encoder_output = self.cvae_encoder(encoder_input)
        mu, log_std = encoder_output.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)  # 稳定化
        return mu, log_std

    def decode(self, obs_feat: torch.Tensor,
               z: torch.Tensor) -> torch.Tensor:
        """
        Transformer 解码: 观测 + z → 动作序列

        流程:
          1. 将观测特征与 z 拼接作为解码器的"记忆"
          2. 生成 chunk_size 个位置嵌入作为"查询"
          3. Transformer 解码器处理时序
          4. 动作头映射为动作

        Args:
            obs_feat: 观测特征, (B, hidden_dim)
            z: CVAE 隐变量, (B, latent_dim)

        Returns:
            动作 chunk, (B, chunk_size, action_dim)
        """
        B = obs_feat.shape[0]

        # ── 构建解码器输入 ──────────────────────────────────────────────────
        # 将 obs 和 z 融合为条件特征
        cond = torch.cat([obs_feat, z], dim=-1)
        cond_proj = nn.Linear(
            obs_feat.shape[-1] + z.shape[-1], obs_feat.shape[-1],
            device=obs_feat.device,
        )(cond).unsqueeze(1)  # (B, 1, hidden_dim)

        # 生成位置查询
        pos_ids = torch.arange(self.chunk_size, device=obs_feat.device)
        pos_emb = self.pos_embedding(pos_ids)  # (chunk_size, hidden_dim)
        query = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, hidden_dim)

        # ── Transformer 解码 ────────────────────────────────────────────────
        # memory = cond 作为 Transformer 的"记忆" (key/value)
        # tgt = query 作为"查询"序列
        decoded = self.transformer_decoder(
            tgt=query,
            memory=cond_proj.expand(-1, self.chunk_size, -1),
        )  # (B, chunk_size, hidden_dim)

        # ── 动作头 ──────────────────────────────────────────────────────────
        actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        return actions


class ACTTrainer(ImitationTrainer):
    """
    ACT 训练器 (ACTTrainer)
    ────────────────────────────────────────────────────────────────────────────
    实现 ACT 模型的训练循环。

    训练损失:
      L = L_recon (动作重建) + β * KL(CVAE 先验)
      其中 KL 散度约束隐变量 z 接近标准正态分布，起到正则化作用。
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int = 10,
        kl_weight: float = 0.001,  # β: KL 散度权重
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.kl_weight = kl_weight

        self.policy = ACTPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个 epoch

        每一步训练:
          1. 编码器从 (obs, 专家动作) 提取 z 的分布
          2. 从分布采样 z
          3. 解码器生成动作 chunk
          4. 计算重建损失 + KL 散度
        """
        self.policy.train()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # ── 提取观测和动作 ──────────────────────────────────────────────
            obs_feat = batch.get("obs_features", batch["obs"]).to(self.device)
            # 动作需要组织为 chunk 格式: (B, chunk_size, action_dim)
            actions = batch["action"].to(self.device)

            # ── CVAE 编码: 得到 z 分布 ─────────────────────────────────────
            action_flat = actions.flatten(1)  # (B, chunk_size * action_dim)
            mu, log_std = self.policy.encode_latent(obs_feat, action_flat)

            # ── 重参数化采样 z ──────────────────────────────────────────────
            std = log_std.exp()
            eps = torch.randn_like(std)
            z = mu + std * eps  # (B, latent_dim)

            # ── 解码: 生成动作 ──────────────────────────────────────────────
            pred_actions = self.policy.decode(obs_feat, z)

            # ── 损失计算 ────────────────────────────────────────────────────
            # 1. 重建损失: 预测动作与专家动作的 MSE
            recon_loss = F.mse_loss(pred_actions, actions)

            # 2. KL 散度: 约束 z 分布接近 N(0,1)
            # KL(N(μ,σ²) || N(0,1)) = -½(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.sum(
                1 + 2 * log_std - mu.pow(2) - (2 * log_std).exp()
            ) / mu.shape[0]

            # 总损失
            total_loss = recon_loss + self.kl_weight * kl_loss

            # ── 反向传播 ────────────────────────────────────────────────────
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        return {
            "act_recon_loss": total_recon_loss / num_batches,
            "act_kl_loss": total_kl_loss / num_batches,
        }

    def validate(self, dataloader) -> Dict[str, float]:
        """验证"""
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                obs_feat = batch["obs_features"].to(self.device)
                actions = batch["action"].to(self.device)

                # 验证时使用先验 z (从 N(0,1) 采样)
                z = torch.randn(obs_feat.shape[0], 32, device=self.device)
                pred_actions = self.policy.decode(obs_feat, z)

                loss = F.mse_loss(pred_actions, actions)
                total_loss += loss.item()
                num_batches += 1

        return {"val_act_loss": total_loss / num_batches}
