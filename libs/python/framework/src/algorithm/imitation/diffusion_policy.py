"""
================================================================================
 扩散策略 (diffusion_policy.py) — Diffusion Policy
 ──────────────────────────────────────────────────────────────────────────────
 扩散策略是近年来最先进的模仿学习方法之一，由 MIT/Columbia 提出。

 核心思想:
   将策略建模为一个条件去噪扩散过程:
   1. 前向过程: 从专家动作开始，逐步添加高斯噪声，最终变成纯噪声
   2. 反向过程: 从纯噪声开始，以观测为条件，逐步去噪恢复动作

 优势:
   - 能建模多峰动作分布: 对同一观测，能输出多种合理动作
   - 训练稳定: 不像 GAN 那样需要对抗训练
   - 高保真度: 生成的轨迹平滑且自然

 网络结构:
   使用 1D U-Net 或 Transformer 作为噪声预测网络 ε_θ(x_t, t, obs)

 参考文献:
   "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
   (Chi et al., 2023)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Callable
import math

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import ImitationTrainer
from src.algorithm.policy import MLP


# ═══════════════════════════════════════════════════════════════════════════════
# 扩散过程 (Diffusion Process)
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionScheduler:
    """
    扩散调度器 (DiffusionScheduler)
    ────────────────────────────────────────────────────────────────────────────
    管理扩散过程的噪声调度 (Noise Schedule)。

    原理:
      β_t (噪声方差): 控制每一步添加多少噪声
      α_t = 1 - β_t
      α̅_t = ∏_{s=1}^{t} α_s (累积保留系数)

    常用的调度策略:
      - Linear: β 从 0.0001 线性增加到 0.02
      - Cosine: β 按余弦函数变化 (更平滑)
      - Sigmoid: β 按 Sigmoid 函数变化
    """
    def __init__(self, num_steps: int = 100, schedule: str = "cosine"):
        """
        Args:
            num_steps: 扩散步数 (越多生成质量越高，但推理越慢)
            schedule: 调度策略 ("linear", "cosine", "sigmoid")
        """
        self.num_steps = num_steps

        if schedule == "linear":
            # β 从 0.0001 到 0.02 线性增长
            self.beta = torch.linspace(0.0001, 0.02, num_steps)
        elif schedule == "cosine":
            # 余弦调度: 起始和结束变化慢，中间变化快
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            f = torch.cos((steps / num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            self.beta = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 预计算扩散系数
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向过程: 从 x0 出发，经过 t 步扩散后得到 x_t

        公式:
          x_t = √(α̅_t) * x0 + √(1 - α̅_t) * ε

        Args:
            x0: 初始干净数据 (专家动作)
            t: 时间步 (0 ~ num_steps-1)

        Returns:
            (x_t, ε): 噪声数据和添加的噪声
        """
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1)

        noise = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise


# ═══════════════════════════════════════════════════════════════════════════════
# 噪声预测网络 (U-Net 风格的 1D CNN)
# ═══════════════════════════════════════════════════════════════════════════════

class NoisePredNet(nn.Module):
    """
    噪声预测网络 (U-Net 风格)
    ────────────────────────────────────────────────────────────────────────────
    预测: ε_θ(x_t, t, obs) → 估计添加的噪声 ε

    输入:
      - x_t: 噪声动作, (B, T, action_dim)
      - t: 时间步索引, (B,)
      - obs: 条件观测, (B, obs_dim)

    输出:
      - ε_hat: 估计的噪声, (B, T, action_dim)
    """
    def __init__(self, action_dim: int, obs_dim: int,
                 chunk_size: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.chunk_size = chunk_size

        # ── 时间编码: 将时间步 t 编码为特征向量 ────────────────────────────
        # 使用 sinusoidal 位置编码 (类似 Transformer)
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),  # Swish 激活函数 (扩散策略常用)
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 观测编码器 ──────────────────────────────────────────────────────
        self.obs_encoder = MLP(
            [obs_dim, hidden_dim, hidden_dim],
            activation=nn.SiLU,
        )

        # ── 1D U-Net 编码器 ─────────────────────────────────────────────────
        # 使用 1D 卷积处理动作序列的时序模式
        self.conv_in = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)

        # FiLM (Feature-wise Linear Modulation) 层: 用 obs + time 调制特征
        self.film = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                obs_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: 噪声动作, (B, chunk_size, action_dim)
            t: 时间步, (B,)
            obs_feat: 观测特征, (B, obs_dim)

        Returns:
            ε_hat: 预测噪声, (B, chunk_size, action_dim)
        """
        B = x_t.shape[0]

        # ── 时间嵌入 ────────────────────────────────────────────────────────
        t_emb = self._sinusoidal_embedding(t, 128)   # (B, 128)
        t_feat = self.time_embed(t_emb)               # (B, hidden_dim)

        # ── 观测嵌入 ────────────────────────────────────────────────────────
        o_feat = self.obs_encoder(obs_feat)           # (B, hidden_dim)

        # ── 条件特征 (时间 + 观测) ──────────────────────────────────────────
        cond = torch.cat([t_feat, o_feat], dim=-1)    # (B, hidden_dim*2)

        # ── U-Net 前向 ──────────────────────────────────────────────────────
        # 1D 卷积: (B, action_dim, chunk_size) → (B, hidden_dim, chunk_size)
        x = x_t.permute(0, 2, 1)                      # (B, action_dim, T)
        h = self.conv_in(x)                            # (B, hidden_dim, T)

        # FiLM 调制: γ * h + β
        gamma, beta = self.film(cond).chunk(2, dim=-1)  # (B, hidden_dim)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = F.silu(h)

        # 中间层
        h = self.conv_mid(h)
        h = F.silu(h)

        # 输出层
        noise = self.conv_out(h)                       # (B, action_dim, T)
        noise = noise.permute(0, 2, 1)                 # (B, T, action_dim)

        return noise

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal 时间编码"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 扩散策略训练器
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionPolicyTrainer(ImitationTrainer):
    """
    扩散策略训练器
    ────────────────────────────────────────────────────────────────────────────
    训练噪声预测网络 ε_θ 来预测添加的噪声。

    训练损失 (简单形式):
      L = E_{t, x0, ε} [ || ε - ε_θ(√(α̅_t)*x0 + √(1-α̅_t)*ε, t, obs) ||² ]

    即在随机选择的时间步 t 上，最小化预测噪声与真实噪声的 MSE。
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        num_diffusion_steps: int = 100,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # ── 扩散调度器 ──────────────────────────────────────────────────────
        self.scheduler = DiffusionScheduler(
            num_steps=num_diffusion_steps,
            schedule="cosine",
        )

        # ── 噪声预测网络 ────────────────────────────────────────────────────
        self.noise_net = NoisePredNet(
            action_dim=action_dim,
            obs_dim=obs_dim,
            chunk_size=chunk_size,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.noise_net.parameters(), lr=lr
        )

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个 epoch

        每次迭代:
          1. 从 batch 中取专家动作 x0
          2. 随机采样时间步 t ~ Uniform(0, T)
          3. 前向过程: x_t = √(α̅_t) * x0 + √(1 - α̅_t) * ε
          4. 预测噪声: ε_hat = ε_θ(x_t, t, obs)
          5. 损失: || ε - ε_hat ||²
        """
        self.noise_net.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # ── 准备数据 ────────────────────────────────────────────────────
            obs_feat = batch["obs_features"].to(self.device)
            x0 = batch["action"].to(self.device)  # 干净专家动作

            B = x0.shape[0]

            # ── 随机采样时间步 ──────────────────────────────────────────────
            t = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)

            # ── 前向扩散: 添加噪声 ──────────────────────────────────────────
            x_t, noise = self.scheduler.add_noise(x0, t)

            # ── 预测噪声 ────────────────────────────────────────────────────
            noise_pred = self.noise_net(x_t, t, obs_feat)

            # ── 损失: 预测噪声与真实噪声的 MSE ──────────────────────────────
            loss = F.mse_loss(noise_pred, noise)

            # ── 反向传播 ────────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.noise_net.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"diffusion_loss": total_loss / num_batches}

    def validate(self, dataloader) -> Dict[str, float]:
        """验证"""
        self.noise_net.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                obs_feat = batch["obs_features"].to(self.device)
                x0 = batch["action"].to(self.device)
                B = x0.shape[0]
                t = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)
                x_t, noise = self.scheduler.add_noise(x0, t)
                noise_pred = self.noise_net(x_t, t, obs_feat)
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()

        return {"val_diffusion_loss": total_loss / len(dataloader)}

    @torch.no_grad()
    def sample(self, obs_feat: torch.Tensor,
               num_steps: Optional[int] = None) -> torch.Tensor:
        """
        推理: 从纯噪声开始，逐步去噪生成动作

        Args:
            obs_feat: 观测特征, (B, obs_dim)
            num_steps: 去噪步数 (默认使用训练时步数)

        Returns:
            生成的动作, (B, chunk_size, action_dim)

        去噪过程 (DDPM 采样):
          x_T ~ N(0, I) → x_{T-1} → ... → x_0
          每一步: x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-α̅_t) * ε_θ(x_t, t))
        """
        self.noise_net.eval()
        num_steps = num_steps or self.scheduler.num_steps

        B = obs_feat.shape[0]
        chunk_size = self.noise_net.chunk_size
        action_dim = self.noise_net.conv_out.out_channels

        # ── 从纯噪声开始 ────────────────────────────────────────────────────
        x_t = torch.randn(B, chunk_size, action_dim, device=self.device)

        # ── 逐步去噪 ────────────────────────────────────────────────────────
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.noise_net(x_t, t_tensor, obs_feat)

            # DDPM 去噪一步
            alpha = self.scheduler.alpha[t].to(self.device)
            alpha_bar = self.scheduler.alpha_bar[t].to(self.device)
            beta = self.scheduler.beta[t].to(self.device)

            # x_{t-1} = 1/√α_t * (x_t - β_t/√(1-α̅_t) * ε_θ)
            x_t = (1 / alpha.sqrt()) * (
                x_t - beta / (1 - alpha_bar).sqrt() * noise_pred
            )

            # 添加随机噪声 (t > 0 时)
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + beta.sqrt() * noise

        return x_t
