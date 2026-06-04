"""
================================================================================
 SAC 算法实现 (sac.py) — Soft Actor-Critic
 ──────────────────────────────────────────────────────────────────────────────
 SAC 是一种基于最大熵 (Maximum Entropy) 框架的 Off-policy 强化学习算法。

 核心思想:
   在标准 RL 目标 (最大化累积奖励) 之外，同时最大化策略的熵:
     J(π) = Σ E[ r(s,a) + α * H(π(·|s)) ]
   这鼓励策略在多个好动作之间保持随机性，提升探索效率。

 算法特点:
   - Off-policy: 使用经验回放缓冲，样本利用效率高
   - 最大熵: 鼓励探索，避免过早收敛到局部最优
   - 自动调节温度 α: 自适应调整熵的重要性权重

 网络结构 (共 5 个网络):
   - Actor (策略): π_φ(s) → 动作
   - Critic 1 & 2: Q_θ1(s,a), Q_θ2(s,a) — 两个 Q 网络缓解过估计
   - Target Critic 1 & 2: 目标 Q 网络 (软更新)

 参考文献:
   "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
   (Haarnoja et al., 2018)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Tuple, Any
from collections import deque

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import RLTrainer, VecEnv
from src.algorithm.policy import MLP, GaussianPolicy


# ═══════════════════════════════════════════════════════════════════════════════
# Q 网络 (Critic)
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleQNetwork(nn.Module):
    """
    双重 Q 网络 (Double Q-Network)

    SAC 使用两个 Q 网络来缓解价值函数的高估偏差:
      - 目标 Q 值 = min(Q1(s,a), Q2(s,a))
      - 两个网络独立训练，取最小值作为目标
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 网络
        self.q1 = MLP([obs_dim + action_dim, hidden_dim, hidden_dim, 1])
        # Q2 网络
        self.q2 = MLP([obs_dim + action_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: 观测, (B, obs_dim)
            action: 动作, (B, action_dim)

        Returns:
            (q1, q2): 两个 Q 网络的输出
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 经验回放缓冲 (Replay Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    经验回放缓冲 (ReplayBuffer)
    ────────────────────────────────────────────────────────────────────────────
    存储过去的经验元组 (s, a, r, s', d)，供 Off-policy 算法采样训练。

    为什么需要回放缓冲?
      - 打破样本间的时间相关性 (i.i.d. 假设)
      - 提高数据利用效率 (一条经验可以多次使用)
      - 稳定训练过程
    """
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """存储一条经验"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """随机采样一个 batch"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in batch]

        obs = torch.stack([s[0] for s in samples]).to(device)
        action = torch.stack([s[1] for s in samples]).to(device)
        reward = torch.tensor([s[2] for s in samples], device=device, dtype=torch.float32)
        next_obs = torch.stack([s[3] for s in samples]).to(device)
        done = torch.tensor([s[4] for s in samples], device=device, dtype=torch.float32)

        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        }

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# SAC 训练器
# ═══════════════════════════════════════════════════════════════════════════════

class SACTrainer(RLTrainer):
    """
    SAC 训练器
    ────────────────────────────────────────────────────────────────────────────

    SAC 的更新流程:
      1. 从回放缓冲采样 batch
      2. 通过目标 Q 网络计算目标值:
         y = r + γ * (min(Q1'(s',a'), Q2'(s',a')) - α * log π(a'|s'))
      3. 更新 Q 网络: minimize (Q(s,a) - y)²
      4. 更新策略: maximize E[ min(Q1,Q2) - α * log π(a|s) ]
      5. 软更新目标网络: θ' ← τ*θ + (1-τ)*θ'
      6. (可选) 更新温度 α
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,       # 软更新系数
        alpha: float = 0.2,        # 初始温度
        auto_alpha: bool = True,   # 自动调节温度
        target_entropy: Optional[float] = None,
        batch_size: int = 256,
        replay_capacity: int = 1_000_000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # ── 策略网络 (Actor) ────────────────────────────────────────────────
        self.actor = GaussianPolicy(obs_dim, action_dim, [hidden_dim, hidden_dim])
        self.actor.to(self.device)

        # ── Q 网络 (Critic) × 2 ────────────────────────────────────────────
        self.q_net = DoubleQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = DoubleQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        # 硬拷贝 Q 参数到目标网络
        self.q_target.load_state_dict(self.q_net.state_dict())

        # ── 优化器 ──────────────────────────────────────────────────────────
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # ── 自动温度调节 ────────────────────────────────────────────────────
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.alpha = alpha

        # ── 经验回放缓冲 ────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(replay_capacity)

    def collect(self, env: VecEnv, num_steps: int) -> Dict[str, Any]:
        """
        从环境中收集经验并存入回放缓冲
        """
        obs = env.reset()
        total_reward = 0.0
        num_episodes = 0

        for _ in range(num_steps):
            # ── 策略推理 ────────────────────────────────────────────────────
            obs_tensor = torch.stack([o["robot_state"] for o in obs]).to(self.device)

            with torch.no_grad():
                # SAC 在训练时从策略分布采样 (探索)
                action = self.actor(obs_tensor, deterministic=False)

            # ── 环境步进 ────────────────────────────────────────────────────
            next_obs, rewards, dones, infos = env.step(action.cpu())

            # ── 存入回放缓冲 ────────────────────────────────────────────────
            for i in range(env.num_envs):
                self.replay_buffer.push(
                    obs_tensor[i].cpu(),
                    action[i].cpu(),
                    rewards[i].item(),
                    torch.tensor(next_obs[i]["robot_state"]),
                    dones[i].item(),
                )

            total_reward += rewards.sum().item()
            num_episodes += dones.sum().item()
            obs = next_obs

        return {
            "total_reward": total_reward,
            "num_episodes": num_episodes,
            "buffer_size": len(self.replay_buffer),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        SAC 单步更新 (从回放缓冲采样 batch)

        Returns:
            训练指标字典
        """
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # ══════════════════════════════════════════════════════════════════════
        # 1. 更新 Q 网络
        # ══════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            # 下一状态的动作 (来自当前策略)
            next_action = self.actor(next_obs, deterministic=False)
            # 目标 Q 值: r + γ * (min(Q') - α * log π)
            next_q1, next_q2 = self.q_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2)

            # 策略熵: -log π(a'|s')
            # 由于 GaussianPolicy 直接输出带噪声的动作, 计算对数概率
            dist = torch.distributions.Normal(
                self.actor.mean_net(next_obs),
                self.actor.log_std.exp().clamp(1e-6)
            )
            log_prob = dist.log_prob(next_action).sum(dim=-1)

            # Q 目标: y = r + γ * (1-d) * (min Q' - α * log π')
            target_q = reward + self.gamma * (1 - done) * (next_q - self.alpha * log_prob)

        # 当前 Q 值
        current_q1, current_q2 = self.q_net(obs, action)

        # Q 损失: MSE
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # ══════════════════════════════════════════════════════════════════════
        # 2. 更新策略网络 (Actor)
        # ══════════════════════════════════════════════════════════════════════
        # SAC 策略目标: maximize E[ min(Q1,Q2) - α * log π(a|s) ]
        new_action = self.actor(obs, deterministic=False)
        new_q1, new_q2 = self.q_net(obs, new_action)
        new_q = torch.min(new_q1, new_q2)

        # 策略熵
        dist = torch.distributions.Normal(
            self.actor.mean_net(obs),
            self.actor.log_std.exp().clamp(1e-6)
        )
        log_prob = dist.log_prob(new_action).sum(dim=-1)

        # 策略损失: -E[ min Q - α * log π ]
        actor_loss = -(new_q - self.alpha * log_prob).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ══════════════════════════════════════════════════════════════════════
        # 3. 软更新目标网络
        # ══════════════════════════════════════════════════════════════════════
        # θ' ← τ * θ + (1-τ) * θ'
        for target_param, param in zip(
            self.q_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # ══════════════════════════════════════════════════════════════════════
        # 4. (可选) 更新温度 α
        # ══════════════════════════════════════════════════════════════════════
        metrics = {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha_loss"] = alpha_loss.item()

        return metrics

    def evaluate(self, env: VecEnv, num_episodes: int) -> Dict[str, float]:
        """评估策略 (使用确定性模式)"""
        self.actor.eval()
        total_reward = 0.0

        for _ in range(num_episodes):
            obs = env.reset()
            done = torch.zeros(env.num_envs)
            episode_reward = 0.0

            while not done.all():
                with torch.no_grad():
                    obs_tensor = torch.stack([o["robot_state"] for o in obs]).to(self.device)
                    action = self.actor(obs_tensor, deterministic=True)
                    obs, rewards, dones, _ = env.step(action.cpu())

                episode_reward += rewards.sum().item()
                done = torch.logical_or(done, dones.to(done.device))

            total_reward += episode_reward

        return {"eval_avg_reward": total_reward / max(num_episodes, 1)}
