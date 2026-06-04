"""
================================================================================
 PPO 算法实现 (ppo.py) — Proximal Policy Optimization
 ──────────────────────────────────────────────────────────────────────────────
 PPO 是 OpenAI 提出的强化学习算法，在机器人控制领域广泛使用。

 核心思想:
   策略更新时限制每次更新的步长，避免策略"崩溃"。
   通过裁剪 (Clipping) 旧策略与新策略的比率来实现。

 算法伪代码:
   for iteration in range(max_iterations):
       # 1. 收集数据
       trajectories = collect_trajectories(env, policy, T)

       # 2. 计算优势函数
       advantages = compute_gae(trajectories, value_net)

       # 3. 优化策略 (多 epoch)
       for epoch in range(K):
           for minibatch in trajectories:
               # 计算裁剪后的策略损失
               ratio = π_θ(a|s) / π_θ_old(a|s)
               L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
               # 更新策略和价值网络

 参考文献:
   "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple, Any
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
# PPO 网络 (Actor-Critic)
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络 (PPO 使用)

    包含两个子网络:
      - Actor (策略): π(a|s) → 动作分布
      - Critic (价值): V(s) → 状态价值估计
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # ── 共享特征提取器 ──────────────────────────────────────────────────
        self.feature_extractor = MLP(
            [obs_dim, hidden_dim, hidden_dim],
            dropout=0.0,
        )

        # ── Actor: 输出动作分布参数 ────────────────────────────────────────
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic: 输出状态价值 V(s) ──────────────────────────────────────
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: 观测, (B, obs_dim)
        Returns:
            (action, log_prob, value)
        """
        features = self.feature_extractor(obs)

        # Actor: 高斯策略
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(-20, 2)
        std = log_std.exp()

        # 重参数化采样
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Critic
        value = self.critic(features).squeeze(-1)

        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率和熵 (用于 PPO 更新)

        Args:
            obs: 观测, (B, obs_dim)
            action: 动作, (B, action_dim)
        Returns:
            (log_prob, entropy, value)
        """
        features = self.feature_extractor(obs)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(-20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(features).squeeze(-1)

        return log_prob, entropy, value


# ═══════════════════════════════════════════════════════════════════════════════
# 广义优势估计 (GAE)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """
    广义优势估计 (Generalized Advantage Estimation)

    原理:
      GAE 在 TD(λ) 和蒙特卡洛之间做插值:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
      其中 λ=0 时退化为 TD(0), λ=1 时退化为 MC。

    Args:
        rewards: 奖励序列, (T,)
        values: 价值估计, (T+1,) 包含最后一个状态的 V(s_{T+1})
        dones: 终止标志, (T,)
        gamma: 折扣因子
        gae_lambda: GAE λ 参数

    Returns:
        优势估计, (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    # 反向递推计算 GAE
    for t in reversed(range(T)):
        # TD 误差: δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    return advantages


# ═══════════════════════════════════════════════════════════════════════════════
# PPO 训练器
# ═══════════════════════════════════════════════════════════════════════════════

class PPOTrainer(RLTrainer):
    """
    PPO 训练器
    ────────────────────────────────────────────────────────────────────────────
    实现完整的 PPO 训练循环。

    超参数说明:
      - lr: 学习率 (通常 3e-4)
      - gamma: 折扣因子 (0.99)
      - gae_lambda: GAE λ (0.95)
      - clip_epsilon: PPO 裁剪阈值 (0.2)
      - value_coef: 价值损失权重 (0.5)
      - entropy_coef: 熵正则化权重 (0.01)
      - max_grad_norm: 梯度裁剪上限 (0.5)
      - update_epochs: 每次采样后更新轮数 (10)
      - batch_size: 小批量大小 (64)
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # ── 网络 ────────────────────────────────────────────────────────────
        self.ac = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def collect(self, env: VecEnv, num_steps: int) -> Dict[str, Any]:
        """
        从环境中收集 rollout 数据

        Args:
            env: 向量化环境
            num_steps: 每个环境收集的步数

        Returns:
            收集统计信息
        """
        obs_list, action_list, reward_list, done_list, value_list, log_prob_list = (
            [], [], [], [], [], []
        )

        obs = env.reset()
        total_reward = 0.0
        num_episodes = 0

        for _ in range(num_steps):
            # ── 策略推理 ────────────────────────────────────────────────────
            obs_tensor = torch.stack([o["robot_state"] for o in obs]).to(self.device)
            action, log_prob, value = self.ac(obs_tensor)

            # ── 环境步进 ────────────────────────────────────────────────────
            next_obs, rewards, dones, infos = env.step(action.cpu())

            # ── 存储 ────────────────────────────────────────────────────────
            obs_list.append(obs_tensor)
            action_list.append(action)
            reward_list.append(rewards.to(self.device))
            done_list.append(dones.to(self.device))
            value_list.append(value)
            log_prob_list.append(log_prob)

            total_reward += rewards.sum().item()
            num_episodes += dones.sum().item()
            obs = next_obs

        # ── 计算最后一个状态的价值 (用于 GAE) ─────────────────────────────
        last_obs = torch.stack([o["robot_state"] for o in obs]).to(self.device)
        _, _, last_value = self.ac(last_obs)

        # ── 转换为 Tensor ──────────────────────────────────────────────────
        rewards = torch.stack(reward_list)
        dones = torch.stack(done_list)
        values = torch.stack(value_list)
        actions = torch.stack(action_list)
        log_probs = torch.stack(log_prob_list)

        # ── 计算 GAE ────────────────────────────────────────────────────────
        values_aug = torch.cat([values, last_value.unsqueeze(0)])
        advantages = compute_gae(rewards, values_aug, dones, self.gamma, self.gae_lambda)
        returns = advantages + values

        # ── 存储到经验缓冲区 ────────────────────────────────────────────────
        self._buffer = {
            "obs": torch.cat(obs_list),
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

        return {
            "total_reward": total_reward,
            "num_episodes": num_episodes,
            "avg_reward": total_reward / max(num_episodes, 1),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO 单步更新

        PPO 的四个损失:
          1. 策略损失 (clipped surrogate objective)
          2. 价值损失 (MSE)
          3. 熵奖励 (鼓励探索)
        """
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # ── 归一化优势函数 (稳定训练) ──────────────────────────────────────
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 评估当前策略 ────────────────────────────────────────────────────
        log_probs, entropy, values = self.ac.evaluate(obs, actions)

        # ── 重要性采样比率 ──────────────────────────────────────────────────
        # r_t(θ) = π_θ(a|s) / π_{θ_old}(a|s)
        ratio = (log_probs - old_log_probs).exp()

        # ── 1. Clipped Surrogate Objective (策略损失) ───────────────────────
        # L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── 2. 价值损失 ─────────────────────────────────────────────────────
        value_loss = F.mse_loss(values, returns)

        # ── 3. 熵奖励 ───────────────────────────────────────────────────────
        entropy_loss = -entropy.mean()

        # ── 总损失 ──────────────────────────────────────────────────────────
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

    def train_on_buffer(self) -> Dict[str, float]:
        """
        在收集的 buffer 数据上执行多 epoch PPO 更新
        """
        if not hasattr(self, "_buffer"):
            raise RuntimeError("没有可用的 buffer，请先调用 collect()")

        buffer = self._buffer
        total_samples = buffer["obs"].shape[0]

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for epoch in range(self.update_epochs):
            # ── 打乱并创建 mini-batch ──────────────────────────────────────
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch = {k: v[batch_idx] for k, v in buffer.items()}

                # ── 更新 ────────────────────────────────────────────────────
                self.optimizer.zero_grad()
                step_metrics = self.train_step(batch)
                loss = step_metrics["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                for k in metrics:
                    metrics[k] += step_metrics.get(k, 0)

        n_updates = self.update_epochs * (total_samples // self.batch_size)
        for k in metrics:
            metrics[k] /= max(n_updates, 1)

        return metrics

    def evaluate(self, env: VecEnv, num_episodes: int) -> Dict[str, float]:
        """评估策略性能"""
        self.ac.eval()
        total_reward = 0.0
        success_count = 0

        for _ in range(num_episodes):
            obs = env.reset()
            done = torch.zeros(env.num_envs)
            episode_reward = 0.0

            while not done.all():
                with torch.no_grad():
                    obs_tensor = torch.stack([o["robot_state"] for o in obs]).to(self.device)
                    action, _, _ = self.ac(obs_tensor)
                    obs, rewards, dones, _ = env.step(action.cpu())

                episode_reward += rewards.sum().item()
                done = torch.logical_or(done, dones.to(done.device))

            total_reward += episode_reward
            success_count += 1

        return {
            "eval_avg_reward": total_reward / max(success_count, 1),
            "eval_success_rate": success_count / max(num_episodes, 1),
        }
