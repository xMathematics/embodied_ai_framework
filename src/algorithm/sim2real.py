"""
================================================================================
 Sim-to-Real 迁移模块 (sim2real.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供将仿真环境中训练的策略迁移到真实机器人上的工具。

 Sim-to-Real 的核心挑战:
   仿真与现实之间的"差距" (Reality Gap) 会导致策略在真实环境中失效。
   本模块提供多种弥合此差距的技术。

 迁移技术:
   1. 域随机化 (Domain Randomization): 仿真中变化各种参数，使策略学会适应
   2. 域适应 (Domain Adaptation): 通过 GAN 等对齐仿真和真实的观测分布
   3. 系统辨识 (System Identification): 调整仿真参数匹配真实机器人
   4. 动作补偿 (Action Compensation): 补偿仿真与真实的动力学差异
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Callable, Any

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import Policy


class DomainAdaptation(nn.Module):
    """
    域适应模块 (Domain Adaptation)
    ────────────────────────────────────────────────────────────────────────────
    使用对抗训练对齐仿真和真实环境的观测分布。

    原理 (GAN 风格):
      1. 一个判别器 D 尝试区分: 输入的观测来自仿真还是真实
      2. 一个观测编码器 E 尝试"欺骗"D，使仿真观测的编码看起来像真实的
      3. 最终效果: 策略看到的仿真观测编码与真实观测编码分布一致

    网络结构:
      仿真观测 ─→ 编码器 ─→ 判别器 → 真/假
      真实观测 ─→ 编码器 ─→ 判别器 → 真/假
                   ↓
               共享特征 (送到策略网络)
    """
    def __init__(self, obs_dim: int, feature_dim: int = 64):
        super().__init__()

        # ── 观测编码器 (共享) ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # ── 域判别器 ────────────────────────────────────────────────────────
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 输出 0=仿真, 1=真实
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """编码观测为域无关的特征"""
        return self.encoder(obs)

    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        """域判别: 特征来自仿真还是真实"""
        return self.discriminator(features)


class ActionDelayCompensation:
    """
    动作延迟补偿 (Action Delay Compensation)
    ────────────────────────────────────────────────────────────────────────────
    真实机器人通常比仿真有更大的控制延迟 (传感器处理、通信、电机响应)。
    本模块通过在仿真中引入模拟延迟，使策略适应延迟环境。

    常用的延迟补偿策略:
      1. 在仿真训练时随机化延迟大小
      2. 在策略观测中显式包含历史动作 (让策略学会"看"到延迟)
    """
    def __init__(self, max_delay_steps: int = 3):
        self.max_delay = max_delay_steps
        self.action_buffer = []  # 存储历史动作

    def apply_delay(self, action: torch.Tensor,
                    delay: Optional[int] = None) -> torch.Tensor:
        """
        模拟动作执行延迟

        原理:
          不立即执行当前动作，而是执行 delay 步之前的动作。
          这迫使策略学习在延迟环境下仍然有效的行为。

        Args:
            action: 当前动作
            delay: 显式指定延迟步数, None 则随机

        Returns:
            实际执行的动作 (可能是历史动作)
        """
        if delay is None:
            import random
            delay = random.randint(0, self.max_delay)

        self.action_buffer.append(action)
        if len(self.action_buffer) > self.max_delay + 1:
            self.action_buffer.pop(0)

        # 返回 delay 步之前的动作
        idx = max(0, len(self.action_buffer) - 1 - delay)
        return self.action_buffer[idx]


class Sim2RealPipeline:
    """
    Sim-to-Real 流水线 (Sim2RealPipeline)
    ────────────────────────────────────────────────────────────────────────────
    统筹完整的 Sim-to-Real 迁移流程。

    典型工作流:
      1. 在随机化的仿真环境中训练策略
      2. 应用域适应 (可选)
      3. 在真实机器人上测试
      4. 根据测试结果调整参数
      5. 重复迭代
    """
    def __init__(self, policy: Policy, config: Dict[str, Any]):
        self.policy = policy
        self.config = config
        self.delay_comp = ActionDelayCompensation(
            max_delay_steps=config.get("max_delay", 3)
        )

    def deploy_to_real(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        在真实机器人上部署策略

        与仿真推理的区别:
          - 增加延迟补偿
          - (可选) 使用域适应编码器
          - 动作平滑 (低通滤波)
        """
        action = self.policy.act(obs, deterministic=True)

        # 延迟补偿
        action = self.delay_comp.apply_delay(action)

        # 动作平滑 (指数移动平均)
        if hasattr(self, "_last_action"):
            alpha = self.config.get("smooth_alpha", 0.3)
            action = alpha * action + (1 - alpha) * self._last_action
        self._last_action = action

        return action
