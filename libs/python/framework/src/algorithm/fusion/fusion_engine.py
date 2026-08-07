"""
================================================================================
 融合引擎 (FusionEngine)
 ──────────────────────────────────────────────────────────────────────────────
 核心职责: 根据配置文件中的 fusion_mode，将 VLA、RL、世界模型三个模块
 按不同策略组合成统一的决策系统。

 支持四种融合模式:
   1. vla_only:         仅使用 VLA 端到端推理
   2. vla_rl:           VLA 高层规划 + RL 底层控制
   3. vla_world_model:  VLA + 世界模型想象推演
   4. all:              VLA + RL + 世界模型三模块闭环

 支持四种融合策略:
   1. weighted_average: 加权平均动作融合
   2. bayesian:         基于置信度的贝叶斯融合
   3. priority:         优先级切换路由
   4. cascade:          级联决策 (VLA → RL → 世界模型)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Any

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np


class FusionMode(str, Enum):
    """融合模式枚举"""
    VLA_ONLY = "vla_only"
    VLA_RL = "vla_rl"
    VLA_WORLD_MODEL = "vla_world_model"
    ALL = "all"


class FusionStrategy(str, Enum):
    """融合策略枚举"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"
    PRIORITY = "priority"
    CASCADE = "cascade"


@dataclass
class FusionConfig:
    """
    融合引擎配置
    ────────────────────────────────────────────────────────────────────────────
    """
    # ── 融合模式 ──
    mode: FusionMode = FusionMode.VLA_ONLY
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE

    # ── 加权平均策略参数 ──
    vla_weight: float = 0.4          # VLA 权重
    rl_weight: float = 0.6           # RL 权重
    adaptive_weight: bool = True     # 是否根据置信度自适应调整权重

    # ── 贝叶斯融合策略参数 ──
    vla_confidence_threshold: float = 0.7   # VLA 置信度阈值
    rl_confidence_scale: float = 1.0        # RL 置信度缩放因子

    # ── 优先级路由策略参数 ──
    default_controller: str = "rl"          # 默认控制器
    fallback_threshold: float = 0.3         # 切换阈值

    # ── 级联决策策略参数 ──
    planning_horizon: int = 10              # VLA 规划步长
    subgoal_dim: int = 3                    # 子目标维度


class FusionEngine(ABC):
    """
    模块融合引擎抽象基类
    ────────────────────────────────────────────────────────────────────────────
    职责: 将 VLA、RL、世界模型三个模块按配置组合成统一的决策系统。

    使用方式:
      engine = FusionEngine.create(cfg)
      action = engine.select_action(obs, instruction="pick up the cube")
    """
    def __init__(self, config: FusionConfig):
        self.config = config

    @abstractmethod
    def select_action(
        self,
        obs: Dict,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        """
        统一的决策入口

        Args:
            obs: 观测字典
            instruction: 自然语言指令 (VLA 模式必需)

        Returns:
            融合后的动作
        """
        ...

    @classmethod
    def create(cls, config: FusionConfig) -> "FusionEngine":
        """
        工厂方法: 根据配置创建对应的融合引擎实例

        Args:
            config: 融合引擎配置

        Returns:
            对应融合模式的引擎实例
        """
        if config.mode == FusionMode.VLA_ONLY:
            return VLAOnlyEngine(config)
        elif config.mode == FusionMode.VLA_RL:
            return VLARLFusionEngine(config)
        elif config.mode == FusionMode.VLA_WORLD_MODEL:
            return VLAWorldModelFusionEngine(config)
        elif config.mode == FusionMode.ALL:
            return TripleFusionEngine(config)
        else:
            raise ValueError(f"Unknown fusion mode: {config.mode}")


class VLAOnlyEngine(FusionEngine):
    """
    纯 VLA 引擎
    ────────────────────────────────────────────────────────────────────────────
    仅使用 VLA 大模型进行端到端推理，适合已有成熟 VLA 模型的场景。
    """
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.vla_policy = None  # 将在外部注入

    def select_action(
        self,
        obs: Dict,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        if self.vla_policy is None:
            raise RuntimeError("VLA policy not injected into engine")
        output = self.vla_policy.act_with_language(obs, instruction or "")
        return output.action


class VLARLFusionEngine(FusionEngine):
    """
    VLA + RL 融合引擎
    ────────────────────────────────────────────────────────────────────────────
    VLA 大模型提供高层语义理解与动作先验，
    RL 策略提供精细运动控制。

    融合策略:
      - weighted_average: a = w * a_vla + (1-w) * a_rl
      - bayesian: 基于置信度的贝叶斯融合
      - priority: 高置信度时用 VLA，低置信度时用 RL
    """
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.vla_policy = None
        self.rl_policy = None

    def select_action(
        self,
        obs: Dict,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        # 1. 获取 VLA 动作
        vla_output = self.vla_policy.act_with_language(obs, instruction or "")
        a_vla = vla_output.action
        conf = vla_output.confidence

        # 2. 获取 RL 动作
        a_rl = self.rl_policy.act(obs)

        # 3. 根据策略融合
        if self.config.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            w = self.config.vla_weight
            if self.config.adaptive_weight:
                w = conf * self.config.vla_weight
            return w * a_vla + (1 - w) * a_rl

        elif self.config.strategy == FusionStrategy.BAYESIAN:
            # 贝叶斯融合: 假设两个高斯分布
            # VLA: N(a_vla, σ_v), RL: N(a_rl, σ_r)
            sigma_v = 1.0 - conf  # 置信度越高，方差越小
            sigma_r = 1.0
            w_v = (1.0 / sigma_v) / (1.0 / sigma_v + 1.0 / sigma_r)
            return w_v * a_vla + (1 - w_v) * a_rl

        elif self.config.strategy == FusionStrategy.PRIORITY:
            if conf > self.config.vla_confidence_threshold:
                return a_vla
            else:
                return a_rl

        else:
            return a_vla  # fallback


class VLAWorldModelFusionEngine(FusionEngine):
    """
    VLA + 世界模型融合引擎
    ────────────────────────────────────────────────────────────────────────────
    VLA 生成多个候选动作，世界模型在"想象"中对每个候选动作进行推演，
    选择最优动作。实现"先想后做"的安全决策机制。
    """
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.vla_policy = None
        self.world_model = None

    def select_action(
        self,
        obs: Dict,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        # 1. VLA 生成多个候选动作
        vla_output = self.vla_policy.act_with_language(obs, instruction or "")

        # 2. 如果有动作块，在世界模型中进行推演评估
        if vla_output.action_chunk is not None and self.world_model is not None:
            candidates = vla_output.action_chunk  # (n_candidates, action_dim)
            # 在世界模型中对每个候选动作进行推演
            scores = []
            for i in range(candidates.shape[0]):
                imagined_trajectory = self.world_model.imagine(
                    obs, candidates[i], horizon=self.config.planning_horizon
                )
                score = imagined_trajectory["expected_reward"]
                scores.append(score)

            # 选择最优动作
            best_idx = torch.tensor(scores).argmax()
            return candidates[best_idx]

        return vla_output.action


class TripleFusionEngine(FusionEngine):
    """
    三模块全闭环引擎 (VLA + RL + 世界模型)
    ────────────────────────────────────────────────────────────────────────────
    VLA 做语义理解与高层规划，世界模型做想象推演，
    RL 做精细控制。三模块形成正向循环。

    决策流程:
      VLA 理解任务 → 设定子目标 → 世界模型规划运动轨迹 →
      RL 跟踪轨迹 → 实际执行 → 反馈给 VLA 和世界模型更新
    """
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        self.vla_policy = None
        self.rl_policy = None
        self.world_model = None

    def select_action(
        self,
        obs: Dict,
        instruction: Optional[str] = None
    ) -> torch.Tensor:
        # 阶段 1: VLA 生成子目标
        vla_output = self.vla_policy.act_with_language(obs, instruction or "")
        subgoal = vla_output.action[:self.config.subgoal_dim]

        # 阶段 2: 世界模型在子目标条件下进行运动规划
        if self.world_model is not None:
            planned_traj = self.world_model.imagine(
                obs, subgoal, horizon=self.config.planning_horizon
            )
            # 提取当前时间步的参考动作
            reference_action = planned_traj["action"]
        else:
            reference_action = vla_output.action

        # 阶段 3: RL 策略跟踪参考动作
        if self.rl_policy is not None:
            rl_action = self.rl_policy.act(obs)
            # 加权融合: RL 跟踪参考动作加上探索噪声
            return 0.7 * reference_action + 0.3 * rl_action

        return reference_action
