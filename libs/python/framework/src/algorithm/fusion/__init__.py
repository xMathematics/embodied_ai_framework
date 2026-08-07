"""
================================================================================
 模块融合引擎 (Fusion Engine)
 ──────────────────────────────────────────────────────────────────────────────
 将 VLA 大模型、强化学习、世界模型三个核心模块按配置组合成统一的决策系统。

 核心组件:
   - FusionEngine: 融合引擎基类，定义统一的决策接口
   - ModuleRegistry: 模块注册中心，管理所有算法的注册与实例化
   - VLARLFusion: VLA + RL 混合决策
   - VLAWorldModelFusion: VLA + 世界模型想象推演
   - TripleFusion: VLA + RL + 世界模型三模块全闭环

 融合策略:
   - weighted_average: 加权平均融合
   - bayesian: 贝叶斯融合
   - priority: 优先级路由
   - cascade: 级联决策

 设计原则:
   用户只需要在配置文件中指定 fusion_mode，融合引擎自动
   选择并组合对应的模块，实现"一键切换"不同算法范式。
================================================================================
"""

from src.algorithm.fusion.fusion_engine import (
    FusionEngine,
    FusionConfig,
    ModuleSelectionConfig,
)
from src.algorithm.fusion.module_registry import ModuleRegistry

__all__ = [
    "FusionEngine",
    "FusionConfig",
    "ModuleSelectionConfig",
    "ModuleRegistry",
]
