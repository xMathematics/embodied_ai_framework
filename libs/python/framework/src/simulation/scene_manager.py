"""
================================================================================
 场景管理器模块 (scene_manager.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 管理仿真场景的加载、配置和程序化生成。

 核心功能:
   1. 场景库管理: 预定义的标准任务场景 (开柜门、桌面操作等)
   2. 程序化场景生成: 根据任务描述自动组装场景元素
   3. 领域随机化: 随机化场景中的物体位置、材质、光照等
   4. 场景描述序列化: 场景 ↔ SceneDescription 对象的转换

 场景描述格式:
   支持三种格式:
     - USD (Universal Scene Description): Isaac Sim 原生格式
     - MJCF (MuJoCo XML): MuJoCo 场景格式
     - URDF + 程序化生成: PyBullet 常用

 设计模式:
   建造者模式 (Builder Pattern) —— SceneBuilder 逐步构建场景，
   支持链式调用，方便定义复杂的场景配置。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import SceneDescription, ConfigDict


# ═══════════════════════════════════════════════════════════════════════════════
# 场景构建器
# ═══════════════════════════════════════════════════════════════════════════════

class SceneBuilder:
    """
    场景构建器 (SceneBuilder)
    ────────────────────────────────────────────────────────────────────────────
    使用建造者模式，提供链式调用 API 来构建复杂场景。

    用法:
        scene = (SceneBuilder("pick_place")
                 .with_robot("franka")
                 .with_object("cube", position=(0.3, 0.0, 0.02))
                 .with_target(position=(0.5, 0.0, 0.1))
                 .randomize(lighting=True, physics=True)
                 .build())
    """
    def __init__(self, task_name: str = "default"):
        self._scene = SceneDescription(
            task_name=task_name,
        )
        self._objects: List[Dict[str, Any]] = []
        self._randomize_config: Dict[str, bool] = {}

    def with_robot(self, robot_name: str, pose: Optional[Tuple] = None) -> "SceneBuilder":
        """指定机器人型号和初始位姿"""
        robot_urdf_map = {
            "franka": "franka_panda/panda.urdf",
            "xarm": "xarm/xarm6.urdf",
            "ur5": "ur5/ur5.urdf",
        }
        if robot_name in robot_urdf_map:
            self._scene.robot_urdf = robot_urdf_map[robot_name]
        if pose is not None:
            self._scene.init_pose = pose
        return self

    def with_object(self, name: str, urdf: str = "",
                    position: Tuple = (0, 0, 0),
                    orientation: Tuple = (0, 0, 0, 1)) -> "SceneBuilder":
        """添加场景物体"""
        self._objects.append({
            "name": name,
            "urdf": urdf,
            "position": position,
            "orientation": orientation,
        })
        return self

    def with_target(self, position: Tuple = (0, 0, 0)) -> "SceneBuilder":
        """设置目标位置"""
        self._scene.metadata["target_position"] = position
        return self

    def randomize(self, lighting: bool = False, physics: bool = False,
                  textures: bool = False) -> "SceneBuilder":
        """启用领域随机化"""
        self._scene.randomize = True
        self._randomize_config = {
            "lighting": lighting,
            "physics": physics,
            "textures": textures,
        }
        return self

    def build(self) -> SceneDescription:
        """构建最终的场景描述"""
        self._scene.metadata["objects"] = self._objects
        self._scene.metadata["randomize_config"] = self._randomize_config
        return self._scene


# ═══════════════════════════════════════════════════════════════════════════════
# 领域随机化工具
# ═══════════════════════════════════════════════════════════════════════════════

class DomainRandomizer:
    """
    领域随机化工具 (DomainRandomizer)
    ────────────────────────────────────────────────────────────────────────────
    职责: 对场景参数进行随机化，提升策略的泛化能力。

    领域随机化是 Sim-to-Real 迁移的关键技术之一:
      通过在仿真中随机化那些在真实世界中会变化的参数，
      策略学会关注与任务相关的特征，忽略无关的视觉/物理变化。

    随机化维度 (从设计架构文档):
      - 材质纹理: 随机更换桌面和物体纹理
      - 光照: 随机调整光照方向和强度
      - 物理参数: 摩擦系数、阻尼、质量
      - 相机内参: 噪声、焦距、位置
    """
    def __init__(self, config: ConfigDict):
        self.config = config
        self._rng = np.random.default_rng()

    def randomize_physics(self, scene: SceneDescription) -> SceneDescription:
        """随机化物理参数"""
        import numpy as np
        # 摩擦系数: 随机范围 [0.2, 1.5]
        scene.physics_params["friction"] = float(
            self._rng.uniform(0.2, 1.5)
        )
        # 阻尼: 随机范围 [0.5, 2.0]
        scene.physics_params["damping"] = float(
            self._rng.uniform(0.5, 2.0)
        )
        return scene

    def randomize_lighting(self) -> Dict[str, Any]:
        """生成随机光照参数"""
        return {
            "light_intensity": float(self._rng.uniform(0.5, 1.5)),
            "light_direction": self._rng.uniform(-1, 1, size=3).tolist(),
            "light_color": self._rng.uniform(0.8, 1.0, size=3).tolist(),
        }

    def randomize_camera(self) -> Dict[str, Any]:
        """生成随机相机参数"""
        return {
            "noise_std": float(self._rng.uniform(0.0, 0.02)),
            "fov": float(self._rng.uniform(40, 80)),
            "position_offset": self._rng.uniform(-0.05, 0.05, size=3).tolist(),
        }
