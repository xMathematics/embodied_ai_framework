"""
================================================================================
 Isaac Sim 仿真后端 (isaac_sim.py)
 ──────────────────────────────────────────────────────────────────────────────
 功能: 封装 NVIDIA Isaac Sim (Omniverse 平台)，实现 SimulatorBackend 接口。

 Isaac Sim 简介:
   - 平台: NVIDIA Omniverse (基于 USD 的 3D 协作平台)
   - 特点: 最高保真度的物理仿真 + 光线追踪渲染
           基于 PhysX 引擎，支持 GPU 并行
           原生支持 USD 场景描述，可与数字孪生流程对接
   - 适用: Sim-to-Real 迁移、高保真验证、数字孪生
   - 语言: Python 3.10 (通过 Omniverse Kit 扩展)

 注意事项:
   - Isaac Sim 需要独立安装 (NVIDIA Container Toolkit 或本地安装)
   - 需要 NVIDIA GPU (RTX 系列及以上)
   - 启动较慢 (首次加载约 1-2 分钟)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import SimulatorBackend
from src.common.types import SceneDescription, ConfigDict


class IsaacSimBackend(SimulatorBackend):
    """
    Isaac Sim 仿真后端
    ────────────────────────────────────────────────────────────────────────────
    将 Isaac Sim 的 Omniverse API 封装为 SimulatorBackend 统一接口。

    初始化流程:
      1. 启动 Omniverse Kit 进程 (或连接到已有进程)
      2. 加载 USD 场景
      3. 注册传感器和机器人
    """
    def __init__(self, config: ConfigDict):
        """
        Args:
            config: 配置字典
                - "usd_path": 场景 USD 文件路径
                - "headless": 是否无头模式 (bool, 默认 True)
                - "physics_dt": 物理步长 (float, 默认 1/60)
                - "render_dt": 渲染步长 (float, 默认 1/30)
        """
        self.config = config
        self._dt = config.get("physics_dt", 1.0 / 60.0)
        self._headless = config.get("headless", True)
        self._timeline = None
        self._world = None

        # ── 初始化 Isaac Sim ────────────────────────────────────────────────
        self._init_isaac_sim()

    def _init_isaac_sim(self):
        """
        初始化 Isaac Sim 环境

        原理:
          Isaac Sim 基于 Omniverse Kit SDK。初始化过程:
          1. 启动或连接到 Omniverse 进程
          2. 创建 PhysicsContext (物理场景)
          3. 创建 RenderContext (渲染上下文)

        注意:
          Isaac Sim 的 Python API 需要在特定环境下导入 (Ominverse Kit 扩展环境)。
          以下代码假设已在 Isaac Sim 的 Python 解释器中运行。
        """
        try:
            # ── 导入 Isaac Sim 模块 ─────────────────────────────────────────
            # 这些导入仅在 Isaac Sim 环境中可用
            from omni.isaac.kit import SimulationApp
            from omni.isaac.core import World

            # ── 启动仿真应用 ────────────────────────────────────────────────
            self._app = SimulationApp(
                {"headless": self._headless}
            )

            # ── 创建世界 ────────────────────────────────────────────────────
            self._world = World(
                physics_dt=self._dt,
                rendering_dt=self.config.get("render_dt", 1.0 / 30.0),
            )

            # ── 加载场景 USD ────────────────────────────────────────────────
            usd_path = self.config.get("usd_path", "")
            if usd_path:
                self._world.stage.DefinePrim(
                    "/World", "Xform"
                )
                # 使用 omni.usd API 加载 USD 文件
                from pxr import UsdGeom, Sdf
                import omni.usd
                omni.usd.get_context().open_usd(usd_path)

            print(f"[IsaacSim] 初始化完成 (headless={self._headless})")

        except ImportError:
            print("[IsaacSim] 警告: Isaac Sim 环境未安装或未启动。")
            print("[IsaacSim] 请确保在 Isaac Sim 的 Python 环境中运行。")
            self._app = None
            self._world = None

    def _ensure_initialized(self):
        """确保 Isaac Sim 已正确初始化"""
        if self._world is None:
            raise RuntimeError(
                "Isaac Sim 未正确初始化。"
                "请在 Isaac Sim 的 Ominverse Kit 环境中运行此代码。"
            )

    def load_scene(self, scene: SceneDescription) -> Any:
        """加载场景"""
        self._ensure_initialized()

        if scene.scene_file:
            import omni.usd
            omni.usd.get_context().open_usd(scene.scene_file)

        # 加载机器人 URDF
        if scene.robot_urdf:
            from omni.isaac.core.robots import Robot
            from omni.isaac.core.utils.stage import add_reference_to_stage

            # 将 URDF 导入为 USD 引用
            robot_prim_path = "/World/Robot"
            add_reference_to_stage(
                scene.robot_urdf, robot_prim_path
            )
            self._robot = Robot(
                prim_path=robot_prim_path,
                name="robot",
            )
            self._world.scene.add(self._robot)

        return self._world.stage

    def spawn_robot(self, urdf_path: str, position: Tuple[float, ...],
                    quaternion: Tuple[float, ...]) -> Any:
        """在场景中生成机器人"""
        self._ensure_initialized()
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage

        import carb
        robot_prim_path = "/World/Robot"
        add_reference_to_stage(urdf_path, robot_prim_path)

        self._robot = Robot(
            prim_path=robot_prim_path,
            name="robot",
            position=np.array(position[:3]),
            orientation=np.array(quaternion),
        )
        self._world.scene.add(self._robot)
        return self._robot

    def step(self, action: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行一步 Isaac Sim 仿真

        原理:
          Isaac Sim 使用 World.step() 驱动物理和渲染同步前进。
          动作通过机器人的关节控制接口发送。
        """
        self._ensure_initialized()

        # ── 应用动作 ────────────────────────────────────────────────────────
        if hasattr(self, "_robot") and self._robot is not None:
            if "joint_positions" in action:
                self._robot.set_joint_positions(
                    action["joint_positions"].cpu().numpy()
                )

        # ── 步进物理世界 ────────────────────────────────────────────────────
        self._world.step(render=not self._headless)

        # ── 读取观测 ────────────────────────────────────────────────────────
        obs = {}
        if hasattr(self, "_robot") and self._robot is not None:
            obs["joint_pos"] = torch.tensor(
                self._robot.get_joint_positions(), dtype=torch.float32
            )
            obs["joint_vel"] = torch.tensor(
                self._robot.get_joint_velocities(), dtype=torch.float32
            )

        return {
            "obs": obs,
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def get_observation(self, sensor_spec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """获取观测"""
        self._ensure_initialized()
        obs = {}

        if hasattr(self, "_robot") and self._robot is not None:
            obs["joint_pos"] = torch.tensor(
                self._robot.get_joint_positions(), dtype=torch.float32
            )
            obs["joint_vel"] = torch.tensor(
                self._robot.get_joint_velocities(), dtype=torch.float32
            )

        return obs

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置物理状态"""
        self._ensure_initialized()
        if hasattr(self, "_robot") and self._robot is not None:
            self._robot.set_joint_positions(qpos)
            self._robot.set_joint_velocities(qvel)

    def reset(self) -> None:
        """重置世界"""
        self._ensure_initialized()
        self._world.reset()

    def close(self) -> None:
        """关闭仿真并释放资源"""
        if hasattr(self, "_app") and self._app is not None:
            self._app.close()
            print("[IsaacSim] 仿真已关闭")

    @property
    def dt(self) -> float:
        return self._dt
