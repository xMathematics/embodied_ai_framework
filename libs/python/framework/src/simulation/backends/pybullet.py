"""
================================================================================
 PyBullet 仿真后端 (pybullet.py)
 ──────────────────────────────────────────────────────────────────────────────
 功能: 封装 PyBullet 物理引擎，实现 SimulatorBackend 接口。

 PyBullet 简介:
   - 开发: Erwin Coumans (开源)
   - 特点: 基于 Bullet 物理引擎的 Python 封装
           安装简单 (pip install pybullet)
           支持多种渲染模式 (GUI, 离屏, 无头)
           适合快速原型和小规模实验
   - 限制: 大规模并行能力有限，GPU 加速不完善
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import numpy as np
import torch
import pybullet as p
import pybullet_data  # 提供官方模型和 URDF

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import SimulatorBackend
from src.common.types import SceneDescription, ConfigDict


class PyBulletBackend(SimulatorBackend):
    """
    PyBullet 仿真后端
    ────────────────────────────────────────────────────────────────────────────
    将 PyBullet 的客户端-服务器 API 封装为 SimulatorBackend 统一接口。

    PyBullet 采用客户端-服务器架构:
      - 物理引擎运行在服务器端 (可本地进程或远程)
      - 客户端通过 API 发送命令和接收状态
      - 支持 DIRECT (本地进程内) 和 GUI (带可视化) 两种模式
    """
    def __init__(self, config: ConfigDict):
        """
        Args:
            config: 配置字典
                - "gui": 是否启用 GUI (bool, 默认 False)
                - "time_step": 物理步长 (float, 默认 1/240)
                - "robot_urdf": 机器人 URDF 路径 (str)
        """
        self.config = config
        self._dt = config.get("time_step", 1.0 / 240.0)
        self._robot_id = None

        # ── 连接 PyBullet ───────────────────────────────────────────────────
        gui_mode = config.get("gui", False)
        self._client_id = p.connect(p.GUI if gui_mode else p.DIRECT)

        # 设置搜索路径 (用于查找官方 URDF)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 设置重力
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.setTimeStep(self._dt, physicsClientId=self._client_id)

        print(f"[PyBullet] 初始化完成 (GUI={gui_mode})")

    def load_scene(self, scene: SceneDescription) -> Any:
        """加载场景 (添加平面和物体)"""
        if scene.scene_file:
            # 加载 URDF/SDF 场景
            plane_id = p.loadURDF(
                scene.scene_file,
                useFixedBase=True,
                physicsClientId=self._client_id,
            )
            return plane_id

        # 默认: 加载地面平面
        plane_id = p.loadURDF(
            "plane.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        return plane_id

    def spawn_robot(self, urdf_path: str, position: Tuple[float, ...],
                    quaternion: Tuple[float, ...]) -> Any:
        """加载机器人 URDF"""
        self._robot_id = p.loadURDF(
            urdf_path,
            basePosition=position[:3],
            baseOrientation=quaternion[:4] if len(quaternion) >= 4 else [0, 0, 0, 1],
            useFixedBase=False,
            physicsClientId=self._client_id,
        )
        return self._robot_id

    def step(self, action: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行一步仿真

        原理:
          1. 通过 setJointMotorControlArray 设置关节目标
          2. stepSimulation 推进物理
          3. getJointStates 读取关节状态
        """
        if self._robot_id is None:
            raise RuntimeError("机器人未加载，请先调用 spawn_robot")

        # ── 应用动作 ────────────────────────────────────────────────────────
        if "joint_positions" in action:
            joint_poses = action["joint_positions"].cpu().numpy()
            num_joints = p.getNumJoints(
                self._robot_id, physicsClientId=self._client_id
            )
            joint_indices = list(range(num_joints))

            p.setJointMotorControlArray(
                self._robot_id,
                jointIndices=joint_indices[:len(joint_poses)],
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_poses[:num_joints],
                physicsClientId=self._client_id,
            )

        # ── 步进仿真 ────────────────────────────────────────────────────────
        p.stepSimulation(physicsClientId=self._client_id)

        # ── 读取观测 ────────────────────────────────────────────────────────
        num_joints = p.getNumJoints(
            self._robot_id, physicsClientId=self._client_id
        )
        joint_states = p.getJointStates(
            self._robot_id,
            list(range(num_joints)),
            physicsClientId=self._client_id,
        )
        joint_positions = np.array([s[0] for s in joint_states])
        joint_velocities = np.array([s[1] for s in joint_states])

        # 读取末端执行器位姿 (假设最后一个关节为末端)
        ee_state = p.getLinkState(
            self._robot_id,
            num_joints - 1,
            physicsClientId=self._client_id,
        )
        ee_pos = ee_state[0]   # 位置 (x, y, z)
        ee_orn = ee_state[1]   # 姿态 (四元数)

        obs = {
            "joint_pos": torch.tensor(joint_positions, dtype=torch.float32),
            "joint_vel": torch.tensor(joint_velocities, dtype=torch.float32),
            "end_effector_pos": torch.tensor(ee_pos, dtype=torch.float32),
            "end_effector_orn": torch.tensor(ee_orn, dtype=torch.float32),
        }

        return {
            "obs": obs,
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def get_observation(self, sensor_spec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """获取观测"""
        return self.step({"joint_positions": torch.zeros(0)})["obs"]

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """直接设置关节状态"""
        if self._robot_id is not None:
            num_joints = p.getNumJoints(
                self._robot_id, physicsClientId=self._client_id
            )
            for i in range(min(len(qpos), num_joints)):
                p.resetJointState(
                    self._robot_id,
                    i,
                    targetValue=float(qpos[i]),
                    targetVelocity=float(qvel[i]) if i < len(qvel) else 0.0,
                    physicsClientId=self._client_id,
                )

    def reset(self) -> None:
        """重置仿真"""
        p.resetSimulation(physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        # 重新加载平面和机器人
        self.load_scene(SceneDescription())
        if self.config.get("robot_urdf"):
            self.spawn_robot(
                self.config["robot_urdf"],
                (0, 0, 0),
                (0, 0, 0, 1),
            )

    def close(self) -> None:
        """断开 PyBullet 连接"""
        if hasattr(self, "_client_id"):
            p.disconnect(physicsClientId=self._client_id)
            print("[PyBullet] 仿真已断开")

    @property
    def dt(self) -> float:
        return self._dt
