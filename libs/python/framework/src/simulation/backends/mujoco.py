"""
================================================================================
 MuJoCo 仿真后端 (mujoco.py)
 ──────────────────────────────────────────────────────────────────────────────
 功能: 封装 MuJoCo 物理引擎，实现 SimulatorBackend 接口。

 MuJoCo 简介:
   - 全称: Multi-Joint dynamics with Contact (多关节动力学与接触)
   - 开发: DeepMind (开源)
   - 特点: 极致的仿真速度 (单 CPU 核可达数万 FPS)
           支持 MJX (JAX 加速，GPU 并行数百个环境)
           适用于强化学习的大规模并行训练
   - 场景描述: MJCF (MuJoCo XML) 或 URDF

 本后端同时支持:
   1. 经典 MuJoCo (CPU): 通过 mujoco 和 dm_control 接口
   2. MJX (GPU): 通过 mujoco.mjx 接口，支持大规模并行

 设计考虑:
   MuJoCo 的 API 非常底层，基于 mjModel (模型) 和 mjData (状态) 结构体。
   本适配器封装这些底层细节，对外提供 SimulatorBackend 统一接口。
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


class MuJoCoBackend(SimulatorBackend):
    """
    MuJoCo 仿真后端实现
    ────────────────────────────────────────────────────────────────────────────
    将 MuJoCo 的底层 API 封装为 SimulatorBackend 统一接口。
    """
    def __init__(self, config: ConfigDict):
        """
        初始化 MuJoCo 后端

        Args:
            config: 配置字典
                - "model_path": MJCF/URDF 模型路径 (str)
                - "time_step": 仿真步长 (float, 默认 0.002s)
                - "use_mjx": 是否使用 MJX GPU 加速 (bool, 默认 False)
                - "render_width/height": 渲染分辨率 (int)
        """
        self.config = config
        self._dt = config.get("time_step", 0.002)  # 默认 500Hz
        self._model = None
        self._data = None
        self._renderer = None

        # ── 加载模型 ────────────────────────────────────────────────────────
        model_path = config.get("model_path", "")
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        加载 MuJoCo 模型

        原理:
          MuJoCo 使用 mjModel 描述机器人/场景的物理属性 (关节、体、碰撞几何等)，
          mjData 存储运行时状态 (位置、速度、力等)。
          两者关系: mjModel = 模板 (不可变), mjData = 实例 (可变)
        """
        import mujoco

        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        print(f"[MuJoCo] 模型加载完成: {model_path}")
        print(f"[MuJoCo] 自由度: {self._model.nq}, "
              f"关节数: {self._model.nu}")

    def _ensure_initialized(self):
        """确保后端已初始化"""
        if self._model is None:
            raise RuntimeError("MuJoCo 后端未初始化，请提供 model_path 配置")

    def load_scene(self, scene: SceneDescription) -> Any:
        """加载场景 (调用 _load_model)"""
        if scene.scene_file:
            self._load_model(scene.scene_file)
        return self._model

    def spawn_robot(self, urdf_path: str, position: Tuple[float, ...],
                    quaternion: Tuple[float, ...]) -> Any:
        """在场景中生成机器人 (MuJoCo 通过加载包含机器人的 MJCF 实现)"""
        self._load_model(urdf_path)
        return self._model

    def step(self, action: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行一步 MuJoCo 仿真

        原理:
          1. 将动作指令写入 mjData.ctrl (控制量)
          2. 调用 mj_step 前进一步物理模拟
          3. 从 mjData 读取新的传感器观测
        """
        self._ensure_initialized()
        import mujoco

        # ── 写入控制量 ──────────────────────────────────────────────────────
        # action 可能是关节位置/速度/力矩，取决于控制模式
        for key, value in action.items():
            if key == "ctrl":
                # 设置关节控制量 (位置/速度/力矩)
                ctrl = value.cpu().numpy().flatten()
                n_ctrl = min(len(ctrl), self._model.nu)
                self._data.ctrl[:n_ctrl] = ctrl[:n_ctrl]

        # ── 前进一步 ────────────────────────────────────────────────────────
        mujoco.mj_step(self._model, self._data)

        # ── 读取观测 ────────────────────────────────────────────────────────
        # 注意: 必须用 mj_name2id 判断 body 是否存在，不能用 hasattr(self._data, "body")
        # (mjData.body 是包装器属性，永远存在，直接用 data.body("xxx") 会抛 KeyError)
        has_end_effector = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,
                              "end_effector") != -1
        )
        obs = {
            "joint_pos": torch.tensor(self._data.qpos.copy(), dtype=torch.float32),
            "joint_vel": torch.tensor(self._data.qvel.copy(), dtype=torch.float32),
            "end_effector": torch.tensor(
                self._data.body("end_effector").xpos.copy()
                if has_end_effector
                else np.zeros(3),
                dtype=torch.float32
            ),
        }

        # ── 渲染图像 (可选) ─────────────────────────────────────────────────
        if self._renderer is not None:
            rgb = self._render()
            obs["rgb"] = rgb

        return {
            "obs": obs,
            "reward": 0.0,       # 由环境层计算
            "done": False,        # 由环境层判断
            "info": {},
        }

    def _render(self) -> torch.Tensor:
        """离屏渲染 RGB 图像"""
        import mujoco

        if self._renderer is None:
            # 初始化离屏渲染器
            width = self.config.get("render_width", 256)
            height = self.config.get("render_height", 256)
            self._renderer = mujoco.Renderer(self._model, height, width)

        self._renderer.update_scene(self._data)
        rgb = self._renderer.render().copy()
        # HWC → CHW, uint8 → float32
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return rgb

    def get_observation(self, sensor_spec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """获取当前观测"""
        self._ensure_initialized()

        obs = {
            "joint_pos": torch.tensor(self._data.qpos.copy(), dtype=torch.float32),
            "joint_vel": torch.tensor(self._data.qvel.copy(), dtype=torch.float32),
        }

        if sensor_spec.get("render", False):
            obs["rgb"] = self._render()

        return obs

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """设置物理状态"""
        self._ensure_initialized()

        self._data.qpos[:] = qpos[:self._model.nq]
        self._data.qvel[:] = qvel[:self._model.nv]
        # 前向运动学更新
        import mujoco
        mujoco.mj_forward(self._model, self._data)

    def reset(self) -> None:
        """重置到初始状态"""
        self._ensure_initialized()
        import mujoco
        mujoco.mj_resetData(self._model, self._data)
        # 可选: 领域随机化初始状态
        if self.config.get("randomize_init", False):
            self._randomize_initial_state()

    def _randomize_initial_state(self):
        """随机化初始状态 (领域随机化)"""
        nq = self._model.nq
        nv = self._model.nv
        # 在 ±5% 范围内随机化初始关节位置
        noise_q = np.random.uniform(-0.05, 0.05, size=nq)
        noise_v = np.random.uniform(-0.01, 0.01, size=nv)
        self._data.qpos[:nq] += noise_q
        self._data.qvel[:nv] += noise_v

    def close(self) -> None:
        """释放资源"""
        self._renderer = None
        self._data = None
        self._model = None
        print("[MuJoCo] 资源已释放")

    @property
    def dt(self) -> float:
        return self._dt
