"""
================================================================================
 仿真后端核心模块 (backend.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 定义仿真引擎抽象接口，并实现仿真引擎工厂和后端选择逻辑。

 本模块包含:
   1. SimulatorBackend 抽象类 — 所有引擎适配器的父类 (已在 interfaces.py 中定义)
   2. SimBackendFactory — 仿真后端工厂，根据配置创建对应引擎的后端实例
   3. 仿真任务基类 — 封装具体的仿真任务 (如 pick-place, open-drawer)

 设计模式:
   工厂模式 (Factory Pattern) + 策略模式 (Strategy Pattern)
   - 工厂: 根据配置字符串创建对应的引擎后端
   - 策略: 运行时可以动态切换仿真引擎 (如 MuJoCo 快速调试 → Isaac Sim 高保真验证)
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Type, Optional, Any
from pathlib import Path

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import SimulatorBackend
from src.common.types import ConfigDict


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真后端工厂
# ═══════════════════════════════════════════════════════════════════════════════

class SimBackendFactory:
    """
    仿真后端工厂 (SimBackendFactory)
    ────────────────────────────────────────────────────────────────────────────
    职责: 根据配置字符串或配置字典，自动创建对应的仿真后端实例。

    用法:
        factory = SimBackendFactory()
        # 注册后端实现
        factory.register("mujoco", MuJoCoBackend)
        factory.register("isaac_sim", IsaacSimBackend)

        # 创建后端实例
        backend = factory.create("mujoco", {"time_step": 0.002})

    扩展性:
      新增仿真引擎时只需:
        1. 继承 SimulatorBackend 实现新后端
        2. 在工厂中注册
      无需修改已有代码，符合开闭原则。
    """
    def __init__(self):
        self._backends: Dict[str, Type[SimulatorBackend]] = {}

    def register(self, name: str, backend_cls: Type[SimulatorBackend]) -> None:
        """
        注册仿真后端实现类

        Args:
            name: 后端名称 (如 "mujoco", "isaac_sim", "pybullet")
            backend_cls: 后端实现类 (必须继承 SimulatorBackend)
        """
        self._backends[name] = backend_cls
        print(f"[SimFactory] 已注册仿真后端: {name} -> {backend_cls.__name__}")

    def create(self, name: str, config: Optional[ConfigDict] = None) -> SimulatorBackend:
        """
        创建仿真后端实例

        Args:
            name: 后端名称
            config: 后端配置参数

        Returns:
            仿真后端实例

        Raises:
            ValueError: 后端名称未注册
        """
        if name not in self._backends:
            backends = list(self._backends.keys())
            raise ValueError(
                f"未知的仿真后端: '{name}'。"
                f"已注册的后端: {backends}"
            )
        backend_cls = self._backends[name]
        config = config or {}
        return backend_cls(config)

    def list_available(self) -> list[str]:
        """列出所有已注册的仿真后端"""
        return list(self._backends.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Gymnasium 风格环境封装
# ═══════════════════════════════════════════════════════════════════════════════

class RobotEnv:
    """
    机器人仿真环境 (RobotEnv)
    ────────────────────────────────────────────────────────────────────────────
    基于 Gymnasium 接口封装，将 SimulatorBackend 适配为算法层熟悉的
    gym.Env 接口。

    为什么需要这个封装?
      - 算法层 (强化学习) 期望的是 gym.Env 接口 (reset, step, render)
      - 但仿真引擎适配器 (SimulatorBackend) 是更低层的接口
      - RobotEnv 起到"适配器"作用，将低层引擎接口翻译为高层算法接口

    典型用法:
        backend = SimBackendFactory().create("mujoco", config)
        env = RobotEnv(backend, task_cfg)
        obs = env.reset()
        next_obs, reward, done, info = env.step(action)

    参考:
      Gymnasium API: https://gymnasium.farama.org/
    """
    def __init__(self, sim_backend: SimulatorBackend, config: ConfigDict):
        """
        Args:
            sim_backend: 仿真后端实例 (已初始化)
            config: 环境配置 (任务参数、奖励函数、最大步数等)
        """
        self.backend = sim_backend
        self.config = config
        self.max_steps = config.get("max_steps", 500)
        self._step_count = 0
        self._action_space = None  # 将由子类或配置指定
        self._observation_space = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境到初始状态
        ────────────────────────────────────────────────────────────────────────
        工作流程:
          1. 调用仿真后端 reset() 重置物理状态
          2. (可选) 使用领域随机化改变初始条件
          3. 获取初始观测并返回

        Args:
            seed: 随机种子
            options: 额外选项 (如初始位姿覆盖)

        Returns:
            obs: 初始观测字典
            info: 额外信息字典
        """
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

        self.backend.reset()
        self._step_count = 0

        # 获取初始观测
        obs = self.backend.get_observation({})
        info = {"step": 0}
        return obs, info

    def step(self, action):
        """
        执行一步仿真
        ────────────────────────────────────────────────────────────────────────
        工作流程:
          1. 将动作发送到仿真后端
          2. 物理引擎前进一步，更新世界状态
          3. 获取新的观测、奖励和终止信号

        Args:
            action: 动作指令 (通常是关节位置/末端位姿/力矩)

        Returns:
            obs: 新观测
            reward: 奖励值
            terminated: 是否终止 (如任务成功/失败)
            truncated: 是否截断 (如达到最大步数)
            info: 额外信息 (任务进度、碰撞检测等)
        """
        self._step_count += 1

        # 执行动作
        result = self.backend.step(action)

        obs = result.get("obs", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        info = result.get("info", {})

        # 判断终止条件
        terminated = done
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """
        渲染当前画面
        具体实现依赖于后端引擎:
          - Isaac Sim: 使用 Omniverse 渲染器
          - MuJoCo: 使用 OpenGL 离屏渲染
          - PyBullet: 使用内置相机渲染
        """
        obs = self.backend.get_observation({"render": True})
        return obs.get("rgb")

    def close(self):
        """关闭环境，释放资源"""
        self.backend.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真任务基类
# ═══════════════════════════════════════════════════════════════════════════════

class SimTask:
    """
    仿真任务基类 (SimTask)
    ────────────────────────────────────────────────────────────────────────────
    表示一个具体的仿真任务 (如 "将方块放到目标位置")。

    职责:
      1. 定义任务的成功/失败条件
      2. 计算每一步的奖励
      3. 生成任务相关的场景配置
      4. 重置时随机化初始条件

    具体任务继承此类:
        class PickPlaceTask(SimTask):
            def reward(self, obs):
                return 1.0 if self._is_object_placed(obs) else -0.01
    """
    def __init__(self, config: ConfigDict):
        self.config = config
        self.name = config.get("name", "default_task")

    def reward(self, obs: Dict[str, Any]) -> float:
        """
        计算当前步的奖励
        ────────────────────────────────────────────────────────────────────────
        奖励函数的设计是 RL 训练中最关键的部分之一:
          - 稀疏奖励 (sparse): 只有成功/失败时获得 +1/-1
          - 密集奖励 (dense): 每一步都有小奖励，引导策略逐步接近目标
        """
        return 0.0  # 由子类实现具体奖励逻辑

    def is_success(self, obs: Dict[str, Any]) -> bool:
        """判断任务是否成功完成"""
        return False

    def is_failure(self, obs: Dict[str, Any]) -> bool:
        """判断任务是否失败 (如机器人跌倒、物体掉落)"""
        return False
