"""
================================================================================
 向量化环境管理器 (vec_env.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 同时管理多个并行的仿真环境，实现高效的数据并行采样。

 为什么需要向量化环境?
   在强化学习中，策略主循环是: 观测 → 策略推理 → 动作 → 环境步进。
   其中环境步进 (仿真) 通常是瓶颈。向量化环境通过同时运行 N 个环境
   (每个环境可以有不同的随机种子)，将采样效率提升 N 倍。

 三种并行策略:
   1. SubprocVecEnv:  多进程并行 (每个环境一个进程) — CPU 密集型
   2. DummyVecEnv:    单进程串行 (调试用)
   3. MJXVecEnv:      GPU 并行 (MuJoCo MJX 专用) — GPU 密集型

 设计参考:
   OpenAI Gym 的 VecEnv 接口规范
   https://github.com/openai/gym/tree/master/gym/vector
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import List, Callable, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import VecEnv
from src.simulation.backend import RobotEnv


class DummyVecEnv(VecEnv):
    """
    单进程向量化环境 (DummyVecEnv)
    ────────────────────────────────────────────────────────────────────────────
    最简单的实现: 在单进程中串行执行多个环境。
    适用于:
      - 调试和测试
      - 环境数量少 (< 8) 的场景
      - 不需要真正并行的场景

    原理:
      只是把多个环境包装在一个列表中，step 时循环调用每个环境的 step。
      没有真正的并行加速，但代码接口与其他 VecEnv 一致。
    """
    def __init__(self, env_fns: List[Callable[[], RobotEnv]]):
        """
        Args:
            env_fns: 环境工厂函数列表，每个函数返回一个 RobotEnv 实例
        """
        self.envs = [fn() for fn in env_fns]
        self._num_envs = len(self.envs)

    def reset(self) -> List[Dict[str, torch.Tensor]]:
        """重置所有环境"""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return observations

    def step(self, actions: torch.Tensor) -> Tuple[
        List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, List[Dict]
    ]:
        """
        在所有环境中步进

        动作的 batch 维度对应不同的环境:
          actions[i] → envs[i].step(actions[i])
        """
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.envs):
            action = actions[i]  # 取出第 i 个环境的动作
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

        return obs_list, torch.tensor(rewards), torch.tensor(dones), infos

    def close(self) -> None:
        """关闭所有环境"""
        for env in self.envs:
            env.close()

    @property
    def num_envs(self) -> int:
        return self._num_envs


class SubprocVecEnv(VecEnv):
    """
    多进程向量化环境 (SubprocVecEnv)
    ────────────────────────────────────────────────────────────────────────────
    使用多进程技术实现真正的并行环境采样。

    原理:
      每个环境运行在独立的子进程中，通过 Pipe (管道) 与主进程通信。
      主进程将动作发送给所有子进程，子进程并行执行后返回结果。

    性能:
      - 适用于 CPU 密集型的仿真引擎 (PyBullet, MuJoCo CPU)
      - 加速比 ≈ min(N, CPU核心数)
      - 需要小心进程间通信开销 (大数据的传输)

    注意事项:
      - 子进程不能共享 GPU 资源 (每个进程需要自己的 GPU context)
      - 环境需要在 __main__ 中创建 (Python 多进程限制)
    """
    def __init__(self, env_fns: List[Callable[[], RobotEnv]]):
        """
        Args:
            env_fns: 环境工厂函数列表
        """
        self._num_envs = len(env_fns)
        self._processes = []
        self._pipes = []

        for env_fn in env_fns:
            # ── 创建进程间通信管道 ──────────────────────────────────────────
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(
                target=self._worker,
                args=(env_fn, child_pipe),
                daemon=True,  # 主进程结束时自动终止子进程
            )
            proc.start()
            child_pipe.close()  # 父进程关闭子端
            self._processes.append(proc)
            self._pipes.append(parent_pipe)

    @staticmethod
    def _worker(env_fn: Callable[[], RobotEnv], pipe: mp.Connection):
        """
        子进程的工作函数

        无限循环:
          1. 从管道接收指令 ("step", "reset", "close")
          2. 执行对应操作
          3. 将结果发送回管道
        """
        env = env_fn()
        while True:
            cmd, data = pipe.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                pipe.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset()
                pipe.send((obs, info))
            elif cmd == "close":
                env.close()
                pipe.close()
                break

    def reset(self) -> List[Dict[str, torch.Tensor]]:
        """并行重置所有环境"""
        for pipe in self._pipes:
            pipe.send(("reset", None))
        return [pipe.recv()[0] for pipe in self._pipes]

    def step(self, actions: torch.Tensor) -> Tuple[
        List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, List[Dict]
    ]:
        """并行步进所有环境"""
        # ── 发送动作到所有子进程 ───────────────────────────────────────────
        for i, pipe in enumerate(self._pipes):
            pipe.send(("step", actions[i]))

        # ── 收集结果 ────────────────────────────────────────────────────────
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for pipe in self._pipes:
            obs, reward, terminated, truncated, info = pipe.recv()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

        return obs_list, torch.tensor(rewards), torch.tensor(dones), infos

    def close(self) -> None:
        """关闭所有子进程"""
        for pipe in self._pipes:
            pipe.send(("close", None))
        for proc in self._processes:
            proc.join(timeout=1.0)

    @property
    def num_envs(self) -> int:
        return self._num_envs


def make_vec_env(env_fn: Callable[[], RobotEnv],
                 num_envs: int = 8,
                 mode: str = "dummy") -> VecEnv:
    """
    创建向量化环境的工厂函数

    Args:
        env_fn: 单个环境的工厂函数
        num_envs: 并行环境数量
        mode: 并行模式 ("dummy" = 单进程, "subproc" = 多进程)

    Returns:
        VecEnv 实例
    """
    env_fns = [env_fn for _ in range(num_envs)]

    if mode == "subproc":
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)
