"""
================================================================================
 评估器模块 (evaluator.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 在仿真环境中批量评估策略性能，生成详细的评估报告。

 评估功能:
   1. 批量 Rollout: 在多个仿真场景中运行策略，收集统计数据
   2. 指标计算: 成功率、平均奖励、任务完成时间、碰撞次数等
   3. 视频录制: 渲染策略执行过程为视频，用于定性分析
   4. 对比评估: 比较多个策略或基线在同一任务上的表现
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
from dataclasses import dataclass

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import Policy, VecEnv
from src.common.utils import ensure_dir, timestamp


@dataclass
class EvalResult:
    """
    评估结果数据结构
    ────────────────────────────────────────────────────────────────────────────
    存储一次评估的所有结果，包括统计指标和原始数据。
    """
    success_rate: float               # 成功率
    avg_reward: float                 # 平均奖励
    avg_episode_length: float         # 平均轨迹长度 (步数)
    total_episodes: int               # 总测试回合数
    rewards: List[float]              # 每个回合的奖励
    successes: List[bool]             # 每个回合的成功标志
    video_paths: List[str]            # 渲染视频路径 (可选)
    config: Dict[str, Any]            # 评估配置

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的字典"""
        return {
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_episode_length": self.avg_episode_length,
            "total_episodes": self.total_episodes,
            "config": self.config,
        }


class Evaluator:
    """
    策略评估器 (Evaluator)
    ────────────────────────────────────────────────────────────────────────────
    在指定的仿真环境中批量评估策略。
    """
    def __init__(self, env_fn: Callable[[], Any],
                 render: bool = False,
                 video_dir: Optional[str] = None):
        """
        Args:
            env_fn: 创建环境的工厂函数
            render: 是否渲染视频
            video_dir: 视频输出目录
        """
        self.env_fn = env_fn
        self.render = render
        self.video_dir = ensure_dir(video_dir) if video_dir else None

    def evaluate(self, policy: Policy, num_episodes: int = 50,
                 max_steps: int = 500) -> EvalResult:
        """
        评估策略性能

        Args:
            policy: 待评估的策略
            num_episodes: 测试回合数
            max_steps: 每个回合的最大步数

        Returns:
            EvalResult 评估结果
        """
        env = self.env_fn()
        rewards = []
        successes = []
        episode_lengths = []
        video_paths = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            frames = []

            for step in range(max_steps):
                # ── 策略推理 ────────────────────────────────────────────────
                action = policy.act(obs, deterministic=True)

                # ── 环境步进 ────────────────────────────────────────────────
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                # ── 录制帧 ──────────────────────────────────────────────────
                if self.render:
                    frame = env.render()
                    frames.append(frame)

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            successes.append(info.get("success", False))

            # ── 保存视频 ────────────────────────────────────────────────────
            if self.render and self.video_dir and frames:
                video_path = self._save_video(frames, ep)
                video_paths.append(video_path)

        env.close()

        # ── 计算统计指标 ────────────────────────────────────────────────────
        success_rate = sum(successes) / len(successes) if successes else 0.0
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

        return EvalResult(
            success_rate=success_rate,
            avg_reward=float(avg_reward),
            avg_episode_length=float(avg_length),
            total_episodes=num_episodes,
            rewards=rewards,
            successes=successes,
            video_paths=video_paths,
            config={"num_episodes": num_episodes, "max_steps": max_steps},
        )

    def _save_video(self, frames: List[np.ndarray],
                    episode_id: int) -> str:
        """将渲染帧保存为视频文件"""
        try:
            import cv2
            video_path = f"{self.video_dir}/episode_{episode_id}_{timestamp()}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # 30 FPS
                (w, h),
            )
            for frame in frames:
                # RGB → BGR (OpenCV 格式)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return video_path
        except ImportError:
            print("[Evaluator] cv2 未安装，无法保存视频")
            return ""
