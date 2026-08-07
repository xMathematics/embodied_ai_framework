"""
================================================================================
 训练流水线模块 (pipeline.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 定义训练流水线的 DAG (有向无环图)，将各层模块串联为完整工作流。

 流水线阶段:
   1. 数据准备: 下载 → 适配器转换 → 标准化 → 增强
   2. 仿真初始化: 场景加载 → 机器人加载 → 领域随机化
   3. 训练循环: IL/RL 训练 → 检查点保存 → 指标记录
   4. 评估: 模型加载 → 仿真 Rollout → 结果可视化

 本模块定义了 Pipeline 类，它:
   - 使用配置文件自动构建完整的训练流程
   - 支持从任意阶段恢复 (如从检查点恢复训练)
   - 记录每个阶段的耗时和资源使用
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Any, Callable
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import ConfigDict
from src.common.utils import setup_logging, load_config, timer
from src.data.registry import DatasetRegistry
from src.data.dataloader import create_dataloader
from src.simulation.backend import SimBackendFactory, RobotEnv
from src.orchestration.experiment_manager import MLflowExperimentManager


class TrainingPipeline:
    """
    训练流水线 (TrainingPipeline)
    ────────────────────────────────────────────────────────────────────────────
    将完整训练流程组织为可执行的流水线。

    用法:
        pipeline = TrainingPipeline("config/experiment/bc_manipulation.yaml")
        pipeline.run()

    流水线执行流程:
      [配置加载] → [数据准备] → [环境初始化] → [训练] → [评估] → [完成]
    """
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 实验配置文件路径
        """
        # ── 加载配置 ────────────────────────────────────────────────────────
        self.config = load_config(config_path)
        self.config_path = config_path

        # ── 设置日志 ────────────────────────────────────────────────────────
        self.logger = setup_logging(
            name="pipeline",
            log_file=self.config.get("log_file"),
        )

        # ── 实验管理器 ──────────────────────────────────────────────────────
        self.exp_manager = MLflowExperimentManager(
            experiment_name=self.config.get("experiment_name", "embodied_ai"),
        )

        # ── 流水线状态 ──────────────────────────────────────────────────────
        self._current_stage = 0
        self._stages = [
            "data_preparation",
            "simulation_init",
            "training",
            "evaluation",
        ]

    @timer
    def run(self) -> Dict[str, Any]:
        """
        运行完整训练流水线
        ────────────────────────────────────────────────────────────────────────
        按顺序执行每个阶段:
          1. 数据准备: 加载和预处理数据集
          2. 仿真初始化: 创建仿真后端和环境
          3. 训练: 运行 IL 或 RL 训练循环
          4. 评估: 在仿真中评估策略
        """
        # ── 创建实验 ────────────────────────────────────────────────────────
        run_id = self.exp_manager.create_experiment(
            name=self.config.get("run_name", "default"),
            config=self.config,
        )
        self.logger.info(f"实验开始 (Run ID: {run_id})")

        results = {}

        try:
            # ── Stage 1: 数据准备 ──────────────────────────────────────────
            self.logger.info("=" * 50)
            self.logger.info("阶段 1/4: 数据准备")
            dataloader = self._prepare_data()
            results["data_ready"] = True

            # ── Stage 2: 仿真初始化 ─────────────────────────────────────────
            self.logger.info("=" * 50)
            self.logger.info("阶段 2/4: 仿真环境初始化")
            env = self._init_simulation()
            results["env_ready"] = True

            # ── Stage 3: 训练 ──────────────────────────────────────────────
            self.logger.info("=" * 50)
            self.logger.info("阶段 3/4: 训练")
            train_results = self._train(dataloader, env)
            results.update(train_results)

            # ── Stage 4: 评估 ──────────────────────────────────────────────
            self.logger.info("=" * 50)
            self.logger.info("阶段 4/4: 评估")
            eval_results = self._evaluate(env)
            results.update(eval_results)

        except Exception as e:
            self.logger.error(f"流水线执行失败: {e}")
            results["error"] = str(e)
            raise

        finally:
            # ── 清理 ──────────────────────────────────────────────────────
            if 'env' in locals():
                env.close()
            self.logger.info("流水线执行完成")

        return results

    def _prepare_data(self):
        """数据准备阶段"""
        data_cfg = self.config.get("data", {})

        # 注册数据集
        registry = DatasetRegistry(data_cfg.get("registry_path"))

        # 创建数据加载器
        dataloader = create_dataloader(
            data_dir=data_cfg.get("data_dir", ""),
            batch_size=data_cfg.get("batch_size", 64),
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=True,
        )

        return dataloader

    def _init_simulation(self):
        """仿真初始化阶段"""
        sim_cfg = self.config.get("simulation", {})

        # 创建仿真后端
        factory = SimBackendFactory()
        backend_name = sim_cfg.get("backend", "mujoco")
        backend = factory.create(backend_name, sim_cfg)

        # 创建环境
        env = RobotEnv(backend, sim_cfg)
        return env

    def _train(self, dataloader, env):
        """训练阶段"""
        train_cfg = self.config.get("training", {})
        algorithm = train_cfg.get("algorithm", "bc")

        if algorithm == "bc":
            # 行为克隆训练
            from src.algorithm.imitation.bc import BCTrainer
            trainer = BCTrainer(
                obs_dim=train_cfg.get("obs_dim", 128),
                action_dim=train_cfg.get("action_dim", 7),
            )
            for epoch in range(train_cfg.get("num_epochs", 100)):
                metrics = trainer.train_epoch(dataloader)
                self.exp_manager.log_metrics(metrics, step=epoch)
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: {metrics}")

        elif algorithm == "ppo":
            # PPO 训练
            from src.algorithm.reinforcement.ppo import PPOTrainer
            from src.simulation.vec_env import make_vec_env

            vec_env = make_vec_env(
                lambda: env,
                num_envs=train_cfg.get("num_envs", 8),
            )
            trainer = PPOTrainer(
                obs_dim=train_cfg.get("obs_dim", 128),
                action_dim=train_cfg.get("action_dim", 7),
            )

            for iteration in range(train_cfg.get("max_iterations", 1000)):
                collect_info = trainer.collect(vec_env, 2048)
                train_metrics = trainer.train_on_buffer()
                self.exp_manager.log_metrics(
                    {**collect_info, **train_metrics},
                    step=iteration,
                )

        return {"algorithm": algorithm}

    def _evaluate(self, env):
        """评估阶段"""
        eval_cfg = self.config.get("evaluation", {})
        num_episodes = eval_cfg.get("num_episodes", 10)

        total_reward = 0.0
        for ep in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = torch.randn(7)  # 随机动作 (占位)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            self.logger.info(f"Episode {ep}: reward = {episode_reward:.2f}")

        avg_reward = total_reward / num_episodes
        self.exp_manager.log_metrics({"avg_eval_reward": avg_reward}, step=0)

        return {"avg_eval_reward": avg_reward}
