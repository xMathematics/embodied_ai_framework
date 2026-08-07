"""
================================================================================
 实验管理器模块 (experiment_manager.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 管理实验的全生命周期 (创建、配置、运行、记录、恢复)。

 本模块将 Hydra (配置管理) 和 MLflow (实验追踪) 集成在一起:
   - Hydra: 用于层次化配置，支持命令行覆盖和配置文件组合
   - MLflow: 用于记录指标、超参数、模型产物

 设计思路:
   实验管理是科学可复现性的基础。每次实验记录:
     - 完整配置 (超参数、数据路径、环境版本)
     - 训练指标曲线 (损失、奖励、成功率)
     - 模型权重检查点
     - 评估视频和日志
  所有这些信息存储在 MLflow 中，方便后续分析和比较。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, Any
from pathlib import Path
import os
import json

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import ExperimentManager
from src.common.utils import timestamp, ensure_dir


class MLflowExperimentManager(ExperimentManager):
    """
    MLflow 实验管理器
    ────────────────────────────────────────────────────────────────────────────
    使用 MLflow 追踪实验。

    MLflow 追踪的核心概念:
      - Experiment: 一个实验 (包含多次 run)
      - Run: 一次实验运行 (唯一 ID)
      - Params: 超参数 (key-value)
      - Metrics: 训练指标 (随时间变化的数值)
      - Artifacts: 产物文件 (模型权重、视频、配置等)
    """
    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: str = "embodied_ai"):
        """
        Args:
            tracking_uri: MLflow 追踪 URI (如 "file:///mlruns" 或 "http://localhost:5000")
            experiment_name: 实验名称
        """
        self.experiment_name = experiment_name
        self._current_run_id = None

        try:
            import mlflow
            self.mlflow = mlflow

            # ── 设置追踪 URI ────────────────────────────────────────────────
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            # ── 创建或获取实验 ──────────────────────────────────────────────
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"[MLflow] 创建实验: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"[MLflow] 使用已有实验: {experiment_name}")

        except ImportError:
            print("[MLflow] 未安装 mlflow，使用本地日志模式")
            self.mlflow = None
            self._log_dir = ensure_dir(f"logs/{experiment_name}")

    def create_experiment(self, name: str, config: ConfigDict) -> str:
        """创建新的实验运行"""
        if self.mlflow is not None:
            # ── mlflow 模式 ─────────────────────────────────────────────────
            run = self.mlflow.start_run(run_name=name)
            self._current_run_id = run.info.run_id
            self.mlflow.log_params(self._flatten_config(config))
            return self._current_run_id
        else:
            # ── 本地日志模式 ────────────────────────────────────────────────
            run_id = f"{name}_{timestamp()}"
            self._current_run_id = run_id
            run_dir = Path(self._log_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # 保存配置
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

            return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录超参数"""
        if self.mlflow is not None:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """记录指标"""
        if self.mlflow is not None and self._current_run_id:
            with self.mlflow.start_run(run_id=self._current_run_id):
                for key, value in metrics.items():
                    self.mlflow.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str) -> None:
        """记录产物"""
        if self.mlflow is not None and self._current_run_id:
            with self.mlflow.start_run(run_id=self._current_run_id):
                self.mlflow.log_artifact(file_path)

    def get_best_checkpoint(self, metric: str, mode: str = "max") -> Optional[str]:
        """获取最佳检查点"""
        # 实际应用中需要查询 MLflow 后台
        return None

    def _flatten_config(self, config: Dict, prefix: str = "") -> Dict:
        """展平嵌套配置 (MLflow 要求扁平 key)"""
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        return flat


# 类型别名 (抑制 IDE 警告)
ConfigDict = Dict[str, Any]
