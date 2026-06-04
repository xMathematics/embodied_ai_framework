"""
================================================================================
 容器化支持模块 (container.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供 Docker/Kubernetes 环境的集成支持，包括容器内 GPU 检测、
       环境变量管理、K8s Pod 信息解析等。

 设计目标:
   1. 自动感知是否运行在容器内，适配资源访问方式
   2. 解析 K8s 下发的环境变量 (如 Pod 名称、命名空间、资源限制)
   3. 为 Docker 镜像构建提供入口脚本和健康检查支持

 背景:
   在 K8s 环境中，每个模块 (数据预处理、训练、评估) 运行在独立的 Pod 中，
   通过共享存储 (PVC) 交换数据。本模块负责解析容器运行时的环境信息，
   使框架能自适应地调整资源使用策略。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# 容器环境信息
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContainerInfo:
    """容器运行时环境信息
    ────────────────────────────────────────────────────────────────────────────
    用于在容器内运行时获取平台和环境上下文。
    """
    in_container: bool          # 是否运行在容器内
    in_kubernetes: bool         # 是否运行在 K8s 中
    pod_name: str               # Pod 名称 (仅在 K8s 中有意义)
    pod_namespace: str          # Pod 命名空间
    cpu_limit: float            # CPU 限制 (核数)
    memory_limit_gb: float      # 内存限制 (GB)
    gpu_count: int              # 可用 GPU 数量


def detect_container_env() -> ContainerInfo:
    """
    检测当前运行环境是否为容器
    ────────────────────────────────────────────────────────────────────────────
    原理:
      通过检查以下标志判断是否在容器内:
        1. /.dockerenv 文件存在 (Docker 容器)
        2. /proc/1/cgroup 包含 "docker" 或 "kubepods" 关键字
        3. K8s 环境变量 HOSTNAME 与 KUBERNETES_SERVICE_HOST 存在

    Returns:
        ContainerInfo: 容器环境信息
    """
    # ── 检测容器环境 ────────────────────────────────────────────────────────
    in_docker = Path("/.dockerenv").exists()
    in_kubernetes = "KUBERNETES_SERVICE_HOST" in os.environ

    in_container = in_docker or in_kubernetes

    # ── K8s 信息 ────────────────────────────────────────────────────────────
    pod_name = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", "unknown"))
    pod_namespace = os.environ.get("POD_NAMESPACE", "default")

    # ── 资源限制 ────────────────────────────────────────────────────────────
    cpu_limit = float(os.environ.get("CPU_LIMIT", os.cpu_count() or 1))
    mem_limit_str = os.environ.get("MEMORY_LIMIT", "0")
    try:
        memory_limit_gb = float(mem_limit_str) / (1024**3)
    except ValueError:
        memory_limit_gb = 0.0

    gpu_count = int(os.environ.get("NVIDIA_VISIBLE_DEVICES",
                   os.environ.get("GPU_COUNT", "0")))

    return ContainerInfo(
        in_container=in_container,
        in_kubernetes=in_kubernetes,
        pod_name=pod_name,
        pod_namespace=pod_namespace,
        cpu_limit=cpu_limit,
        memory_limit_gb=memory_limit_gb,
        gpu_count=gpu_count,
    )


def is_running_in_container() -> bool:
    """快速判断是否运行在容器中"""
    return detect_container_env().in_container


# ═══════════════════════════════════════════════════════════════════════════════
# 环境变量管理
# ═══════════════════════════════════════════════════════════════════════════════

def get_env_with_default(key: str, default: Any = None) -> Any:
    """
    获取环境变量，支持从 ConfigMap (K8s) 和 .env 文件获取
    ────────────────────────────────────────────────────────────────────────────
    优先顺序:
      1. 操作系统环境变量
      2. .env 文件 (如果存在)
      3. 默认值

    Args:
        key: 环境变量名
        default: 默认值

    Returns:
        环境变量值或默认值
    """
    # ── 检查 .env 文件 ──────────────────────────────────────────────────────
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.exists() and key not in os.environ:
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip().strip("\"'")

    return os.environ.get(key, default)
