"""
================================================================================
 计算资源调度模块 (compute.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 管理训练和仿真任务的计算资源分配，支持单机多卡和分布式集群。

 核心功能:
   1. GPU 设备检测与自动选择
   2. 分布式训练环境初始化 (PyTorch DDP / Ray)
   3. 计算资源上下文管理器 (确保资源正确释放)

 设计思路:
   使用上下文管理器 (Context Manager) 模式管理计算资源，确保：
     - 进入上下文时自动分配设备
     - 退出上下文时自动释放 (防止 GPU 内存泄漏)
   同时支持通过环境变量手动指定 GPU (兼容 SLURM 等调度系统)。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import platform
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 设备信息数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeviceInfo:
    """
    设备信息类 (DeviceInfo)
    ────────────────────────────────────────────────────────────────────────────
    存储当前计算设备的详细信息，用于运行时决策和日志记录。
    例如: 决定是否启用混合精度、选择数据加载器工作进程数等。
    """
    device_type: str                    # "cuda", "cpu", "mps" (Apple Silicon)
    device_id: int = 0                  # 设备索引 (多 GPU 时)
    device_name: str = ""               # 设备名称 (如 "NVIDIA A100 80GB")
    total_memory_gb: float = 0.0        # 总显存/内存 (GB)
    distributed: bool = False            # 是否分布式训练
    world_size: int = 1                  # 总进程数 (分布式)
    rank: int = 0                        # 当前进程编号 (分布式)

    @property
    def device(self) -> torch.device:
        """返回 PyTorch 设备对象"""
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.device_id}")
        return torch.device(self.device_type)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def detect_devices() -> DeviceInfo:
    """
    自动检测可用计算设备
    ────────────────────────────────────────────────────────────────────────────
    原理:
      按优先级检测: CUDA GPU > Apple MPS > CPU
      检测结果决定了后续所有张量操作的默认设备。

    优先级策略:
      - NVIDIA GPU: 优先选择显存最大的设备
      - Apple Silicon: 使用 MPS (Metal Performance Shaders) 加速
      - CPU: 兜底方案

    Returns:
        DeviceInfo: 检测到的设备信息
    """
    if torch.cuda.is_available():
        # ── CUDA 可用 ───────────────────────────────────────────────────────
        device_count = torch.cuda.device_count()
        # 选择显存最大的 GPU
        free_memories = []
        for i in range(device_count):
            torch.cuda.set_device(i)
            free_memories.append(torch.cuda.get_device_properties(i).total_memory)
        best_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", str(free_memories.index(max(free_memories)))))

        props = torch.cuda.get_device_properties(best_gpu)
        return DeviceInfo(
            device_type="cuda",
            device_id=best_gpu,
            device_name=props.name,
            total_memory_gb=props.total_memory / (1024**3),  # 字节 → GB
            distributed="WORLD_SIZE" in os.environ,
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
        )
    elif torch.backends.mps.is_available():
        # ── Apple Silicon MPS ───────────────────────────────────────────────
        import psutil
        return DeviceInfo(
            device_type="mps",
            device_name=f"Apple {platform.processor()}",
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
        )
    else:
        # ── CPU 兜底 ────────────────────────────────────────────────────────
        import psutil
        return DeviceInfo(
            device_type="cpu",
            device_name=f"{platform.processor()}",
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
        )


def get_default_device() -> torch.device:
    """
    获取默认计算设备
    ────────────────────────────────────────────────────────────────────────────
    这是框架中最常用的函数之一——所有张量创建和模型加载都使用它来确定设备。

    环境变量控制:
      CUDA_VISIBLE_DEVICES=0,1   # 限制可见 GPU
      FORCE_CPU=1                 # 强制使用 CPU (调试用)

    Returns:
        torch.device
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    info = detect_devices()
    return info.device


# ═══════════════════════════════════════════════════════════════════════════════
# 分布式训练支持
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed(backend: str = "nccl", init_method: str = "env://") -> DeviceInfo:
    """
    初始化分布式训练环境
    ────────────────────────────────────────────────────────────────────────────
    原理:
      使用 PyTorch DDP (Distributed Data Parallel) 的初始化流程。
      支持两种启动方式:
        1. torch.distributed.launch: 通过环境变量传递参数 (默认)
        2. SLURM 调度系统: 自动解析 SLURM 环境变量

    Args:
        backend: 通信后端
            - "nccl":  NVIDIA GPU 专用，最高性能
            - "gloo":  CPU 或跨平台兼容
            - "mpi":   高性能计算集群
        init_method: 初始化方法

    Returns:
        DeviceInfo: 分布式设备信息
    """
    # ── 检测是否在分布式环境中 ──────────────────────────────────────────────
    if not torch.distributed.is_available():
        print("[Warning] PyTorch 分布式不可用，回退到单机模式")
        return detect_devices()

    # ── 检测 SLURM 环境 ──────────────────────────────────────────────────────
    if "SLURM_PROCID" in os.environ:
        # SLURM 环境下自动设置标准环境变量
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # ── 初始化进程组 ────────────────────────────────────────────────────────
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
        )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ── 设置当前设备 ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DeviceInfo(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        device_id=local_rank,
        device_name=torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu",
        distributed=True,
        world_size=world_size,
        rank=rank,
    )


def cleanup_distributed() -> None:
    """
    清理分布式训练环境
    ────────────────────────────────────────────────────────────────────────────
    原理:
      销毁进程组并清理资源。必须在训练结束后调用，否则 GPU 内存无法释放。
    应该在 finally 块中调用，确保异常时也能释放资源。
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
# 计算资源上下文管理器
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def device_context(device_id: Optional[int] = None):
    """
    计算资源上下文管理器
    ────────────────────────────────────────────────────────────────────────────
    原理:
      使用 Python 上下文管理器确保资源生命周期管理。
      进入上下文时设置设备，退出时清理 (适用于临时 GPU 任务)。

    用法:
        with device_context(0):
            model = MyModel().to("cuda:0")
            # 训练代码...
        # 离开上下文后自动清理

    Args:
        device_id: GPU 设备编号，None 时自动选择
    """
    # ── 进入: 配置设备 ──────────────────────────────────────────────────────
    if device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"[Infra] 使用 GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        info = detect_devices()
        print(f"[Infra] 使用设备: {info.device_type} - {info.device_name}")

    try:
        yield
    finally:
        # ── 退出: 清理 ──────────────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放未使用的缓存显存
            print("[Infra] GPU 缓存已清理")


# ═══════════════════════════════════════════════════════════════════════════════
# 混合精度训练支持
# ═══════════════════════════════════════════════════════════════════════════════

def get_autocast_context(dtype: Optional[torch.dtype] = None):
    """
    获取自动混合精度上下文
    ────────────────────────────────────────────────────────────────────────────
    原理:
      混合精度训练使用 FP16/BF16 进行前向和反向传播，FP32 更新权重。
      可以在几乎不影响收敛质量的情况下，将训练速度提升 1.5-3 倍，
      并将显存占用降低近一半。

    Args:
        dtype: 精度类型
            - torch.float16: 传统半精度 (需要梯度缩放)
            - torch.bfloat16: BF16 (不需要梯度缩放，推荐 Ampere+ 架构)

    Returns:
        autocast 上下文管理器
    """
    if dtype is None:
        # 自动选择: Ampere 架构及以上优先 BF16
        if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    return torch.cuda.amp.autocast(dtype=dtype)
