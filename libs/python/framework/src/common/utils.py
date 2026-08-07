"""
================================================================================
 通用工具函数模块 (utils.py)
 ──────────────────────────────────────────────────────────────────────────────
 提供框架各层共用的工具函数，包括日志配置、配置加载、文件操作等。
 设计为纯函数集合，不维护状态，便于复用和测试。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 日志系统
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(
    name: str = "embodied_ai",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    配置全局日志系统
    ────────────────────────────────────────────────────────────────────────────
    原理:
      使用 Python 标准 logging 模块，配置统一的日志格式和级别。
      同时输出到控制台 (便于实时监控) 和文件 (便于事后审计)。

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径，None 则只输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    # ── 创建 Logger ────────────────────────────────────────────────────────
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 清除已有处理器，避免重复

    # ── 定义日志格式 ────────────────────────────────────────────────────────
    # 格式: [时间] [级别] [模块:行号] - 消息
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── 控制台处理器 ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── 文件处理器 (可选) ───────────────────────────────────────────────────
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    ────────────────────────────────────────────────────────────────────────────
    原理:
      使用 PyYAML 库加载 YAML 格式的配置文件。YAML 支持层次化结构、
      注释和锚点引用，非常适合复杂实验的配置管理。

    支持的格式:
      - .yaml / .yml: 使用 PyYAML 加载
      - .json: 使用 json 模块加载

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 不支持的格式
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            # YAML 支持锚点 (&) 和别名 (*) 实现配置复用
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并配置字典
    ────────────────────────────────────────────────────────────────────────────
    原理:
      递归合并两个配置字典，override 的键会覆盖 base 的同名键。
      这实现了"基础配置 + 局部覆盖"的配置管理策略，避免重复定义。

    典型用法:
      config = merge_config(default_config, experiment_specific_config)

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_config(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 文件与路径工具
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: str) -> str:
    """确保目录存在，如果不存在则创建 (类似 mkdir -p)
    Args:
        path: 目录路径
    Returns:
        规范化后的目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return os.path.abspath(path)


def get_project_root() -> Path:
    """
    获取项目根目录
    ────────────────────────────────────────────────────────────────────────────
    原理:
      从当前文件向上查找，直到找到包含特定标记文件 (如 .project_root) 的目录。
      如果没有标记文件，则回退到当前文件所在目录的父目录。

    Returns:
        项目根目录的 Path 对象
    """
    current = Path(__file__).resolve()
    # 向上查找，最多 10 层
    for _ in range(10):
        if (current / ".project_root").exists():
            return current
        current = current.parent
    # 回退: 假设项目根目录为 src/ 的父目录
    return Path(__file__).resolve().parent.parent.parent


def timestamp() -> str:
    """返回当前时间戳字符串，用于文件名和日志"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ═══════════════════════════════════════════════════════════════════════════════
# 性能工具
# ═══════════════════════════════════════════════════════════════════════════════

def timer(func):
    """
    函数执行时间装饰器
    ────────────────────────────────────────────────────────────────────────────
    原理:
      使用装饰器模式，在函数执行前后记录时间戳，计算耗时并输出日志。
      用于性能分析和瓶颈定位。

    用法:
        @timer
        def train_epoch():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[Timer] {func.__name__} 耗时: {elapsed:.3f} 秒")
        return result
    return wrapper


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    ────────────────────────────────────────────────────────────────────────────
    分别统计可训练参数和总参数量，帮助评估模型复杂度。

    Args:
        model: PyTorch 模型

    Returns:
        {"trainable": 可训练参数量, "total": 总参数量}
    """
    import torch
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total}
