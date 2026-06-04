#!/usr/bin/env python3
"""
================================================================================
 安装配置 (setup.py)
 ──────────────────────────────────────────────────────────────────────────────
 定义 Python 包元信息和依赖关系。使用 pip install -e . 安装为开发模式。
================================================================================
"""

from setuptools import setup, find_packages

setup(
    name="embodied-ai-framework",
    version="0.1.0",
    description="具身智能从开源数据集到模拟仿真的全流程框架",
    author="Embodied AI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # ── 深度学习框架 ────────────────────────────────────────────────────
        "torch>=2.0.0",

        # ── 数据处理 ────────────────────────────────────────────────────────
        "numpy>=1.24",
        "pyarrow>=12.0",           # Arrow 列式数据格式
        "h5py>=3.8",               # HDF5 格式支持
        "datasets>=2.10",          # HuggingFace 数据集
        "PyYAML>=6.0",             # YAML 配置解析

        # ── 图像处理 ────────────────────────────────────────────────────────
        "torchvision>=0.15",
        "Pillow>=10.0",

        # ── 仿真引擎 ────────────────────────────────────────────────────────
        # MuJoCo (pip install mujoco)
        # PyBullet (pip install pybullet)
        # Isaac Sim 需要单独安装 (Omniverse 平台)

        # ── 实验管理 ────────────────────────────────────────────────────────
        "mlflow>=2.3",             # 实验追踪

        # ── 工具 ─────────────────────────────────────────────────────────────
        "psutil>=5.9",             # 系统资源监控
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "flake8",
            "mypy",
        ],
        "mujoco": ["mujoco>=3.0"],
        "pybullet": ["pybullet>=3.2"],
        "s3": ["s3fs>=2023.0"],
        "all": [
            "mujoco>=3.0",
            "pybullet>=3.2",
            "s3fs>=2023.0",
            "opencv-python",
            "tensorboard",
        ],
    },
    entry_points={
        "console_scripts": [
            "prepare-data=scripts.prepare_data:main",
            "train=scripts.train:main",
            "eval=scripts.eval:main",
        ],
    },
)
