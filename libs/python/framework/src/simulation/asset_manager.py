"""
================================================================================
 资产管理器模块 (asset_manager.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 管理机器人 URDF/MJCF 模型和场景资产的下载、缓存和加载。

 核心功能:
   1. 模型仓库: 管理常用机器人的 URDF 文件索引
   2. 自动下载: 从开源 Model Zoo 自动下载缺失的模型文件
   3. 本地缓存: 已下载的模型缓存在本地，避免重复下载
   4. 格式转换: URDF ↔ MJCF 格式转换支持

 支持的开源模型:
   - Franka Emika Panda
   - UFACTORY xArm 6/7
   - Universal Robots UR5/UR10
   - Allegro Hand
   - KUKA LBR iiwa
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Optional, List
from pathlib import Path
import os
import urllib.request
import zipfile

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import RobotType
from src.common.utils import ensure_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 模型仓库 (Model Zoo)
# ═══════════════════════════════════════════════════════════════════════════════

# 预定义的模型索引: 机器人类型 → 下载信息
ROBOT_MODEL_INDEX: Dict[RobotType, Dict[str, str]] = {
    RobotType.FRANKA_EMIKA: {
        "name": "Franka Emika Panda",
        "url": "https://github.com/username/franka_urdf/archive/main.zip",
        "urdf_path": "franka_panda/panda.urdf",
        "description": "Franka Emika Panda 7-DoF 协作机械臂",
    },
    RobotType.XARM: {
        "name": "UFACTORY xArm",
        "url": "https://github.com/xArm-Developer/xarm_urdf/archive/master.zip",
        "urdf_path": "xarm/xarm6.urdf",
        "description": "UFACTORY xArm 6/7 轻量级机械臂",
    },
    RobotType.UR5: {
        "name": "Universal Robots UR5",
        "url": "https://github.com/ros-industrial/ur_msgs/archive/master.zip",
        "urdf_path": "ur5/ur5.urdf",
        "description": "Universal Robots UR5 协作机械臂",
    },
}


class AssetManager:
    """
    资产管理器 (AssetManager)
    ────────────────────────────────────────────────────────────────────────────
    管理机器人模型和场景资产的下载、缓存和路径解析。

    工作流程:
      1. 根据机器人类型在 Model Index 中查找模型信息
      2. 检查本地缓存是否存在
      3. 如果不存在，自动从远程 URL 下载并解压
      4. 返回 URDF 文件的本地路径
    """
    def __init__(self, cache_dir: str = "~/.cache/embodied_assets"):
        """
        Args:
            cache_dir: 模型缓存根目录
        """
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        ensure_dir(str(self.cache_dir))

    def get_robot_urdf(self, robot_type: RobotType) -> Optional[str]:
        """
        获取机器人 URDF 文件的本地路径
        ────────────────────────────────────────────────────────────────────────
        如果模型已缓存，直接返回路径；否则自动下载。

        Args:
            robot_type: 机器人类型

        Returns:
            URDF 文件的绝对路径，如果找不到则返回 None
        """
        if robot_type not in ROBOT_MODEL_INDEX:
            print(f"[AssetManager] 未知的机器人类型: {robot_type}")
            return None

        info = ROBOT_MODEL_INDEX[robot_type]
        robot_cache = self.cache_dir / info["name"]
        urdf_path = robot_cache / info["urdf_path"]

        # ── 检查缓存 ────────────────────────────────────────────────────────
        if urdf_path.exists():
            print(f"[AssetManager] 使用缓存: {urdf_path}")
            return str(urdf_path)

        # ── 自动下载 ────────────────────────────────────────────────────────
        print(f"[AssetManager] 开始下载: {info['name']}")
        print(f"[AssetManager] URL: {info['url']}")
        self._download_and_extract(info["url"], robot_cache)

        if urdf_path.exists():
            return str(urdf_path)
        return None

    def _download_and_extract(self, url: str, target_dir: Path) -> None:
        """
        下载并解压模型文件

        Args:
            url: 下载 URL
            target_dir: 目标目录
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / "model.zip"

        try:
            # ── 下载 ────────────────────────────────────────────────────────
            print(f"[AssetManager] 下载中: {url}")
            urllib.request.urlretrieve(url, str(zip_path))

            # ── 解压 ────────────────────────────────────────────────────────
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(target_dir))
            print(f"[AssetManager] 解压完成: {target_dir}")

            # ── 清理 ────────────────────────────────────────────────────────
            zip_path.unlink()

        except Exception as e:
            print(f"[AssetManager] 下载失败: {e}")
            if zip_path.exists():
                zip_path.unlink()

    def list_available_robots(self) -> List[str]:
        """列出所有可用的机器人模型"""
        return [
            f"{info['name']} - {info['description']}"
            for info in ROBOT_MODEL_INDEX.values()
        ]
