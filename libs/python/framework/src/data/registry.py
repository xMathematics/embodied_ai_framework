"""
================================================================================
 数据集注册中心模块 (registry.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 集中管理所有可用数据集的信息，包括下载地址、数据格式、机器人类型、
       传感器模态、许可证等元信息。

 核心功能:
   1. 从 YAML 配置文件加载数据集注册表
   2. 按条件过滤和搜索数据集 (如按机器人类型、任务类型)
   3. 自动下载与版本管理 (通过 HuggingFace datasets / DVC)
   4. 为数据加载提供元信息查询接口

 设计模式:
   注册中心模式 (Registry Pattern) —— 所有数据集信息集中存储和管理，
   新增数据集只需在 YAML 配置文件中添加一条记录，无需修改代码。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.types import DatasetSource, RobotType, SensorModality


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集注册元信息
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetMeta:
    """
    数据集元信息类 (DatasetMeta)
    ────────────────────────────────────────────────────────────────────────────
    描述一个数据集的所有元信息，用于本地加载、自动化适配器选择和路径管理。

    字段说明:
      name:          数据集名称 (简写，如 "bridge_v2")
      source:        数据来源 (Open X-Embodiment, RoboNet 等)
      description:   数据集简短描述
      local_path:    【关键】标准化数据的本地路径
                     - 指向 prepare_data.py 输出的 Arrow IPC 文件目录
                     - 训练时: UnifiedDataset(data_dir=local_path) 直接读取
                     - 支持相对路径和绝对路径
      raw_path:      原始数据本地缓存路径
                     - 指向原始数据集文件 (HDF5/TFRecord/RLDS 等)
                     - 仅 prepare_data.py 的适配器需要读取
      url:           在线下载地址 (HTTP/S3/HF Dataset ID)
                     - 仅首次获取数据时需要
      format:        原始数据格式 (TFRecord, HDF5, RLDS, pickle 等)
      robot_type:    使用的机器人类型
      modalities:    包含的传感器模态
      num_trajs:     轨迹数量 (近似值)
      version:       版本号
      citation:      引用信息 (学术论文引用用)
      adapter_name:  对应的适配器类名
      license:       开源许可证

    路径优先级 (从高到低):
      1. 命令行参数 (--local_path)
      2. 实验配置 (config.data.data_dir)
      3. 本字段 (local_path)
      4. 内置默认 (data/unified/{name})
    """
    name: str                          # 数据集名称
    source: DatasetSource              # 数据来源枚举
    description: str = ""              # 描述
    # ── 本地路径字段 (核心) ─────────────────────────────────────────────────
    local_path: str = ""               # 【本地路径】标准化 Arrow 数据目录
    raw_path: str = ""                 # 【本地路径】原始数据缓存目录
    # ── 远程来源 ───────────────────────────────────────────────────────────
    url: str = ""                      # 在线下载地址 (仅首次需要)
    # ── 数据格式信息 ────────────────────────────────────────────────────────
    format: str = ""                   # 原始格式
    robot_type: RobotType = RobotType.CUSTOM  # 机器人类型
    modalities: List[SensorModality] = field(default_factory=list)  # 传感器模态
    num_trajs: int = 0                 # 轨迹数
    version: str = "latest"            # 版本
    citation: str = ""                 # 引用
    adapter_name: str = ""             # 适配器类名
    license: str = "unknown"           # 许可证

    def to_dict(self) -> Dict[str, Any]:
        """转为字典 (用于序列化和日志)"""
        return {
            "name": self.name,
            "source": self.source.name,
            "description": self.description,
            "local_path": self.local_path,          # 本地标准化数据路径
            "raw_path": self.raw_path,               # 本地原始数据路径
            "format": self.format,
            "robot": self.robot_type.name,
            "modalities": [m.name for m in self.modalities],
            "num_trajs": self.num_trajs,
            "version": self.version,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集注册中心
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetRegistry:
    """
    数据集注册中心 (DatasetRegistry)
    ────────────────────────────────────────────────────────────────────────────
    职责: 管理所有注册数据集的生命周期。

    工作流程:
      1. 初始化时加载 YAML 配置文件中的预定义数据集列表
      2. 提供查询接口 (按名称/来源/机器人类型过滤)
      3. 运行时支持动态注册新数据集

    配置示例 (config/dataset/registry.yaml):
        datasets:
          - name: bridge_v2
            source: OPEN_X_EMBODIMENT
            url: "https://huggingface.co/datasets/bridge_v2"
            format: "rlds"
            robot_type: FRANKA_EMIKA
            modalities: ["RGB", "ROBOT_STATE"]
            adapter_name: BridgeV2Adapter
    """
    def __init__(self, registry_path: Optional[str] = None):
        """
        Args:
            registry_path: 注册表 YAML 文件路径
                           None 时搜索默认位置 config/dataset/registry.yaml
        """
        self._datasets: Dict[str, DatasetMeta] = {}

        # ── 从默认配置路径加载 ──────────────────────────────────────────────
        if registry_path is None:
            default_path = Path(__file__).resolve().parent.parent.parent / \
                "config" / "dataset" / "registry.yaml"
            if default_path.exists():
                registry_path = str(default_path)

        if registry_path and os.path.exists(registry_path):
            self.load_from_yaml(registry_path)

    def load_from_yaml(self, yaml_path: str) -> None:
        """
        从 YAML 文件加载数据集注册表
        ────────────────────────────────────────────────────────────────────────
        原理:
          解析 YAML 中定义的 datasets 列表，转换为 DatasetMeta 对象并存入字典。

        Args:
            yaml_path: YAML 配置文件路径
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for item in data.get("datasets", []):
            meta = DatasetMeta(
                name=item["name"],
                source=DatasetSource[item["source"]],
                description=item.get("description", ""),
                # ── 本地路径: 从 YAML 读取，为空时使用默认值 ────────────────
                local_path=item.get("local_path", ""),  # 如 "data/unified/bridge_v2"
                raw_path=item.get("raw_path", ""),      # 如 "data/raw/bridge_v2"
                # ── 远程来源 ────────────────────────────────────────────────
                url=item.get("url", ""),
                # ── 数据格式信息 ────────────────────────────────────────────
                format=item.get("format", ""),
                robot_type=RobotType[item.get("robot_type", "CUSTOM")],
                modalities=[SensorModality[m] for m in item.get("modalities", [])],
                num_trajs=item.get("num_trajs", 0),
                version=item.get("version", "latest"),
                citation=item.get("citation", ""),
                adapter_name=item.get("adapter_name", ""),
                license=item.get("license", "unknown"),
            )
            self.register(meta)

    def register(self, meta: DatasetMeta) -> None:
        """
        注册一个数据集
        Args:
            meta: 数据集元信息
        """
        self._datasets[meta.name] = meta
        print(f"[Registry] 已注册数据集: {meta.name} "
              f"(来源: {meta.source.name}, 适配器: {meta.adapter_name})")

    def get(self, name: str) -> Optional[DatasetMeta]:
        """按名称获取数据集元信息"""
        return self._datasets.get(name)

    def list_all(self) -> List[DatasetMeta]:
        """列出所有注册的数据集"""
        return list(self._datasets.values())

    def filter_by_source(self, source: DatasetSource) -> List[DatasetMeta]:
        """按数据来源过滤"""
        return [d for d in self._datasets.values() if d.source == source]

    def filter_by_robot(self, robot: RobotType) -> List[DatasetMeta]:
        """按机器人类型过滤"""
        return [d for d in self._datasets.values() if d.robot_type == robot]

    def get_adapter_name(self, dataset_name: str) -> Optional[str]:
        """获取数据集对应的适配器类名"""
        meta = self.get(dataset_name)
        return meta.adapter_name if meta else None

    @property
    def count(self) -> int:
        """已注册数据集数量"""
        return len(self._datasets)
