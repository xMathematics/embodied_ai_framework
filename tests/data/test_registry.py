"""
================================================================================
 数据集注册中心测试 (test_registry.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/data/registry.py 中的 DatasetRegistry 类。

 测试重点:
   1. 从 YAML 文件加载注册表
   2. 按条件过滤数据集
   3. 运行时动态注册新数据集
   4. 边界情况处理 (空注册表、找不到数据集等)
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import yaml

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.data.registry import DatasetRegistry, DatasetMeta
from src.common.types import DatasetSource, RobotType, SensorModality


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数: 创建测试用 YAML 配置文件
# ═══════════════════════════════════════════════════════════════════════════════

def create_test_registry_yaml(tmp_path: Path, datasets: list = None) -> str:
    """
    创建测试用的注册表 YAML 文件
    ────────────────────────────────────────────────────────────────────────────
    在临时目录中创建一个模拟的数据集注册表配置。
    避免测试依赖真实的外部配置文件。
    """
    if datasets is None:
        datasets = [
            {
                "name": "test_dataset_1",
                "source": "OPEN_X_EMBODIMENT",
                "description": "测试数据集 1",
                "url": "https://test.url/dataset1",
                "format": "rlds",
                "robot_type": "FRANKA_EMIKA",
                "modalities": ["RGB", "ROBOT_STATE"],
                "adapter_name": "TestAdapter1",
                "num_trajs": 1000,
                "version": "1.0",
                "license": "MIT",
                "citation": "Test Citation 1",
            },
            {
                "name": "test_dataset_2",
                "source": "ROBOTURK",
                "description": "测试数据集 2",
                "url": "https://test.url/dataset2",
                "format": "hdf5",
                "robot_type": "UR5",
                "modalities": ["RGB", "DEPTH"],
                "adapter_name": "TestAdapter2",
                "num_trajs": 500,
                "version": "2.0",
                "license": "CC-BY",
                "citation": "Test Citation 2",
            },
        ]

    config = {"datasets": datasets}
    config_path = tmp_path / "test_registry.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集注册中心测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetRegistry:
    """
    DatasetRegistry 综合测试
    ────────────────────────────────────────────────────────────────────────────
    覆盖注册中心的全部核心功能。
    """

    def test_empty_registry(self):
        """
        测试空注册表
        ────────────────────────────────────────────────────────────────────────
        验证: 未加载任何配置时，注册表为空
        """
        registry = DatasetRegistry(registry_path=None)
        assert registry.count == 0

    def test_load_from_yaml(self, tmp_path: Path):
        """测试从 YAML 加载"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        assert registry.count == 2

    def test_get_existing_dataset(self, tmp_path: Path):
        """测试获取已注册的数据集"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        meta = registry.get("test_dataset_1")
        assert meta is not None
        assert meta.name == "test_dataset_1"
        assert meta.source == DatasetSource.OPEN_X_EMBODIMENT

    def test_get_nonexistent_dataset(self, tmp_path: Path):
        """测试获取不存在的数据集应返回 None"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        meta = registry.get("nonexistent_dataset")
        assert meta is None

    def test_filter_by_source(self, tmp_path: Path):
        """测试按来源过滤"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        results = registry.filter_by_source(DatasetSource.OPEN_X_EMBODIMENT)
        assert len(results) == 1
        assert results[0].name == "test_dataset_1"

    def test_filter_by_robot(self, tmp_path: Path):
        """测试按机器人类型过滤"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        results = registry.filter_by_robot(RobotType.UR5)
        assert len(results) == 1
        assert results[0].name == "test_dataset_2"

    def test_register_new_dataset(self):
        """测试动态注册新数据集"""
        registry = DatasetRegistry(registry_path=None)
        assert registry.count == 0

        meta = DatasetMeta(
            name="custom_dataset",
            source=DatasetSource.CUSTOM,
            robot_type=RobotType.CUSTOM,
            modalities=[SensorModality.RGB],
            adapter_name="CustomAdapter",
        )
        registry.register(meta)
        assert registry.count == 1
        retrieved = registry.get("custom_dataset")
        assert retrieved is not None
        assert retrieved.adapter_name == "CustomAdapter"

    def test_list_all(self, tmp_path: Path):
        """测试列出所有数据集"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        all_datasets = registry.list_all()
        assert len(all_datasets) == 2

    def test_get_adapter_name(self, tmp_path: Path):
        """测试获取适配器名称"""
        yaml_path = create_test_registry_yaml(tmp_path)
        registry = DatasetRegistry(yaml_path)
        adapter = registry.get_adapter_name("test_dataset_1")
        assert adapter == "TestAdapter1"
        # 不存在的数据集
        assert registry.get_adapter_name("unknown") is None

    def test_yaml_not_found(self):
        """测试 YAML 文件不存在时不会报错"""
        registry = DatasetRegistry("/nonexistent/path.yaml")
        assert registry.count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetMeta 数据结构测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetMeta:
    """
    数据集元信息数据结构测试
    """

    def test_create_minimal_meta(self):
        """测试创建最小必需的元信息"""
        meta = DatasetMeta(
            name="test",
            source=DatasetSource.CUSTOM,
        )
        assert meta.name == "test"
        # 验证默认值
        assert len(meta.modalities) == 0
        assert meta.version == "latest"
        assert meta.license == "unknown"

    def test_to_dict_serialization(self):
        """测试序列化为字典"""
        meta = DatasetMeta(
            name="test",
            source=DatasetSource.MANISKILL,
            num_trajs=5000,
        )
        data = meta.to_dict()
        assert data["name"] == "test"
        assert data["source"] == "MANISKILL"
        assert data["num_trajs"] == 5000
        assert "version" in data
