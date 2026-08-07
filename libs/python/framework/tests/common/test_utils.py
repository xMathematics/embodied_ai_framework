"""
================================================================================
 工具函数测试 (test_utils.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/common/utils.py 中的所有工具函数:
   - setup_logging:    日志配置
   - load_config:      配置加载
   - merge_config:     配置合并
   - ensure_dir:       目录创建
   - get_project_root: 项目根路径检测
   - timer:            计时装饰器
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import json
import tempfile
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import yaml

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.common.utils import (
    setup_logging,
    load_config,
    merge_config,
    ensure_dir,
    get_project_root,
    count_parameters,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 日志系统测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetupLogging:
    """
    日志配置测试
    ────────────────────────────────────────────────────────────────────────────
    验证日志系统是否正确初始化:
      - Logger 名称是否正确
      - 日志级别是否正确
      - 是否可以写入日志文件
    """

    def test_logger_creation(self):
        """测试创建日志记录器"""
        logger = setup_logging("test_logger")
        # 验证: Logger 名称正确
        assert logger.name == "test_logger"
        # 验证: 默认级别为 INFO
        assert logger.level == 20  # INFO

    def test_logger_level(self):
        """测试日志级别设置"""
        logger = setup_logging("test_debug", level=10)  # DEBUG
        assert logger.level == 10

    def test_logger_file_output(self, tmp_path: Path):
        """测试日志文件输出"""
        log_file = str(tmp_path / "test.log")
        logger = setup_logging("test_file", log_file=log_file)
        # 写入测试日志
        logger.info("测试日志消息")
        # 验证: 文件已创建且包含日志内容
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
        assert "测试日志消息" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 配置加载测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadConfig:
    """
    配置文件加载测试
    ────────────────────────────────────────────────────────────────────────────
    验证 YAML 和 JSON 配置文件的正确加载。
    """

    def test_load_yaml(self, tmp_path: Path):
        """测试加载 YAML 配置文件"""
        config = {"key": "value", "number": 42, "nested": {"inner": True}}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(str(config_path))
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        assert loaded["nested"]["inner"] is True

    def test_load_json(self, tmp_path: Path):
        """测试加载 JSON 配置文件"""
        config = {"key": "json_value", "list": [1, 2, 3]}
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_config(str(config_path))
        assert loaded["key"] == "json_value"
        assert loaded["list"] == [1, 2, 3]

    def test_file_not_found(self):
        """测试文件不存在时抛出异常"""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_unsupported_format(self, tmp_path: Path):
        """测试不支持的格式抛出异常"""
        config_path = tmp_path / "test.txt"
        config_path.write_text("some data")
        with pytest.raises(ValueError, match="不支持的配置文件格式"):
            load_config(str(config_path))


# ═══════════════════════════════════════════════════════════════════════════════
# 配置合并测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMergeConfig:
    """
    配置合并功能测试
    ────────────────────────────────────────────────────────────────────────────
    验证 merge_config 函数的正确性:
      1. 基础覆盖: 覆盖简单键
      2. 嵌套递归: 递归合并嵌套字典
      3. 新增键: 添加基础配置中没有的键
      4. 不变性: 不修改原始配置
    """

    def test_simple_override(self):
        """测试简单键的覆盖"""
        base = {"lr": 0.01, "batch_size": 32}
        override = {"lr": 0.001}
        merged = merge_config(base, override)
        assert merged["lr"] == 0.001  # 被覆盖
        assert merged["batch_size"] == 32  # 保持原值

    def test_nested_merge(self):
        """测试嵌套字典的递归合并"""
        base = {"training": {"lr": 0.01, "epochs": 10}, "data": {"path": "/data"}}
        override = {"training": {"lr": 0.001}}  # 只覆盖 lr
        merged = merge_config(base, override)
        assert merged["training"]["lr"] == 0.001
        assert merged["training"]["epochs"] == 10  # 未被覆盖
        assert merged["data"]["path"] == "/data"

    def test_new_keys_added(self):
        """测试添加新键"""
        base = {"existing": 1}
        override = {"new_key": "new_value"}
        merged = merge_config(base, override)
        assert merged["existing"] == 1
        assert merged["new_key"] == "new_value"

    def test_original_unchanged(self):
        """
        测试源字典不变性
        ────────────────────────────────────────────────────────────────────────
        验证: merge_config 不修改原始字典
        """
        base = {"key": "original"}
        override = {"key": "override"}
        merge_config(base, override)
        assert base["key"] == "original", "源字典不应被修改"


# ═══════════════════════════════════════════════════════════════════════════════
# 文件操作测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsureDir:
    """目录创建测试"""

    def test_create_new_dir(self, tmp_path: Path):
        """测试创建新目录"""
        new_dir = str(tmp_path / "new" / "nested" / "dir")
        result = ensure_dir(new_dir)
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
        # verify: 返回规范化的绝对路径
        assert os.path.isabs(result)

    def test_existing_dir(self, tmp_path: Path):
        """测试已存在目录 (不应报错)"""
        existing = str(tmp_path / "existing")
        os.makedirs(existing)
        # 对已存在的目录调用 ensure_dir 不应抛出异常
        result = ensure_dir(existing)
        assert result == os.path.abspath(existing)


class TestGetProjectRoot:
    """项目根路径检测测试"""

    def test_root_has_project_marker(self):
        """
        测试项目根包含标记文件
        ────────────────────────────────────────────────────────────────────────
        验证: 项目根目录下存在 .project_root 标记文件
        """
        root = get_project_root()
        # get_project_root() 应返回 src/common/utils.py 上三级目录
        assert (root / ".project_root").exists()

    def test_root_contains_src(self):
        """测试项目根包含 src 目录"""
        root = get_project_root()
        assert (root / "src").is_dir()


# ═══════════════════════════════════════════════════════════════════════════════
# 模型参数量统计测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountParameters:
    """模型参数量统计测试"""

    def test_count_small_model(self):
        """
        测试统计小模型参数量
        ────────────────────────────────────────────────────────────────────────
        创建一个已知结构的小网络，验证参数量计算正确。
        """
        import torch.nn as nn
        model = nn.Linear(10, 5)
        counts = count_parameters(model)
        # Linear(10, 5) 有 10*5 + 5 = 55 个参数
        assert counts["total"] == 55
        assert counts["trainable"] == 55

    def test_frozen_layers(self):
        """
        测试冻结层参数统计
        ────────────────────────────────────────────────────────────────────────
        验证: 当某些层被冻结 (requires_grad=False) 时，
        可训练参数 < 总参数。
        """
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        # 冻结第一层
        for param in model[0].parameters():
            param.requires_grad = False

        counts = count_parameters(model)
        total_expected = 10 * 20 + 20 + 20 * 5 + 5  # = 325
        trainable_expected = 20 * 5 + 5  # = 105
        assert counts["total"] == total_expected
        assert counts["trainable"] == trainable_expected
