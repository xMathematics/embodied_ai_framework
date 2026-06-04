"""
================================================================================
 存储管理模块测试 (test_storage.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/infra/storage.py 中的存储后端和缓存管理功能。

 测试内容:
   1. LocalStorage: 本地文件系统操作 (读/写/复制/删除)
   2. CacheManager: 数据集缓存管理 (缓存命中/未命中/逐出)

 设计原则:
   测试中使用 tmp_path (pytest内置fixture) 创建临时目录，
   确保测试不影响实际文件系统，且结束后自动清理。
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from pathlib import Path

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.infra.storage import LocalStorage, CacheManager


# ═══════════════════════════════════════════════════════════════════════════════
# 本地存储后端测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalStorage:
    """
    本地存储后端测试
    ────────────────────────────────────────────────────────────────────────────
    验证 LocalStorage 的文件操作功能:
      - exists: 文件存在性检查
      - read_bytes / write_bytes: 二进制文件读写
      - list_dir: 目录内容列表
      - delete: 文件/目录删除
      - copy: 文件复制
    """

    @pytest.fixture(autouse=True)
    def setup_storage(self, tmp_path: Path):
        """每个测试前创建存储实例 (使用临时目录)"""
        self.storage = LocalStorage(str(tmp_path))
        self.tmp_path = tmp_path

    def test_write_and_read(self):
        """测试文件写入和读取"""
        test_data = b"Hello, Embodied AI Framework!"
        self.storage.write_bytes("test.txt", test_data)
        read_data = self.storage.read_bytes("test.txt")
        assert read_data == test_data, "写入和读取的数据不一致!"

    def test_exists(self):
        """测试文件存在性检查"""
        # 文件不存在
        assert not self.storage.exists("nonexistent.txt")
        # 写入后应存在
        self.storage.write_bytes("exists.txt", b"data")
        assert self.storage.exists("exists.txt")

    def test_list_dir(self):
        """测试目录列表"""
        # 空目录
        assert self.storage.list_dir(".") == []
        # 创建文件后
        self.storage.write_bytes("file1.txt", b"data")
        self.storage.write_bytes("file2.txt", b"data")
        files = self.storage.list_dir(".")
        assert len(files) == 2

    def test_delete_file(self):
        """测试删除文件"""
        self.storage.write_bytes("delete_me.txt", b"data")
        assert self.storage.exists("delete_me.txt")
        self.storage.delete("delete_me.txt")
        assert not self.storage.exists("delete_me.txt")

    def test_copy_file(self):
        """测试复制文件"""
        self.storage.write_bytes("source.txt", b"copy test data")
        self.storage.copy("source.txt", "dest.txt")
        assert self.storage.exists("dest.txt")
        assert self.storage.read_bytes("dest.txt") == b"copy test data"

    def test_write_creates_dirs(self):
        """
        测试写入时自动创建目录
        ────────────────────────────────────────────────────────────────────────
        验证: 目标目录不存在时，write_bytes 自动创建父目录
        """
        self.storage.write_bytes("nested/deep/dir/file.txt", b"auto created dirs")
        assert self.storage.exists("nested/deep/dir/file.txt")

    def test_binary_data_integrity(self):
        """
        测试二进制数据完整性
        ────────────────────────────────────────────────────────────────────────
        验证: 二进制数据 (非文本) 写入和读取后内容完整
        """
        import struct
        # 写入二进制结构: 4字节整数 + 8字节浮点数
        binary_data = struct.pack("if", 42, 3.14)
        self.storage.write_bytes("binary.bin", binary_data)
        read_back = self.storage.read_bytes("binary.bin")
        # 验证数据长度和内容
        assert len(read_back) == len(binary_data)
        unpacked_int, unpacked_float = struct.unpack("if", read_back)
        assert unpacked_int == 42
        assert abs(unpacked_float - 3.14) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 缓存管理器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheManager:
    """
    缓存管理器测试
    ────────────────────────────────────────────────────────────────────────────
    验证 CacheManager 的缓存功能:
      - 缓存未命中: 返回 None
      - 缓存命中: 返回正确的缓存路径
      - 缓存逐出: 删除缓存
      - 缓存键唯一性: 不同参数生成不同缓存键
    """

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path: Path):
        """每个测试前创建缓存管理器"""
        self.cache = CacheManager(str(tmp_path / "cache"))
        self.cache_dir = tmp_path / "cache"

    def test_cache_miss(self):
        """
        测试缓存未命中
        ────────────────────────────────────────────────────────────────────────
        验证: 未缓存的数据集返回 None
        """
        cached_path = self.cache.get_cached_path("nonexistent_dataset")
        assert cached_path is None, "不应命中不存在的缓存"

    def test_cache_hit(self):
        """
        测试缓存命中
        ────────────────────────────────────────────────────────────────────────
        验证: 缓存设置后可以正确命中
        """
        # 先设置缓存
        self.cache.set_cached("test_dataset", ".")
        # 再次获取应命中
        cached_path = self.cache.get_cached_path("test_dataset")
        assert cached_path is not None, "缓存应命中"
        assert cached_path.exists(), "缓存路径应存在"

    def test_cache_eviction(self):
        """测试缓存逐出"""
        self.cache.set_cached("evict_me", ".")
        assert self.cache.get_cached_path("evict_me") is not None
        self.cache.evict("evict_me")
        assert self.cache.get_cached_path("evict_me") is None

    def test_cache_key_uniqueness(self):
        """
        测试缓存键唯一性
        ────────────────────────────────────────────────────────────────────────
        验证: 不同参数生成不同的缓存键
        """
        path1 = self.cache.get_cached_path("ds", "v1")
        path2 = self.cache.get_cached_path("ds", "v2")
        # 两个不同版本的缓存键不同
        # (未缓存时都为 None，但缓存键内部不同)

        # 验证: 通过缓存键目录名判断唯一性
        self.cache.set_cached("ds", ".", "v1")
        self.cache.set_cached("ds", ".", "v2")
        cache_dir = self.cache_dir
        # 应有两个不同的缓存子目录
        assert len(list(cache_dir.iterdir())) == 2
