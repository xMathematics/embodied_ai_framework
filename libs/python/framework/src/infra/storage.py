"""
================================================================================
 存储管理模块 (storage.py)
 ──────────────────────────────────────────────────────────────────────────────
 职责: 提供统一的存储抽象层，管理数据集、模型权重、日志等文件的存储访问。

 核心功能:
   1. 本地文件系统与远端存储 (S3/MinIO/CephFS) 的统一接口
   2. 数据集版本管理 (DVC 集成)
   3. 路径与缓存管理 (避免重复下载)

 设计思路:
   采用"存储后端"模式——定义抽象存储接口，本地磁盘和远端对象存储各实现一套。
   数据层和算法层通过 StorageBackend 接口访问数据，无需关心数据实际存储位置。
   这支持无缝切换开发环境 (本地) 和生产环境 (K8s + 对象存储)。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import shutil
from pathlib import Path
from typing import Optional, List, BinaryIO, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
# 可选依赖: s3fs / boto3 (用于 S3 兼容存储)
try:
    import s3fs
    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False


# ═══════════════════════════════════════════════════════════════════════════════
# 存储后端抽象接口
# ═══════════════════════════════════════════════════════════════════════════════

class StorageBackend(ABC):
    """
    存储后端抽象基类
    ────────────────────────────────────────────────────────────────────────────
    定义统一的文件操作接口，支持本地文件系统和远端对象存储。
    所有文件路径在框架内部统一使用 POSIX 风格 "/" 分隔符。
    """
    @abstractmethod
    def exists(self, path: str) -> bool:
        """检查文件或目录是否存在"""
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """读取二进制文件"""
        pass

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """写入二进制文件"""
        pass

    @abstractmethod
    def list_dir(self, path: str) -> List[str]:
        """列出目录内容"""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """删除文件或目录"""
        pass

    @abstractmethod
    def copy(self, src: str, dst: str) -> None:
        """复制文件"""
        pass


class LocalStorage(StorageBackend):
    """
    本地文件系统存储后端
    ────────────────────────────────────────────────────────────────────────────
    适用于开发环境和单机训练。
    所有路径相对于项目根目录或为绝对路径。
    """
    def __init__(self, root_dir: Optional[str] = None):
        self.root = Path(root_dir) if root_dir else Path.cwd()

    def _resolve(self, path: str) -> Path:
        """将相对路径解析为绝对路径"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.root / p

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def list_dir(self, path: str) -> List[str]:
        target = self._resolve(path)
        if not target.exists():
            return []
        return [str(p) for p in target.iterdir()]

    def delete(self, path: str) -> None:
        target = self._resolve(path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    def copy(self, src: str, dst: str) -> None:
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


class S3Storage(StorageBackend):
    """
    S3 兼容对象存储后端
    ────────────────────────────────────────────────────────────────────────────
    适用于生产环境和高性能分布式存储。
    支持 MinIO、CephFS、AWS S3 等兼容 S3 API 的对象存储系统。

    优势:
      - 对象存储天然具备高可用性和扩展性
      - 数据集和模型权重可直接在集群间共享，无需手动传输
      - 支持版本管理 (需要 S3 版本控制)
    """
    def __init__(self, bucket: str, endpoint: Optional[str] = None,
                 key: Optional[str] = None, secret: Optional[str] = None):
        if not HAS_S3FS:
            raise ImportError("s3fs 未安装。请执行: pip install s3fs")
        self.bucket = bucket
        self.fs = s3fs.S3FileSystem(
            endpoint_url=endpoint,
            key=key,
            secret=secret,
        )

    def s3_path(self, path: str) -> str:
        """将路径转换为 S3 完整路径"""
        return f"{self.bucket}/{path.lstrip('/')}"

    def exists(self, path: str) -> bool:
        return self.fs.exists(self.s3_path(path))

    def read_bytes(self, path: str) -> bytes:
        return self.fs.read_bytes(self.s3_path(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        self.fs.write_bytes(self.s3_path(path), data)

    def list_dir(self, path: str) -> List[str]:
        s3_path = self.s3_path(path)
        if not self.fs.exists(s3_path):
            return []
        return self.fs.ls(s3_path)

    def delete(self, path: str) -> None:
        self.fs.delete(self.s3_path(path), recursive=True)

    def copy(self, src: str, dst: str) -> None:
        self.fs.copy(self.s3_path(src), self.s3_path(dst))


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集缓存管理器
# ═══════════════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    数据集缓存管理器 (CacheManager)
    ────────────────────────────────────────────────────────────────────────────
    职责: 管理已下载和处理的数据集的本地缓存，避免重复下载和重复处理。

    缓存策略:
      1. 首次访问时从远端下载并缓存到本地高速存储 (NVMe / tmpfs)
      2. 再次访问时直接使用缓存，跳过下载和处理步骤
      3. 支持缓存逐出策略 (LRU) 以控制磁盘使用量

    原理:
      为每个数据集生成唯一缓存键 (基于数据集名+版本+处理参数)，在本地缓存
      目录检查是否存在对应的缓存文件。这显著减少了重复的数据准备时间。
    """
    def __init__(self, cache_dir: str = "/tmp/embodied_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, dataset_name: str, version: str = "latest",
                   transforms_hash: str = "") -> str:
        """生成缓存键 (用于唯一标识缓存版本)
        Args:
            dataset_name: 数据集名称
            version: 版本号
            transforms_hash: 预处理流程的哈希值 (参数变化时缓存失效)
        Returns:
            缓存键字符串
        """
        return f"{dataset_name}_{version}_{transforms_hash}"

    def get_cached_path(self, dataset_name: str, version: str = "latest",
                        transforms_hash: str = "") -> Optional[Path]:
        """获取缓存路径 (如果存在)
        Returns:
            缓存路径，如果不存在则返回 None
        """
        key = self._cache_key(dataset_name, version, transforms_hash)
        cached = self.cache_dir / key
        return cached if cached.exists() else None

    def set_cached(self, dataset_name: str, source_path: str,
                   version: str = "latest", transforms_hash: str = "") -> Path:
        """将数据存入缓存
        Args:
            source_path: 源数据路径
        Returns:
            缓存路径
        """
        key = self._cache_key(dataset_name, version, transforms_hash)
        cache_path = self.cache_dir / key
        cache_path.mkdir(parents=True, exist_ok=True)

        # ── 如果源路径存在，复制到缓存 ──────────────────────────────────────
        src = Path(source_path)
        if src.exists():
            if src.is_file():
                shutil.copy2(src, cache_path / src.name)
            else:
                shutil.copytree(src, cache_path / src.name, dirs_exist_ok=True)

        return cache_path

    def evict(self, dataset_name: str, version: str = "latest") -> None:
        """清除指定数据集的缓存"""
        key = self._cache_key(dataset_name, version)
        target = self.cache_dir / key
        if target.exists():
            shutil.rmtree(target)
            print(f"[Cache] 已清除缓存: {key}")
