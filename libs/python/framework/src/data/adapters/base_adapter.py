"""
================================================================================
 适配器基类与工厂模块 (base_adapter.py)
 ──────────────────────────────────────────────────────────────────────────────
 提供适配器的基础设施:
   1. 适配器基类: 封装通用的数据集处理逻辑 (下载、缓存、格式检测)
   2. 适配器工厂 (AdapterFactory): 根据数据集名称自动选择和实例化适配器

 工厂模式说明:
   AdapterFactory 实现了简单工厂模式——只需提供数据集名称，工厂自动
   找到对应的适配器类并返回实例。这使得数据加载代码与具体适配器解耦。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict, Type, Optional, Iterator, Any

# ─── 内部导入 ───────────────────────────────────────────────────────────────
from src.common.interfaces import DatasetAdapter
from src.common.types import UnifiedSample


# ═══════════════════════════════════════════════════════════════════════════════
# 适配器工厂
# ═══════════════════════════════════════════════════════════════════════════════

class AdapterFactory:
    """
    适配器工厂类 (AdapterFactory)
    ────────────────────────────────────────────────────────────────────────────
    职责: 根据数据集名称自动创建对应的适配器实例。

    原理:
      维护一个 {数据集名: 适配器类} 的映射表。当请求某个数据集的适配器时，
      通过映射表找到对应的类并实例化。新增适配器需要在工厂中注册。

    典型用法:
        factory = AdapterFactory()
        adapter = factory.create("bridge_v2")
        for sample in adapter.load("/data/bridge_v2"):
            ...
    """
    def __init__(self):
        # 适配器注册表: 数据集名称 → 适配器类
        self._adapters: Dict[str, Type[DatasetAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[DatasetAdapter]) -> None:
        """
        注册适配器类
        Args:
            name: 数据集名称 (如 "bridge_v2")
            adapter_cls: 适配器类 (必须继承 DatasetAdapter)
        """
        self._adapters[name] = adapter_cls
        print(f"[AdapterFactory] 已注册适配器: {name} -> {adapter_cls.__name__}")

    def create(self, name: str, **kwargs) -> Optional[DatasetAdapter]:
        """
        创建适配器实例
        Args:
            name: 数据集名称
            **kwargs: 传递给适配器构造函数的参数
        Returns:
            适配器实例，如果未注册则返回 None
        """
        adapter_cls = self._adapters.get(name)
        if adapter_cls is None:
            print(f"[AdapterFactory] 未找到适配器: {name}")
            return None
        return adapter_cls(**kwargs)

    def can_handle(self, name: str) -> bool:
        """检查是否有对应的适配器"""
        return name in self._adapters
