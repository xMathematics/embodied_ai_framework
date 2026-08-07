"""
================================================================================
 模块注册中心 (ModuleRegistry)
 ──────────────────────────────────────────────────────────────────────────────
 管理所有可用算法的注册与实例化，支持通过名称动态创建算法实例。

 设计模式: 注册表模式 (Registry Pattern) + 工厂方法

 使用示例:
   @ModuleRegistry.register("openvla")
   class OpenVLAAdapter(VLABase): ...

   # 运行时根据名称创建
   vla_model = ModuleRegistry.create("openvla", model_name="openvla-7b")
   rl_model = ModuleRegistry.create("ppo", obs_dim=128, action_dim=7)
================================================================================
"""

from __future__ import annotations

from typing import Dict, Type, Any


class ModuleRegistry:
    """
    模块注册中心
    ────────────────────────────────────────────────────────────────────────────
    支持三种模块类型的注册:
      - vla: VLA 大模型
      - rl: 强化学习算法
      - world_model: 世界模型
      - imitation: 模仿学习算法

    每个模块类型下可注册多个具体实现。
    """
    _registries: Dict[str, Dict[str, Type]] = {
        "vla": {},
        "rl": {},
        "world_model": {},
        "imitation": {},
    }

    @classmethod
    def register(cls, module_type: str, name: str):
        """
        装饰器: 注册算法模块到指定类型下

        Args:
            module_type: 模块类型 (vla / rl / world_model / imitation)
            name: 模块名称

        使用: @ModuleRegistry.register("vla", "openvla")
        """
        def decorator(module_class):
            if module_type not in cls._registries:
                cls._registries[module_type] = {}
            cls._registries[module_type][name] = module_class
            return module_class
        return decorator

    @classmethod
    def create(cls, module_type: str, name: str, **kwargs) -> Any:
        """
        根据模块类型和名称创建实例

        Args:
            module_type: 模块类型
            name: 模块名称
            **kwargs: 传递给模块构造函数的参数

        Returns:
            模块实例

        Raises:
            KeyError: 如果模块类型或名称未注册
        """
        if module_type not in cls._registries:
            raise KeyError(f"Unknown module type: {module_type}. "
                          f"Available: {list(cls._registries.keys())}")

        if name not in cls._registries[module_type]:
            raise KeyError(f"Unknown {module_type} module: {name}. "
                          f"Available: {list(cls._registries[module_type].keys())}")

        module_class = cls._registries[module_type][name]
        return module_class(**kwargs)

    @classmethod
    def list_modules(cls, module_type: Optional[str] = None) -> Dict[str, list]:
        """
        列出已注册的模块

        Args:
            module_type: 指定模块类型，None 则列出所有

        Returns:
            模块名称列表
        """
        if module_type:
            return {module_type: list(cls._registries.get(module_type, {}).keys())}
        return {
            t: list(modules.keys())
            for t, modules in cls._registries.items()
        }
