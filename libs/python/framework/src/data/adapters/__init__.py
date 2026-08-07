"""
================================================================================
 数据集适配器包 (Adapters)
 ──────────────────────────────────────────────────────────────────────────────
 本包包含所有数据集的适配器实现。每个适配器继承 DatasetAdapter 接口，
 将特定数据集的原始格式转换为框架统一的 UnifiedSample 格式。

 适配器模式 (Adapter Pattern) 是本层的核心设计模式：
   - 目标: 将不同的接口 (Open X-Embodiment, ManiSkill, RoboTurk 等)
           统一为框架内部的 UnifiedSample 接口
   - 效果: 新增数据集时只需编写一个新 Adapter 类，其余框架代码完全不变

 内置适配器:
   - base_adapter.py:    适配器基类和工厂类
   - bridge_v2_adapter.py:  BridgeData V2 (Open X-Embodiment 子集)
   - roboturk_adapter.py:   RoboTurk 数据集
================================================================================
"""
