"""
================================================================================
 VLA 大模型模块 (Vision-Language-Action Models)
 ──────────────────────────────────────────────────────────────────────────────
 集成多种 VLA 大模型，提供统一的接入、微调、推理接口。

 支持的模型:
   - OpenVLA: 开源的 7B VLA 模型 (Stanford)
   - Octo: 多任务 Transformer VLA 模型 (UC Berkeley)
   - RT-2-X: Google DeepMind 的 VLA 模型
   - π0 (Pi0): Physical Intelligence 的通用机器人基础模型
   - 自定义 VLA: 基于 HuggingFace Transformers 的自定义 VLA 模型

 核心类:
   - VLABase: VLA 模型基类，定义统一的训练和推理接口
   - OpenVLAAdapter: OpenVLA 模型的适配器实现
   - OctoAdapter: Octo 模型的适配器实现

 设计原则:
   所有 VLA 适配器继承 VLABase，提供统一的 act_with_language 接口，
   上层调用无需关心具体 VLA 模型的实现细节。
================================================================================
"""

from src.algorithm.vla.vla_base import VLABase, VLAInput, VLAOutput
from src.algorithm.vla.openvla_adapter import OpenVLAAdapter
from src.algorithm.vla.octo_adapter import OctoAdapter

__all__ = [
    "VLABase",
    "VLAInput",
    "VLAOutput",
    "OpenVLAAdapter",
    "OctoAdapter",
]
