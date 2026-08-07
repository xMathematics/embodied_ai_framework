"""
================================================================================
 VLA 模型基类 (VLABase)
 ──────────────────────────────────────────────────────────────────────────────
 定义所有 VLA 大模型的统一抽象接口，包括输入输出格式、训练、推理、微调方法。

 设计思路:
   VLA 模型的核心能力是"视觉 + 语言 → 动作"的映射。
   不同 VLA 模型 (OpenVLA, Octo, RT-2) 的实现细节不同，
   但对外接口应该统一。本基类通过模板方法模式 (Template Method)
   固定调用流程，子类只需实现具体的模型加载和前向逻辑。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# VLA 数据接口
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VLAInput:
    """
    VLA 模型的统一输入格式
    ────────────────────────────────────────────────────────────────────────────
    字段:
      - rgb_images: 多视角 RGB 图像序列 (T, V, C, H, W)
      - language_instruction: 自然语言指令
      - robot_state: 本体感受状态 (关节位置/速度/力矩)
      - history_frames: 历史帧数
    """
    rgb_images: List[torch.Tensor]       # 多视角 RGB 图像序列
    language_instruction: str             # 自然语言指令
    robot_state: Optional[torch.Tensor] = None  # 本体感受状态
    history_frames: int = 2              # 历史帧数


@dataclass
class VLAOutput:
    """
    VLA 模型的统一输出格式
    ────────────────────────────────────────────────────────────────────────────
    字段:
      - action: 预测动作
      - action_chunk: 动作块 (多步预测)
      - language_reasoning: 中间推理文本 (VLM 的思维链)
      - confidence: 动作置信度 (用于融合决策)
    """
    action: torch.Tensor                       # 预测动作 (action_dim,)
    action_chunk: Optional[torch.Tensor] = None  # 动作块 (future_steps, action_dim)
    language_reasoning: Optional[str] = None    # 中间推理文本
    confidence: float = 1.0                    # 动作置信度 [0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# VLA 模型基类
# ═══════════════════════════════════════════════════════════════════════════════

class VLABase(nn.Module, ABC):
    """
    VLA 大模型抽象基类
    ────────────────────────────────────────────────────────────────────────────
    职责: 提供统一的 VLA 模型接入接口，所有具体的 VLA 模型适配器继承此类。

    子类必须实现:
      - _load_model(): 加载具体的 VLA 模型权重
      - _encode_vision(): 编码视觉输入为特征
      - _encode_language(): 编码语言指令为特征
      - _predict_action(): 从多模态特征预测动作

    可选覆写:
      - _get_confidence(): 计算模型对当前决策的置信度
      - _generate_reasoning(): 生成中间推理文本
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.is_loaded = False

    # ── 公共接口 ──────────────────────────────────────────────────────────

    def act_with_language(
        self,
        obs: Dict,
        instruction: str,
        deterministic: bool = False
    ) -> VLAOutput:
        """
        接收语言指令的决策接口

        Args:
            obs: 观测字典，包含 "rgb", "depth", "robot_state" 等键
            instruction: 自然语言指令
            deterministic: 是否确定性推理

        Returns:
            VLAOutput: 包含动作和元信息的输出
        """
        if not self.is_loaded:
            raise RuntimeError(f"VLA model {self.model_name} is not loaded. Call load() first.")

        # 1. 编码视觉观测
        rgb_images = obs.get("rgb", [])
        vis_features = self._encode_vision(rgb_images)

        # 2. 编码语言指令
        lang_features = self._encode_language(instruction)

        # 3. 获取机器人状态
        robot_state = obs.get("robot_state", None)

        # 4. 预测动作
        action, action_chunk = self._predict_action(
            vis_features, lang_features, robot_state, deterministic
        )

        # 5. 可选: 计算置信度
        confidence = self._get_confidence(vis_features, lang_features)

        # 6. 可选: 生成推理文本
        reasoning = self._generate_reasoning(vis_features, lang_features) if deterministic else None

        return VLAOutput(
            action=action,
            action_chunk=action_chunk,
            language_reasoning=reasoning,
            confidence=confidence,
        )

    # ── 模型生命周期 ──────────────────────────────────────────────────────

    @abstractmethod
    def load(self, pretrained_path: Optional[str] = None, **kwargs):
        """加载预训练权重"""
        ...

    def save(self, path: str):
        """保存模型权重"""
        torch.save(self.state_dict(), path)

    def set_lora_weights(self, adapter_path: str):
        """
        动态加载/切换 LoRA 适配器
        支持多任务快速切换: 不同任务使用不同的 LoRA 适配器
        """
        raise NotImplementedError("LoRA weight switching not implemented for this model")

    # ── 抽象方法 ───────────────────────────────────────────────────────────

    @abstractmethod
    def _encode_vision(self, rgb_images: List[torch.Tensor]) -> torch.Tensor:
        """编码视觉观测为特征向量"""
        ...

    @abstractmethod
    def _encode_language(self, instruction: str) -> torch.Tensor:
        """编码语言指令为特征向量"""
        ...

    @abstractmethod
    def _predict_action(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        robot_state: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """从多模态特征预测动作"""
        ...

    def _get_confidence(self, vis_features: torch.Tensor,
                        lang_features: torch.Tensor) -> float:
        """
        计算模型对当前决策的置信度
        默认实现: 返回 1.0 (完全置信)，子类可以基于 logits 熵计算置信度
        """
        return 1.0

    def _generate_reasoning(self, vis_features: torch.Tensor,
                            lang_features: torch.Tensor) -> Optional[str]:
        """
        生成中间推理文本 (思维链)
        默认实现: 返回 None，子类 (如 RT-2 类 VLM) 可以覆写
        """
        return None

    # ── 训练支持 ───────────────────────────────────────────────────────────

    def get_trainable_params(self, method: str = "lora"):
        """
        根据微调方式返回可训练参数

        Args:
            method: 微调方式
                - "full": 全参数微调
                - "lora": LoRA 参数高效微调
                - "qlora": QLoRA 量化微调

        Returns:
            需要优化的参数列表
        """
        if method == "full":
            return self.parameters()
        elif method in ("lora", "qlora"):
            # 返回 LoRA 参数 (需要模型实现 apply_lora 方法)
            return [p for n, p in self.named_parameters() if "lora" in n]
        else:
            raise ValueError(f"Unknown finetune method: {method}")
