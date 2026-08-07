"""
================================================================================
 数据类型定义模块 (types.py)
 ──────────────────────────────────────────────────────────────────────────────
 本模块定义框架中所有核心数据结构和类型别名，确保各层之间数据交换格式统一。
 采用 dataclass 定义结构化数据，支持序列化和类型检查。
 这是数据流的基础——所有模块通过这里定义的类型进行通信。
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举类型定义
# ═══════════════════════════════════════════════════════════════════════════════

class SensorModality(Enum):
    """传感器模态枚举
    定义机器人可能搭载的所有传感器类型，用于统一描述观测空间。
    每种模态对应特定的数据处理方式和网络编码器。
    """
    RGB = auto()           # 彩色图像, shape=(H,W,3), uint8
    DEPTH = auto()         # 深度图, shape=(H,W,1), float32 (米)
    POINT_CLOUD = auto()   # 点云, shape=(N,3), float32 (xyz)
    ROBOT_STATE = auto()   # 机器人本体状态 (关节角度/力矩), shape=(D,)
    FORCE_TORQUE = auto()  # 力/力矩传感器, shape=(6,)
    TACTILE = auto()       # 触觉传感器
    AUDIO = auto()         # 音频信号
    LANGUAGE = auto()      # 语言指令 (tokenized)


class RobotType(Enum):
    """机器人类型枚举
    用于数据集注册和仿真配置中标识机器人形态。
    不同机器人的运动学模型、控制频率、观测空间不同。
    """
    FRANKA_EMIKA = auto()      # Franka Emika Panda 7-DoF 机械臂
    XARM = auto()              # UFACTORY xArm 系列
    UR5 = auto()               # Universal Robots UR5
    KUKA_IIWA = auto()         # KUKA LBR iiwa
    ALLEGRO_HAND = auto()      # Allegro Hand 灵巧手
    BOSTON_DYNAMICS_SPOT = auto()  # 四足机器人
    DIFFERENTIAL_DRIVE = auto()     # 差分驱动移动机器人
    CUSTOM = auto()            # 自定义机器人


class ActionSpace(Enum):
    """动作空间类型枚举
    描述机器人动作的表示方式，不同算法/任务需要不同的动作表示。
    """
    JOINT_POSITION = auto()   # 关节位置控制, 维度 = 关节数
    JOINT_VELOCITY = auto()   # 关节速度控制
    JOINT_TORQUE = auto()     # 关节力矩控制
    EE_POSE = auto()          # 末端执行器位姿 (位置+姿态), 维度=7
    EE_DELTA_POSE = auto()    # 末端增量位姿控制
    GRIPPER = auto()          # 夹爪开合控制, 维度=1
    COMPOUND = auto()         # 复合动作 (如 EE_POSE + GRIPPER)


class DatasetSource(Enum):
    """数据集来源枚举
    标识数据集的原始出处，用于适配器选择和元信息追踪。
    """
    OPEN_X_EMBODIMENT = auto()  # Open X-Embodiment 项目 (含 BridgeData V2 等 60+ 子集)
    ROBONET = auto()            # RoboNet 数据集
    ROBOTURK = auto()           # RoboTurk 远程操控数据集
    MANISKILL = auto()          # ManiSkill 仿真原生数据
    RLBENCH = auto()            # RLBench 仿真数据
    LANGUAGE_TABLE = auto()     # Language-Table 语言条件数据
    CUSTOM = auto()             # 用户自定义数据集


# ═══════════════════════════════════════════════════════════════════════════════
# 核心数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedSample:
    """
    统一数据样本格式 (UnifiedSample)
    ────────────────────────────────────────────────────────────────────────────
    这是框架中最核心的数据结构，贯穿数据层→算法层的整个数据流。
    设计目标:
      1. 统一不同数据集 (BridgeData, RoboTurk, ManiSkill 等) 的异构格式
      2. 同时支持真实数据回放和仿真数据生成
      3. 兼容模仿学习(IL)和强化学习(RL)的训练接口

    字段说明:
      obs:           观测字典，包含多模态传感器数据
      action:        动作张量，表示机器人执行的控制指令
      reward:        奖励值 (RL 使用)，IL 中一般为 0
      done:          轨迹是否终止 ( episode 结束标志 )
      language_embed: 语言指令的嵌入向量 (可选)，用于语言条件策略
      metadata:      额外元信息，如时间戳、场景ID、任务描述等
    """
    obs: Dict[str, torch.Tensor]        # 观测字典: "rgb"→(C,H,W), "depth"→(1,H,W), "robot_state"→(D,)
    action: torch.Tensor                # 动作张量: (action_dim,) 或 (action_chunk, action_dim) 用于 ACT
    reward: float = 0.0                 # 奖励值 (标量)
    done: bool = False                  # 终止标志
    language_embed: Optional[torch.Tensor] = None  # 语言嵌入: (embed_dim,)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元信息

    def to_dict(self) -> Dict[str, Any]:
        """将样本转为字典，方便序列化和存储"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedSample":
        """从字典恢复样本"""
        return cls(**data)


@dataclass
class Trajectory:
    """
    完整轨迹数据结构 (Trajectory)
    ────────────────────────────────────────────────────────────────────────────
    表示一个完整的 episode 轨迹，由多个时间步的 UnifiedSample 组成。
    用于:
      1. 从开源数据集加载完整的演示轨迹
      2. 在仿真环境中执行 rollout 后收集的轨迹
      3. 离线 RL 的经验回放缓冲区中的完整 episode

    设计思路:
      将时间步维度分离到 Tensor 的第一维 (T, ...)，便于批量处理和时序建模。
    """
    observations: Dict[str, torch.Tensor]  # 观测字典，每个值 shape=(T, ...)
    actions: torch.Tensor                  # 动作序列, shape=(T, action_dim)
    rewards: torch.Tensor                  # 奖励序列, shape=(T,)
    dones: torch.Tensor                    # 终止标志, shape=(T,)
    language_embeds: Optional[torch.Tensor] = None  # 语言嵌入, shape=(T, embed_dim) 或 (embed_dim,) 广播
    task_id: Optional[str] = None          # 任务标识符
    success: Optional[bool] = None         # 任务是否成功 (评估用)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元信息

    @property
    def length(self) -> int:
        """轨迹长度 (时间步数)"""
        return self.actions.shape[0]

    def __len__(self) -> int:
        return self.length

    def to_list(self) -> List[UnifiedSample]:
        """将轨迹拆分为单个样本列表"""
        samples = []
        for t in range(self.length):
            sample = UnifiedSample(
                obs={k: v[t] for k, v in self.observations.items()},
                action=self.actions[t],
                reward=float(self.rewards[t]),
                done=bool(self.dones[t]),
                language_embed=self.language_embeds[t] if self.language_embeds is not None else None,
                metadata={**self.metadata, "timestep": t}
            )
            samples.append(sample)
        return samples


@dataclass
class SceneDescription:
    """
    场景描述数据结构 (SceneDescription)
    ────────────────────────────────────────────────────────────────────────────
    描述仿真环境的完整配置，用于仿真层的场景加载和领域随机化。
    支持 USD/MJCF/程序化生成三种场景定义方式。

    字段说明:
      scene_file: 场景文件路径 (USD/MJCF/URDF)
      robot_urdf: 机器人 URDF 模型文件路径
      init_pose:  机器人初始位姿 [x, y, z, qx, qy, qz, qw]
      task_name:  任务名称，用于自动匹配场景模板
      randomize:  是否启用领域随机化
      physics_params: 物理参数覆盖 (摩擦、阻尼、质量等)
    """
    scene_file: Optional[str] = None           # 场景文件 (.usd / .mjcf / .xml)
    robot_urdf: Optional[str] = None           # 机器人 URDF 文件
    init_pose: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # (x,y,z,qx,qy,qz,qw)
    task_name: str = "default"                 # 任务名称
    randomize: bool = False                    # 是否领域随机化
    physics_params: Dict[str, float] = field(default_factory=dict)  # 物理参数
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 类型别名 — 提高代码可读性
# ═══════════════════════════════════════════════════════════════════════════════

# 观测空间定义: 字典 {传感器名: (通道数, 维度描述)}
# 例如 {"rgb": (3, 128, 128), "robot_state": (7,)}
ObsSpace = Dict[str, Tuple[int, ...]]

# 动作空间定义: (动作维度, 动作空间类型)
# 例如 (7, ActionSpace.EE_POSE)
ActionSpec = Tuple[int, ActionSpace]

# 配置字典: 灵活的嵌套字典，用于 Hydra/OmegaConf 配置
ConfigDict = Dict[str, Any]

# 奖励函数签名: (观测, 动作, 下一观测) -> 标量奖励
RewardFunc = callable  # type: ignore

# 数据批次: DataLoader 产出的批次格式
DataBatch = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
