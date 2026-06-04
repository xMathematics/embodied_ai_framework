"""
================================================================================
 抽象接口定义模块 (interfaces.py)
 ──────────────────────────────────────────────────────────────────────────────
 本模块定义框架中各层之间的核心抽象接口 (Abstract Base Class)。
 采用 Python 的 ABC 机制实现契约式编程，各模块只需继承并实现抽象方法即可
 接入框架，实现了"面向接口编程"的设计原则。

 设计思想:
   - 每个抽象类定义一组清晰的 method signature，作为层与层之间的合约
   - 新增实现时无需修改框架核心代码，符合开闭原则 (Open/Closed Principle)
   - 方便单元测试——可注入 Mock 实现进行独立测试
================================================================================
"""

from __future__ import annotations

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import torch
import numpy as np

# ─── 内部导入 ─────────────────────────────────────────────────────────────
from src.common.types import (
    UnifiedSample, Trajectory, SceneDescription,
    ObsSpace, ActionSpec, ConfigDict
)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据层接口
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetAdapter(ABC):
    """
    数据集适配器抽象基类 (DatasetAdapter)
    ────────────────────────────────────────────────────────────────────────────
    职责: 将特定格式的原始数据集转换为框架统一的 UnifiedTrajectory 格式。

    为什么需要适配器模式?
      不同开源数据集 (Open X-Embodiment→TFRecord/RLDS, ManiSkill→HDF5, RoboTurk→pickle)
      的存储格式和字段命名完全不同。适配器模式将"转换逻辑"封装到独立的类中，
      新增数据集只需编写一个新的 Adapter 类，无需修改数据层核心代码。

    子类必须实现:
      can_handle(source) -> bool   : 判断是否能处理该数据源
      load(source) -> Iterator[UnifiedSample] : 加载并转换为统一格式
    """
    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """判断该适配器是否能处理指定的数据源
        Args:
            source: 数据源路径或标识符
        Returns:
            True 如果该适配器能处理此数据源
        """
        pass

    @abstractmethod
    def load(self, source: str, **kwargs) -> Iterator[UnifiedSample]:
        """加载原始数据并转换为统一格式
        Args:
            source: 数据源路径或标识符 (本地路径 / S3 URI / HF Dataset ID)
            **kwargs: 额外加载参数 (如采样比例、帧率控制)
        Yields:
            UnifiedSample: 统一格式的逐帧样本
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """返回数据集元信息 (机器人类型、传感器模态、总帧数等)"""
        pass


class DataAugmenter(ABC):
    """
    数据增强器抽象基类 (DataAugmenter)
    ────────────────────────────────────────────────────────────────────────────
    职责: 对标准化后的数据进行增强处理，提升策略泛化能力。

    增强分为两类:
      1. 图像级增强: 随机裁剪、色彩抖动、遮挡模拟 (适用于 RGB/Depth)
      2. 轨迹级增强: 时间重采样、视角合成、动作噪声注入

    使用策略模式 (Strategy Pattern)，不同增强方式可自由组合。
    """
    @abstractmethod
    def augment_sample(self, sample: UnifiedSample) -> UnifiedSample:
        """增强单个样本
        Args:
            sample: 输入的标准化样本
        Returns:
            增强后的样本
        """
        pass

    @abstractmethod
    def augment_trajectory(self, traj: Trajectory) -> Trajectory:
        """增强完整轨迹 (轨迹级增强)
        Args:
            traj: 输入的标准化轨迹
        Returns:
            增强后的轨迹
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真层接口
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatorBackend(ABC):
    """
    仿真后端抽象基类 (SimulatorBackend)
    ────────────────────────────────────────────────────────────────────────────
    职责: 统一封装不同物理仿真引擎 (Isaac Sim, MuJoCo, PyBullet, SAPIEN) 的 API。

    为什么需要抽象仿真后端?
      不同仿真引擎的 API 风格差异巨大:
        - Isaac Sim: 基于 Omniverse Kit, Python API 通过 carb 和 omni 模块调用
        - MuJoCo: 轻量级 mjModel/mjData 结构体驱动
        - PyBullet: 基于 b3Client 的远端调用模式
      通过抽象接口，算法层只需与 SimulatorBackend 交互，无需关心底层引擎差异。
      这也使得切换仿真引擎只需修改一行配置。
    """
    @abstractmethod
    def __init__(self, config: ConfigDict):
        """初始化仿真后端
        Args:
            config: 仿真配置参数 (引擎路径、渲染设置、物理步长等)
        """
        pass

    @abstractmethod
    def load_scene(self, scene: SceneDescription) -> Any:
        """加载场景
        Args:
            scene: 场景描述对象
        Returns:
            SceneHandle: 场景句柄，用于后续操作
        """
        pass

    @abstractmethod
    def spawn_robot(self, urdf_path: str, position: Tuple[float, ...],
                    quaternion: Tuple[float, ...]) -> Any:
        """在场景中生成机器人
        Args:
            urdf_path: URDF 模型路径
            position: 初始位置 (x, y, z)
            quaternion: 初始姿态 (qx, qy, qz, qw)
        Returns:
            RobotHandle: 机器人句柄
        """
        pass

    @abstractmethod
    def step(self, action: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """执行一步仿真
        这是仿真引擎最核心的接口——接收动作指令，返回观测和奖励。

        Args:
            action: 动作字典 {关节名/动作名: 控制量}
        Returns:
            观测字典，包含:
                - "obs": 传感器观测
                - "reward": 当前步奖励 (标量)
                - "done": 是否终止
                - "info": 额外信息 (碰撞检测、任务进度等)
        """
        pass

    @abstractmethod
    def get_observation(self, sensor_spec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """获取指定传感器的观测数据
        Args:
            sensor_spec: 传感器规格 {传感器名: 配置参数}
        Returns:
            观测数据字典
        """
        pass

    @abstractmethod
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """直接设置物理状态 (用于重置、初始化、插入轨迹)
        Args:
            qpos: 关节位置数组
            qvel: 关节速度数组
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置仿真环境到初始状态"""
        pass

    @abstractmethod
    def close(self) -> None:
        """释放仿真引擎资源 (GPU 内存、渲染上下文等)"""
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """仿真时间步长 (秒)"""
        pass


class VecEnv(ABC):
    """
    向量化环境抽象接口 (VecEnv)
    ────────────────────────────────────────────────────────────────────────────
    职责: 同时管理多个并行的仿真环境，实现数据并行采样。

    为什么需要 VecEnv?
      强化学习的训练效率高度依赖并行采样——同时运行 N 个环境可以让 GPU
      利用率饱和。VecEnv 提供统一的"多环境单控制"接口，算法层看到的是一个
      批量环境，内部可使用多进程/多线程/多 GPU 实现真正的并行。
    """
    @abstractmethod
    def reset(self) -> List[Dict[str, torch.Tensor]]:
        """重置所有环境
        Returns:
            每个环境的初始观测列表 [obs_0, obs_1, ..., obs_{N-1}]
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[
        List[Dict[str, torch.Tensor]],  # 观测
        torch.Tensor,                   # 奖励, shape=(N,)
        torch.Tensor,                   # 终止标志, shape=(N,)
        List[Dict]                      # info 列表
    ]:
        """在所有环境中执行一步动作
        这是并行采样的核心——一次调用驱动 N 个环境各前进一步。

        Args:
            actions: 动作批量, shape=(N, action_dim)
        Returns:
            (obs_list, rewards, dones, infos) 四元组
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭所有环境"""
        pass

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """并行环境数量"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 算法层接口
# ═══════════════════════════════════════════════════════════════════════════════

class Policy(ABC):
    """
    策略抽象基类 (Policy)
    ────────────────────────────────────────────────────────────────────────────
    职责: 定义所有策略网络的统一接口。无论是模仿学习还是强化学习策略，
          都通过此接口与环境和训练循环交互。

    设计考虑:
      - act() 接收观测输出动作，是策略的核心推理方法
      - deterministic 参数控制随机/确定性推理，评估时通常用确定性模式
      - save/load 方法统一模型序列化接口，支持 ONNX/TorchScript 导出
    """
    @abstractmethod
    def act(self, obs: Dict[str, torch.Tensor],
            deterministic: bool = False) -> torch.Tensor:
        """根据观测输出动作
        Args:
            obs: 观测字典 (来自环境)
            deterministic: 是否确定性推理
                True: 取均值/最大概率动作 (评估时使用)
                False: 从分布采样 (训练时使用)
        Returns:
            动作张量, shape=(action_dim,)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型权重
        Args:
            path: 保存路径 (.pt / .onnx / .torchscript)
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型权重
        Args:
            path: 权重文件路径
        """
        pass


class RLTrainer(ABC):
    """
    强化学习训练器抽象基类 (RLTrainer)
    ────────────────────────────────────────────────────────────────────────────
    职责: 统一不同 RL 算法 (PPO, SAC, DrQ-v2, DreamerV3) 的训练流程。

    RL 训练的核心循环:
      1. collect: 策略在环境中采样，收集经验
      2. train:   从经验池采样 batch，更新策略/价值网络
      3. evaluate: 在验证环境中评估当前策略
    """
    @abstractmethod
    def collect(self, env: VecEnv, num_steps: int) -> Dict[str, Any]:
        """从环境中收集经验
        Args:
            env: 向量化环境
            num_steps: 收集步数
        Returns:
            收集统计信息 (总奖励、步数等)
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练
        Args:
            batch: 从经验池采样的训练数据
        Returns:
            训练指标字典 (loss, value_loss, entropy 等)
        """
        pass

    @abstractmethod
    def evaluate(self, env: VecEnv, num_episodes: int) -> Dict[str, float]:
        """评估策略性能
        Args:
            env: 评估环境
            num_episodes: 测试回合数
        Returns:
            评估指标 (成功率、平均奖励等)
        """
        pass


class ImitationTrainer(ABC):
    """
    模仿学习训练器抽象基类 (ImitationTrainer)
    ────────────────────────────────────────────────────────────────────────────
    职责: 统一不同 IL 算法 (BC, ACT, Diffusion Policy) 的训练流程。

    IL 训练的特点:
      - 不需要与环境交互 (与 RL 的最大区别)
      - 直接监督学习: 以专家轨迹的动作为标签，最小化预测误差
      - 数据集是固定的，不需要经验回放
    """
    @abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """训练一个 epoch
        Args:
            dataloader: 数据加载器，提供专家轨迹数据
        Returns:
            训练指标 (loss, accuracy 等)
        """
        pass

    @abstractmethod
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """验证集评估
        Args:
            dataloader: 验证数据加载器
        Returns:
            验证指标
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 编排层接口
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentManager(ABC):
    """
    实验管理器抽象接口 (ExperimentManager)
    ────────────────────────────────────────────────────────────────────────────
    职责: 管理实验全生命周期 (创建、运行、记录、恢复)。

    使用实验管理器的好处:
      - 自动记录每次实验的超参数、代码版本、环境信息
      - 支持实验中断后恢复 (resume)
      - 所有实验指标集中管理，便于比较和分析
    """
    @abstractmethod
    def create_experiment(self, name: str, config: ConfigDict) -> str:
        """创建新实验
        Args:
            name: 实验名称
            config: 实验配置
        Returns:
            experiment_id: 实验唯一标识
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录超参数"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """记录指标 (损失、奖励、成功率等)"""
        pass

    @abstractmethod
    def log_artifact(self, file_path: str) -> None:
        """记录产物 (模型权重、视频、配置文件等)"""
        pass

    @abstractmethod
    def get_best_checkpoint(self, metric: str, mode: str = "max") -> Optional[str]:
        """获取最佳检查点路径
        Args:
            metric: 指标名
            mode: "max" 越大越好, "min" 越小越好
        Returns:
            最佳检查点路径
        """
        pass
