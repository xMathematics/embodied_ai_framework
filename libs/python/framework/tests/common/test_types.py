"""
================================================================================
 数据类型测试 (test_types.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/common/types.py 中定义的所有核心数据结构和枚举类型。

 为什么先测试数据类型?
   数据类型是框架的"地基"——所有模块都依赖 UnifiedSample、Trajectory 等
   核心数据结构。如果基础数据类型有问题，上层所有逻辑都会受影响。
   因此数据类型的测试价值最高，应该在所有测试之前运行通过。
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
from typing import Dict

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.common.types import (
    # ── 枚举类型 ──
    SensorModality,
    RobotType,
    ActionSpace,
    DatasetSource,
    # ── 数据类型 ──
    UnifiedSample,
    Trajectory,
    SceneDescription,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举类型测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensorModality:
    """
    传感器模态枚举测试
    ────────────────────────────────────────────────────────────────────────────
    验证枚举成员的正确性和完整性。确保所有传感器类型都被正确定义。
    """

    def test_all_modalities_defined(self):
        """测试所有必需传感器模态是否定义完整"""
        # 列出框架需要支持的所有传感器类型
        required_modalities = [
            "RGB", "DEPTH", "POINT_CLOUD", "ROBOT_STATE",
            "FORCE_TORQUE", "TACTILE", "AUDIO", "LANGUAGE",
        ]
        for mod in required_modalities:
            # 验证: SensorModality 枚举包含该成员
            assert hasattr(SensorModality, mod), f"缺少传感器模态: {mod}"

    def test_modalities_unique(self):
        """测试枚举值唯一性 (auto() 应生成不同的值)"""
        values = [m.value for m in SensorModality]
        # 验证: 所有枚举值互不相同
        assert len(values) == len(set(values)), "存在重复的枚举值!"

    def test_modalities_count(self):
        """测试模态数量与预期一致"""
        assert len(SensorModality) == 8, (
            f"预期 8 种传感器模态, 实际 {len(SensorModality)}"
        )


class TestRobotType:
    """机器人类型枚举测试"""

    def test_all_robots_defined(self):
        """测试所有必需的机器人类型是否定义"""
        required = [
            "FRANKA_EMIKA", "XARM", "UR5", "KUKA_IIWA",
            "ALLEGRO_HAND", "BOSTON_DYNAMICS_SPOT",
            "DIFFERENTIAL_DRIVE", "CUSTOM",
        ]
        for robot in required:
            assert hasattr(RobotType, robot), f"缺少机器人类型: {robot}"

    def test_robot_types_count(self):
        assert len(RobotType) == 8, f"预期 8 种机器人类型, 实际 {len(RobotType)}"


class TestActionSpace:
    """动作空间枚举测试"""

    def test_all_action_spaces_defined(self):
        required = [
            "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE",
            "EE_POSE", "EE_DELTA_POSE", "GRIPPER", "COMPOUND",
        ]
        for space in required:
            assert hasattr(ActionSpace, space), f"缺少动作空间: {space}"

    def test_action_spaces_count(self):
        assert len(ActionSpace) == 7, f"预期 7 种动作空间, 实际 {len(ActionSpace)}"


class TestDatasetSource:
    """数据集来源枚举测试"""

    def test_all_sources_defined(self):
        required = [
            "OPEN_X_EMBODIMENT", "ROBONET", "ROBOTURK",
            "MANISKILL", "RLBENCH", "LANGUAGE_TABLE", "CUSTOM",
        ]
        for src in required:
            assert hasattr(DatasetSource, src), f"缺少数据来源: {src}"


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedSample 数据结构测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnifiedSample:
    """
    UnifiedSample 数据结构测试
    ────────────────────────────────────────────────────────────────────────────
    UnifiedSample 是框架中最核心的数据结构，贯穿数据层→算法层的整个数据流。
    它的正确性直接影响所有上层模块。
    """

    def test_create_basic_sample(self):
        """
        测试创建基本样本
        ────────────────────────────────────────────────────────────────────────
        验证: 使用最小必需参数创建 UnifiedSample 是否正常工作
        """
        sample = UnifiedSample(
            obs={"rgb": torch.rand(3, 128, 128)},
            action=torch.rand(7),
        )
        # 验证: 必填字段赋值正确
        assert "rgb" in sample.obs
        assert sample.obs["rgb"].shape == (3, 128, 128)
        assert sample.action.shape == (7,)

    def test_create_sample_with_all_fields(self):
        """
        测试创建包含所有字段的完整样本
        """
        sample = UnifiedSample(
            obs={
                "rgb": torch.rand(3, 128, 128),
                "depth": torch.rand(1, 128, 128),
                "robot_state": torch.rand(7),
            },
            action=torch.rand(8),
            reward=0.5,
            done=True,
            language_embed=torch.rand(512),
            metadata={"task": "pick_place", "scene": "kitchen"},
        )
        # 验证: 所有字段正确赋值
        assert sample.reward == 0.5
        assert sample.done is True
        assert sample.language_embed.shape == (512,)
        assert sample.metadata["task"] == "pick_place"

    def test_default_values(self):
        """
        测试默认值
        ────────────────────────────────────────────────────────────────────────
        验证: 未显式提供的可选字段使用了正确的默认值
        """
        sample = UnifiedSample(
            obs={"rgb": torch.rand(3, 128, 128)},
            action=torch.rand(7),
        )
        # reward 默认应为 0.0
        assert sample.reward == 0.0, f"期望 reward=0.0, 实际={sample.reward}"
        # done 默认应为 False
        assert sample.done is False, f"期望 done=False, 实际={sample.done}"
        # language_embed 默认应为 None
        assert sample.language_embed is None
        # metadata 默认应为空字典
        assert sample.metadata == {}

    def test_to_dict_serialization(self):
        """
        测试序列化为字典
        ────────────────────────────────────────────────────────────────────────
        验证: to_dict() 方法正确地将 dataclass 转换为可序列化的字典
        这在保存数据到文件或通过网络传输时非常关键。
        """
        sample = UnifiedSample(
            obs={"rgb": torch.rand(3, 128, 128)},
            action=torch.rand(7),
            reward=1.0,
            done=True,
        )
        data = sample.to_dict()
        # 验证: 字典包含所有关键字段
        assert isinstance(data, dict)
        assert "obs" in data
        assert "action" in data
        assert data["reward"] == 1.0
        assert data["done"] is True

    def test_from_dict_deserialization(self):
        """
        测试从字典恢复
        ────────────────────────────────────────────────────────────────────────
        验证: from_dict() 方法正确地从字典恢复 UnifiedSample 对象
        测试序列化→反序列化的往返 (round-trip) 一致性。
        """
        original = UnifiedSample(
            obs={"rgb": torch.rand(3, 128, 128)},
            action=torch.rand(7),
            reward=0.8,
            done=False,
            metadata={"key": "value"},
        )
        # 序列化→反序列化
        data = original.to_dict()
        restored = UnifiedSample.from_dict(data)
        # 验证: 恢复后的数据与原始一致
        assert restored.reward == original.reward
        assert restored.done == original.done
        assert restored.metadata == original.metadata
        # 注意: torch.Tensor 需要逐元素比较
        assert torch.equal(restored.action, original.action)

    def test_obs_dict_structure(self):
        """
        测试观测字典结构
        ────────────────────────────────────────────────────────────────────────
        验证: UnifiedSample 接受各种类型的 obs (Python dataclass 不强制运行时
        类型检查，但代码逻辑应正确保存传入的值)
        """
        # Python dataclass 不强制运行时类型检查 (PEP 484 仅做静态类型提示)
        # 这里验证的是: 传入的值被正确保存
        sample = UnifiedSample(obs="invalid_obs", action=torch.rand(7))  # type: ignore
        assert sample.obs == "invalid_obs"  # dataclass 直接保存传入的值

    def test_action_types(self):
        """
        测试动作类型灵活接受
        ────────────────────────────────────────────────────────────────────────
        验证: UnifiedSample 的 action 字段可以接受各种可转换为 Tensor 的类型
        (实际使用中应在使用时进行类型检查)
        """
        # Python dataclass 不强制类型检查，验证值被正确保存
        sample = UnifiedSample(obs={"rgb": torch.rand(3, 128, 128)}, action=[1, 2, 3])  # type: ignore
        assert sample.action == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory 数据结构测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectory:
    """
    Trajectory 数据结构测试
    ────────────────────────────────────────────────────────────────────────────
    Trajectory 表示一个完整的 episode 轨迹，由多个时间步组成。
    它是 IL 训练中的基本数据单元。
    """

    def test_create_trajectory(self):
        """测试创建轨迹对象"""
        traj = Trajectory(
            observations={
                "rgb": torch.rand(10, 3, 128, 128),         # (T, C, H, W)
                "robot_state": torch.rand(10, 7),            # (T, D)
            },
            actions=torch.rand(10, 7),                        # (T, action_dim)
            rewards=torch.rand(10),                           # (T,)
            dones=torch.zeros(10, dtype=torch.bool),          # (T,)
        )
        assert traj.length == 10
        assert len(traj) == 10

    def test_empty_trajectory_detection(self):
        """
        测试空轨迹检测
        ────────────────────────────────────────────────────────────────────────
        验证: 长度为 0 的轨迹能被正确识别
        """
        traj = Trajectory(
            observations={"rgb": torch.rand(0, 3, 128, 128)},
            actions=torch.rand(0, 7),
            rewards=torch.rand(0),
            dones=torch.zeros(0, dtype=torch.bool),
        )
        assert traj.length == 0
        assert len(traj) == 0

    def test_to_list_conversion(self):
        """
        测试轨迹拆分为样本列表
        ────────────────────────────────────────────────────────────────────────
        验证: to_list() 方法将完整的轨迹拆分为单个 UnifiedSample 列表。
        这是从"轨迹级"到"样本级"转换的关键方法。
        """
        traj = Trajectory(
            observations={
                "rgb": torch.rand(5, 3, 128, 128),
                "robot_state": torch.rand(5, 7),
            },
            actions=torch.rand(5, 7),
            rewards=torch.rand(5),
            dones=torch.tensor([False, False, False, False, True]),
        )
        samples = traj.to_list()
        # 验证: 拆分出的样本数量正确
        assert len(samples) == 5
        # 验证: 每个样本都是 UnifiedSample 类型
        assert all(isinstance(s, UnifiedSample) for s in samples)
        # 验证: 最后一个样本应该是 done=True
        assert samples[-1].done is True

    def test_trajectory_immutability(self):
        """
        测试轨迹不可变性 (部分)
        ────────────────────────────────────────────────────────────────────────
        验证: Trajectory 的字段类型是固定的
        """
        traj = Trajectory(
            observations={"rgb": torch.rand(3, 3, 128, 128)},
            actions=torch.rand(3, 7),
            rewards=torch.rand(3),
            dones=torch.zeros(3, dtype=torch.bool),
        )
        # 验证: actions 是 Tensor 类型
        assert isinstance(traj.actions, torch.Tensor)
        # 验证: dones 是 bool 类型
        assert traj.dones.dtype == torch.bool


# ═══════════════════════════════════════════════════════════════════════════════
# SceneDescription 数据结构测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSceneDescription:
    """
    场景描述数据结构测试
    """

    def test_create_scene(self):
        """测试创建场景描述"""
        scene = SceneDescription(
            scene_file="/path/to/scene.usd",
            robot_urdf="franka_panda/panda.urdf",
            task_name="pick_place",
            randomize=True,
            physics_params={"friction": 0.5, "damping": 1.0},
        )
        assert scene.task_name == "pick_place"
        assert scene.randomize is True
        assert scene.physics_params["friction"] == 0.5

    def test_default_init_pose(self):
        """
        测试默认初始位姿
        ────────────────────────────────────────────────────────────────────────
        验证: 默认初始位姿是单位四元数 (无旋转)
        """
        scene = SceneDescription()
        assert len(scene.init_pose) == 7
        # 单位四元数: qw=1.0, qx=qz=qy=0.0
        assert scene.init_pose[3:] == (0.0, 0.0, 0.0, 1.0), "默认姿态应为单位四元数"
