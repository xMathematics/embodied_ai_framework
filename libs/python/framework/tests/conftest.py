"""
================================================================================
 Pytest 全局配置与共享 Fixture (conftest.py)
 ──────────────────────────────────────────────────────────────────────────────
 conftest.py 是 pytest 的配置文件，在此定义的 fixture 可以被同目录及
 子目录下所有测试文件共享，无需重复导入。

 Fixture 的概念:
   Fixture 是 pytest 的核心功能，用于管理测试所需的"前提条件"。
   它类似于构造函数 + 析构函数:
   - yield 之前: setUp (测试准备)
   - yield 之后: tearDown (测试清理)

 本文件定义的共享 Fixture:
   - 测试数据生成器 (合成模拟数据，不依赖真实数据集)
   - 模拟对象 (Mock) 工厂
   - 临时目录管理
   - 随机种子固定
================================================================================
"""

# ─── 标准库导入 ─────────────────────────────────────────────────────────────
import os
import sys
import tempfile
import random
from pathlib import Path
from typing import Dict, Generator, Tuple

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch
import numpy as np

# ─── 将项目根目录添加到 Python 路径 ────────────────────────────────────────
# 这样测试文件可以直接 "from src.xxx import xxx" 导入
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ═══════════════════════════════════════════════════════════════════════════════
# 全局初始化 Fixture (autouse=True 表示所有测试自动应用)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def fix_random_seed():
    """
    固定随机种子 Fixture
    ────────────────────────────────────────────────────────────────────────────
    原理:
      在每个测试执行前固定 Python/torch/numpy 的随机种子。
      这确保测试结果的可重复性——无论执行多少次，相同测试总是产生相同结果。

    为什么重要:
      深度学习中的随机性来源很多 (权重初始化、dropout、数据打乱等)。
      如果不固定种子，测试可能偶尔失败 (由于不好的随机采样)，
      导致"flaky test"问题。
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ── 设置确定性算法 (牺牲一点性能换取完全可复现) ──────────────────────
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield  # 测试执行在此处
    # 测试结束后无需额外清理


# ═══════════════════════════════════════════════════════════════════════════════
# 临时目录 Fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def tmp_data_dir() -> Generator[str, None, None]:
    """
    临时数据目录 Fixture
    ────────────────────────────────────────────────────────────────────────────
    创建一个临时目录，测试结束后自动清理。
    用于需要读写文件的测试 (如数据标准化、检查点保存)。

    scope="function": 每个测试函数使用独立的临时目录
    这避免了测试间的数据污染。
    """
    with tempfile.TemporaryDirectory(prefix="embodied_test_") as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def tmp_shared_dir() -> Generator[str, None, None]:
    """
    会话级临时目录 Fixture
    ────────────────────────────────────────────────────────────────────────────
    与 tmp_data_dir 的区别:
      scope="session": 整个测试会话共用一个临时目录
    适用于所有测试共享的大型缓存数据。
    """
    with tempfile.TemporaryDirectory(prefix="embodied_shared_") as tmp_dir:
        yield tmp_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟数据 Fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def mock_obs() -> Dict[str, torch.Tensor]:
    """
    模拟观测数据 Fixture
    ────────────────────────────────────────────────────────────────────────────
    生成一个模拟的观测字典，包含 RGB 图像和机器人状态。
    用于测试策略网络和环境交互。

    Returns:
        {"rgb": (3,128,128) float32, "robot_state": (7,) float32}
    """
    return {
        # RGB 图像: 随机像素值 [0,1]
        "rgb": torch.rand(3, 128, 128),
        # 机器人状态: 7 维 (关节位置)
        "robot_state": torch.rand(7),
    }


@pytest.fixture(scope="function")
def mock_action() -> torch.Tensor:
    """
    模拟动作数据 Fixture
    ────────────────────────────────────────────────────────────────────────────
    生成一个模拟的 7 维动作向量 (末端位姿 + 夹爪)。
    """
    return torch.rand(7)


@pytest.fixture(scope="function")
def mock_trajectory(length: int = 50) -> "Trajectory":
    """
    模拟轨迹数据 Fixture
    ────────────────────────────────────────────────────────────────────────────
    生成一条完整的模拟轨迹，包含观测序列、动作序列和奖励。
    用于测试数据加载器和训练循环。

    Args:
        length: 轨迹长度 (时间步数)

    Returns:
        Trajectory 对象
    """
    from src.common.types import Trajectory

    return Trajectory(
        observations={
            "rgb": torch.rand(length, 3, 128, 128),
            "robot_state": torch.rand(length, 7),
        },
        actions=torch.rand(length, 7),
        rewards=torch.rand(length),
        dones=torch.zeros(length, dtype=torch.bool),
        task_id="test_task",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 设备 Fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request) -> torch.device:
    """
    设备 Fixture (参数化)
    ────────────────────────────────────────────────────────────────────────────
    使用 pytest 的参数化功能，自动在 CPU 和 CUDA 上分别运行测试。
    如果 CUDA 不可用，跳过 CUDA 测试。

    参数化效果:
      - test_xxx[cpu]    # 在 CPU 上运行
      - test_xxx[cuda]   # 在 CUDA 上运行 (如不可用则跳过)
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 不可用")
    return torch.device(request.param)


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟对象 (Mock) Fixture
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_sim_backend(mocker):
    """
    模拟仿真后端 Fixture
    ────────────────────────────────────────────────────────────────────────────
    使用 pytest-mock 创建一个 SimulatorBackend 的模拟对象。
    这样在测试上层逻辑 (如 RobotEnv) 时，不需要启动真实的仿真引擎。

    模拟对象 (Mock) 的原理:
      - 创建一个"假"的对象，它实现了 SimulatorBackend 的接口
      - 每个方法的返回值由我们指定 (不涉及真实物理引擎)
      - 可以验证调用参数和调用次数
    """
    from src.common.interfaces import SimulatorBackend

    mock = mocker.create_autospec(SimulatorBackend)

    # ── 配置模拟行为 ────────────────────────────────────────────────────────
    # 当调用 step() 时，返回固定的模拟观测
    mock.step.return_value = {
        "obs": {"joint_pos": torch.rand(7)},
        "reward": 0.0,
        "done": False,
        "info": {},
    }
    mock.get_observation.return_value = {
        "rgb": torch.rand(3, 128, 128),
        "robot_state": torch.rand(7),
    }
    mock.dt = 0.002

    return mock


@pytest.fixture
def mock_env(mock_sim_backend):
    """
    模拟仿真环境 Fixture
    ────────────────────────────────────────────────────────────────────────────
    基于 mock_sim_backend 创建一个 RobotEnv 实例。
    用于测试训练循环、策略推理等不依赖真实物理引擎的逻辑。
    """
    from src.simulation.backend import RobotEnv

    return RobotEnv(mock_sim_backend, {"max_steps": 100})


@pytest.fixture
def mock_dataset_adapter(mocker):
    """
    模拟数据集适配器 Fixture
    ────────────────────────────────────────────────────────────────────────────
    用于测试数据加载器时，不需要真实数据集。
    模拟的适配器直接生成随机的 UnifiedSample。
    """
    from src.common.interfaces import DatasetAdapter

    mock = mocker.create_autospec(DatasetAdapter)
    mock.can_handle.return_value = True
    mock.metadata.return_value = {"name": "mock_dataset", "num_samples": 1000}

    # ── 让 load() 返回模拟的样本流 ─────────────────────────────────────────
    def mock_load(source, **kwargs):
        from src.common.types import UnifiedSample
        for i in range(100):
            yield UnifiedSample(
                obs={
                    "rgb": torch.rand(3, 128, 128),
                    "robot_state": torch.rand(7),
                },
                action=torch.rand(7),
                reward=0.0,
                done=(i == 99),
            )

    mock.load.side_effect = mock_load
    return mock
