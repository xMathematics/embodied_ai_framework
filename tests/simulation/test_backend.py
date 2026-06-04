"""
================================================================================
 仿真后端测试 (test_backend.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/simulation/backend.py 中的:
   1. SimBackendFactory: 仿真后端工厂
   2. RobotEnv: Gymnasium 风格环境封装
   3. SimTask: 仿真任务基类

 测试策略:
   使用 Mock (模拟) 后端来测试 RobotEnv 的逻辑，避免依赖真实仿真引擎。
   真实后端测试 (MuJoCo/PyBullet) 在单独的集成测试中。
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.simulation.backend import SimBackendFactory, RobotEnv, SimTask
from src.common.interfaces import SimulatorBackend
from src.common.types import SceneDescription, ConfigDict


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真后端工厂测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimBackendFactory:
    """
    仿真后端工厂测试
    ────────────────────────────────────────────────────────────────────────────
    验证 SimBackendFactory 能正确注册和创建后端实例。
    """

    def test_register_and_create(self, mocker):
        """
        测试注册和创建后端
        ────────────────────────────────────────────────────────────────────────
        验证: 注册后端类后，create() 返回正确的实例
        """
        factory = SimBackendFactory()
        # 创建一个 Mock 后端类
        mock_backend_cls = mocker.MagicMock(spec=SimulatorBackend)
        mock_backend_cls.return_value = mocker.MagicMock(spec=SimulatorBackend)

        factory.register("mock_backend", mock_backend_cls)
        instance = factory.create("mock_backend", {"param": "value"})

        # 验证: create 返回了注册类的实例
        mock_backend_cls.assert_called_once_with({"param": "value"})
        assert instance is not None

    def test_create_unknown_backend(self):
        """
        测试创建未注册的后端
        ────────────────────────────────────────────────────────────────────────
        验证: 抛出 ValueError
        """
        factory = SimBackendFactory()
        with pytest.raises(ValueError, match="未知的仿真后端"):
            factory.create("unknown_backend")

    def test_list_available(self):
        """测试列出可用后端"""
        factory = SimBackendFactory()
        assert factory.list_available() == []

        # 注册两个后端
        factory.register("backend_a", None)  # type: ignore
        factory.register("backend_b", None)  # type: ignore
        available = factory.list_available()
        assert "backend_a" in available
        assert "backend_b" in available


# ═══════════════════════════════════════════════════════════════════════════════
# RobotEnv 环境封装测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobotEnv:
    """
    RobotEnv 环境封装测试
    ────────────────────────────────────────────────────────────────────────────
    验证 RobotEnv 正确地:
      - 封装 SimulatorBackend 为 Gymnasium 风格接口
      - 管理 episode 步数和终止条件
      - 处理 reset 和 step 的返回值格式
    """

    def test_reset_returns_obs_and_info(self, mock_sim_backend):
        """
        测试 reset 返回格式
        ────────────────────────────────────────────────────────────────────────
        验证: reset() 返回 (obs, info) 二元组 (Gymnasium 规范)
        """
        env = RobotEnv(mock_sim_backend, {"max_steps": 100})
        obs, info = env.reset()
        assert isinstance(obs, dict), "观测应为字典类型"
        assert isinstance(info, dict), "info 应为字典类型"

    def test_step_returns_five_tuple(self, mock_sim_backend):
        """
        测试 step 返回格式
        ────────────────────────────────────────────────────────────────────────
        验证: step() 返回 (obs, reward, terminated, truncated, info) 五元组
        (Gymnasium 规范: match)
        """
        env = RobotEnv(mock_sim_backend, {"max_steps": 100})
        env.reset()
        action = torch.rand(7)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_counts_correctly(self, mock_sim_backend):
        """
        测试步数计数正确
        ────────────────────────────────────────────────────────────────────────
        验证: 每次 step 后步数增加，超过 max_steps 后 truncated=True
        """
        env = RobotEnv(mock_sim_backend, {"max_steps": 5})
        env.reset()
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(torch.rand(7))
            if i < 4:
                assert not truncated, f"第 {i} 步不应 truncated"
            else:
                assert truncated, "第 5 步应 truncated"

    def test_render_returns_rgb(self, mock_sim_backend):
        """
        测试渲染功能
        ────────────────────────────────────────────────────────────────────────
        验证: render() 返回 RGB 图像
        """
        env = RobotEnv(mock_sim_backend, {"max_steps": 100})
        rgb = env.render()
        assert rgb is not None

    def test_close(self, mock_sim_backend):
        """测试关闭环境 (应不抛出异常)"""
        env = RobotEnv(mock_sim_backend, {"max_steps": 100})
        env.close()
        # 验证: close 调用了后端的 close
        mock_sim_backend.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 仿真任务基类测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimTask:
    """
    仿真任务基类测试
    ────────────────────────────────────────────────────────────────────────────
    验证 SimTask 的默认行为 (子类应覆盖这些方法)
    """

    def test_default_config(self):
        """测试默认配置"""
        task = SimTask({})
        assert task.name == "default_task"

    def test_default_reward(self):
        """测试默认奖励为 0"""
        task = SimTask({"name": "test"})
        reward = task.reward({})
        assert reward == 0.0

    def test_default_success_false(self):
        """测试默认成功判断为 False"""
        task = SimTask({})
        assert task.is_success({}) is False

    def test_default_failure_false(self):
        """测试默认失败判断为 False"""
        task = SimTask({})
        assert task.is_failure({}) is False
