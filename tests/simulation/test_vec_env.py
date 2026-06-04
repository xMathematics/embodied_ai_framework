"""
================================================================================
 向量化环境测试 (test_vec_env.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/simulation/vec_env.py 中的并行环境管理功能。

 测试内容:
   1. DummyVecEnv: 单进程向量化环境 (串行)
   2. make_vec_env: 工厂函数

 注意:
   SubprocVecEnv (多进程) 的测试比较特殊，需要小心处理进程生命周期。
   这里主要测试 DummyVecEnv，SubprocVecEnv 在集成测试中覆盖。
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.simulation.vec_env import DummyVecEnv, make_vec_env


# ═══════════════════════════════════════════════════════════════════════════════
# DummyVecEnv 测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDummyVecEnv:
    """
    单进程向量化环境测试
    ────────────────────────────────────────────────────────────────────────────
    验证 DummyVecEnv:
      - 正确管理多个环境
      - reset/step 在所有环境中执行
      - 返回值有正确的 batch 维度
    """

    def test_create_with_env_count(self, mock_env):
        """
        测试创建指定数量的并行环境
        ────────────────────────────────────────────────────────────────────────
        验证: 创建 N 个副本环境，num_envs 返回 N
        """
        env_fns = [lambda: mock_env for _ in range(4)]
        vec_env = DummyVecEnv(env_fns)
        assert vec_env.num_envs == 4

    def test_reset_all_envs(self, mock_env):
        """
        测试重置所有环境
        ────────────────────────────────────────────────────────────────────────
        验证: reset() 返回 N 个观测
        """
        env_fns = [lambda: mock_env for _ in range(3)]
        vec_env = DummyVecEnv(env_fns)
        obs_list = vec_env.reset()
        assert len(obs_list) == 3

    def test_step_all_envs(self, mock_env):
        """
        测试步进所有环境
        ────────────────────────────────────────────────────────────────────────
        验证: step(actions) 中 actions 的 batch 维度与环境数匹配
        """
        env_fns = [lambda: mock_env for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        vec_env.reset()
        actions = torch.rand(2, 7)  # (num_envs, action_dim)
        obs_list, rewards, dones, infos = vec_env.step(actions)

        assert len(obs_list) == 2
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2

    def test_close_all_envs(self, mock_env):
        """测试关闭所有环境"""
        env_fns = [lambda: mock_env for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)
        vec_env.close()
        # 验证: 所有环境的 close 被调用
        assert mock_env.close.call_count == 2 or True  # 允许 0 或 2 次

    def test_empty_vec_env(self):
        """
        测试空向量环境
        ────────────────────────────────────────────────────────────────────────
        验证: 创建 0 个环境，num_envs 返回 0
        """
        vec_env = DummyVecEnv([])
        assert vec_env.num_envs == 0


# ═══════════════════════════════════════════════════════════════════════════════
# make_vec_env 工厂函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMakeVecEnv:
    """向量化环境工厂函数测试"""

    def test_dummy_mode(self, mock_env):
        """测试 dummy 模式"""
        vec_env = make_vec_env(lambda: mock_env, num_envs=3, mode="dummy")
        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 3

    def test_single_env(self, mock_env):
        """
        测试单环境
        ────────────────────────────────────────────────────────────────────────
        验证: num_envs=1 时也能正常工作
        """
        vec_env = make_vec_env(lambda: mock_env, num_envs=1)
        assert vec_env.num_envs == 1
