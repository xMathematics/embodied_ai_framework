"""
================================================================================
 计算资源模块测试 (test_compute.py)
 ──────────────────────────────────────────────────────────────────────────────
 测试 src/infra/compute.py 中的设备和资源管理功能。

 这些测试涉及 GPU 检测，需要处理两种情况:
   1. 有 GPU 的环境: 验证 GPU 检测正确
   2. 无 GPU 的环境: 验证回退到 CPU 正确

 使用 pytest.mark.skipif 条件跳过: 当条件满足时跳过测试
 这确保测试在无 GPU 的 CI 环境中也不会失败。
================================================================================
"""

# ─── 第三方库导入 ───────────────────────────────────────────────────────────
import pytest
import torch

# ─── 被测试模块导入 ─────────────────────────────────────────────────────────
from src.infra.compute import (
    detect_devices,
    get_default_device,
    device_context,
    get_autocast_context,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 设备检测测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectDevices:
    """
    设备检测测试
    ────────────────────────────────────────────────────────────────────────────
    验证 detect_devices() 函数能正确识别当前环境的计算设备。
    """

    def test_device_type_is_valid(self):
        """
        测试设备类型合法性
        ────────────────────────────────────────────────────────────────────────
        验证: 返回的设备类型是 "cuda", "cpu", 或 "mps" 之一
        """
        info = detect_devices()
        assert info.device_type in ("cuda", "cpu", "mps"), (
            f"无效的设备类型: {info.device_type}"
        )

    def test_device_name_not_empty(self):
        """测试设备名称非空"""
        info = detect_devices()
        assert len(info.device_name) > 0, "设备名称为空"

    def test_cuda_specific_fields(self):
        """测试 CUDA 特有的字段 (仅在 GPU 环境下)"""
        info = detect_devices()
        if info.device_type == "cuda":
            # 验证: device_id 是有效整数
            assert isinstance(info.device_id, int)
            assert info.device_id >= 0
            # 验证: 显存大于 0
            assert info.total_memory_gb > 0
        else:
            # 非 CUDA 设备: device_id 应为默认值 0
            assert info.device_id == 0

    def test_not_distributed_by_default(self):
        """
        测试默认非分布式
        ────────────────────────────────────────────────────────────────────────
        验证: 在非分布式环境中，distributed=False
        """
        info = detect_devices()
        assert info.distributed is False


class TestGetDefaultDevice:
    """
    默认设备获取测试
    ────────────────────────────────────────────────────────────────────────────
    验证 get_default_device() 返回的 torch.device 对象正确。
    """

    def test_returns_torch_device(self):
        """测试返回类型为 torch.device"""
        device = get_default_device()
        assert isinstance(device, torch.device)

    def test_force_cpu_env(self, monkeypatch):
        """
        测试 FORCE_CPU 环境变量
        ────────────────────────────────────────────────────────────────────────
        原理:
          使用 monkeypatch 模拟设置环境变量 FORCE_CPU=1，
          即使有 GPU 也应返回 CPU 设备。

        monkeypatch 是 pytest 内置的 fixture，用于:
          - 修改环境变量: monkeypatch.setenv()
          - 修改对象属性: monkeypatch.setattr()
          - 临时修改，测试结束后自动恢复
        """
        monkeypatch.setenv("FORCE_CPU", "1")
        device = get_default_device()
        assert device.type == "cpu", f"FORCE_CPU 时预期 cpu, 实际 {device.type}"


# ═══════════════════════════════════════════════════════════════════════════════
# 上下文管理器测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeviceContext:
    """
    设备上下文管理器测试
    ────────────────────────────────────────────────────────────────────────────
    验证 device_context 上下文管理器的正确性:
      - 进入时设置设备
      - 退出时清理 GPU 缓存
    """

    def test_context_enters_and_exits(self):
        """
        测试上下文进入和退出
        ────────────────────────────────────────────────────────────────────────
        验证: 上下文管理器可以正常进入和退出，不抛出异常
        """
        with device_context():
            pass  # 上下文内什么都不做
        # 如果走到这里，说明上下文管理器正常执行

    def test_device_context_works(self):
        """测试设备上下文的功能"""
        with device_context(0):
            # 在上下文中执行一个简单的张量操作
            t = torch.tensor([1.0, 2.0])
            assert t.device.type in ("cuda", "cpu")


class TestAutocastContext:
    """
    混合精度上下文测试
    """

    def test_autocast_returns_context(self):
        """测试返回 autocast 上下文管理器"""
        ctx = get_autocast_context()
        # 验证: 返回的是一个上下文管理器
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_autocast_with_fp16(self):
        """测试显式指定 fp16"""
        ctx = get_autocast_context(dtype=torch.float16)
        # autocast 上下文应能正常使用
        with ctx:
            t = torch.tensor([1.0, 2.0])
            assert t is not None
