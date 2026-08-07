# Mobile Manipulation 项目 (移动操作)

**状态**: 骨架 — 待填充

## 范围
移动底盘 + 机械臂的复合操作任务（导航到目标 → 操作物体）。

## 子目录
- `python/` — 分层策略、导航+操作融合训练
- `cpp/`    — 底盘运动学 / 里程计
- `ros/`    — 导航 (Nav2) 与操作节点集成
- `cuda/`   — 感知加速 (如 3D 占用网络)
- `config/` — 实验配置
- `scripts/` — 运行脚本
- `tests/`  — 测试

## 依赖
- Python 框架: `libs/python/framework`
- ROS2 接口: `libs/ros/embodied_interfaces`
- C++/CUDA 库: `libs/cpp`, `libs/cuda`
