# Humanoid 项目 (人形机器人)

**状态**: 骨架 — 待填充

## 范围
人形机器人全身控制：行走、平衡、灵巧操作。

## 子目录
- `python/` — 全身策略 (RL)、模仿学习、仿真训练
- `cpp/`    — 实时 MPC / 全身动力学
- `ros/`    — 传感器与状态估计节点
- `cuda/`   — GPU 加速物理/仿真
- `config/` — 实验配置
- `scripts/` — 运行脚本
- `tests/`  — 测试

## 依赖
- Python 框架: `libs/python/framework`
- ROS2 接口: `libs/ros/embodied_interfaces`
- C++/CUDA 库: `libs/cpp`, `libs/cuda`
