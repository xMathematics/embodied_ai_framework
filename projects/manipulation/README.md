# Manipulation 项目 (机械臂操作)

**状态**: 骨架 — 待填充

## 范围
桌面级机械臂操作任务：抓取、插拔、叠放、工具使用。

## 子目录
- `python/` — 模仿学习 / 强化学习 / VLA 训练与推理
- `cpp/`    — 实时运动学 / 轨迹规划
- `ros/`    — 操作节点 (感知→规划→执行)
- `cuda/`   — GPU 加速算子 (如抓取位姿检测)
- `config/` — 实验配置
- `scripts/` — 运行脚本
- `tests/`  — 测试

## 依赖
- Python 框架: `libs/python/framework`
- ROS2 接口: `libs/ros/embodied_interfaces`
- C++/CUDA 库: `libs/cpp`, `libs/cuda`
