# 具身智能全流程框架 (Embodied AI Framework)

从开源数据集输入到模拟仿真的全流程项目架构，采用**五层四流**模块化设计。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    应用编排层 (Orchestration)            │
│  实验管理 · 超参搜索 · 批量训练 · CI/CD 流水线          │
├─────────────────────────────────────────────────────────┤
│                        算法层 (Algorithm)                │
│  策略网络 · 模仿学习 (BC, ACT, Diffusion Policy)        │
│  强化学习 (PPO, SAC, Dreamer) · 世界模型 · Sim2Real     │
├─────────────────────────────────────────────────────────┤
│                        仿真层 (Simulation)               │
│  仿真引擎适配器 (Isaac Sim, MuJoCo, SAPIEN, PyBullet)   │
│  场景管理 · 资产导入 · 物理参数随机化 · 并行环境        │
├─────────────────────────────────────────────────────────┤
│                         数据层 (Data)                    │
│  数据集注册中心 · 格式标准化 · 轨迹重放                 │
│  数据增强 · 数据转换 (RGBD, 点云, 本体感知)             │
├─────────────────────────────────────────────────────────┤
│                      基础资源层 (Infra)                  │
│  计算集群 (GPU/NPU) · 分布式存储 · Docker/K8s           │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
embodied_ai_framework/
├── src/                          # 源代码
│   ├── infra/                    # 基础资源层
│   │   ├── compute.py            # 计算资源调度 (GPU检测/分布式/DDP)
│   │   ├── storage.py            # 存储管理 (本地/S3/缓存)
│   │   └── container.py          # 容器化支持 (Docker/K8s环境感知)
│   ├── data/                     # 数据层
│   │   ├── registry.py           # 数据集注册中心
│   │   ├── adapters/             # 数据集适配器 (BridgeV2, RoboTurk等)
│   │   ├── standardization.py    # 数据标准化 (Arrow IPC格式)
│   │   ├── augmentation.py       # 数据增强 (图像/轨迹级)
│   │   └── dataloader.py         # 数据加载器 (混合采样/多进程)
│   ├── simulation/               # 仿真层
│   │   ├── backend.py            # 仿真后端抽象 + RobotEnv封装
│   │   ├── backends/             # 引擎适配 (MuJoCo/Isaac Sim/PyBullet)
│   │   ├── scene_manager.py      # 场景构建器 + 领域随机化
│   │   ├── asset_manager.py      # 机器人模型管理
│   │   └── vec_env.py            # 向量化环境 (多进程并行)
│   ├── algorithm/                # 算法层
│   │   ├── policy.py             # 策略网络 (MLP/CNN/高斯策略)
│   │   ├── imitation/            # 模仿学习 (BC/ACT/Diffusion Policy)
│   │   ├── reinforcement/        # 强化学习 (PPO/SAC)
│   │   ├── world_model.py        # 世界模型 (RSSM/Dreamer)
│   │   └── sim2real.py           # Sim-to-Real迁移 (域适应/延迟补偿)
│   ├── orchestration/            # 编排层
│   │   ├── experiment_manager.py # 实验管理 (MLflow追踪)
│   │   ├── pipeline.py           # 训练流水线 (DAG工作流)
│   │   └── evaluator.py          # 批量评估 (Rollout/视频录制)
│   └── common/                   # 公共模块
│       ├── types.py              # 统一数据类型 (UnifiedSample等)
│       ├── interfaces.py         # 抽象接口 (ABC契约)
│       └── utils.py              # 工具函数 (日志/配置/计时)
├── config/                       # 配置文件
│   ├── experiment/               # 实验配置 (YAML)
│   └── dataset/                  # 数据集注册表
├── scripts/                      # 可执行脚本
│   ├── prepare_data.py           # 数据准备
│   ├── train.py                  # 训练入口
│   └── eval.py                   # 评估入口
├── setup.py                      # Python 包安装
└── requirements.txt              # 依赖列表
```

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repo_url>
cd embodied_ai_framework

# 安装核心依赖
pip install -r requirements.txt

# (推荐) 开发模式安装
pip install -e .

# (可选) 安装仿真引擎
pip install mujoco pybullet
```

### 数据准备

```bash
# 准备 BridgeData V2 数据集
python scripts/prepare_data.py --datasets=bridge_v2 --output=/data/unified

# 查看可用数据集
python scripts/prepare_data.py --list
```

### 训练

```bash
# 行为克隆训练
python scripts/train.py --config config/experiment/bc_manipulation.yaml

# PPO 强化学习
python scripts/train.py --config config/experiment/ppo_pick_place.yaml
```

### 评估

```bash
# 评估训练好的策略
python scripts/eval.py --checkpoint=checkpoints/bc/best.pt --sim=mujoco --render
```

## 技术栈

| 层次         | 技术                                    | 备注                     |
|-------------|-----------------------------------------|--------------------------|
| 编程语言     | Python 3.10+                            | 生态丰富                 |
| 仿真引擎     | MuJoCo 3.0, Isaac Sim, PyBullet         | 可插拔后端               |
| 数据管理     | HuggingFace Datasets, Arrow, DVC        | 高效零拷贝               |
| 深度学习     | PyTorch 2.x + TorchRL                   |                          |
| 实验管理     | MLflow + Hydra                          | 可复现实验               |
| 并行训练     | Ray / PyTorch DDP / SubprocVecEnv       | 多级并行                 |
| 可视化       | Rerun.io, TensorBoard                   | 传感器数据可视化         |

## 关键设计模式

- **适配器模式**: 数据集适配器、仿真后端适配器，新增来源只需编写新适配器
- **工厂模式**: `AdapterFactory`, `SimBackendFactory` 根据配置自动创建实例
- **策略模式**: 算法、增强器可灵活组合和切换
- **建造者模式**: `SceneBuilder` 链式构建复杂仿真场景
- **抽象接口模式**: 各层通过 ABC 契约解耦

## 许可证

MIT
