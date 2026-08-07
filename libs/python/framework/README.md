# 具身智能全流程框架 (Embodied AI Framework)

从开源数据集输入到模拟仿真的全流程项目架构，采用**五层四流**模块化设计。
支持模仿学习 (BC/ACT/Diffusion Policy)、强化学习 (PPO/SAC)、VLA 大模型 (OpenVLA/Octo) 及模块融合训练。

---

## 📋 目录

1. [环境搭建](#1-环境搭建)
2. [数据准备与查看](#2-数据准备与查看)
3. [训练工作流](#3-训练工作流)
4. [模型评估](#4-模型评估)
5. [配置文件指南](#5-配置文件指南)
6. [常见问题](#6-常见问题)
7. [架构概览](#7-架构概览)
8. [技术栈](#8-技术栈)

---

> 📖 **完整分步教程**：参见 [`docs/usage_tutorial.md`](docs/usage_tutorial.md)
> ，涵盖 BridgeData V2 和 RoboTurk 两个数据集的训练与推理逐步计算过程及详细解释。

---

## 1. 环境搭建

### 1.1 基础环境

```bash
# 克隆项目
git clone <repo_url>
cd embodied_ai_framework

# 创建 Conda 环境 (推荐 Python 3.10+)
conda create -n embodied python=3.10 -y
conda activate embodied

# 安装核心依赖
pip install -r requirements.txt

# (推荐) 开发模式安装，方便导入 src/ 包
pip install -e .
```

### 1.2 可选仿真引擎

```bash
# MuJoCo (轻量级，推荐首次使用)
pip install mujoco

# PyBullet (备选)
pip install pybullet

# Isaac Sim (需 NVIDIA GPU 及单独安装)
# 参考: https://docs.omniverse.nvidia.com/isaacsim/latest/
```

### 1.3 数据解析依赖

```bash
# 查看 RLDS/TFRecord 格式数据时需要
pip install tfrecord

# 数据处理所需
pip install pyarrow  # Arrow IPC 格式支持
```

### 1.4 国内下载加速

```bash
# 如果在国内下载 HuggingFace 模型/数据速度慢，可设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 数据准备与查看

> 框架采用**本地优先 (Local-First)** 策略，所有数据从本地磁盘加载，路径由配置文件统一管理。

### 2.1 查看已注册数据集

```bash
# 列出所有已注册数据集及其本地状态
python scripts/inspect_data.py --list
```

输出示例：
```
📋 已注册数据集一览 (共 4 个)
  名称         来源                    本地状态     轨迹数    数据大小
  ──────────── ─────────────────────── ────────── ──────── ──────────
  bridge_v2    OPEN_X_EMBODIMENT       📁 目录      60000        0 B
  roboturk     ROBOTURK                📦 TFRecord   8000  207.59 MB
  maniskill    MANISKILL               ❌ 不存在    100000        0 B
  language_table LANGUAGE_TABLE         ❌ 不存在    450000        0 B
```

### 2.2 数据准备

```bash
# 场景 A：本地已有原始数据，转换为标准化 Arrow 格式
python scripts/prepare_data.py --datasets=bridge_v2 \
    --raw_path=data/raw/bridge_v2 \
    --local_path=data/unified/bridge_v2

# 场景 B：从远程下载并转换（首次使用）
python scripts/prepare_data.py --datasets=bridge_v2 \
    --download --local_path=data/unified/bridge_v2

# 场景 C：仅查看可用数据集
python scripts/prepare_data.py --list
```

### 2.3 数据查看与调试

准备前后可用 `inspect_data.py` 详细浏览数据集内容：

```bash
# ★ 综合摘要（推荐）— 包含注册信息 + 本地状态 + 轨迹统计
python scripts/inspect_data.py --dataset=bridge_v2

# 目录结构
python scripts/inspect_data.py --dataset=roboturk --tree

# 数据 Schema（列名、类型、形状）
python scripts/inspect_data.py --dataset=roboturk --schema

# 统计信息（文件数、轨迹长度分布、动作/状态范围）
python scripts/inspect_data.py --dataset=roboturk --stats

# 预览样本内容
python scripts/inspect_data.py --dataset=roboturk --samples=3

# 指定非默认路径
python scripts/inspect_data.py --dataset=bridge_v2 --local_path=/mnt/nvme/data/bridge_v2
```

**支持的数据格式**：

| 格式 | 查看能力 |
|------|---------|
| Arrow (`.arrow`) | Schema、轨迹长度分布、样本数值、图像信息 |
| TFRecord/RLDS (`.tfrecord*`) | 自动检测特征键、JPEG 图像尺寸解析、episode/step 统计、语言指令解码 |
| HDF5 (`.h5`) | 文件大小和分布信息 |

### 2.4 RoboTurk 数据集专节

> RoboTurk 是 Stanford 大学发布的 Franka 远程操控数据集，包含 8000+ 条桌面操控轨迹（抓取、推、开抽屉等）。原始格式为 HDF5，每文件一条轨迹。

#### 数据集特点

| 属性 | 值 |
|------|-----|
| 机器人 | Franka Emika Panda (7-DoF) |
| 传感器 | RGB (256x256), 深度图, 关节角度/速度 |
| 动作空间 | 14 维 [关节位置(7) + 夹爪(1)] × 2 |
| 轨迹数 | ~8,000 |
| 原始格式 | HDF5 |

#### 获取原始数据

```bash
# 方案 A：从 Stanford 官网下载
# 参考: https://roboturk.stanford.edu/
# 下载后放置到 data/raw/roboturk/ 目录

# 方案 B：使用演示数据（当前目录已有示例数据）
# data/unified/roboturk/ 目录下存有 BridgeData 格式的 RLDS/TFRecord 文件
# 可直接用 inspect_data.py 查看
python scripts/inspect_data.py --dataset=roboturk --schema
```

#### 转换为标准化 Arrow 格式

```bash
# 确保原始 HDF5 文件在 data/raw/roboturk/ 目录下
ls data/raw/roboturk/*.hdf5

# 运行适配器转换
python scripts/prepare_data.py --datasets=roboturk \
    --raw_path=data/raw/roboturk \
    --local_path=data/unified/roboturk
```

> **注意**：`RoboTurkAdapter` 期望 HDF5 目录结构为：
> ```
> data/raw/roboturk/
> ├── trajectory_0.hdf5    # 观测: /observations/images/top, /observations/qpos
> ├── trajectory_1.hdf5    # 动作: /action (T, 14)
> └── ...                  # 图像: (T, 256, 256, 3) uint8
> ```

#### 使用 RLDS/TFRecord 格式的现有数据

当前 `data/unified/roboturk/` 目录下包含的是 BridgeData 格式的 TFRecord 文件（与 registry 中 `format: hdf5` 不同），`inspect_data.py` 已自动支持 RLDS 解析：

```bash
# 查看 RLDS 特征 schema（自动检测 20+ 特征键）
python scripts/inspect_data.py --dataset=roboturk --schema

# 统计 episode/step 信息
python scripts/inspect_data.py --dataset=roboturk --stats

# 预览样本内容（含语言指令、JPEG 图像尺寸、动作/状态值）
python scripts/inspect_data.py --dataset=roboturk --samples=2
```

输出节选：
```
📊 Schema: roboturk
  Context 特征: episode_metadata/episode_id, file_path, has_image_0/1/2/3, has_language
  Step 特征:
    steps/observation/image_0    JPEG 256x256  ~37KB/step
    steps/observation/state      float32  (196,) → 每步 7 维
    steps/action                 float32  (196,) → 每步 7 维
    steps/language_instruction   示例: "put small spoon from basket to tray"

📈 Stats: roboturk
  总 Episode 数: 104 (采样)
  轨迹长度范围: 18 ~ 89 步, 平均 38.3 步
  动作 dim[0]: range=[-0.08, 0.08]  dim[6]: range=[-0.01, 1.00]
  图像 JPEG 大小: 1.61KB ~ 37.15KB

🔍 Samples: roboturk
  Episode 1: "put small spoon from basket to tray" (28 步)
    步 1: action=[0,0,0,0,0,0,1.0] state=[0.30,0.09,0.09,0.42,-0.14,0.04,0.99]
           image_0=JPEG 256x256  image_1=JPEG 256x256
    步 2: action=[0.008,0.003,-0.010,-0.048,-0.026,0.136,1.0]
```

#### 使用 RoboTurk 数据训练

```bash
# BC 训练（使用 HDF5 格式原始数据）
python scripts/train.py --config config/experiment/bc_manipulation.yaml \
    data.data_dir=data/unified/roboturk \
    data.dataset_name=roboturk \
    training.action_dim=14         # RoboTurk 动作 14 维

# VLA 微调（使用 TFRecord 格式现有数据）
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=octo \
    data.data_dir=data/unified/roboturk \
    training.action_dim=7
```

#### 适配器说明

`RoboTurkAdapter`（`src/data/adapters/roboturk_adapter.py`）实现：

```python
class RoboTurkAdapter(DatasetAdapter):
    """将 HDF5 格式的 RoboTurk 数据转换为 UnifiedSample"""
    # 关键数据映射:
    #   HDF5 路径                     → UnifiedSample 字段
    #   /observations/images/top      → obs["rgb"]          (T,256,256,3)
    #   /observations/qpos            → obs["robot_state"]   (T,7)
    #   /action                       → action               (T,14)
    #   无奖励信号                    → reward = 0.0
    #   最后一步                      → done = True
```

---

## 3. 训练工作流

### 3.1 模仿学习 (Behavior Cloning)

```bash
# 基础用法
python scripts/train.py --config config/experiment/bc_manipulation.yaml

# 覆盖参数（学习率、batch size 等）
python scripts/train.py --config config/experiment/bc_manipulation.yaml \
    training.learning_rate=0.0005 \
    data.batch_size=128
```

**训练流程**：
1. 加载本地 Arrow 数据 → `UnifiedDataset`
2. `DataLoader` 多进程预取 → batch
3. 策略网络前向传播 → 行为克隆损失 (MSE/交叉熵)
4. 反向传播更新策略

### 3.2 强化学习 (PPO)

```bash
# 从示范数据启动
python scripts/train.py --config config/experiment/ppo_pick_place.yaml

# 指定仿真后端
python scripts/train.py experiment=ppo_pick_place sim=mujoco

# 启用渲染
python scripts/train.py experiment=ppo_pick_place \
    simulation.backend=mujoco \
    evaluation.render=true
```

**训练流程**：
1. `VecEnv` 启动多个并行仿真环境
2. 策略采样动作 → 环境 step → 经验池存储
3. PPO 算法更新策略（GAE 优势估计 + clip 目标）
4. 定期评估并保存 checkpoint

### 3.3 VLA 大模型微调

```bash
# OpenVLA 微调 (7B 参数，需 16GB+ 显存)
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=openvla-7b \
    vla.finetune_method=lora

# Octo 微调 (更轻量，推荐先跑通)
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=octo \
    vla.finetune_method=lora

# 覆盖数据路径和训练参数
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=octo \
    data.data_dir=data/unified/bridge_v2 \
    training.num_epochs=100
```

### 3.4 VLA + RL 联合训练

```bash
# 默认：Octo + PPO 加权平均融合
python scripts/train.py --config config/experiment/fusion/vla_rl_fusion.yaml

# 切换到 OpenVLA + PPO
python scripts/train.py experiment=fusion/vla_rl_fusion \
    vla.model=openvla-7b \
    vla.load_in_8bit=true \
    rl.algorithm=ppo

# 切换融合策略
python scripts/train.py experiment=fusion/vla_rl_fusion \
    fusion.strategy=bayesian      # 贝叶斯融合
    # fusion.strategy=priority    # 优先级路由
    # fusion.strategy=weighted_average  # 加权平均（默认）

# 调整 VLA 指导权重
python scripts/train.py experiment=fusion/vla_rl_fusion \
    fusion.weighted_average.vla_weight=0.6  # 增大 VLA 影响
```

**融合策略说明**：

| 策略 | 公式 | 适用场景 |
|------|------|---------|
| **加权平均** | `a = w·a_vla + (1-w)·a_rl` | VLA 与 RL 互补 |
| **贝叶斯融合** | 基于置信度的高斯融合 | VLA 置信度可靠时 |
| **优先级路由** | 置信度高用 VLA，否则 RL | VLA 偶有失误时 |

### 3.5 VLA + 世界模型联合训练

```bash
python scripts/train.py experiment=fusion/vla_world_model \
    vla.model=openvla-7b \
    world_model=dreamer \
    fusion.mode=vla_world_model \
    fusion.vla_world_model.imagine_horizon=50
```

**原理**：VLA 生成多个候选动作 → 世界模型在隐空间中想象推演 → 选择推演奖励最高的动作。

### 3.6 三模块全闭环 (VLA + 世界模型 + RL)

```bash
python scripts/train.py experiment=fusion/triple_fusion \
    vla.model=openvla-7b \
    rl.algorithm=sac \
    world_model=td_mpc2 \
    fusion.mode=all
```

**级联决策流程**：
```
VLA (高层语义规划) → 子目标
    ↓
世界模型 (运动规划) → 轨迹
    ↓
RL (精细跟踪控制) → 动作
```

---

## 4. 模型评估

### 4.1 评估训练好的策略

```bash
# 基础评估
python scripts/eval.py --checkpoint=checkpoints/bc/best.pt --sim=mujoco

# 评估并渲染视频
python scripts/eval.py --checkpoint=checkpoints/ppo/best.pt \
    --sim=mujoco --render --video=videos/eval

# 比较多个策略
python scripts/eval.py --checkpoint=checkpoints/bc/best.pt,checkpoints/ppo/best.pt \
    --num_episodes=100

# 指定配置文件（用于环境配置）
python scripts/eval.py --checkpoint=checkpoints/vla_rl_fusion/best.pt \
    --config config/experiment/fusion/vla_rl_fusion.yaml \
    --sim=mujoco --render
```

### 4.2 评估指标

| 指标 | 说明 |
|------|------|
| 成功率 (Success Rate) | 任务完成比例 |
| 平均奖励 (Avg Reward) | 每 episode 累积奖励 |
| 轨迹长度 (Episode Length) | 完成任务的步数 |
| 动作平滑度 (Action Smoothness) | 相邻动作的差异 |

---

## 5. 配置文件指南

### 5.1 配置文件结构

所有实验配置位于 `config/experiment/`，采用 YAML 格式：

```
config/experiment/
├── bc_manipulation.yaml          # 行为克隆
├── ppo_pick_place.yaml           # PPO 强化学习
└── fusion/
    ├── vla_finetune.yaml         # VLA 微调
    ├── vla_rl_fusion.yaml        # VLA + RL 融合
    ├── vla_world_model.yaml      # VLA + 世界模型
    └── triple_fusion.yaml        # 三模块全闭环
```

### 5.2 核心配置字段

```yaml
# 算法模块选择
algorithm:
  primary: "vla"               # 主算法: vla / rl / imitation
  secondary: ["rl"]            # 辅助算法列表
  fusion_mode: "vla_rl"        # 融合模式

# 数据配置
data:
  data_dir: "data/unified/bridge_v2"  # ← 本地 Arrow 数据路径
  batch_size: 128                     # batch size
  num_workers: 4                      # 数据加载进程数

# 仿真配置
simulation:
  backend: "mujoco"            # 仿真后端: mujoco / isaac_sim / pybullet
  num_envs: 32                 # 并行环境数
  max_steps: 500               # 每 episode 最大步数

# 训练配置
training:
  num_epochs: 200              # 训练轮数
  learning_rate: 0.0003        # 学习率
  checkpoint_dir: "checkpoints/"  # 模型保存路径
```

### 5.3 路径配置优先级

数据路径按以下优先级确定（从高到低）：

1. **命令行参数**：`--local_path=/custom/path`
2. **实验配置文件**：`config.data.data_dir`
3. **数据集注册表**：`config/dataset/registry.yaml` → `datasets[].local_path`
4. **内置默认**：`data/unified/{dataset_name}`

---

## 6. 常见问题

### Q1: ModuleNotFoundError: No module named 'torch'

**原因**：Python 环境中未安装 PyTorch。
**解决**：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 或 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q2: 国内下载 HuggingFace 模型/数据慢

**解决**：
```bash
# 方法 1: 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 在代码中设置
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### Q3: 运行脚本时报错 `No module named 'src'`

**原因**：Python 找不到项目包。
**解决**：
```bash
# 方法 1: 开发模式安装
pip install -e .

# 方法 2: 手动添加 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Q4: 如何添加新的数据集？

1. 在 `config/dataset/registry.yaml` 中添加数据集元信息
2. 在 `src/data/adapters/` 中创建适配器类（继承 `DatasetAdapter`）
3. 在 `scripts/prepare_data.py` 中注册适配器
4. 运行 `python scripts/prepare_data.py --datasets=新数据集名`

### Q5: VLA 模型需要多少显存？

| 模型 | 推理显存 | 训练显存 (LoRA) | 训练显存 (全量) |
|------|---------|----------------|----------------|
| Octo | ~2GB | ~4-6GB | ~8GB |
| OpenVLA 7B | ~8GB (8-bit) | ~12GB (8-bit) | ~16GB+ |
| RT-2-X | ~12GB | ~16GB | ~24GB+ |

---

## 7. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    应用编排层 (Orchestration)            │
│  实验管理 · 超参搜索 · 批量训练 · CI/CD 流水线          │
├─────────────────────────────────────────────────────────┤
│                        算法层 (Algorithm)                │
│  VLA大模型 (RT-2/OpenVLA/Octo) · VLM基础模型           │
│  策略网络 · 模仿学习 (BC/ACT/Diffusion Policy)         │
│  强化学习 (PPO/SAC/Dreamer) · 世界模型 · Sim2Real      │
│  模块化融合引擎 (VLA+RL / VLA+世界模型 / 三模块闭环)   │
├─────────────────────────────────────────────────────────┤
│                        仿真层 (Simulation)               │
│  仿真引擎适配器 (Isaac Sim, MuJoCo, SAPIEN, PyBullet)   │
│  场景管理 · 资产导入 · 物理参数随机化 · 并行环境        │
├─────────────────────────────────────────────────────────┤
│                         数据层 (Data)                    │
│  数据集注册中心 · 格式标准化 (Arrow IPC) · 轨迹重放     │
│  数据增强 (图像/轨迹级) · TFRecord/RLDS 解析            │
├─────────────────────────────────────────────────────────┤
│                      基础资源层 (Infra)                  │
│  计算集群 (GPU/NPU) · 分布式存储 · Docker/K8s           │
└─────────────────────────────────────────────────────────┘
```

### 项目结构

```
embodied_ai_framework/
├── src/                          # 源代码
│   ├── algorithm/                # 算法层
│   │   ├── vla/                  # VLA 大模型 (OpenVLA/Octo 适配器)
│   │   ├── fusion/               # 融合引擎 (VLA+RL+世界模型)
│   │   ├── imitation/            # 模仿学习 (BC/ACT/Diffusion Policy)
│   │   ├── reinforcement/        # 强化学习 (PPO/SAC)
│   │   ├── policy.py             # 策略网络
│   │   ├── world_model.py        # 世界模型
│   │   └── sim2real.py           # Sim-to-Real 迁移
│   ├── data/                     # 数据层
│   │   ├── registry.py           # 数据集注册中心
│   │   ├── adapters/             # 数据集适配器
│   │   ├── standardization.py    # 数据标准化 (Arrow IPC)
│   │   ├── augmentation.py       # 数据增强
│   │   └── dataloader.py         # 数据加载器
│   ├── simulation/               # 仿真层
│   │   ├── backends/             # 仿真引擎适配 (MuJoCo/Isaac/PyBullet)
│   │   ├── scene_manager.py      # 场景管理
│   │   ├── vec_env.py            # 向量化环境
│   │   └── asset_manager.py      # 资产导入
│   ├── orchestration/            # 编排层
│   │   ├── pipeline.py           # 训练流水线
│   │   ├── evaluator.py          # 批量评估
│   │   └── experiment_manager.py # 实验管理 (MLflow)
│   ├── infra/                    # 基础资源层
│   ├── common/                   # 公共模块
│   │   ├── types.py              # 统一数据类型
│   │   ├── interfaces.py         # 抽象接口
│   │   └── utils.py              # 工具函数
├── config/                       # 配置文件
│   ├── experiment/               # 实验配置 (YAML)
│   │   └── fusion/               # 融合实验配置
│   └── dataset/registry.yaml     # 数据集注册表
├── scripts/                      # 可执行脚本
│   ├── prepare_data.py           # 数据准备
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 评估入口
│   └── inspect_data.py           # 数据查看
├── data/                         # 数据目录 (本地优先)
│   ├── unified/                  # 标准化 Arrow 数据
│   └── raw/                      # 原始数据缓存
├── docs/                         # 文档
│   ├── CHANGELOG.md              # 变更日志
│   └── 测试过程文档.md
├── setup.py
└── requirements.txt
```

---

## 8. 技术栈

| 层次         | 技术                                    | 备注                     |
|-------------|-----------------------------------------|--------------------------|
| 编程语言     | Python 3.10+                            |                          |
| 深度学习     | PyTorch 2.x                             |                          |
| 仿真引擎     | MuJoCo 3.0, Isaac Sim, PyBullet         | 可插拔后端               |
| 数据格式     | Apache Arrow IPC, TFRecord/RLDS         | 零拷贝内存映射           |
| VLA 模型     | OpenVLA 7B, Octo, RT-2-X               | HuggingFace 集成         |
| 实验管理     | MLflow + Hydra                          | 可复现实验               |
| 并行训练     | Ray / PyTorch DDP / SubprocVecEnv       | 多级并行                 |
| 可视化       | Rerun.io, TensorBoard                   | 传感器数据可视化         |

## 关键设计模式

- **适配器模式**: 数据集适配器、仿真后端适配器，新增来源只需编写新适配器
- **工厂模式**: `AdapterFactory`, `SimBackendFactory` 根据配置自动创建实例
- **策略模式**: 算法、融合策略可灵活组合和切换
- **建造者模式**: `SceneBuilder` 链式构建复杂仿真场景
- **抽象接口模式**: 各层通过 ABC 契约解耦

## 许可证

MIT
