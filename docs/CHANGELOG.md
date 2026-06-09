# 变更日志 (Changelog)

> **项目**: 具身智能从开源数据集到模拟仿真的全流程框架  
> **文档更新日期**: 2026-06-08

---

## 目录

1. [变更概述](#1-变更概述)
2. [第 1 次更新: VLA 大模型 + 模块融合引擎](#2-第-1-次更新-vla-大模型--模块融合引擎)
   - [设计架构文档更新](#21-设计架构文档更新)
   - [新建源代码文件](#22-新建源代码文件)
   - [新建实验配置文件](#23-新建实验配置文件)
   - [修改的现有文件](#24-修改的现有文件)
3. [第 2 次更新: 本地优先数据加载](#3-第-2-次更新-本地优先数据加载)
   - [设计架构文档更新](#31-设计架构文档更新)
   - [代码及配置文件更新](#32-代码及配置文件更新)
4. [第 3 次更新: 变更日志文档](#4-第-3-次更新-变更日志文档)
5. [第 4 次更新: 数据集查看脚本](#5-第-4-次更新-数据集查看脚本)
6. [文件变更总表](#6-文件变更总表)

---

## 1. 变更概述

本次迭代对具身智能框架进行了 **四次重大更新**：

| # | 更新主题 | 涉及文件数 | 核心变更 |
|---|---------|-----------|---------|
| 1 | **VLA 大模型 + 模块融合引擎** | 14 | 新增 VLA 大模型模块、融合引擎、VLA+RL+世界模型三模块融合机制 |
| 2 | **本地优先数据加载** | 10 | 数据加载从云存储下载改为本地路径加载，路径由配置文件统一管理 |
| 3 | **变更日志文档** | 1 | 创建本文档，汇总所有修改记录 |
| 4 | **数据集查看脚本** | 2+ → 3 | 新增 inspect_data.py 数据查看脚本 + RLDS 格式解析能力，更新设计架构文档 |

---

## 2. 第 1 次更新: VLA 大模型 + 模块融合引擎

### 2.1 设计架构文档更新

**文件**: `设计架构.md`

#### §2 总体架构图 — 算法层扩展

原算法层仅包含 `策略网络 · 模仿学习 · 强化学习 · 世界模型 · Sim2Real`，更新后新增：

```
│  VLA大模型 (RT-2/OpenVLA/Octo)  ·  VLM基础模型 (CLIP/DINOv2)│
│  ─── 模块化融合引擎 ───                                      │
│  VLA+RL 融合  ·  世界模型+语言规划  ·  多模态联合训练       │
```

#### §3.4.2 强化学习管道 — 新增 VLA+RL 融合

新增 **VLA+RL 融合** 段落，包含三种融合方式：
- **VLA 作为高层策略**：输出语言化的子目标或动作先验，RL 作为底层控制器执行精细动作
- **语言驱动的奖励塑形**：VLA 通过视觉-语言对齐打分替代手工奖励函数
- **正向循环**：VLA 指导 RL 探索 → RL 数据增强 VLA

#### §3.4.3 世界模型 — 扩展为"世界模型与 Sim-to-Real & VLA 融合"

从单一的世界模型描述扩展为六个要点：
- **世界模型训练**（原有）
- **VLA 增强的世界模型**：将 VLM 语义特征注入世界模型隐状态
- **世界模型辅助 VLA**：VLA 输出动作前在世界模型中快速推演
- **世界模型作为 VLA 数据增强器**：合成"想象轨迹"扩展训练数据
- **Sim-to-Real 模块**（原有）
- **三模块闭环**：VLA ↔ 世界模型 ↔ RL 形成三角闭环

#### §3.4.5 VLA 大模型 (全新小节)

新增完整的 VLA 大模型小节，包含：
- **概述**：VLA 将视觉理解、语言推理和动作生成统一到端到端模型
- **支持的 VLA 模型**：RT-2 / RT-2-X、OpenVLA、Octo、π0 (Pi0)、自定义 VLA
- **VLA 训练管道**：四阶段图示（VLM 预训练 → 动作头适配 → 联合微调 → 交互式后训练）
- **VLA 输入输出接口**：`VLAInput` 和 `VLAOutput` 数据类定义
- **三种部署方式**：在线推理、离线指导、混合推理

#### §4 数据流 — 新增 VLA 训练流程

在训练循环中新增三种 VLA 训练方式：
- 纯 VLA 微调：语言标注轨迹 → VLM 权重 → LoRA/QLoRA
- VLA + RL 联合训练
- VLA + 世界模型联合训练

#### §4.2 控制流示意图 — 更新

`[训练循环 (IL/RL)]` → `[训练循环 (IL/RL/VLA)]`

#### §5 技术栈选型 — 新增三行

| 层次 | 推荐技术 | 备注 |
|------|---------|------|
| VLA 大模型 | OpenVLA, RT-2-X, Octo, π0, HuggingFace Transformers | 视觉-语言-动作统一模型 |
| VLM 基础模型 | CLIP, DINOv2, SigLIP, Prismatic VLM, LLaVA | 视觉-语言对齐与语义特征提取 |
| 参数高效微调 | LoRA, QLoRA, DoRA, AdaLoRA | VLA 模型的低成本微调 |

#### §6 关键接口定义 — 新增 4 个接口

- **§6.4 VLA 策略接口**：`VLAPolicy` 扩展 `Policy`，新增 `act_with_language()`、`get_confidence()`、`set_lora_weights()`
- **§6.5 模块选择配置接口**：`ModuleSelectionConfig` 数据类，定义 `algorithm`、`vla_model`、`fusion_mode`、`world_model` 等字段
- **§6.6 模块工厂与注册**：`ModuleRegistry` 类，支持装饰器注册和按名称实例化
- **§6.7 融合引擎接口**：`FusionEngine` 抽象基类，统一的 `select_action()` 决策入口

#### §8 部署与运行示例 — 新增 4 条命令

```bash
# 5. VLA 大模型微调训练
python train.py experiment=vla_finetune vla_model=openvla-7b ...

# 6. VLA + RL 联合训练
python train.py experiment=vla_rl_fusion vla_model=octo rl_algorithm=ppo ...

# 7. VLA + 世界模型联合训练
python train.py experiment=vla_world_model vla_model=openvla-7b world_model=dreamer ...

# 8. 三模块全闭环训练
python train.py experiment=triple_fusion vla_model=openvla-7b rl_algorithm=sac world_model=td_mpc2 ...
```

#### §9 扩展与未来方向 — 新增 3 条

- **VLA + 代码生成**：VLA 不仅输出动作，还能生成机器人控制代码
- **在线 VLA 学习**：部署后通过在线交互数据持续自我改进
- **具身基础模型**：VLA、世界模型、RL 三模块深度融合为统一基础模型

#### §10 模块选择与融合机制 (全新章节)

完整的模块选择和融合机制设计，包含：

- **§10.1 设计理念**：7 种融合模式的模块选择矩阵
- **§10.2 配置驱动**：`vla_rl_fusion.yaml` 和 `triple_fusion.yaml` 完整配置示例
- **§10.3 运行时模块调度**：融合引擎架构图（VLA → RL → 世界模型 → FusionEngine → 动作输出）
- **§10.4 模块间数据流**：数据流图（语言指令 → VLA → 融合引擎 → RL/世界模型 → 仿真环境）
- **§10.5 融合策略的 YAML 配置**：四种融合策略（加权平均、贝叶斯融合、优先级路由、级联决策）的完整参数配置

---

### 2.2 新建源代码文件

#### `src/algorithm/vla/__init__.py`

VLA 模块入口文件，导出 `VLABase`、`VLAInput`、`VLAOutput`、`OpenVLAAdapter`、`OctoAdapter`。

#### `src/algorithm/vla/vla_base.py`

VLA 模型基类，定义统一的抽象接口：

| 类/方法 | 说明 |
|---------|------|
| `VLAInput` | 统一输入格式：`rgb_images`、`language_instruction`、`robot_state`、`history_frames` |
| `VLAOutput` | 统一输出格式：`action`、`action_chunk`、`language_reasoning`、`confidence` |
| `VLABase` | 抽象基类，继承 `nn.Module` 和 `ABC`，定义 `act_with_language()` 模板方法 |
| `load()` / `save()` | 模型生命周期管理 |
| `set_lora_weights()` | 动态切换 LoRA 适配器，支持多任务快速切换 |
| `_encode_vision()` / `_encode_language()` | 抽象方法：编码视觉和语言输入 |
| `_predict_action()` | 抽象方法：从多模态特征预测动作 |
| `get_trainable_params()` | 根据微调方式（full/lora/qlora）返回可训练参数 |

#### `src/algorithm/vla/openvla_adapter.py`

OpenVLA 模型适配器，继承 `VLABase`：
- 加载 OpenVLA 7B 模型（Prismatic VLM + Llama 2 + LoRA）
- 使用 SigLIP + DINOv2 编码视觉特征
- 支持 8-bit 量化加载
- 动作头输出连续动作值 + 动作块

#### `src/algorithm/vla/octo_adapter.py`

Octo 模型适配器，继承 `VLABase`：
- 加载 Octo Base 模型（Multi-task Transformer）
- 使用 ViT 编码视觉特征
- 使用 T5 编码语言指令
- Readout head 输出动作块

#### `src/algorithm/fusion/__init__.py`

融合引擎模块入口，导出 `FusionEngine`、`FusionConfig`、`ModuleSelectionConfig`、`ModuleRegistry`。

#### `src/algorithm/fusion/fusion_engine.py`

融合引擎核心实现，包含：

| 类 | 说明 |
|----|------|
| `FusionMode` 枚举 | `vla_only` / `vla_rl` / `vla_world_model` / `all` |
| `FusionStrategy` 枚举 | `weighted_average` / `bayesian` / `priority` / `cascade` |
| `FusionConfig` | 融合引擎完整配置数据类 |
| `FusionEngine` | 抽象基类，含工厂方法 `create()` |
| `VLAOnlyEngine` | 纯 VLA 推理 |
| `VLARLFusionEngine` | VLA + RL 融合：加权平均 / 贝叶斯 / 优先级路由 |
| `VLAWorldModelFusionEngine` | VLA + 世界模型：候选动作 → 想象推演 → 最优选择 |
| `TripleFusionEngine` | 三模块闭环：VLA 子目标 → 世界模型运动规划 → RL 跟踪控制 |

**融合策略算法**：

- **加权平均**：`a = w * a_vla + (1-w) * a_rl`，支持自适应权重
- **贝叶斯融合**：基于置信度的高斯分布融合
- **优先级路由**：VLA 置信度高于阈值时切换，否则使用 RL
- **级联决策**：VLA 做高层任务规划 → RL 拆解为子任务 → 世界模型做运动规划

#### `src/algorithm/fusion/module_registry.py`

模块注册中心：
- 支持四种模块类型：`vla` / `rl` / `world_model` / `imitation`
- 装饰器 `@ModuleRegistry.register(type, name)` 注册模块
- `create(type, name, **kwargs)` 按名称动态实例化
- `list_modules()` 列出所有已注册模块

---

### 2.3 新建实验配置文件

#### `config/experiment/fusion/vla_finetune.yaml`

VLA 大模型微调实验配置：
- 支持 `full` / `lora` / `qlora` 三种微调方式
- 支持两阶段训练：`action_head`（仅训练动作头）/ `full`（端到端）
- LoRA 参数可配置：`lora_rank`、`lora_alpha`、`target_modules`
- VLA 模型使用小 batch size（8-16），配合 `gradient_accumulation_steps`

#### `config/experiment/fusion/vla_rl_fusion.yaml`

VLA + RL 联合训练实验配置：
- 融合策略：`weighted_average` / `bayesian` / `priority`
- VLA 指导权重 `vla_guidance_weight`（0=纯 RL，1=纯 VLA）
- 支持 VLA 打分作为奖励信号 `reward_from_vla: true`
- PPO 算法 + 32 个并行仿真环境

#### `config/experiment/fusion/vla_world_model.yaml`

VLA + 世界模型联合训练实验配置：
- 世界模型使用 Dreamer 算法
- `imagine_horizon: 50` 步推演
- VLA 生成 `num_candidates: 8` 个候选动作，世界模型选择最优

#### `config/experiment/fusion/triple_fusion.yaml`

三模块全闭环实验配置：
- VLA（OpenVLA 7B）+ RL（SAC）+ 世界模型（TD-MPC2）
- 级联决策策略：VLA 设定子目标 → 世界模型规划轨迹 → RL 跟踪控制
- 300 个 epoch 训练，100 个 episode 评估

---

### 2.4 修改的现有文件

#### `src/algorithm/__init__.py`

算法层文档字符串更新，模块列表新增：
- `vla/`：VLA 大模型子模块（vla_base、openvla_adapter、octo_adapter）
- `fusion/`：模块融合引擎子模块（fusion_engine、module_registry）

---

## 3. 第 2 次更新: 本地优先数据加载

### 3.1 设计架构文档更新

**文件**: `设计架构.md`

#### §4 数据流与控制流 — 全面重写

原数据流描述为"从云存储下载原始数据"，全面重写为**本地优先 (Local-First)** 策略：

##### §4.1 数据加载方式：从本地路径加载

新增完整的数据加载策略总览图：
```
加载模式: 本地优先 (Local-First)
数据来源: 本地磁盘 (SSD/NVMe) → 内存映射 (Zero-Copy) → GPU 直接加载
路径配置: 统一由 YAML 配置文件和 Hydra 实验配置管理
数据格式: Arrow IPC 列式格式 (高性能列存 + 内存映射)
目录结构: data/unified/{dataset_name}/*.arrow
```

**数据路径配置优先级**（从高到低）：
1. 命令行参数：`--data_dir=/custom/path`
2. 实验配置文件：`experiment.yaml` → `data.data_dir`
3. 数据集注册表：`registry.yaml` → `datasets[].local_path`
4. 框架内置默认路径：`data/unified/`

**配置示例**：
- 数据集注册表中的本地路径配置（`registry.yaml` 完整示例）
- 实验配置文件中的数据路径覆盖（`bc_manipulation.yaml` 示例）

##### §4.2 典型训练流 — 步骤 1 重写

数据准备阶段从"云存储下载"改为：
1. 从实验配置读取 `data.data_dir` 本地路径
2. `UnifiedDataset` 直接初始化本地数据集
3. 如果本地路径无数据，触发 `prepare_data.py` 数据准备流程
4. Arrow 文件通过内存映射 (mmap) 加载，无需完全读入 RAM

##### §4.3 控制流示意图 — 重写

从简单的线性流程图重写为详细的 ASCII 流程图：
- 标注每个步骤的本地路径来源
- 标注关键代码路径（`UnifiedDataset(data_dir)`、`SimBackend(cfg.simulation)`）
- 标注数据流方向（本地 Arrow 文件 → DataLoader → 训练循环）

##### §4.4 本地数据加载核心机制 (全新)

新增详细的注释式说明，展示数据加载的三步核心代码路径：
1. 从配置文件读取本地路径
2. UnifiedDataset 初始化（扫描 .arrow 文件）
3. DataLoader 多进程批量产出

---

### 3.2 代码及配置文件更新

#### `scripts/prepare_data.py` — 全面重写

**新增命令行参数**：

| 参数 | 说明 |
|------|------|
| `--local_path` | 标准化数据的本地输出/读取目录（核心参数） |
| `--raw_path` | 原始数据本地缓存路径 |
| `--download` | 允许从远程下载（默认 false，仅本地加载） |

**主函数重写为本地优先流程**：

```
步骤 1: 确定本地路径 (命令行 > registry.yaml > 内置默认)
步骤 2: 检查本地是否已有标准化数据 → 有则跳过
步骤 3: 确定原始数据路径
步骤 4: 创建适配器并加载数据
步骤 5: 写入标准化 Arrow IPC 格式
```

**关键逻辑**：
- 自动检测本地 `.arrow` 文件是否存在，存在则跳过
- 区分 `local_only` 和远程下载两种加载模式
- 详细的错误提示（文件未找到、网络失败、适配器缺失）

#### `src/data/dataloader.py` — 注释重写

**`UnifiedDataset` 类**：
- 文档字符串全面重写，新增本地路径说明
- 解释为什么使用 `IterableDataset`（TB 级数据、流式读取、mmap）
- 新增数据路径说明段：`data_dir` 来源、目录结构约定
- 新增工作流程图：扫描文件 → 建立索引 → mmap 打开 → 迭代产出

**`create_dataloader()` 函数**：
- 文档字符串全面重写，新增三种典型调用方式示例
- 新增 `num_workers` 推荐值（HDD: 2-4, SSD: 4-8, NVMe: 8-16）
- 新增性能调优建议

#### `src/data/standardization.py` — 注释重写

**`TrajectoryReader` 类**：
- 文档字符串全面重写，新增本地文件路径说明
- 新增性能特性说明（mmap、零拷贝、随机访问）
- `__init__` 参数增加详细注释

**`TrajectoryWriter` 类**：
- 文档字符串全面重写，新增数据写入流程说明
- 新增输出目录来源说明（`--local_path` 或 `registry.yaml`）
- 新增 Arrow IPC 格式优势说明

#### `src/data/registry.py` — 新增 local_path 字段

**`DatasetMeta` 数据类**：
- 新增 `local_path` 字段：标准化数据的本地路径
- 新增 `raw_path` 字段：原始数据本地缓存目录
- 新增路径优先级说明注释
- `to_dict()` 方法新增 `local_path` 和 `raw_path` 输出
- `load_from_yaml()` 方法解析新增字段

#### `config/dataset/registry.yaml` — 新增路径字段

每个数据集新增两个字段：
```yaml
local_path: "data/unified/{dataset_name}"   # 标准化数据位置
raw_path: "data/raw/{dataset_name}"          # 原始数据缓存位置
```

受影响的数据集：
- `bridge_v2`：`data/unified/bridge_v2`
- `roboturk`：`data/unified/roboturk`
- `maniskill`：`data/unified/maniskill`
- `language_table`：`data/unified/language_table`

#### 实验配置文件 — 注释增强

| 文件 | 更新内容 |
|------|---------|
| `config/experiment/bc_manipulation.yaml` | 数据配置段新增完整中文注释，说明路径格式、来源、batch_size 建议值 |
| `config/experiment/ppo_pick_place.yaml` | 数据配置段新增注释，区分 Offline/Online RL 数据来源 |
| `config/experiment/fusion/vla_finetune.yaml` | 数据配置段新增注释，说明 VLA 训练特殊性（小 batch、梯度累积） |
| `config/experiment/fusion/vla_rl_fusion.yaml` | 数据配置段新增注释，说明 VLA+RL 数据使用方式 |
| `config/experiment/fusion/vla_world_model.yaml` | 数据配置段新增注释，说明世界模型训练数据来源 |
| `config/experiment/fusion/triple_fusion.yaml` | 数据配置段新增注释，说明三模块的数据流关系 |

---

## 4. 第 3 次更新: 变更日志文档

**文件**: `docs/CHANGELOG.md`（即本文档）

- 汇总第 1 次和第 2 次更新的所有修改记录
- 按文件分类整理变更细节
- 提供文件变更总表，便于快速查阅

---

## 5. 第 4 次更新: 数据集查看脚本

### 5.1 设计架构文档更新

**文件**: `设计架构.md`

#### §8 部署与运行示例 — 新增数据查看命令

新增数据查看命令示例，并调整原有命令编号（第 1 条改为数据查看）：

```bash
# 1. 查看已注册数据集信息
python scripts/inspect_data.py --list

# 2. 查看数据集内容
python scripts/inspect_data.py --dataset=bridge_v2 --samples=5

# 3. 查看数据集统计
python scripts/inspect_data.py --dataset=roboturk --stats
```

---

### 5.2 新建脚本文件

#### `scripts/inspect_data.py` — 数据集查看工具

**功能定位**：数据调试和探索工具，位于架构的**数据层 (Data Layer)**，与 `prepare_data.py`（数据准备）配套使用。

**支持的功能**：

| 命令 | 作用 |
|------|------|
| `--list` / `-l` | 列出所有已注册数据集及其本地状态 |
| `--dataset=NAME` / `-d NAME` | 查看指定数据集（默认显示综合摘要） |
| `--info` | 查看数据集注册元信息 |
| `--tree` / `-t` | 查看数据集目录结构树 |
| `--schema` | 查看 Arrow 数据文件的 Schema（列名、类型、形状） |
| `--stats` | 查看统计信息（文件数、大小、轨迹长度分布） |
| `--samples=N` / `-n N` | 预览 N 条样本内容 |
| `--summary` / `-s` | 查看综合摘要 |
| `--all-datasets` / `-a` | 查看所有已注册数据集的综合摘要 |
| `--local_path=PATH` | 覆盖数据路径 |

**支持的三种数据格式**：
1. **Arrow IPC** (`.arrow`) — 标准化格式，显示完整 Schema、轨迹长度分布、样本数值
2. **TFRecord** (`.tfrecord*`) — 原始 RLDS 格式，解析 episode/step 级别内容
3. **HDF5** (`.h5`/`.hdf5`) — 原始格式，显示文件信息

**依赖安装**：
```bash
# inspect_data.py 需要以下库解析 RLDS/TFRecord 格式
pip install tfrecord
```

**RLDS 格式解析能力**（第 4 次更新增强）：

| 查看模式 | RLDS 支持内容 |
|---------|--------------|
| `--schema` | 自动检测 20+ 特征键名，区分 context/step 特征；显示每个特征的 dtype、shape、每步维度；解码 JPEG 图像尺寸 (256x256)；显示语言指令示例 |
| `--stats` | 统计总 Episode 数、总 Step 数、轨迹长度分布 (均值/标准差)；每维动作/状态的数据范围；图像 JPEG 压缩大小分布 |
| `--samples` | 逐 Episode 显示语言指令；前 3 步的完整特征数值（动作向量、状态向量、图像尺寸）；标量特征 (reward/discount/is_first 等) |
| `--summary` | 综合展示注册信息 + 本地状态 + RLDS 轨迹统计 |

**RLDS 解析技术实现**：
- 使用 `tfrecord` 库直接解析本地 TFRecord 文件，无需安装 TensorFlow
- `_detect_rlds_features()`：自动检测 RLDS 特征键名，区分 step 和 context
- `_build_rlds_description()`：根据键名后缀自动推断数据类型（byte/float/int）
- `_get_jpeg_info()`：从 JPEG 字节流解析图像尺寸，无需 PIL/OpenCV
- 自动识别 flattened 多维特征（如 `steps/action` 被展平为 `steps*7`）

**与现有工具的关系**：
- `prepare_data.py`：负责 **写入** 标准化数据
- `inspect_data.py`：负责 **读取和查看** 标准化数据（以及原始数据文件信息）
- `train.py`：使用 UnifiedDataset 直接加载标准化数据进行训练

---

## 6. 文件变更总表

| # | 文件路径 | 变更类型 | 所属更新 |
|---|---------|---------|---------|
| 1 | `设计架构.md` | 修改 | 第 1 次 + 第 2 次 + 第 4 次 |
| 2 | `src/algorithm/vla/__init__.py` | **新建** | 第 1 次 |
| 3 | `src/algorithm/vla/vla_base.py` | **新建** | 第 1 次 |
| 4 | `src/algorithm/vla/openvla_adapter.py` | **新建** | 第 1 次 |
| 5 | `src/algorithm/vla/octo_adapter.py` | **新建** | 第 1 次 |
| 6 | `src/algorithm/fusion/__init__.py` | **新建** | 第 1 次 |
| 7 | `src/algorithm/fusion/fusion_engine.py` | **新建** | 第 1 次 |
| 8 | `src/algorithm/fusion/module_registry.py` | **新建** | 第 1 次 |
| 9 | `config/experiment/fusion/vla_finetune.yaml` | **新建** | 第 1 次 |
| 10 | `config/experiment/fusion/vla_rl_fusion.yaml` | **新建** | 第 1 次 |
| 11 | `config/experiment/fusion/vla_world_model.yaml` | **新建** | 第 1 次 |
| 12 | `config/experiment/fusion/triple_fusion.yaml` | **新建** | 第 1 次 |
| 13 | `src/algorithm/__init__.py` | 修改 | 第 1 次 |
| 14 | `scripts/prepare_data.py` | 修改 | 第 2 次 |
| 15 | `src/data/dataloader.py` | 修改 | 第 2 次 |
| 16 | `src/data/standardization.py` | 修改 | 第 2 次 |
| 17 | `src/data/registry.py` | 修改 | 第 2 次 |
| 18 | `config/dataset/registry.yaml` | 修改 | 第 2 次 |
| 19 | `config/experiment/bc_manipulation.yaml` | 修改 | 第 2 次 |
| 20 | `config/experiment/ppo_pick_place.yaml` | 修改 | 第 2 次 |
| 21 | `docs/CHANGELOG.md` | **新建** | 第 3 次 |
| 22 | `scripts/inspect_data.py` | **新建** | 第 4 次 |
| 23 | `设计架构.md` | 修改 | 第 4 次 |

**统计**：新建 13 个文件，修改 10 个文件，共计 23 个文件变更。

---

> **文档更新日期**: 2026-06-09
