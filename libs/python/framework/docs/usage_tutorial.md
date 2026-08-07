# 工程项目完整使用教程

> 涵盖 BridgeData V2 和 RoboTurk 两个数据集的训练与推理全流程，
> 每一步附带详细的数学原理和代码解释。

---

## 📖 教程概述

本教程通过两个典型数据集，演示框架的完整工作流：

| 数据集 | 机器人 | 传感器 | 动作空间 | 算法路线 |
|--------|--------|--------|---------|---------|
| **BridgeData V2** | WidowX 250 | RGB + 关节状态 | 7-DoF 末端位姿 | BC → PPO → VLA |
| **RoboTurk** | Franka Panda | RGB + Depth + 关节 | 14-DoF 关节+夹爪 | BC → VLA+RL 融合 |

**学习路线**：
```
数据准备 → 数据查看 → BC 训练(桥接) → BC 推理 → PPO 训练 → PPO 推理 → VLA 微调 → VLA+RL 融合
```

---

## 第一部分：环境与数据准备

### 1.1 环境搭建

```bash
# 创建专属环境
conda create -n embodied python=3.10 -y
conda activate embodied

# 安装项目 (开发模式，使 import src 可用)
cd embodied_ai_framework
pip install -e .

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyarrow tfrecord mujoco
```

### 1.2 数据准备

#### BridgeData V2

```
原始数据 (TFRecord/RLDS)                  标准化数据 (Arrow IPC)
data/raw/bridge_v2/             →         data/unified/bridge_v2/
  bridge_dataset-train.            prepare_data.py          bridge_v2_traj_0000.arrow
  tfrecord-00000-of-01024      ─────────────────→           bridge_v2_traj_0001.arrow
  bridge_dataset-train.                                       ...
  tfrecord-00001-of-01024
```

**准备命令**：

```bash
python scripts/prepare_data.py --datasets=bridge_v2 \
    --local_path=data/unified/bridge_v2 \
    --raw_path=data/raw/bridge_v2
```

**发生什么？**
1. `DatasetRegistry` 从 `config/dataset/registry.yaml` 读取 bridge_v2 配置
2. `AdapterFactory` 创建 `BridgeV2Adapter` 实例
3. 适配器从 `data/raw/bridge_v2/` 读取原始 TFRecord/RLDS 数据
4. 逐帧产出 `UnifiedSample`（观测、动作、奖励、终止标志）
5. `TrajectoryWriter` 将样本流聚合为完整轨迹，写入 Arrow IPC 文件

#### RoboTurk

**注意**：RoboTurk 原始格式为 HDF5，适配器读取以下 HDF5 路径：

```
data/raw/roboturk/
├── trajectory_000.hdf5
│   ├── /observations/images/top    →  RGB 图像 (T, 256, 256, 3) uint8
│   ├── /observations/qpos          →  关节位置 (T, 7) float32
│   ├── /observations/qvel          →  关节速度 (T, 7) float32
│   └── /action                     →  动作     (T, 14) float32
├── trajectory_001.hdf5
└── ...
```

**准备命令**：

```bash
python scripts/prepare_data.py --datasets=roboturk \
    --raw_path=data/raw/roboturk \
    --local_path=data/unified/roboturk
```

**适配器内部工作流** (`RoboTurkAdapter.load()`)：

```
for each .hdf5 file:
    with h5py.File(path, "r") as f:
        images = f["observations/images/top"][()]    # (T, 256, 256, 3)
        qpos   = f["observations/qpos"][()]          # (T, 7)
        actions = f["action"][()]                     # (T, 14)

        for t in range(T):
            rgb = process_image(images[t])  # uint8 → float32 [0,1], HWC→CHW
            obs = {"rgb": rgb, "robot_state": qpos[t]}
            done = (t == T-1)

            yield UnifiedSample(obs=obs, action=actions[t], reward=0.0, done=done)
```

### 1.3 数据查看与验证

使用 `inspect_data.py` 验证数据完整性：

```bash
# BridgeData V2 综合摘要
python scripts/inspect_data.py --dataset=bridge_v2

# RoboTurk Schema（自动检测 RLDS 特征键）
python scripts/inspect_data.py --dataset=roboturk --schema

# RoboTurk 统计
python scripts/inspect_data.py --dataset=roboturk --stats

# RoboTurk 样本预览
python scripts/inspect_data.py --dataset=roboturk --samples=2
```

---

## 第二部分：行为克隆 (BC) — 训练与推理

> **行为克隆**是最简单的模仿学习方法：将策略学习视为监督学习，
> 让神经网络直接学习从"观测"到"动作"的映射。

### 2.1 数学原理

**核心公式**：

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[ \mathcal{L}\big(\pi_\theta(o), a\big) \right]
$$

其中：
- $\mathcal{D}$ = 专家演示数据集（从第 1 部分准备的 Arrow 文件加载）
- $\pi_\theta$ = 参数为 $\theta$ 的策略网络
- $\mathcal{L}$ = 损失函数，对于连续动作使用 **均方误差 (MSE)**：

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N \|\pi_\theta(o_i) - a_i\|^2
$$

**策略网络结构** (`GaussianPolicy`)：

```
观测 (obs_dim=128)
    ↓
MLP [128 → 256 → 256 → 32]     ← 3 层隐藏层，ReLU 激活
    ↓
高斯策略头:
  ├── mean: Linear(32 → action_dim)   ← 动作均值
  └── log_std: Parameter(action_dim)  ← 动作对数标准差
    ↓
采样: a = mean + std * ε, ε ~ N(0, I)  ← 训练时采样，推理时直接用 mean
```

### 2.2 BC 训练（以 BridgeData V2 为例）

**启动命令**：

```bash
python scripts/train.py --config config/experiment/bc_manipulation.yaml
```

**逐步计算过程**：

```
Step 1: 加载配置
        └── config/experiment/bc_manipulation.yaml
            ├── data.data_dir = "data/unified/bridge_v2"  ← 数据路径
            ├── data.batch_size = 256                      ← 批次大小
            └── training.learning_rate = 0.0001             ← 学习率

Step 2: 初始化数据加载器
        └── UnifiedDataset(data_dir)        ← 扫描 .arrow 文件，建立索引
            └── TrajectoryReader            ← mmap 打开每个文件
                └── 随机采样轨迹 → 逐帧拆分 → 返回 UnifiedSample

Step 3: 初始化策略网络
        └── GaussianPolicy(obs_dim=128, action_dim=7)
            ├── MLP: 128 → 256 → 256 → 32   ← 特征提取
            ├── mean_head: Linear(32 → 7)    ← 均值网络
            └── log_std: Parameter(7,)       ← 标准差（可学习）
            → 参数量 ≈ 128*256 + 256*256 + 256*32 + 32*7 + 7 = 107,751

Step 4: 训练循环 (每个 epoch)
        ┌─────────────────────────────────────────────────────┐
        │ for batch in dataloader:                            │
        │     obs_feat = extract_features(batch)  # (256,128) │
        │     pred_actions = policy(obs_feat)     # (256,7)   │
        │     loss = MSE(pred_actions, expert_actions)        │
        │     loss.backward()                                 │
        │     optimizer.step()                                │
        └─────────────────────────────────────────────────────┘

Step 5: 损失计算细节
        └── 假设 batch 中第一个样本:
            pred = [0.12, -0.03, 0.45, -0.21, 0.08, 0.33, 0.92]
            expert = [0.10, -0.05, 0.42, -0.19, 0.07, 0.30, 0.95]
            MSE = 1/7 * [(0.12-0.10)² + (-0.03+0.05)² + ...]
                = 1/7 * [0.0004 + 0.0004 + 0.0009 + 0.0004 + 0.0001 + 0.0009 + 0.0009]
                = 1/7 * 0.0040
                = 0.00057

Step 6: 反向传播
        └── loss.backward() → 计算梯度
            └── torch.nn.utils.clip_grad_norm_(max_norm=1.0)  ← 梯度裁剪
            └── optimizer.step() → θ ← θ - lr * ∇θ           ← 参数更新
```

### 2.3 BC 推理（评估）

**启动命令**：

```bash
python scripts/eval.py --checkpoint=checkpoints/bc/best.pt --sim=mujoco --render
```

**逐步计算过程**：

```
Step 1: 加载模型
        └── checkpoint = torch.load("checkpoints/bc/best.pt")
            └── policy.load_state_dict(checkpoint["policy"])
            └── policy.eval()  ← 切换到评估模式 (禁用 Dropout/BN)

Step 2: 初始化仿真环境
        └── SimBackendFactory.create("mujoco")
            └── RobotEnv(backend, config)
                ├── scene.load("pick_place")      ← 加载场景
                ├── robot.spawn("franka", ...)     ← 生成机器人
                └── reset(): 回到初始状态

Step 3: Rollout 循环
        ┌────────────────────────────────────────────────────┐
        │ obs = env.reset()                                 │
        │ done = False                                      │
        │ while not done:                                   │
        │     with torch.no_grad():                         │
        │         obs_feat = extract(obs)                   │
        │         action = policy(obs_feat, deterministic=True)  ← 取均值
        │     obs, reward, done, info = env.step(action)    │
        │     total_reward += reward                        │
        └────────────────────────────────────────────────────┘

Step 4: 动作执行细节
        └── policy 输出的 action (7-DoF):
            [dx, dy, dz, droll, dpitch, dyaw, gripper]
            ↓
            仿真引擎接收动作，更新物理状态:
            q_new = q_current + Δq  (关节位置增量控制)
            ↓
            渲染新图像 → 返回下一帧观测
```

---

## 第三部分：PPO 强化学习 — 训练与推理

> **PPO (Proximal Policy Optimization)** 是当前最流行的强化学习算法之一，
> 通过"裁剪"策略更新幅度来保证训练稳定性。

### 3.1 数学原理

**PPO 的 Clip 目标函数**：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是**新旧策略比率**
- $\hat{A}_t$ 是**优势函数估计**（GAE 计算）
- $\epsilon$ 是裁剪阈值（通常 0.2）

**GAE (广义优势估计)**：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

其中 $\lambda=0$ 退化为 TD(0)，$\lambda=1$ 退化为蒙特卡洛。

**总损失**：

$$
L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)
$$

| 项 | 含义 | 系数 |
|----|------|------|
| $L^{\text{CLIP}}$ | 策略损失（最大化期望回报） | 1.0 |
| $L^{\text{VF}}$ | 价值损失（MSE，让价值估计更准） | 0.5 |
| $S[\pi_\theta]$ | 熵奖励（鼓励探索） | 0.01 |

### 3.2 PPO 训练

**启动命令**：

```bash
python scripts/train.py --config config/experiment/ppo_pick_place.yaml
```

**Actor-Critic 网络结构**：

```
观测 (obs_dim=128)
    ↓
共享 MLP [128 → 256 → 256]     ← 特征提取器
    ├──────────┬──────────┐
    ↓          ↓          ↓
Actor        Actor     Critic
mean_head   log_std    Linear(256 → 1)
Linear      Parameter     ↓
(256 → 7)   (7,)      V(s) 标量
    ↓          ↓
动作均值    标准差
```

**逐步训练计算过程**：

```
外循环: for iteration in range(max_iterations):
    内循环: for step in range(num_steps):
        ┌────────────────────────────────────────────────────┐
        │ action, log_prob, value = actor_critic(obs)      │
        │ next_obs, reward, done, _ = env.step(action)     │
        │ 存储 (obs, action, reward, done, log_prob, value) │
        │ obs = next_obs                                   │
        └────────────────────────────────────────────────────┘

    当收集了 N 步经验后:

    Step 1: 计算 GAE 优势
            ┌────────────────────────────────────────────────┐
            │ advantages = compute_gae(rewards, values,     │
            │                            dones, gamma, λ)   │
            │                                              │
            │ 例如 γ=0.99, λ=0.95:                         │
            │   δ_T = r_T - V(s_T)                         │
            │   δ_{T-1} = r_{T-1} + γ*V(s_T) - V(s_{T-1}) │
            │   ...                                        │
            │   A_t = δ_t + γλ*δ_{t+1} + (γλ)²*δ_{t+2}+...│
            └────────────────────────────────────────────────┘

    Step 2: 内循环 PPO 更新 (K=10 epoch)
            for epoch in range(10):
                for minibatch in dataloader(trajectories):
                    ┌────────────────────────────────────────┐
                    │ # 计算新旧策略比率                      │
                    │ new_log_prob, entropy, value =          │
                    │     actor_critic.evaluate(obs, action)  │
                    │ ratio = exp(new_log_prob - old_log_prob)│
                    │                                         │
                    │ # 裁剪后的策略损失                      │
                    │ surr1 = ratio * advantages              │
                    │ surr2 = clip(ratio, 0.8, 1.2) * adv    │
                    │ policy_loss = -mean(min(surr1, surr2))  │
                    │                                         │
                    │ # 价值损失                             │
                    │ value_loss = MSE(value, returns)        │
                    │                                         │
                    │ # 熵奖励                               │
                    │ entropy_loss = -mean(entropy)           │
                    │                                         │
                    │ # 总损失                               │
                    │ loss = policy_loss                       │
                    │      + 0.5 * value_loss                 │
                    │      + 0.01 * entropy_loss              │
                    │                                         │
                    │ loss.backward()                         │
                    │ clip_grad_norm_(max_norm=0.5)           │
                    │ optimizer.step()                        │
                    └────────────────────────────────────────┘
```

**数值示例**（假设一个 minibatch 中的单个样本）：

```
假设:
  old_log_prob = -0.5 (旧策略下该动作的概率密度对数)
  new_log_prob = -0.3 (新策略下该动作的概率密度对数)
  优势 A = 2.0 (该动作比平均好)

计算:
  ratio = exp(-0.3 - (-0.5)) = exp(0.2) = 1.221
  surr1 = 1.221 × 2.0 = 2.442
  surr2 = clip(1.221, 0.8, 1.2) × 2.0 = 1.2 × 2.0 = 2.400
  policy_loss = -min(2.442, 2.400) = -2.400
               ↑ 取裁剪后的值，限制更新幅度

如果 ratio 太大 (比如 1.5):
  surr1 = 1.5 × 2.0 = 3.0
  surr2 = clip(1.5, 0.8, 1.2) × 2.0 = 1.2 × 2.0 = 2.4
  policy_loss = -min(3.0, 2.4) = -2.4   ← 被裁剪了！
  这防止了策略一步更新过大。
```

### 3.3 PPO 推理

```bash
python scripts/eval.py --checkpoint=checkpoints/ppo/best.pt --sim=mujoco --render
```

**与 BC 推理的区别**：
- BC 推理直接输出动作均值（确定性）
- PPO 推理也是取均值（`deterministic=True`），但训练时 PPO 通过熵奖励学到的动作分布通常更平滑

---

## 第四部分：VLA 大模型微调 — 训练与推理

> VLA (Vision-Language-Action) 模型将视觉理解、语言推理和动作生成
> 统一在单个端到端模型中。

### 4.1 Octo 模型微调

**启动命令**：

```bash
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=octo \
    data.data_dir=data/unified/bridge_v2 \
    vla.finetune_method=lora
```

**逐步计算过程**：

```
Step 1: 加载 Octo 模型
        └── OctoAdapter.load()
            ├── 从 HuggingFace 下载 octo-models/octo-base
            ├── ViT 视觉编码器 (加载预训练权重)
            ├── T5 文本编码器 (加载预训练权重)
            └── Transformer backbone (加载预训练权重)
                ↓
        └── 添加 Readout head: Linear(隐藏层 → action_dim)
            └── 仅这个头是随机初始化的，其他权重冻结

Step 2: 注入 LoRA 适配器
        └── 在 Transformer 的 Q/K/V/O 投影层插入 LoRA:
            W' = W + BA   其中 B∈R^{d×r}, A∈R^{r×k}, r << min(d,k)
            └── 原始权重 W 冻结，只训练低秩矩阵 B 和 A
            └── 参数量从 500M 降至 ~1M (r=32)

Step 3: 数据加载
        └── UnifiedDataset(data_dir) 加载 BridgeData V2
            └── 每条轨迹包含:
                ├── RGB 图像序列
                ├── 关节状态
                ├── 语言指令 (如 "put spoon on tray")
                └── 专家动作

Step 4: 前向传播
        ┌────────────────────────────────────────────────┐
        │ images = batch["rgb"]          # (B, 3, 128, 128)│
        │ instruction = batch["text"]    # ["put spoon..."]│
        │                                                  │
        │ # 视觉编码 (ViT)                                │
        │ vis_feat = vit_encoder(images)  # (B, 256, 512) │
        │                                                  │
        │ # 语言编码 (T5)                                 │
        │ lang_feat = t5_encoder(instruction) # (B, 512)  │
        │                                                  │
        │ # 特征融合 (Transformer)                        │
        │ fused = transformer(vis_feat, lang_feat)         │
        │                                                  │
        │ # Readout head → 动作                           │
        │ pred_action = readout(fused)   # (B, 7)         │
        │                                                  │
        │ # 损失: L2 损失 vs 专家动作                     │
        │ loss = MSE(pred_action, expert_action)           │
        └────────────────────────────────────────────────┘

Step 5: 反向传播
        └── 仅更新 LoRA 参数 + Readout head 参数
            └── 其他参数冻结 (requires_grad=False)
```

### 4.2 OpenVLA 7B 微调

```bash
python scripts/train.py experiment=fusion/vla_finetune \
    vla.model=openvla-7b \
    vla.finetune_method=lora \
    vla.load_in_8bit=true
```

**与 Octo 的区别**：

| 特性 | Octo | OpenVLA 7B |
|------|------|-----------|
| 视觉编码器 | ViT | SigLIP + DINOv2 |
| 语言编码器 | T5 | Llama 2 |
| 参数量 | ~500M | ~7B |
| 显存需求 | 4-6 GB | 12-16 GB (8-bit) |
| 推理速度 | 快 | 慢 5-10x |
| 语言理解 | 中等 | 强 |

### 4.3 VLA 推理

```bash
python scripts/eval.py --checkpoint=checkpoints/vla/best.pt \
    --config config/experiment/fusion/vla_finetune.yaml \
    --sim=mujoco --render
```

**VLA 推理流程**：

```
观测 (RGB 图像 + 机器人状态)
    │
    ▼
语言指令: "pick up the red cube"
    │
    ▼
[VLA 模型前向]
    ├── ViT 编码图像 → 视觉 token 序列
    ├── T5 编码语言 → 语言 token
    ├── Transformer 多模态融合
    └── Readout head → 动作 token
    │
    ▼
输出: action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
      confidence = 0.85  (模型对动作的置信度)
      ↓
仿真环境执行动作
```

---

## 第五部分：VLA + RL 融合 — 训练与推理

> **VLA + RL 融合**是框架的核心创新之一：VLA 提供高层语义理解
> 和动作先验，RL 提供精细运动控制。

### 5.1 融合策略原理

#### 策略 1：加权平均 (Weighted Average)

$$
a_{\text{final}} = w \cdot a_{\text{VLA}} + (1-w) \cdot a_{\text{RL}}
$$

其中权重 $w$ 可以固定或根据 VLA 置信度自适应调整：
$$
w = \text{confidence}_{\text{VLA}} \cdot w_{\text{base}}
$$

**数值示例**（假设 7-DoF 动作）：

```
VLA 动作:       [0.10, -0.05, 0.42, -0.19, 0.07, 0.30, 0.95]  置信度=0.85
RL 动作:        [0.08, -0.03, 0.38, -0.15, 0.05, 0.25, 0.90]
w_base = 0.4
自适应 w = 0.85 × 0.4 = 0.34

融合动作:
  a[0] = 0.34 × 0.10 + 0.66 × 0.08 = 0.087
  a[1] = 0.34 × (-0.05) + 0.66 × (-0.03) = -0.037
  a[6] = 0.34 × 0.95 + 0.66 × 0.90 = 0.917
  ...
```

#### 策略 2：贝叶斯融合 (Bayesian Fusion)

将 VLA 和 RL 的输出视为高斯分布：

$$
a_{\text{VLA}} \sim \mathcal{N}(\mu_V, \sigma_V^2), \quad
a_{\text{RL}} \sim \mathcal{N}(\mu_R, \sigma_R^2)
$$

贝叶斯最优融合：

$$
\mu_F = \frac{\mu_V / \sigma_V^2 + \mu_R / \sigma_R^2}{1/\sigma_V^2 + 1/\sigma_R^2}
$$

其中 $\sigma_V = 1 - \text{confidence}_{\text{VLA}}$（置信度越高，方差越小）。

#### 策略 3：优先级路由 (Priority Routing)

$$
a_{\text{final}} = \begin{cases}
a_{\text{VLA}}, & \text{if } \text{confidence}_{\text{VLA}} > \text{threshold} \\
a_{\text{RL}}, & \text{otherwise}
\end{cases}
$$

### 5.2 VLA + RL 联合训练

**启动命令**：

```bash
python scripts/train.py --config config/experiment/fusion/vla_rl_fusion.yaml
```

**训练框架架构**：

```
┌─────────────────────────────────────────────────────────┐
│                   融合引擎 (FusionEngine)                 │
│                                                         │
│  VLA 模型 (Octo/OpenVLA)         RL 策略 (PPO)          │
│    ↓                                   ↓                │
│  act_with_language(obs, instruction)   act(obs)          │
│    ↓                                   ↓                │
│  VLAOutput(action, confidence)        action             │
│    ↓                                   ↓                │
│  └──────────────┬──────────────────────┘                │
│                 ↓                                       │
│          融合策略 (FusionStrategy)                       │
│          ├── weighted_average: w*a_vla + (1-w)*a_rl     │
│          ├── bayesian: 高斯分布融合                      │
│          └── priority: 置信度 > 阈值 ? VLA : RL         │
│                 ↓                                       │
│           融合后的动作 a_final                           │
└─────────────────────────────────────────────────────────┘
    ↓
仿真环境 step(a_final) → 奖励 + 下一观测
```

**融合训练循环（每个 iteration）**：

```
1. VLA 推理:
   for each env in vec_env:
       vla_output = vla_model.act_with_language(obs, instruction)
       a_vla = vla_output.action
       conf  = vla_output.confidence

2. RL 推理:
   a_rl = rl_policy.act(obs)

3. 融合:
   a_final = fusion_engine.fuse(a_vla, a_rl, conf)
              ↓
          加权平均 / 贝叶斯 / 优先级路由

4. 环境执行:
   next_obs, reward, done = env.step(a_final)

5. RL 更新 (PPO):
   rl_policy.update(obs, a_rl, reward, next_obs, done)

6. 可选: VLA 微调 (使用 RL 采集的新数据):
   if collect_vla_data:
       vla_model.finetune(new_obs, new_actions, new_instructions)
```

### 5.3 VLA + RL 推理

```bash
python scripts/eval.py --checkpoint=checkpoints/vla_rl_fusion/best.pt \
    --config config/experiment/fusion/vla_rl_fusion.yaml \
    --sim=mujoco --render
```

---

## 第六部分：使用 RoboTurk 数据训练

### 6.1 BC 训练（RoboTurk）

**关键差异**：RoboTurk 动作空间是 14 维（7-DoF 关节位置 × 2），需要调整配置。

```bash
python scripts/train.py --config config/experiment/bc_manipulation.yaml \
    data.data_dir=data/unified/roboturk \
    data.dataset_name=roboturk \
    training.action_dim=14         # ← 关键：RoboTurk 为 14 维动作
```

**适配器数据处理流程**：

```
原始 HDF5:
  /observations/images/top:   (T, 256, 256, 3) uint8     ← RGB 图像
  /observations/qpos:         (T, 7) float32              ← 关节位置
  /observations/qvel:         (T, 7) float32              ← 关节速度（未使用）
  /action:                    (T, 14) float32             ← 动作

→ RoboTurkAdapter._process_image():
    1. PIL.Image.fromarray(image)        ← numpy → PIL
    2. img.resize((256, 256))             ← 确保尺寸
    3. TF.to_tensor(img)                  ← uint8 → float32 [0,1], HWC→CHW

→ UnifiedSample 输出:
    obs = {
        "rgb":        torch.float32  (3, 256, 256)  ← 归一化图像
        "robot_state": torch.float32 (7,)             ← 关节位置
    }
    action = torch.float32 (14,)        ← 关节位置(7) + 夹爪宽度(1) 重复两遍
    reward = 0.0                         ← RoboTurk 无奖励信号
    done = True (最后一步) / False (其他)
```

### 6.2 VLA + RL 融合训练（RoboTurk）

```bash
python scripts/train.py experiment=fusion/vla_rl_fusion \
    vla.model=octo \
    data.data_dir=data/unified/roboturk \
    training.action_dim=14 \
    rl.action_dim=14 \
    fusion.strategy=weighted_average
```

---

## 第七部分：端到端完整示例

### 场景：用 BridgeData V2 训练 BC 策略，在 MuJoCo 中评估

```bash
# ====== 第 1 步：环境搭建 ======
conda create -n embodied python=3.10 -y
conda activate embodied
cd embodied_ai_framework
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyarrow mujoco

# ====== 第 2 步：数据准备 ======
python scripts/prepare_data.py --datasets=bridge_v2 \
    --local_path=data/unified/bridge_v2

# ====== 第 3 步：数据验证 ======
python scripts/inspect_data.py --dataset=bridge_v2 --summary
python scripts/inspect_data.py --dataset=bridge_v2 --samples=3

# ====== 第 4 步：BC 训练 ======
python scripts/train.py --config config/experiment/bc_manipulation.yaml

# 训练输出示例:
# [2026-06-09] [INFO] Epoch 1/100 - bc_loss: 0.0523 - lr: 1e-4
# [2026-06-09] [INFO] Epoch 10/100 - bc_loss: 0.0087 - lr: 1e-4
# [2026-06-09] [INFO] Epoch 50/100 - bc_loss: 0.0032 - lr: 1e-4
# [2026-06-09] [INFO] Epoch 100/100 - bc_loss: 0.0018 - lr: 1e-4
# [2026-06-09] [INFO] Checkpoint saved: checkpoints/bc/best.pt

# ====== 第 5 步：评估 ======
python scripts/eval.py --checkpoint=checkpoints/bc/best.pt \
    --sim=mujoco --num_episodes=50 --render

# 评估输出示例:
# [2026-06-09] [INFO] 评估结果 (50 episodes):
# [2026-06-09] [INFO]   Success Rate: 76.0% (38/50)
# [2026-06-09] [INFO]   Avg Reward:   42.3
# [2026-06-09] [INFO]   Avg Episode Length: 128.5 steps
# [2026-06-09] [INFO]   Video saved: videos/bc_eval/episode_0.mp4
```

### 场景：用 RoboTurk 数据训练 VLA+RL 融合策略

```bash
# ====== 第 1 步：数据准备 ======
python scripts/prepare_data.py --datasets=roboturk \
    --raw_path=data/raw/roboturk \
    --local_path=data/unified/roboturk

# ====== 第 2 步：数据查看 ======
python scripts/inspect_data.py --dataset=roboturk --schema
python scripts/inspect_data.py --dataset=roboturk --stats

# ====== 第 3 步：VLA + RL 融合训练 ======
python scripts/train.py --config config/experiment/fusion/vla_rl_fusion.yaml \
    data.data_dir=data/unified/roboturk \
    training.action_dim=14 \
    rl.action_dim=14 \
    fusion.strategy=weighted_average \
    fusion.weighted_average.vla_weight=0.4 \
    fusion.weighted_average.adaptive=true

# ====== 第 4 步：评估 ======
python scripts/eval.py --checkpoint=checkpoints/vla_rl_fusion/best.pt \
    --config config/experiment/fusion/vla_rl_fusion.yaml \
    --sim=mujoco --num_episodes=50 --render
```

---

## 附录

### A. 数据流向图（完整训练流程）

```
[原始数据]                    [配置文件]
   TFRecord/HDF5              experiment.yaml
      │                            │
      ▼                            ▼
[prepare_data.py]          [TrainingPipeline]
      │                            │
      ▼                            │
[Arrow IPC 文件]                  │
   data/unified/*.arrow           │
      │                            │
      ▼                            │
[UnifiedDataset] ◄─────────────────┘
   │                              ▲
   │ 流式读取 .arrow               │
   │ 拆分为 UnifiedSample          │
   ▼                              │
[DataLoader]                      │
   │ 多进程预取                    │
   ▼                              │
[训练循环] ────────────────────────┘
   │
   ├── [IL 模式]:  batch → 策略 → MSE loss → 反向传播
   │
   ├── [RL 模式]:  obs → 策略 → action → env.step → 经验池 → PPO 更新
   │
   └── [VLA 模式]: obs + instruction → VLA → action → 语言条件损失 → LoRA 更新
   │
   ▼
[checkpoint/best.pt] ──→ [eval.py] ──→ [仿真 Rollout] ──→ [评估指标]
```

### B. 关键类与方法速查

| 类 | 文件 | 核心方法 | 作用 |
|----|------|---------|------|
| `DatasetRegistry` | `src/data/registry.py` | `get()`, `list_all()` | 查询数据集元信息 |
| `UnifiedDataset` | `src/data/dataloader.py` | `__iter__()` | 流式读取 Arrow 文件 |
| `BCTrainer` | `src/algorithm/imitation/bc.py` | `train_epoch()`, `validate()` | BC 训练循环 |
| `PPOTrainer` | `src/algorithm/reinforcement/ppo.py` | `train_iteration()` | PPO 训练循环 |
| `VLABase` | `src/algorithm/vla/vla_base.py` | `act_with_language()`, `load()` | VLA 模型接口 |
| `FusionEngine` | `src/algorithm/fusion/fusion_engine.py` | `select_action()`, `create()` | 模块融合引擎 |
| `TrainingPipeline` | `src/orchestration/pipeline.py` | `run()` | 训练流水线编排 |
| `TrajectoryReader` | `src/data/standardization.py` | `read_trajectory()` | 读取 Arrow 轨迹 |

### C. 超参数速查表

| 算法 | 超参数 | 推荐值 | 说明 |
|------|--------|--------|------|
| BC | `learning_rate` | 1e-4 | Adam 优化器 |
| BC | `batch_size` | 128-256 | 越大梯度越稳定 |
| PPO | `lr` | 3e-4 | Actor-Critic 共享学习率 |
| PPO | `gamma` | 0.99 | 折扣因子 |
| PPO | `gae_lambda` | 0.95 | GAE 参数 |
| PPO | `clip_epsilon` | 0.2 | 裁剪阈值 |
| PPO | `entropy_coef` | 0.01 | 熵奖励权重 |
| VLA | `finetune_method` | `lora` | 参数高效微调 |
| VLA | `lora_rank` | 32 | LoRA 秩（越大越强但越慢） |
| VLA+RL | `vla_weight` | 0.3-0.5 | VLA 指导权重 |
| VLA+RL | `adaptive` | `true` | 是否根据置信度自适应 |
