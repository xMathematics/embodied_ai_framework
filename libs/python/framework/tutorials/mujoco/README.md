# MuJoCo 仿真教程：从入门到项目实战

> 本教程是《具身智能全流程框架》的**仿真层实战指南**。从零开始，带你掌握
> DeepMind 的开源物理引擎 **MuJoCo (Multi-Joint dynamics with Contact)**，
> 并最终对接本框架的 `MuJoCoBackend → RobotEnv → VecEnv → PPO` 完整 RL 训练栈。

---

## 📁 目录结构

```
tutorials/mujoco/
├── README.md                     ← 本教程 (你正在看)
├── 调试过程与思路.md             ← 开发调试全记录 (现象→思路→根因→修改→结果)
├── models/                       ← MJCF 模型文件 (代码自动生成/引用)
│   ├── robot_arm.xml             #   自定义 5-DoF 机械臂 + 两指夹爪
│   ├── cartpole.xml              #   倒立摆 (由 02 生成)
│   ├── pendulum.xml              #   单摆 (由 04 生成)
│   └── pick_place_scene.xml      #   抓取场景 (由 make_pick_place_scene 生成)
└── code/                         ← 全部可运行代码 (01~09)
    ├── 01_first_simulation.py     入门: 加载模型/步进/mjModel vs mjData
    ├── 02_build_cartpole.py       入门: MJCF 场景建模 (倒立摆)
    ├── 03_sensors_actuators.py    进阶: 传感器与三种执行器模式
    ├── 04_pendulum_swingup.py     进阶: 力矩控制 + 能量整形起摆
    ├── 05_render_video.py         进阶: 离屏渲染 + 视频录制
    ├── make_pick_place_scene.py   实战: 程序化生成抓取场景
    ├── 06_pick_place_env.py       实战: 抓取环境 PickPlaceEnv (DLS 逆运动学)
    ├── 07_scripted_pick_place.py  实战: 脚本化抓取搬运 + 录视频
    ├── 08_train_cartpole_ppo.py   实战: 对接框架 PPO 训练倒立摆平衡
    └── 09_evaluate_cartpole.py    实战: 评估策略 + 录制平衡视频
```

---

## 0. 环境准备

```bash
# Python 3.10+ (本教程在 3.13 + mujoco 3.9.0 验证)
pip install mujoco numpy torch pillow opencv-python matplotlib
# 本教程还用到框架的 RL 组件 (torch 必装)
pip install -r ../../requirements.txt
```

> 💡 **服务器/容器无显示器也能运行**：MuJoCo 自带 EGL/OSMesa 离屏渲染后端，
> 教程里的渲染与视频录制全部在无显示器环境下验证通过。

验证安装：

```python
import mujoco
print(mujoco.__version__)   # 3.x
```

---

## 1. 入门：第一次仿真

**运行:** `python code/01_first_simulation.py`

### 1.1 两个核心对象：`mjModel` 与 `mjData`

MuJoCo 的一切围绕两个结构体，这是理解整个引擎的钥匙：

| | `MjModel` (模型) | `MjData` (数据) |
|---|---|---|
| 性质 | **静态模板**，不可变 | **动态实例**，可变 |
| 内容 | 质量、几何、关节、执行器、传感器定义 | 当前位置 `qpos`、速度 `qvel`、力等 |
| 类比 | 机器人的"图纸" | 机器人的"当前状态" |
| 关系 | 一个 Model 可派生出**任意多个** Data | 每个 Data 是一个独立仿真实例 |

一个 `MjModel` 派生多个 `MjData` 是 MuJoCo 能极快并行（MJX/JAX）的根本原因。

```python
model = mujoco.MjModel.from_xml_path("models/robot_arm.xml")
data  = mujoco.MjData(model)          # 从 model 创建实例
mujoco.mj_step(model, data)           # 前进一步物理仿真
print(data.qpos)                      # 读取关节位置
```

### 1.2 常用 API 速查

```python
mujoco.mj_step(m, d)                          # 前进一步 (核心!)
mujoco.mj_forward(m, d)                       # 仅更新运动学 (不动物理)
mujoco.mj_resetData(m, d)                     # 重置到初始状态
mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "name")  # 名字→id
mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, id)     # id→名字
mujoco.mj_jacBody(m, d, jacp, None, body_id)  # 雅可比矩阵 (IK 用)
mujoco.mj_contactForce(m, d, i, out)          # 第 i 个接触的力
```

**⏱ 性能提示**：单 CPU 核通常每秒可仿真数千步；配合 MJX 可并行上千个环境。

---

## 2. MJCF 场景建模

**运行:** `python code/02_build_cartpole.py`

MJCF（MuJoCo XML）是 MuJoCo 的场景描述语言。核心标签：

```xml
<mujoco model="cartpole">
  <option timestep="0.02"/>            <!-- 全局物理参数 -->
  <default>
    <joint damping="0.1"/>             <!-- 默认值, 可继承 -->
  </default>
  <worldbody>                          <!-- 世界根节点 -->
    <light pos="0 0 3"/>               <!-- 光源 -->
    <geom name="floor" type="plane"/>  <!-- 静态几何 -->
    <body name="cart" pos="0 0 0.15">  <!-- 刚体 (运动学树节点) -->
      <joint name="slider" type="slide" axis="1 0 0" range="-1.2 1.2"/>
      <geom name="cart_geom" type="box" size="0.25 0.06 0.06"/>
      <body name="pole" pos="0 0 0.06"> <!-- 子刚体: 摆杆 -->
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule" size="0.03 0.25"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_motor" joint="slider" gear="10"/> <!-- 执行器 -->
  </actuator>
</mujoco>
```

### 2.1 建模的三个易错点（本教程踩过的坑，务必记住）

1. **几何体不能互相穿透**：小车与轨道若重叠会产生巨大接触力，把机器人"卡死"。
   模型里小车悬空、由 `slide` 关节约束在 1D 轨道上，就是为避免接触干扰。
2. **相邻连杆要排除自碰撞**：肩关节球体与基座柱必然重叠，若不排除，机器人会被
   **自己的"身体"卡住**。用 `<contact><exclude body1="..." body2="..."/></contact>`。
   （见 `models/robot_arm.xml`，这是本项目机械臂能正常工作的关键。）
3. **初始精确直立是不稳定平衡点**：摆杆在 θ=0 静止时永远不会自己倒下，
   演示"重力让它倒下"需给一个微小初始偏角。

### 2.2 从零构建机械臂

`models/robot_arm.xml` 是一个**手写**的 5-DoF 机械臂（肩/肘/腕 4 个旋转关节 +
2 个手指滑动关节），是第 6 章抓取项目的主角。打开它对照理解每个 `<body>/<joint>/<geom>`。

---

## 3. 传感器与执行器

**运行:** `python code/03_sensors_actuators.py`

### 3.1 传感器（`<sensor>` 元素）

```xml
<sensor>
  <jointpos name="j_pos" joint="slider"/>     <!-- 关节位置 -->
  <jointvel name="j_vel" joint="slider"/>     <!-- 关节速度 -->
  <actuatorfrc name="act_force" actuator="cart_motor"/> <!-- 执行器力 -->
  <framepos name="tip" objtype="site" objname="tip"/>   <!-- 指定点位置 -->
</sensor>
```
值统一读取自 `data.sensordata`（用 `model.sensor_adr[i]` 定位第 i 个传感器）。

### 3.2 三种执行器模式（RL 与控制的根基）

| 类型 | `ctrl` 含义 | 适用场景 |
|---|---|---|
| `motor` | **力/力矩** (× gear) | RL 连续动作、经典控制 |
| `position` | **目标位置** (内部隐式 PD 伺服) | 机械臂关节伺服（本项目机械臂用此） |
| `velocity` | **目标速度** (内部阻尼伺服) | 速度跟踪 |

**切换模式只改一行 `<actuator>`，模型其余部分不变**——这是 MuJoCo 的一大优势。

---

## 4. 经典控制：能量整形起摆

**运行:** `python code/04_pendulum_swingup.py`

单摆只有 1 个自由度、力矩受限，无法直接把摆"抬"到直立。经典解法 **能量整形
(Energy Shaping)**：

$$
E = \tfrac{1}{2}I\omega^2 + mgL\cos\theta,\qquad E^* = mgL
$$

$$
u = k_p\,(E^* - E)\,\operatorname{sign}(\omega)
$$

由 $dE/dt = u\cdot\omega$ 可知：能量不足时让力矩与角速度同号即可"顺着摆动方向推
一把"逐步累积能量；过冲时自动反向制动。当 $|\theta|$ 与 $|E-E^*|$ 都足够小，切换为
线性 PD 锁定。运行结果：单摆 2 秒内起摆并稳定在直立（偏差 < 1°）。

> 这是"经典控制"的完美示例，也是 RL 的"参照系"——第 8 章的 PPO 学的是同一件事。

---

## 5. 离屏渲染与视频录制

**运行:** `python code/05_render_video.py`

```python
renderer = mujoco.Renderer(model, height=480, width=640)  # 离屏渲染器
renderer.update_scene(data, camera="track")               # 同步状态到渲染器
img = renderer.render()                                    # (H, W, 3) uint8
```

- 服务器无显示器也能渲染（EGL/OSMesa）。
- 相机可在 MJCF 里用 `<camera>` 定义（`mode="trackcom"` 可跟随物体）。
- 用 OpenCV `cv2.VideoWriter` 把帧序列写成 mp4，无需额外 ffmpeg。
- **视频用途**：训练可视化 / 论文插图 / Sim2Real 迁移对比。

---

## 6. 项目实战 (一)：抓取搬运 Pick-and-Place

**运行:** `python code/make_pick_place_scene.py` → `python code/06_pick_place_env.py`
→ `python code/07_scripted_pick_place.py`

### 6.1 场景与逆运动学（IK）

`make_pick_place_scene.py` 把机械臂 + 方块 + 目标点组装成完整场景
（方块用 `<freejoint/>` 获得 6 自由度）。

`06_pick_place_env.py` 实现 **阻尼最小二乘逆运动学 (DLS-IK)**，用雅可比矩阵把
"末端目标位置"转成"关节角"：

$$
\Delta q = J^\top (J J^\top + \lambda I)^{-1}\,e,\qquad e = \text{目标} - \text{当前}
$$

- $\lambda$ 是阻尼系数：既避免 $JJ^\top$ 奇异，又限制步长。
- 每轮重算雅可比，迭代至收敛（误差 < 0.2mm）。
- `wrist_pitch` 由解析法设定，保证夹爪始终朝下。

### 6.2 状态机：approach → descend → grasp → lift → transfer → place → release

`07_scripted_pick_place.py` 用一个状态机驱动完整搬运流程，并实时渲染成视频
`pick_place_demo.mp4`。

### 6.3 关于"抓取"的诚实说明（重要）

真实世界中抓取靠**指尖与物体的摩擦**。在 MuJoCo 里，对"放在桌面上的轻方块 +
薄手指"做摩擦夹持会非常不稳健，本教程开发中花大量篇幅验证了这一点：

- 接触**法向**是理解接触的关键。注意：**MuJoCo 接触帧的第一个轴 (frame[0:3]) 才是
  法向**，不是第三个轴！误读会导致把正确的 Y 法向接触当成"退化 X 法向"。
- 高夹持力下手指可能**压穿**轻物体（position 执行器会一直向内推）。
- 因此项目实战采用 **运动学虚拟抓取**（`attach_block()`）：抓取后用 `mj_forward`
  每步把方块位置精确同步到夹爪（等价于 MuJoCo 官方 `mj_attachBody` 的功能），
  抬升/搬运/放置完全可控可靠。这正是很多官方演示的做法，也避开了脆弱的接触建模。

> 想挑战真实摩擦抓取？可以从"增大接触面积 + 软接触 (solimp dmax) + 低夹持力 +
> 缓慢抬升"开始调参——这是机器人学里真实的调参过程。

### 6.4 预期输出

```
🎉 任务成功: 方块已放到目标点！  (误差 < 2cm)
✅ 视频已保存: pick_place_demo.mp4
```

---

## 7. 项目实战 (二)：对接框架 PPO 训练倒立摆

**运行:** `python code/08_train_cartpole_ppo.py` → `python code/09_evaluate_cartpole.py`

这一节把前几节的 MuJoCo 知识接入**本框架的完整 RL 训练栈**：

```
MuJoCoBackend (仿真后端, 框架)
   ↓ 通过 SimBackendFactory 创建
RobotEnv (Gym 风格环境, 框架)     ← 教程自定义 CartPoleEnv
   ↓ 包装
DummyVecEnv (向量化, 框架)        ← 多环境并行采样
   ↓ 驱动
PPOTrainer (策略梯度, 框架)       ← PPO 训练
```

关键点：

- `CartPoleEnv` 继承框架的 `RobotEnv`，内部用 `MuJoCoBackend` 驱动；
  观测用 **sin/cos 表示角度**避免无界累积，动作是连续推力。
- `PPOTrainer` 每轮 `collect → train_on_buffer → evaluate`。
- 训练约 2 分钟即可收敛：评估回报从 ~170 升到**满分 800**（倒立摆完美平衡）。
- `09` 加载训练好的策略、以 `torch.no_grad()` 推理控制，并渲染成
  `cartpole_balance.mp4`。

### 7.1 预期输出

```
🏁 训练完成, 最佳平均回报 = 800.0 (满分 800)
✅ 视频已保存: cartpole_balance.mp4
```

---

## 8. 常见问题与调试技巧

1. **`KeyError: Invalid name 'end_effector'`**：框架后端的 `step()` 里对 body
   是否存在要用 `mujoco.mj_name2id(...) != -1` 判断，不能 `hasattr`。
2. **仿真 NaN/不稳定**：几何体穿透、执行器刚度过高（`kp` 太大而关节惯量太小）、
   时间步长过大。先查接触对（`data.contact`），再降 `kp`/加 `armature`。
3. **手臂动不了/被卡住**：90% 是**自碰撞**没排除，或与桌子穿透。
4. **接触力读不出来**：用 `mujoco.mj_contactForce(m, d, i, out)`（返回
   `[法向力, 切向1, 切向2]`），**法向方向看 `contact.frame[0:3]`**。
5. **方块抬不起来**：见 6.3，薄手指 + 桌面方块用摩擦夹持不可靠，改用虚拟抓取。

---

## 9. 下一步

- 把 `CartPoleEnv` 换成 `PickPlaceEnv` 的观测/奖励，用 PPO 学抓取策略。
- 用 `SubprocVecEnv` 多进程并行，或 MJX 上 GPU 跑上千环境。
- 加入**领域随机化**（随机初始角/方块位置）提升策略鲁棒性。
- 把训练好的策略对接 `src/algorithm/sim2real.py` 做 Sim2Real 迁移。

> 📖 想知道"为什么教程里要这么建模 / 那些坑是怎么踩出来的"，请看
> [《调试过程与思路》](./调试过程与思路.md)——包含每个问题的现象、排查思路、
> 根因、修改方案、预期与实际结果，以及一套可复用的仿真/RL 调试方法论。

祝仿真愉快！🎉
