# 具身智能项目集合 (Embodied AI Monorepo)

面向**具身智能**（Embodied AI）研究的多语言项目集合，覆盖
**Python · C++ · ROS2 · CUDA** 四类代码，形成从开源数据集 → 仿真 → 算法 → 部署的完整闭环。

```
libs/       跨项目共享库 (Python 框架 / C++ / CUDA / ROS2 接口)
projects/   具身智能具体项目 (机械臂操作 / 移动操作 / 人形)
ros/        ROS2 功能节点 (感知 / 控制 / 仿真 / 启动)
data/       全局数据集 (本地优先, gitignore)
tools/      统一构建与检查脚本
docs/       (见 libs/python/framework/docs)
```

---

## 📁 目录结构

```
embodied_ai_framework/
├── CMakeLists.txt              # C++/CUDA 总构建 (cmake -S . -B build)
├── pytest.ini                  # Python 测试配置 (testpaths → libs/python/framework/tests)
├── .vscode/                    # VS Code 配置 (ROS2/CUDA/Python 调试与 IntelliSense)
│
├── libs/                       # ── 跨项目共享库 ──
│   ├── python/
│   │   └── framework/          # 具身智能全流程框架 (五层四流) ★原项目迁入
│   ├── cpp/                    # C++ 共享库 (embodied_cpp: 几何/线性代数)
│   ├── cuda/                   # CUDA 共享算子 (embodied_cuda: GPU 内核)
│   └── ros/
│       └── embodied_interfaces/# 共享消息/服务/动作 (msg/srv/action)
│
├── projects/                   # ── 具体项目 ──
│   ├── manipulation/           #   机械臂操作 (抓取/插拔/叠放)
│   ├── mobile_manipulation/    #   移动操作 (导航+操作)
│   └── humanoid/               #   人形机器人 (全身控制)
│
├── ros/                        # ── ROS2 功能节点 (按功能域散放) ──
│   ├── bringup/                #   启动 launch / 参数
│   ├── perception/             #   感知节点
│   ├── control/                #   控制节点
│   └── sim/                    #   仿真桥接
│
├── data/                       # 全局数据集 (标准化 + 原始缓存, gitignore)
├── tools/                      # 统一工具: lint.sh / format.sh / build_all.sh
└── LICENSE
```

---

## 🚀 各语言快速开始

### 1. Python（深度学习框架）

```bash
cd libs/python/framework
pip install -e .                 # 开发模式安装 (推荐 conda env py313: torch+cu130)
pytest tests -v                  # 运行测试
python scripts/train.py --config config/experiment/bc_manipulation.yaml
```

详见 [`libs/python/framework/README.md`](libs/python/framework/README.md)。

### 2. C++ / CUDA

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 总构建 (libs/cpp + libs/cuda)
cmake --build build -j$(nproc)
ctest --test-dir build                           # 运行 C++/CUDA 测试
```

- C++ 库: `libs/cpp`（含 `embodied/geometry.hpp` 示例）
- CUDA 库: `libs/cuda`（含 `embodied/cuda/vector_add.cu` 示例，按本机 GPU 架构编译）

### 3. ROS2

```bash
source /opt/ros/jazzy/setup.bash
# 同时发现共享接口(libs/ros) 与功能节点(ros/)
colcon build --base-paths libs/ros ros --symlink-install \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.bash
ros2 run <package> <node>
```

也可在 VS Code 中运行任务 **`ROS2: colcon build (Release)`**（详见 `ros/README.md`）。

---

## 🛠 统一工具 (tools/)

| 脚本 | 说明 |
|------|------|
| `tools/build_all.sh` | 一键构建全部语言 (pip + cmake + colcon) |
| `tools/lint.sh`      | 统一检查 (flake8 + clang-tidy 可选) |
| `tools/format.sh`    | 统一格式化 (black + clang-format) |

---

## 💻 VS Code 配置 (.vscode/)

已按本机环境配置（ROS2 Jazzy / CUDA 13 / conda py313 + torch+cu130）：

- **Python**: 解释器 `py313`，Pylance 含 ROS 消息库补全
- **C/C++/CUDA**: 含 `/opt/ros/jazzy/include`、`/usr/local/cuda/include`，优先使用 colcon 的 `compile_commands.json`
- **调试**: Python (debugpy)、ROS2 节点附加、C++ (gdb)
- **任务**: colcon build / pytest / 训练 / CUDA 环境检查

---

## 📚 文档

- 框架使用教程: [`libs/python/framework/docs/usage_tutorial.md`](libs/python/framework/docs/usage_tutorial.md)
- 架构设计: [`设计架构.md`](设计架构.md)
- MuJoCo 教程: [`libs/python/framework/tutorials/mujoco/README.md`](libs/python/framework/tutorials/mujoco/README.md)

## 📄 许可证

MIT
