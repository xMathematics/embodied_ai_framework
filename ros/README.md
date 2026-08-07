# ROS2 功能代码 (ros/)

本目录按**功能域散放** ROS2 包（每个子目录可包含一个或多个 ament 包），
与 `libs/ros/`（共享接口）分开管理。

## 目录

```
ros/
├── bringup/      # 系统启动: launch 文件、参数、组合节点
├── perception/   # 感知: 相机、点云、检测、分割节点
├── control/      # 控制: 运动学/动力学控制器、夹爪控制
└── sim/          # 仿真桥接: 连接 MuJoCo / PyBullet / Isaac Sim
```

## 约定

- 每个子目录下放独立 ament 包（含 `package.xml` + `CMakeLists.txt`）。
- 共享消息/服务/动作请定义在 `libs/ros/embodied_interfaces`。
- 构建:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths libs/ros ros --symlink-install \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.bash
```

- VS Code 中已配置任务 `ROS2: colcon build (Release/Debug)` 与 C++ gdb 调试。
