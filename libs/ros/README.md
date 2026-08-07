# ROS2 共享接口 (libs/ros)

本目录存放**跨项目复用**的 ROS2 接口包，供 `projects/*/ros` 与 `ros/` 下的功能节点共同依赖。

## 结构

```
libs/ros/
├── embodied_interfaces/   # 共享消息 / 服务 / 动作定义
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── msg/  srv/  action/
└── (未来可增加其他共享 ROS2 包)
```

## 构建

接口包与功能节点一起用 colcon 构建（在 monorepo 根执行）：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths libs/ros ros --symlink-install \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.bash
```

> 说明: `--base-paths libs/ros ros` 让 colcon 同时发现 `libs/ros/`（共享接口）与
> `ros/`（功能节点）下的所有包。也可在 VS Code 中运行任务
> `ROS2: colcon build (Release)`。
