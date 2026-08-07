# libs/ — 跨项目共享库

按语言划分的共享代码，供 `projects/*` 与 `ros/` 复用。

```
libs/
├── python/   # Python 共享库
│   └── framework/   # 具身智能全流程框架 (五层四流, 现有框架整体迁入)
├── cpp/      # C++ 共享库 (embodied_cpp: 几何/线性代数等)
├── cuda/     # CUDA 共享算子库 (embodied_cuda)
└── ros/      # ROS2 共享接口 (embodied_interfaces: msg/srv/action)
```

## 约定
- 只放可复用的基础能力，不放具体业务。
- 各语言库独立可构建/测试:
  - Python: `pip install -e libs/python/framework`
  - C++/CUDA: 见根 `CMakeLists.txt` (`cmake -S . -B build`)
  - ROS2: 见 `libs/ros/README.md`
