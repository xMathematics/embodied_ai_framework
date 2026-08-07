# 具身智能项目集合 (projects/)

本目录存放**具体项目**。每个项目独立组织自己的 `python/ cpp/ ros/ cuda/`
子目录与配置，共享代码引用 `libs/`。

| 项目 | 说明 | 状态 |
|------|------|------|
| [`manipulation/`](manipulation/README.md) | 桌面/机械臂操作 (抓取、插拔、叠放) | 骨架 |
| [`mobile_manipulation/`](mobile_manipulation/README.md) | 移动操作 (导航+操作融合) | 骨架 |
| [`humanoid/`](humanoid/README.md) | 人形机器人 (全身控制、行走) | 骨架 |

## 新增项目

```bash
mkdir -p projects/<name>/{python,cpp,ros,cuda,config,scripts,tests}
# 然后复制 README 模板
```

## 项目内部结构

```
projects/<name>/
├── README.md      # 项目说明、依赖、运行方式
├── python/        # Python 代码 (算法、数据、训练)
├── cpp/           # C++ 代码 (实时控制、底层)
├── ros/           # ROS2 节点 / launch
├── cuda/          # CUDA 算子
├── config/        # 项目配置 (yaml)
├── scripts/       # 运行脚本
└── tests/         # 测试
```
