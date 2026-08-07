# Python 共享库 (libs/python)

## framework — 具身智能全流程框架

原项目框架整体迁入，保持自包含（含 `src/ scripts/ tests/ config/ docs/
tutorials/`）。

```
libs/python/framework/
├── src/           # 框架源码 (五层四流)
├── scripts/       # 训练/评估/数据准备脚本
├── tests/         # pytest 测试
├── config/        # 数据集注册表 + 实验配置
├── docs/          # 使用教程/变更日志
├── tutorials/     # 教程代码
├── setup.py
├── requirements.txt
└── pytest.ini
```

### 使用

```bash
# 安装 (开发模式, 解释器建议 conda py313: torch+cu130)
pip install -e libs/python/framework

# 运行测试
pytest libs/python/framework/tests -v

# 训练
python libs/python/framework/scripts/train.py
```

> 数据集统一放在 monorepo 根 `data/`，框架内路径以 `../../data/...` 引用。
