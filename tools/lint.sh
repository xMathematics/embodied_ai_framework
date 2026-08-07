#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 统一代码检查: Python (flake8) + C/C++/CUDA (clang-tidy, 可选)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3 || true)"

echo "[lint] Python (flake8)"
if command -v flake8 >/dev/null 2>&1; then
  (cd "$ROOT/libs/python/framework" && flake8 src scripts tests)
else
  echo "  (flake8 未安装, 跳过; 可运行: pip install flake8)"
fi

echo "[lint] C++/CUDA (clang-tidy, 可选)"
if command -v clang-tidy >/dev/null 2>&1; then
  echo "  clang-tidy 已安装; 请配合 build/compile_commands.json 使用, 例如:"
  echo "  clang-tidy -p build <file>"
else
  echo "  (clang-tidy 未安装, 跳过)"
fi

echo "[lint] 完成 ✅"
