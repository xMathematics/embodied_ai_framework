#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 统一格式化: Python (black) + C/C++/CUDA (clang-format)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[format] Python (black)"
if command -v black >/dev/null 2>&1; then
  (cd "$ROOT/libs/python/framework" && black src scripts tests)
else
  echo "  (black 未安装, 跳过; 可运行: pip install black)"
fi

echo "[format] C/C++/CUDA (clang-format)"
if command -v clang-format >/dev/null 2>&1; then
  find "$ROOT/libs" "$ROOT/projects" "$ROOT/ros" \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
    -print0 | xargs -0 -r clang-format -i
else
  echo "  (clang-format 未安装, 跳过)"
fi

echo "[format] 完成 ✅"
