#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 构建所有语言: Python (pip -e) + C++/CUDA (cmake) + ROS2 (colcon)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> [1/3] Python framework (pip install -e .)"
(cd "$ROOT/libs/python/framework" && python3 -m pip install -e .)

echo "==> [2/3] C++ / CUDA (cmake build)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$(nproc)"

echo "==> [3/3] ROS2 (colcon build: libs/ros + ros/)"
source /opt/ros/jazzy/setup.bash
colcon build --base-paths "$ROOT/libs/ros" "$ROOT/ros" --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "全部构建完成 ✅"
