cmake_minimum_required(VERSION 3.18)
project(embodied_ai_monorepo LANGUAGES CXX CUDA)

# ═══════════════════════════════════════════════════════════════════════════
# 具身智能项目集合 (monorepo) — C++ / CUDA 总构建
# ────────────────────────────────────────────────────────────────────────────
# 构建:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
# 测试:
#   ctest --test-dir build
# ═══════════════════════════════════════════════════════════════════════════

add_subdirectory(libs/cpp)
add_subdirectory(libs/cuda)
