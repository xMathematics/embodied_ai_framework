cmake_minimum_required(VERSION 3.18)

# 按当前 GPU 架构编译 (RTX 5060 Ti → Blackwell sm_120)
set(CMAKE_CUDA_ARCHITECTURES native)

project(embodied_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── 共享 CUDA 算子库 ─────────────────────────────────────────────────────
add_library(embodied_cuda
    kernels/vector_add.cu
)
target_include_directories(embodied_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── 测试 ─────────────────────────────────────────────────────────────────
option(EMBODIED_CUDA_BUILD_TESTS "Build CUDA tests" ON)
if(EMBODIED_CUDA_BUILD_TESTS)
    enable_testing()
    add_executable(test_vector_add tests/test_vector_add.cu)
    target_link_libraries(test_vector_add PRIVATE embodied_cuda)
    add_test(NAME test_vector_add COMMAND test_vector_add)
endif()
