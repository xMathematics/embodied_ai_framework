cmake_minimum_required(VERSION 3.16)
project(embodied_cpp LANGUAGES CXX)

# ── 标准 ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (可选, 提供线性代数) ───────────────────────────────────────────
find_package(Eigen3 QUIET)

# ── 共享库 ───────────────────────────────────────────────────────────────
add_library(embodied_cpp
    src/geometry.cpp
)
target_include_directories(embodied_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(Eigen3_FOUND)
    message(STATUS "embodied_cpp: 链接 Eigen3")
    target_link_libraries(embodied_cpp PUBLIC Eigen3::Eigen)
endif()

# ── 测试 ─────────────────────────────────────────────────────────────────
option(EMBODIED_CPP_BUILD_TESTS "Build C++ tests" ON)
if(EMBODIED_CPP_BUILD_TESTS)
    enable_testing()
    add_executable(test_geometry tests/test_geometry.cpp)
    target_link_libraries(test_geometry PRIVATE embodied_cpp)
    add_test(NAME test_geometry COMMAND test_geometry)
endif()
