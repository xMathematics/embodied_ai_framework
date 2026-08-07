cmake_minimum_required(VERSION 3.8)
project(embodied_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 在此列出所有共享接口文件 (msg / srv / action)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GripperCommand.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
