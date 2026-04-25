cmake_minimum_required(VERSION 3.8)
project(sixdof_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

# Install directories so ROS can find the meshes and launch files
install(
  DIRECTORY launch urdf meshes config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/gripper_joint_state_republisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()