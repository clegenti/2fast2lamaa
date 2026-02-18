cmake_minimum_required(VERSION 3.8)
project(ffastllamaa)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(external_lib)

# Add C++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mbmi2")

# Set release mode
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Ceres 2.2 REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)


find_package(rosidl_default_generators REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/QueryDistField.srv"
  "msg/SubmapInfo.msg"
  DEPENDENCIES std_msgs
)


add_library(lice STATIC
  src/lib/lidar_odometry.cpp
  src/lib/state.cpp
  src/lib/map_distance_field.cpp
  src/lib/submap_bundle_adjustment.cpp
  src/lib/submap_manager.cpp
)

target_link_libraries(lice
  PUBLIC
    Eigen3::Eigen
    Ceres::ceres
)

target_include_directories(lice
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external_lib>
    $<INSTALL_INTERFACE:include>
)

add_executable(lidar_scan_odometry src/lidar_odometry_node.cpp)
add_executable(gp_map src/gp_map_node.cpp)

add_executable(map_cleaner src/map_cleaning.cpp)
add_executable(offline_loop_closure src/offline_loop_closure.cpp)

add_executable(lidar_2d_conversion src/2d_lidar_conversion_node.cpp)

add_executable(pose_graph src/pose_graph_node.cpp)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(lidar_scan_odometry
  lice
)
target_link_libraries(gp_map
  "${cpp_typesupport_target}"
  lice
)

target_link_libraries(pose_graph
  "${cpp_typesupport_target}"
  lice
  ${OpenCV_LIBS}
)

target_link_libraries(map_cleaner
  Eigen3::Eigen
  Ceres::ceres
  lice
)
target_link_libraries(offline_loop_closure
  Eigen3::Eigen
  Ceres::ceres
  lice
)
target_link_libraries(lidar_2d_conversion
  Eigen3::Eigen
)


ament_export_dependencies(rosidl_default_runtime)


ament_target_dependencies(lidar_scan_odometry
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_filters
  std_msgs
  tf2
  tf2_ros
  )
  
ament_target_dependencies(gp_map
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  std_msgs
  tf2
  tf2_ros
  )

ament_target_dependencies(pose_graph
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  std_msgs
  tf2
  tf2_ros
  OpenCV
  )


ament_target_dependencies(map_cleaner
  Eigen3
  Ceres)

ament_target_dependencies(offline_loop_closure
  Eigen3
  Ceres)

ament_target_dependencies(lidar_2d_conversion
  rclcpp
  sensor_msgs
  Eigen3
) 

include_directories(
  include
  external_lib
)

# Enable openMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

install(
    DIRECTORY include/ external_lib/
    DESTINATION include
    )

install(TARGETS
  lice
  lidar_scan_odometry gp_map map_cleaner offline_loop_closure lidar_2d_conversion pose_graph
  DESTINATION lib/${PROJECT_NAME})





install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY cfg DESTINATION share/${PROJECT_NAME}/)


ament_package()
