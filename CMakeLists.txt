cmake_minimum_required(VERSION 2.8.3)
project(pwc_net)

find_package(catkin REQUIRED
  cv_bridge
  image_transport
  nodelet
  optical_flow_msgs
  optical_flow_srvs
  pluginlib
  roscpp
  sensor_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /home/fujimoto/Desktop/flownet2/include
  /usr/local/cuda-10.0/targets/x86_64-linux/include
)

add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}/pwc_net_nodelet.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  /home/fujimoto/Desktop/flownet2/build/lib/libcaffe.so
)

add_executable(pwc_net_node src/pwc_net_node.cpp)
target_link_libraries(pwc_net_node ${catkin_LIBRARIES})