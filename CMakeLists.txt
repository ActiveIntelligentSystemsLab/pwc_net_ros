cmake_minimum_required(VERSION 2.8.3)
project(pwc_net)

find_package(catkin REQUIRED
  cv_bridge
  image_transport
  nodelet
  optical_flow_msgs
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

add_library(pwc_net_nodelet
  src/${PROJECT_NAME}/pwc_net_nodelet.cpp
)

add_dependencies(pwc_net_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(pwc_net_nodelet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  /home/fujimoto/Desktop/flownet2/build/lib/libcaffe.so
)
