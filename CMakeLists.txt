cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(pwc_net CUDA CXX)

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

set(CAFFE_PATH /root/flownet2)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CAFFE_PATH}/include
  ${CAFFE_PATH}/build/src
)

add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}/pwc_net_nodelet.cu
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CAFFE_PATH}/build/lib/libcaffe.so
)

add_executable(pwc_net_node src/pwc_net_node.cpp)
target_link_libraries(pwc_net_node ${catkin_LIBRARIES})
