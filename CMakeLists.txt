cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(pwc_net CUDA CXX)

find_package(catkin REQUIRED
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs
  DEPENDS OpenCV
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# Path for flownet2 with PWC-Net's patch
set(CAFFE_PATH /root/flownet2)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CAFFE_PATH}/include
  ${CAFFE_PATH}/build/src
)

add_library(${PROJECT_NAME} src/pwc_net.cu)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CAFFE_PATH}/build/lib/libcaffe.so
)

add_executable(sample_node
  src/sample_node.cpp
  src/sample_node_main.cpp
)
add_dependencies(sample_node ${PROJECT_NAME})
target_link_libraries(sample_node ${PROJECT_NAME})

