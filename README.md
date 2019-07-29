# PWC-Net ROS

ROS package for estimation of optical flow by [PWC-Net](https://github.com/NVlabs/PWC-Net).

This uses [model definition](https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/pwc_net_test.prototxt) and [trained model](https://github.com/NVlabs/PWC-Net/blob/master/Caffe/model/pwc_net.caffemodel) from [official implementation by Caffe](https://github.com/NVlabs/PWC-Net/tree/master/Caffe).

## Requirements

* ROS Melodic
* CUDA 10.1

## pwc_net_node (Node)

A node estimates dense optical flow from image topic.

### Subscribed topic

* `image` ([sensor_msgs/Image](http://docs.ros.org/api/sensor_msgs/html/msg/Image.html))

  Input image should be remapped. Optical flow is estimated between latest image and it's previous image.

### Published topic

* `optical_flow` ([optical_flow_msgs/DenseOpticalFlow](https://github.com/ActiveIntelligentSystemsLab/optical_flow_msgs/blob/master/msg/DenseOpticalFlow.msg))

  Estimated optical flow.

### Provided services

* `~calculate_dense_optical_flow` ([optical_flow_srvs/CalculateDenseOpticalFlow](https://github.com/ActiveIntelligentSystemsLab/ros_optical_flow/blob/master/optical_flow_srvs/srv/CalculateDenseOpticalFlow.srv))

  Return dense optical flow between input images in request.

### Parameters

* `~image_transport` (string, default: "raw")

  Transport used for the image stream. See [image_transport](http://wiki.ros.org/image_transport).

* `~scale_ratio` (double, default: 1.0)

  For small images, it better set the scale_ratio to be 2.0 or 3.0 so that the input has height/width around 1000.

## pwc_net/pwc_net (Nodelet)

Nodelet version of [pwc_net_node](#pwc_net_node-(Node)).
Parameters and topics are same to it.