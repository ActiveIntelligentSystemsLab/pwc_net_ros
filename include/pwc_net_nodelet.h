#ifndef PWC_NET_NODELET_H_
#define PWC_NET_NODELET_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <optical_flow_msgs/DenseOpticalFlow.h>
#include <optical_flow_srvs/CalculateDenseOpticalFlow.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include <caffe/caffe.hpp>
#include <caffe/layers/input_layer.hpp>

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace pwc_net {

class PWCNetNodelet : public nodelet::Nodelet {
private:
  /**
   * @brief Name of package which contains this nodelet
   */
  const std::string PACKAGE_NAME_ = "pwc_net";
  /**
   * @brief Blob of input layer for older image
   */
  const std::string INPUT_BLOB_OLDER_ = "img0";
  /**
   * @brief Blob of input layer for newer image
   */
  const std::string INPUT_BLOB_NEWER_ = "img1";
  /**
   * @brief Output blob
   */
  const std::string OUTPUT_BLOB_ = "predict_flow_final";
  /**
   * @brief Used to resize input images. Network require that input image width and height are multiples of this number.
   */
  const double RESOLUTION_DIVISOR_ = 64.0;

  /**
   * @brief Size of resized image for network input layer
   * Network require that input image width and height are multiples of PWCNetNodelet::RESOLUTION_DIVISOR_
   */
  int adapted_width_;
  /**
   * @brief Size of resized image for network input layer
   * Network require that input image width and height are multiples of PWCNetNodelet::RESOLUTION_DIVISOR_
   */
  int adapted_height_;

  /**
   * @brief Original size of input image
   */
  int target_width_;
  /**
   * @brief Original size of input image
   */
  int target_height_;

  /**
   * @brief For small images, it better set the scale_ratio to be 2.0 or 3.0 so that the input has height/width around 1000
   *  This value is set by rosparam
   */
  double scale_ratio_;

  using d_type_ = float;
  std::shared_ptr<caffe::Net<d_type_>> net_;

  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_subscriber_;
  ros::Publisher flow_publisher_;

  /**
   * @brief Service server for CalculateDenseOpticalFlow service
   */
  ros::ServiceServer flow_service_server_;

  /**
   * @brief Previous frame image used as input of optical flow calculation
   */
  cv::Mat previous_image_;
  /**
   * @brief Timestamp of previous frame image
   */
  ros::Time previous_stamp_;

  void onInit();

  /**
   * @brief Generate temporary model file from template
   * Template model file is contain some variable so replace them to real number
   * 
   * @param package_path 
   */
  std::string generateTemporaryModelFile(const std::string &package_path);

  /**
   * @brief Called when new image is recieved
   * 
   * @param image_msg Recieved ROS message
   */
  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg);

  /**
   * @brief Initialize network model
   * 
   * @param image_width Width of input images
   * @param image_height Height of input images
   */
  void initializeNetwork(int image_width, int image_height);

  /**
   * @brief Convert output layer blob to optical_flow_msgs/DenseOpticalFlow
   * 
   * @param frame_id Frame id of input images
   * @param newer_stamp Timestamp of newer image
   * @param older_stamp Timestamp of older image
   * @param flow_msg Pointer to ROS msg
   */
  void outputLayerToFlowMsg(const std::string& frame_id, const ros::Time& newer_stamp, const ros::Time& older_stamp, optical_flow_msgs::DenseOpticalFlow* flow_msg);
  
  /**
   * @brief Publish optical flow msg from output of network
   * 
   * @param current_image_header Header of current image msg
   */
  void publishOpticalFlow(const std_msgs::Header& current_image_header);

  bool serviceCallback(optical_flow_srvs::CalculateDenseOpticalFlow::Request& request, optical_flow_srvs::CalculateDenseOpticalFlow::Response& response);

  /**
   * @brief Set input images To input Layer of network
   * 
   * @param older_image 
   * @param newer_image 
   */
  void setImagesToInputLayer(const cv::Mat& older_image, const cv::Mat& newer_image);
};

} // end of pwc_net namespace

#endif
