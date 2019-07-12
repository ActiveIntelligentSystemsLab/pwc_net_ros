#ifndef PWC_NET_NODELET_H_
#define PWC_NET_NODELET_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <optical_flow_msgs/DenseOpticalFlow.h>
#include <sensor_msgs/Image.h>

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
   * @brief Name of MemoryDataLayer for previous image
   */
  const std::string INPUT_LAYER_PREVIOUS_ = "Image1";
  /**
   * @brief Name of MemoryDataLayer for previous image
   */
  const std::string INPUT_LAYER_CURRENT_ = "Image2";
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

  cv::Mat previous_image_;

  /**
   * @brief Blob of InputLayer for previous image
   */
  std::shared_ptr<caffe::Blob<d_type_>> input_blob_previous_;
  /**
   * @brief Blob of InputLayer for current image
   */
  std::shared_ptr<caffe::Blob<d_type_>> input_blob_current_;
  /**
   * @brief Channels of previous input image which is associated to blob
   */
  std::vector<cv::Mat> input_channels_previous_;
  /**
   * @brief Channels of current input image which is associated to blob
   */
  std::vector<cv::Mat> input_channels_current_;

  void onInit();

  /**
   * @brief Convert cv::Mat to input layer of network
   * 
   * @param current_image 
   */
  void convertImagesToNetworkInput(const cv::Mat& current_image);

  /**
   * @brief Called when new image is recieved
   * 
   * @param image_msg Recieved ROS message
   */
  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg);

  /**
   * @brief Initialize network model
   */
  void initializeNetwork();

  /**
   * @brief Associate channels to blob's memory area
   * 
   * @param blob     Blob
   * @param channels It's cotent will be cleared
   */
  void setChannelsToBlob(caffe::Blob<d_type_>* blob, std::vector<cv::Mat>* channels);

  /**
   * @brief Set the Blob To InputLayer
   * 
   * @param blob        A blob set to InputLayer as top
   * @param input_layer Target InputLayer
   */
  void setBlobToInputLayer(caffe::Blob<d_type_>* blob, caffe::Layer<d_type_>* input_layer);
};

} // end of pwc_net namespace

#endif
