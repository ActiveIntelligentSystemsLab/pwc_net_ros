#ifndef PWC_NET__PWC_NET_H_
#define PWC_NET__PWC_NET_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

namespace pwc_net {

class PwcNet {
private:
  const std::string PACKAGE_NAME_ = "pwc_net";

  const std::string SOURCE_IMAGE_BLOB_ = "img0";
  const std::string DIST_IMAGE_BLOB_ = "img1";
  const std::string OUTPUT_BLOB_ = "predict_flow_final";

  /**
   * @brief Used to resize input images. 
   *
   * Network require that input image width and height are multiples of this number.
   */
  const double RESOLUTION_DIVISOR_ = 64.0;

  /**
   * @brief Size of resized image for network input layer
   * 
   * Network require that input image width and height are multiples of PWCNetNodelet::RESOLUTION_DIVISOR_
   */
  int adapted_width_;
  /**
   * @brief Size of resized image for network input layer
   * 
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
   * @brief Generate temporary model file from template
   * Template model file is contain some variable so replace them to real number
   * 
   * @param package_path 
   */
  std::string generateTemporaryModelFile(const std::string &package_path);

  /**
   * @brief Initialize network model
   * 
   * @param image_width Width of input images
   * @param image_height Height of input images
   */
  void initializeNetwork(int image_width, int image_height);

  /**
   * @brief Convert output layer blob to cv::Mat
   */
  void outputLayerToCvMat(cv::Mat& optical_flow);

  /**
   * @brief Set input images To input Layer of network
   * 
   * @param source_image 
   * @param dist_image 
   */
  void setImagesToInputLayer(const cv::Mat& source_image, const cv::Mat& dist_image);
public:
  /**
   * @brief Estimate optical flow from source_image to dist_image by PWC-Net
   *
   * @param optical_flow CV_32FC2. first channel is optical flow's x-axis component, second is y-axis.
   * @return false if input images are invalid, such as not same size
   */
  bool estimateOpticalFlow
  (
    const sensor_msgs::Image& source_image_msg, 
    const sensor_msgs::Image& dist_image_msg,
    cv::Mat& optical_flow
  ); 
  
  /**
   * @brief Visualize optical flow by HSV coloring
   *
   * Direction and magnitude of flow is represented as hue and saturation.
   *
   * @param optical_flow CV_32FC2. first channel is optical flow's x-axis component, second is y-axis.
   * @param visualized_optical_flow CV_8UC3, BGR image.
   * @param max_magnitude Optical flow's magnitude[pixel] mapped to maximum saturation
   */
  static void visualizeOpticalFlow
  (
    const cv::Mat& optical_flow,
    cv::Mat& visualized_optical_flow,
    float max_magnitude
  ); 
};

} // end of pwc_net namespace

#endif
