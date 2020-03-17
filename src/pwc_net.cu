#include "pwc_net/pwc_net.h"

#include <cv_bridge/cv_bridge.h>
#include <ros/console.h>
#include <ros/package.h>

#include <caffe/caffe.hpp>
#include <caffe/layers/input_layer.hpp>

#include <cmath>
#include <fstream>
#include <map>

namespace pwc_net{

// Put here to include pwc_net.h from non-CUDA project
std::shared_ptr<caffe::Net<float>> net_;

std::string PwcNet::generateTemporaryModelFile(const std::string &package_path) {
  std::string model_file_template = package_path + "/model/pwc_net_test.prototxt";
  ROS_INFO_STREAM_NAMED("libpwc_net", "Loading template of model file: " << model_file_template);

  std::ifstream template_ifstream(model_file_template);
  if (!template_ifstream.is_open()) {
    ROS_FATAL_STREAM_NAMED("libpwc_net", "Cannot open template file of model: " << model_file_template);
    ros::shutdown();
    exit(EXIT_FAILURE);
  }

  std::string temporary_model_file = package_path + "/model/tmp_model_file.prototxt";
  ROS_INFO_STREAM_NAMED("libpwc_net", "Generate temporary model file: " << temporary_model_file);

  std::ofstream temporary_ofstream(temporary_model_file);
  if (!temporary_ofstream.is_open()) {
    ROS_FATAL_STREAM_NAMED("libpwc_net", "Cannot open temporary model file: " << temporary_model_file);
    ros::shutdown();
    exit(EXIT_FAILURE);
  }

  std::map<std::string, std::string> replacement_map;
  replacement_map.emplace("$ADAPTED_WIDTH", std::to_string(adapted_width_));
  replacement_map.emplace("$ADAPTED_HEIGHT", std::to_string(adapted_height_));
  replacement_map.emplace("$TARGET_WIDTH", std::to_string(target_width_));
  replacement_map.emplace("$TARGET_HEIGHT", std::to_string(target_height_));
  replacement_map.emplace("$SCALE_WIDTH", std::to_string(1.0 * target_width_ / adapted_width_));
  replacement_map.emplace("$SCALE_HEIGHT", std::to_string(1.0 * target_height_ / adapted_height_));

  for (std::string line; std::getline(template_ifstream, line);) {
    for (auto iterator = replacement_map.begin(); iterator != replacement_map.end(); iterator++) {
      size_t replace_start;
      std::string replaced_text = iterator->first;
      std::string new_text = iterator->second;

      replace_start = line.find(replaced_text);
      if (replace_start != std::string::npos)
        line.replace(replace_start, replaced_text.length(), new_text);
    }

    line += "\n";
    temporary_ofstream.write(line.c_str(), static_cast<long>(line.length()));
  }
  template_ifstream.close();
  temporary_ofstream.close();

  return temporary_model_file;
}

bool PwcNet::estimateOpticalFlow
(
  const sensor_msgs::Image& source_image_msg, 
  const sensor_msgs::Image& dist_image_msg,
  cv::Mat& optical_flow
) 
{
  // Are input images same size?
  if (source_image_msg.width != dist_image_msg.width
    || source_image_msg.height != dist_image_msg.height)
  {
    ROS_ERROR_STREAM_NAMED("libpwc_net",
      "Input images aren't same size!\n" <<
      "source: " << source_image_msg.width << "x" << source_image_msg.height << "\n" <<
      "dist: " << dist_image_msg.width << "x" << dist_image_msg.height
    );

    return false;
  }

  // Initialize network if not
  if (!net_)
    initializeNetwork(dist_image_msg.width, dist_image_msg.height);
  else if (dist_image_msg.width != target_width_ || dist_image_msg.height != target_height_)
  {
    ROS_INFO_STREAM_NAMED("libpwc_net", 
      "Size of input image is not same to first input image which is used to initialize network.\n" << 
      "Reinitialize network for new size\n" << 
      "old: " << target_width_ << "x" << target_height_ << "\n" <<
      "new: " << dist_image_msg.width << "x" << dist_image_msg.height
    );
    initializeNetwork(dist_image_msg.width, dist_image_msg.height);
  }

  // Convert msg to cv::Mat
  cv::Mat dist_image;
  cv::Mat source_image;
  try 
  {
    dist_image = cv_bridge::toCvCopy(dist_image_msg, "bgr8")->image;
    source_image = cv_bridge::toCvCopy(source_image_msg, "bgr8")->image;
  }
  catch(const cv_bridge::Exception& exception) 
  {
    ROS_ERROR_STREAM_NAMED("libpwc_net", exception.what());
    return false;
  }

  // Convert cv::Mat to float and set to input layer
  source_image.convertTo(source_image, CV_32FC3);
  dist_image.convertTo(dist_image, CV_32FC3);
  setImagesToInputLayer(source_image, dist_image);

  net_->Forward();

  outputLayerToCvMat(optical_flow);

  return true;
}

void PwcNet::initializeNetwork(int image_width, int image_height) {
  ROS_INFO_STREAM_NAMED("libpwc_net", "Start network initialization\n"
    << "input image size: " << image_width << "x" << image_height);

  if (image_width <= 0 || image_height <= 0)
  {
    ROS_FATAL_STREAM_NAMED("libpwc_net", "Invalid size is specified to network initialization!\n" 
      << "Specified value: " << image_width << "x" << image_height);
      ros::shutdown();
      std::exit(EXIT_FAILURE);
  }

  target_width_ = image_width;
  target_height_ = image_height;
  adapted_width_ = static_cast<int>(std::ceil(target_width_ / RESOLUTION_DIVISOR_) * RESOLUTION_DIVISOR_);
  adapted_height_ = static_cast<int>(std::ceil(target_height_ / RESOLUTION_DIVISOR_) * RESOLUTION_DIVISOR_);

  std::string package_path = ros::package::getPath(PACKAGE_NAME_);
  if (package_path.empty()) {
    ROS_FATAL_STREAM_NAMED("libpwc_net", "Package not found: " << PACKAGE_NAME_);
    ros::shutdown();
    std::exit(EXIT_FAILURE);
  }

  std::string temporary_model_file = generateTemporaryModelFile(package_path);

  ROS_INFO_NAMED("libpwc_net", "Loading temporary model file");
  net_.reset(new caffe::Net<float>(temporary_model_file, caffe::TEST));

  std::string trained_file = package_path + "/model/pwc_net.caffemodel";
  ROS_INFO_STREAM_NAMED("libpwc_net", "Loading trained file: " << trained_file);
  net_->CopyTrainedLayersFrom(trained_file);

  ROS_INFO_STREAM_NAMED("libpwc_net", "Network initialization is finished");
}

void PwcNet::outputLayerToCvMat(cv::Mat& optical_flow)
{
  const boost::shared_ptr<caffe::Blob<float>> output_blob = net_->blob_by_name(OUTPUT_BLOB_);

  cv::Mat channels[2];

  int height = output_blob->shape(2);
  int width = output_blob->shape(3);

  channels[0].create(cv::Size(width, height), CV_32FC1);
  channels[1].create(cv::Size(width, height), CV_32FC1);

  int total_pixel = height * width;
  const float* x_channel = output_blob->cpu_data();
  const float* y_channel = x_channel + total_pixel;
  size_t channel_size = sizeof(float) * total_pixel;
  std::memcpy(channels[0].data, x_channel, channel_size);
  std::memcpy(channels[1].data, y_channel, channel_size);

  cv::merge(channels, 2, optical_flow);
}

void PwcNet::setImagesToInputLayer(const cv::Mat& source_image, const cv::Mat& dist_image) {
  std::vector<cv::Mat> channels;
  size_t channel_size = source_image.cols * source_image.rows;

  // Set source image
  cv::split(source_image, channels); // Split to BGR channels
  float *input_layer_blob = net_->blob_by_name(SOURCE_IMAGE_BLOB_)->mutable_cpu_data();
  // Store each channels to blob
  for (int i = 0; i < 3; i++)
    memcpy(input_layer_blob + (channel_size * i), channels[i].ptr<float>(), channel_size * sizeof(float));

  // Set dist image
  cv::split(dist_image, channels);
  input_layer_blob = net_->blob_by_name(DIST_IMAGE_BLOB_)->mutable_cpu_data();
  for (int i = 0; i < 3; i++)
    memcpy(input_layer_blob + (channel_size * i), channels[i].ptr<float>(), channel_size * sizeof(float));

  caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

void PwcNet::visualizeOpticalFlow
(
  const cv::Mat& optical_flow,
  cv::Mat& visualized_optical_flow,
  float max_magnitude
) 
{
  cv::Mat hsv_image(optical_flow.rows, optical_flow.cols, CV_8UC3, cv::Vec3b(0, 0, 0));

  int total_pixels = optical_flow.total();
  for (int i = 0; i < total_pixels; i++)
  {
    const cv::Vec2f& flow_at_point = optical_flow.at<cv::Vec2f>(i);

    float flow_magnitude = 
      std::sqrt(flow_at_point[0]*flow_at_point[0] + flow_at_point[1]*flow_at_point[1]);
    float flow_direction = std::atan2(flow_at_point[0], flow_at_point[1]);

    uchar hue = (flow_direction / M_PI + 1.0) / 2.0 * 255;
    uchar saturation = std::min(std::max(flow_magnitude / max_magnitude, 0.0f), 1.0f) * 255;
    uchar value = 255;

    cv::Vec3b &hsv = hsv_image.at<cv::Vec3b>(i);
    hsv[0] = hue;
    hsv[1] = saturation;
    hsv[2] = value;
  }

  cv::cvtColor(hsv_image, visualized_optical_flow, cv::ColorConversionCodes::COLOR_HSV2BGR_FULL);
}

}
