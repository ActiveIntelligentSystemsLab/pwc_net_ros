#include <pluginlib/class_list_macros.h>

#include "pwc_net_nodelet.h"

PLUGINLIB_EXPORT_CLASS(pwc_net::PWCNetNodelet, nodelet::Nodelet)

#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <ros/time.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

namespace pwc_net{

void PWCNetNodelet::onInit() {
  ros::NodeHandle& node_handle = getNodeHandle();
  ros::NodeHandle& private_node_handle = getPrivateNodeHandle();

  scale_ratio_ = private_node_handle.param("scale_ratio", 1.0);

  // Initialize network if image_width and image_height params are available
  int image_width, image_height;
  bool width_param = private_node_handle.getParam("image_width", image_width);
  bool height_param = private_node_handle.getParam("image_height", image_height);
  if (width_param && height_param)
    initializeNetwork(image_width, image_height);

  std::string image_topic = node_handle.resolveName("image");
  image_transport_.reset(new image_transport::ImageTransport(node_handle));
  image_subscriber_ = image_transport_->subscribe(image_topic, 1, &PWCNetNodelet::imageCallback, this);
  
  flow_publisher_ = private_node_handle.advertise<optical_flow_msgs::DenseOpticalFlow>("optical_flow", 1);

  flow_service_server_ = private_node_handle.advertiseService("calculate_dense_optical_flow", &PWCNetNodelet::serviceCallback, this);
}

std::string PWCNetNodelet::generateTemporaryModelFile(const std::string &package_path) {
  std::string model_file_template = package_path + "/model/pwc_net_test.prototxt";
  NODELET_INFO_STREAM("Loading template of model file: " << model_file_template);

  std::ifstream template_ifstream(model_file_template);
  if (!template_ifstream.is_open()) {
    NODELET_FATAL_STREAM("Cannot open template file of model: " << model_file_template);
    ros::shutdown();
    exit(EXIT_FAILURE);
  }

  std::string temporary_model_file = package_path + "/model/tmp/tmp_model_file.prototxt";
  NODELET_INFO_STREAM("Generate temporary model file: " << temporary_model_file);

  std::ofstream temporary_ofstream(temporary_model_file);
  if (!temporary_ofstream.is_open()) {
    NODELET_FATAL_STREAM("Cannot open temporary model file: " << temporary_model_file);
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

void PWCNetNodelet::imageCallback(const sensor_msgs::ImageConstPtr& image_msg) {
  ros::WallTime process_start = ros::WallTime::now();

  cv::Mat current_image;
  try {
    current_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
  }
  catch(const cv_bridge::Exception& exception) {
    NODELET_ERROR_STREAM(exception.what());
    return;
  }

  if (!net_)
    initializeNetwork(current_image.cols, current_image.rows);

  if (current_image.cols != target_width_ || current_image.rows != target_height_) {
    NODELET_ERROR_STREAM("Size of current image is not same to first input image which is used to initialize network.\n" << 
      "first: " << target_width_ << "x" << target_height_ << "\n" <<
      "current: " << current_image.cols << "x" << current_image.rows);
    return;
  }

  // Convert image to float for input layer
  current_image.convertTo(current_image, CV_32FC3);

  if (!previous_image_.empty()) {
    setImagesToInputLayer(previous_image_, current_image);

    ros::WallTime inference_start = ros::WallTime::now();
    net_->Forward();
    ros::WallDuration inference_time = ros::WallTime::now() - inference_start;

    publishOpticalFlow(image_msg->header);
    ros::WallDuration process_time = ros::WallTime::now() - process_start;

    NODELET_INFO_STREAM("Total process time: " << process_time.toSec() << " [s] (inference time: " << inference_time.toSec() << " [s])");
  }
  
  current_image.copyTo(previous_image_);
  previous_stamp_ = image_msg->header.stamp;
}

void PWCNetNodelet::initializeNetwork(int image_width, int image_height) {
  NODELET_INFO_STREAM("Start network initialization\n"
    << "input image size: " << image_width << "x" << image_height);

  if (image_width <= 0 || image_height <= 0)
  {
    NODELET_FATAL_STREAM("Invalid size is specified to network initialization!\n" 
      << "Specified value: " << image_width << "x" << image_height);
      ros::shutdown();
      std::exit(EXIT_FAILURE);
  }

  target_width_ = image_width;
  target_height_ = image_height;
  adapted_width_ = static_cast<int>(std::ceil(target_width_ / RESOLUTION_DIVISOR_ * scale_ratio_) * RESOLUTION_DIVISOR_);
  adapted_height_ = static_cast<int>(std::ceil(target_height_ / RESOLUTION_DIVISOR_ * scale_ratio_) * RESOLUTION_DIVISOR_);

  std::string package_path = ros::package::getPath(PACKAGE_NAME_);
  if (package_path.empty()) {
    NODELET_FATAL_STREAM("Package not found: " << PACKAGE_NAME_);
    ros::shutdown();
    std::exit(EXIT_FAILURE);
  }

  std::string temporary_model_file = generateTemporaryModelFile(package_path);

  NODELET_INFO("Loading temporary model file");
  net_.reset(new caffe::Net<d_type_>(temporary_model_file, caffe::TEST));

  std::string trained_file = package_path + "/model/pwc_net.caffemodel";
  NODELET_INFO_STREAM("Loading trained file: " << trained_file);
  net_->CopyTrainedLayersFrom(trained_file);

  NODELET_INFO_STREAM("Network initialization is finished");
}

void PWCNetNodelet::outputLayerToFlowMsg(const std::string& frame_id, const ros::Time& newer_stamp, const ros::Time& older_stamp, optical_flow_msgs::DenseOpticalFlow* flow_msg)
{
  const boost::shared_ptr<caffe::Blob<d_type_>> output_blob = net_->blob_by_name(OUTPUT_BLOB_);

  flow_msg->header.frame_id = frame_id;
  flow_msg->header.stamp = newer_stamp;
  flow_msg->previous_stamp = older_stamp;

  flow_msg->width = output_blob->shape(3);
  flow_msg->height = output_blob->shape(2);

  size_t flow_num = flow_msg->width * flow_msg->height;
  flow_msg->invalid_map.resize(flow_num, false);
  flow_msg->flow_field.resize(flow_num);

  const float *flow_x = output_blob->cpu_data();
  const float *flow_y = flow_x + flow_num;

  for (int i = 0; i < flow_num; i++) {
    optical_flow_msgs::PixelDisplacement& flow_at_point = flow_msg->flow_field[i];
    flow_at_point.x = flow_x[i];
    flow_at_point.y = flow_y[i];
  }
}

void PWCNetNodelet::publishOpticalFlow(const std_msgs::Header& current_image_header) {
  const boost::shared_ptr<caffe::Blob<d_type_>> output_blob = net_->blob_by_name(OUTPUT_BLOB_);

  optical_flow_msgs::DenseOpticalFlow flow_msg;
  outputLayerToFlowMsg(current_image_header.frame_id, current_image_header.stamp, previous_stamp_, &flow_msg);

  flow_publisher_.publish(flow_msg);
}

bool PWCNetNodelet::serviceCallback(optical_flow_srvs::CalculateDenseOpticalFlow::Request& request, optical_flow_srvs::CalculateDenseOpticalFlow::Response& response)
{
  ros::WallTime process_start = ros::WallTime::now();
  NODELET_INFO("CalculateDenseOpticalFlow service is called.");

  cv::Mat older_image, newer_image;
  try {
    older_image = cv_bridge::toCvCopy(request.older_image, "bgr8")->image;
    newer_image = cv_bridge::toCvCopy(request.newer_image, "bgr8")->image;
  }
  catch(const cv_bridge::Exception& exception) {
    NODELET_ERROR_STREAM(exception.what());
    return false;
  }

  if (older_image.cols != newer_image.cols || older_image.rows != newer_image.rows)
  {
    NODELET_ERROR_STREAM("Two images in request is not same size.\n"
      << "older_image: " << older_image.cols << "x" << older_image.rows << "\n"
      << "newer_image: " << newer_image.cols << "x" << newer_image.rows);
    return false;
  }

  if (!net_ || newer_image.cols != target_width_ || newer_image.rows != target_height_)
    initializeNetwork(newer_image.cols, newer_image.rows);

  // Convert image to float for input layer
  older_image.convertTo(older_image, CV_32FC3);
  newer_image.convertTo(newer_image, CV_32FC3);

  setImagesToInputLayer(older_image, newer_image);

  ros::WallTime inference_start = ros::WallTime::now();
  net_->Forward();
  ros::WallDuration inference_time = ros::WallTime::now() - inference_start;

  std::string& frame_id = request.newer_image.header.frame_id;
  ros::Time& newer_stamp = request.newer_image.header.stamp;
  ros::Time& older_stamp = request.older_image.header.stamp;
  outputLayerToFlowMsg(frame_id, newer_stamp, older_stamp, &response.optical_flow);

  ros::WallDuration process_time = ros::WallTime::now() - process_start;
  NODELET_INFO_STREAM("Service process time: " << process_time.toSec() << " [s] (inference time: " << inference_time.toSec() << " [s])");

  return true;
}

void PWCNetNodelet::setImagesToInputLayer(const cv::Mat& older_image, const cv::Mat& newer_image) {
  std::vector<cv::Mat> channels;
  size_t channel_size = older_image.cols * older_image.rows;

  // Set older image
  cv::split(older_image, channels); // Split to BGR channels
  d_type_ *input_layer_blob = net_->blob_by_name(INPUT_BLOB_OLDER_)->mutable_cpu_data();
  // Store each channels to blob
  for (int i = 0; i < 3; i++)
    memcpy(input_layer_blob + (channel_size * i), channels[i].ptr<d_type_>(), channel_size * sizeof(float));

  // Set current image
  cv::split(newer_image, channels);
  input_layer_blob = net_->blob_by_name(INPUT_BLOB_NEWER_)->mutable_cpu_data();
  for (int i = 0; i < 3; i++)
    memcpy(input_layer_blob + (channel_size * i), channels[i].ptr<d_type_>(), channel_size * sizeof(float));

  caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

}
