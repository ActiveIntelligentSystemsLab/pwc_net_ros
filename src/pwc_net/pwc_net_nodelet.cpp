#include <pluginlib/class_list_macros.h>

#include "pwc_net_nodelet.h"

PLUGINLIB_EXPORT_CLASS(pwc_net::PWCNetNodelet, nodelet::Nodelet)

#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>

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

  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  std::string image_topic = node_handle.resolveName("image");
  image_transport_.reset(new image_transport::ImageTransport(node_handle));
  image_subscriber_ = image_transport_->subscribe(image_topic, 1, &PWCNetNodelet::imageCallback, this);
  
  flow_publisher_ = private_node_handle.advertise<optical_flow_msgs::DenseOpticalFlow>("optical_flow", 1);
}

void PWCNetNodelet::imageCallback(const sensor_msgs::ImageConstPtr& image_msg) {
  cv::Mat current_image;
  try {
    current_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
  }
  catch(const cv_bridge::Exception& exception) {
    NODELET_ERROR_STREAM(exception.what());
    return;
  }

  if (!net_) {
    NODELET_INFO("First image is received and network initialization begins using it's size");
    target_width_ = current_image.cols;
    target_height_ = current_image.rows;

    adapted_width_ = static_cast<int>(std::ceil(target_width_ / RESOLUTION_DIVISOR_ * scale_ratio_) * RESOLUTION_DIVISOR_);
    adapted_height_ = static_cast<int>(std::ceil(target_height_ / RESOLUTION_DIVISOR_ * scale_ratio_) * RESOLUTION_DIVISOR_);

    initializeNetwork();
  }

  if (current_image.cols != target_width_ || current_image.rows != target_height_) {
    NODELET_ERROR_STREAM("Size of current image is not same to first input image which is used to initialize network.\n" << 
      "first: " << target_width_ << "x" << target_height_ << "\n" <<
      "current: " << current_image.cols << "x" << current_image.rows);
    return;
  }
  current_image.convertTo(current_image, CV_32FC3);

  if (!previous_image_.empty()) {
    std::vector<cv::Mat> test;
    cv::split(previous_image_, test);

    float *dest = net_->blob_by_name("img0")->mutable_cpu_data();
    memcpy(dest, test[0].ptr<float>(), target_height_*target_width_*sizeof(float));
    dest += target_height_*target_width_;
    memcpy(dest, test[1].ptr<float>(), target_height_*target_width_*sizeof(float));
    dest += target_height_*target_width_;
    memcpy(dest, test[2].ptr<float>(), target_height_*target_width_*sizeof(float));

    cv::split(current_image, test);
    dest = net_->blob_by_name("img1")->mutable_cpu_data();
    memcpy(dest, test[0].ptr<float>(), target_height_*target_width_*sizeof(float));
    dest += target_height_*target_width_;
    memcpy(dest, test[1].ptr<float>(), target_height_*target_width_*sizeof(float));
    dest += target_height_*target_width_;
    memcpy(dest, test[2].ptr<float>(), target_height_*target_width_*sizeof(float));

    const boost::shared_ptr<caffe::Blob<d_type_>> blob = net_->blob_by_name("img0");

    net_->Forward();

    const boost::shared_ptr<caffe::Blob<d_type_>> output_blob = net_->blob_by_name("predict_flow_final");

    optical_flow_msgs::DenseOpticalFlow flow_msg;
    flow_msg.header.frame_id = image_msg->header.frame_id;
    flow_msg.header.stamp = image_msg->header.stamp;
    flow_msg.previous_stamp = previous_stamp_;

    flow_msg.width = output_blob->shape(3);
    flow_msg.height = output_blob->shape(2);

    size_t flow_num = flow_msg.width * flow_msg.height;
    flow_msg.invalid_map.resize(flow_num, false);
    flow_msg.flow_field.resize(flow_num);

    const float *flow_x = output_blob->cpu_data();
    const float *flow_y = flow_x + flow_num;

    for (int i = 0; i < flow_num; i++) {
      optical_flow_msgs::PixelDisplacement& flow_at_point = flow_msg.flow_field[i];
      flow_at_point.x = flow_x[i];
      flow_at_point.y = flow_y[i];
    }

    flow_publisher_.publish(flow_msg);
  }
  
  current_image.copyTo(previous_image_);
  previous_stamp_ = image_msg->header.stamp;
}

void PWCNetNodelet::initializeNetwork() {
  std::string package_path = ros::package::getPath(PACKAGE_NAME_);
  if (package_path.empty()) {
    NODELET_FATAL_STREAM("Package not found: " << PACKAGE_NAME_);
    ros::shutdown();
    std::exit(EXIT_FAILURE);
  }

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

  NODELET_INFO("Loading temporary model file");
  
  net_.reset(new caffe::Net<d_type_>(temporary_model_file, caffe::TEST));

  std::string trained_file = package_path + "/model/pwc_net.caffemodel";
  NODELET_INFO_STREAM("Loading trained file: " << trained_file);
  net_->CopyTrainedLayersFrom(trained_file);

  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  NODELET_INFO_STREAM("Network initialization is finished");
}

}
