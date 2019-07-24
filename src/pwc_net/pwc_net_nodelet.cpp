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

  if (previous_image_.empty()) {
    current_image.copyTo(previous_image_);
    return;
  }
  
  cv::split(previous_image_, input_channels_previous_);  
  cv::split(current_image, input_channels_current_); 

  caffe::Blob<d_type_>* output_blob = net_->output_blobs()[0];
  NODELET_INFO_STREAM("shape(0): " << output_blob->shape(0));
  NODELET_INFO_STREAM("shape(1): " << output_blob->shape(1));
  NODELET_INFO_STREAM("shape(2): " << output_blob->shape(2));
  NODELET_INFO_STREAM("shape(3): " << output_blob->shape(3));
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

  NODELET_INFO("Allocating memory and set it to input layers");
  std::vector<int> input_blob_shape;
  input_blob_shape.push_back(1);
  input_blob_shape.push_back(3);
  input_blob_shape.push_back(target_width_);
  input_blob_shape.push_back(target_height_);
  input_blob_previous_.reset(new caffe::Blob<d_type_>(input_blob_shape));
  input_blob_current_.reset(new caffe::Blob<d_type_>(input_blob_shape));

  setChannelsToBlob(input_blob_previous_.get(), &input_channels_previous_);
  setChannelsToBlob(input_blob_current_.get(), &input_channels_current_);

  const boost::shared_ptr<caffe::Layer<d_type_>> input_layer_previous = net_->layer_by_name(INPUT_LAYER_PREVIOUS_);
  setBlobToInputLayer(input_blob_previous_.get(), input_layer_previous.get());

  const boost::shared_ptr<caffe::Layer<d_type_>> input_layer_current = net_->layer_by_name(INPUT_LAYER_CURRENT_);
  setBlobToInputLayer(input_blob_current_.get(), input_layer_current.get());

  NODELET_INFO_STREAM("Network initialization is finished");
}

void PWCNetNodelet::setChannelsToBlob(caffe::Blob<d_type_>* blob, std::vector<cv::Mat>* channels) {
  channels->clear();

  float * blob_data = blob->mutable_cpu_data();
  for (int i = 0; i < 3; ++i) {
    cv::Mat single_channel(target_height_, target_width_, CV_32FC1, blob_data);
    channels->push_back(single_channel);
    blob_data += target_width_ * target_height_;
  }
}

void PWCNetNodelet::setBlobToInputLayer(caffe::Blob<d_type_>* blob, caffe::Layer<d_type_>* input_layer) {
  std::vector<caffe::Blob<d_type_>*> blobs_top;
  blobs_top.push_back(blob);

  std::vector<caffe::Blob<d_type_>*> blobs_bottom; // bottom is never used in LayerSetUp

  input_layer->LayerSetUp(blobs_bottom, blobs_top);
}

}
