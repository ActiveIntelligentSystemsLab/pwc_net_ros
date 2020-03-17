#include "sample_node.h"

#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <ros/time.h>
#include <sensor_msgs/image_encodings.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

namespace pwc_net{

SampleNode::SampleNode() {
  ros::NodeHandle node_handle;
  ros::NodeHandle private_node_handle("~");

  std::string image_topic = node_handle.resolveName("image");
  image_transport_.reset(new image_transport::ImageTransport(node_handle));
  image_sub_ = image_transport_->subscribe(image_topic, 1, &SampleNode::imageCallback, this);
  
  flow_pub_ = private_node_handle.advertise<sensor_msgs::Image>("optical_flow", 1);
}

void SampleNode::imageCallback(const sensor_msgs::ImageConstPtr& image) {
  if (previous_image_)
  {
    cv_bridge::CvImage optical_flow(image->header, sensor_msgs::image_encodings::TYPE_32FC2);
    bool success = pwc_net_.estimateOpticalFlow(*previous_image_, *image, optical_flow.image);

    if (success)
      flow_pub_.publish(optical_flow.toImageMsg());
  }
  
  previous_image_ = image;
}


}
