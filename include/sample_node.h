#ifndef PWC_NET__SAMPLE_NODE_H_
#define PWC_NET__SAMPLE_NODE_H_

#include "pwc_net/pwc_net.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

namespace pwc_net {

class SampleNode {
private:
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_sub_;
  ros::Publisher flow_pub_;
  ros::Publisher visualized_flow_pub_;

  PwcNet pwc_net_;

  sensor_msgs::ImageConstPtr previous_image_;

  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg);
public:
  SampleNode();
};

} // namespace pwc_net

#endif // PWC_NET__SAMPLE_NODE_H_

