#include "sample_node.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pwc_net_sample");
  pwc_net::SampleNode sample_node;
  ros::spin();

  return 0;
}
