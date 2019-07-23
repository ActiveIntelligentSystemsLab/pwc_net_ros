#include <ros/ros.h>
#include <nodelet/loader.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pwc_net");

  nodelet::Loader nodelet_loader(false);

  nodelet::M_string remappings;
  nodelet::V_string args(argv + 1, argv + argc);

  nodelet_loader.load(ros::this_node::getName(), "pwc_net/pwc_net", remappings, args);

  ros::spin();
  return 0;
}