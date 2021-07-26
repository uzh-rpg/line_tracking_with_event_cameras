#include <ros/ros.h>
#include <iostream>

#include "line_event_tracker/tracker.hpp"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "line_event_tracker");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::cout << "starting line_event_tracker_node" << std::endl;

  line_event_tracker::Tracker tracker(nh, pnh);

  ros::spin();

  return 0;

}
