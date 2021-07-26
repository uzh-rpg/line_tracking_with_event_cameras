#include "line_event_tracker_visualizer/visualizer.hpp"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "line_event_tracker_visualizer");

  ros::NodeHandle nh;

  std::cout << "starting line_event_tracker_visualizer_node" << std::endl;


  line_event_tracker_visualizer::Visualizer visualizer(nh);

  ros::spin();

  return 0;
}
