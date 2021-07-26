#include "line_event_tracker_controller/controller.hpp"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "line_event_tracker_controller");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");


  std::cout << "starting line_event_tracker_controller_node" << std::endl;

  line_event_tracker_controller::Controller controller(nh, pnh);

  ros::spin();

  return 0;
}
