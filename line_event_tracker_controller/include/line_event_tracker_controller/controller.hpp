#pragma once

#include <ros/ros.h>

#include <mutex>
#include <line_event_tracker_msgs/Lines.h>

namespace line_event_tracker_controller {

/**
 * @brief The controller class used for HW test
 * this controller tracks a spinning line by calculating
 * yaw angular velocity commands
 */
class Controller{

public:
  Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh);
  ~Controller();

private:

  /**
 * @brief ros lines callback, draws lines and their ids into the current image
 */
  void linesCallback(const line_event_tracker_msgs::LinesConstPtr & msg);

  /**
   * @brief controller update, calculated vel cmd and publishes it
   */
  void update();

private:
  // ros
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber line_sub_;
  ros::Publisher velocity_command_pub_;


  // PD controller
  double update_frequency_;
  double k_p_;
  double k_d_;
  double max_vel_cmd_;
  double e_last_ = 0.0;

  // TESTING
  double test_yaw_rate_;

  // line selection
  double theta_upper_;
  double theta_lower_;
  double line_vel_threshold_;
  double line_min_length_;

  std::mutex lines_mutex_;

  line_event_tracker_msgs::Lines lines_;
  ros::Time last_update_;
  ros::Time last_line_stamp_;

  // TESTING
  double last_vel_cmd_;

};

} // namespace