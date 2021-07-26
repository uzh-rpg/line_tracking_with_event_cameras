#include "line_event_tracker_controller/controller.hpp"

#include <thread>
//#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>

namespace line_event_tracker_controller{

Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh), pnh_(pnh)
{
  // parameters
  pnh_.param<double>("controller/update_frequency", update_frequency_,30.0);
  pnh_.param<double>("controller/k_p", k_p_,0.01);
  pnh_.param<double>("controller/k_d", k_d_,0.01);
  pnh_.param<double>("controller/max_vel_cmd", max_vel_cmd_,4.0);
  pnh_.param<double>("controller/test_yaw_rate", test_yaw_rate_,0.0);

  pnh_.param<double>("line/theta_upper", theta_upper_,0.3);
  pnh_.param<double>("line/theta_lower", theta_lower_,-0.3);
  pnh_.param<double>("line/line_vel_threshold", line_vel_threshold_,4.0);
  pnh_.param<double>("line/line_min_length", line_min_length_,40.0);

  line_sub_ = nh_.subscribe("lines", 1, &Controller::linesCallback, this);
  velocity_command_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("vel_cmd", 10);

  std::thread control_loop_thread(&Controller::update, this);
  control_loop_thread.detach();
}

Controller::~Controller()
{
  velocity_command_pub_.shutdown();
}

void Controller::linesCallback(const line_event_tracker_msgs::LinesConstPtr &msg)
{
  std::unique_lock<std::mutex> lock(lines_mutex_);
  lines_ = *msg;
  last_update_ = ros::Time::now();

}

void Controller::update()
{
  ros::Rate r(update_frequency_);

  geometry_msgs::TwistStamped vel_cmd;
  vel_cmd.header.stamp = ros::Time::now();
  vel_cmd.twist.linear.x = 0.0;
  vel_cmd.twist.linear.y = 0.0;
  vel_cmd.twist.linear.z = 0.0;

  vel_cmd.twist.angular.x = 0.0;
  vel_cmd.twist.angular.y = 0.0;
  vel_cmd.twist.angular.z = 0.0;

  last_vel_cmd_ = 0.0;

  while(ros::ok())
  {

    line_event_tracker_msgs::Line longest_line;
    double max_length = 0;

    {
      std::unique_lock<std::mutex> lock(lines_mutex_);

      // longest line
      if ((ros::Time::now() - last_update_).toSec() * 1000 < 30)
      {
        for (auto const &line : lines_.lines) {
          if (line.state == line_event_tracker_msgs::Line::INITIALIZING) continue;

          if (line.theta < theta_lower_ || line.theta > theta_upper_) continue;
          int mid_point_offset = 260 / 2 - line.pos_y;
          if (abs(mid_point_offset) > 25) continue;;

          if (line.length > max_length)
          {
            longest_line = line;
            max_length = line.length;
          }
        }
      }
    }

    if (max_length > 0)
    {

      double yaw_vel = 0.0;
      double mean_mid_point_x = longest_line.pos_x;

      // PD controller
      double e = lines_.width / 2.0 - mean_mid_point_x;
      double time_passed = (ros::Time::now() - last_update_).toSec();
      double e_dot = (e - e_last_) / time_passed;
      if (e_dot < 0) e_dot = 0;
      double p_part = k_p_ * e;
      double d_part = -k_d_ * e_dot;
      yaw_vel = p_part + d_part;

      // limit yaw rate
      if (yaw_vel > max_vel_cmd_)
      {
        yaw_vel = max_vel_cmd_;
      }
      else if (yaw_vel < 0)
      {
        yaw_vel = 0;
      }

      vel_cmd.twist.angular.z = yaw_vel;
      last_vel_cmd_ = yaw_vel;
      std::cout << "LINE DETECTED: " << longest_line.id << ", VEL CMD: " << yaw_vel  << ", P: " << p_part << ", D: " << d_part << std::endl;
    }
    else
    {
      vel_cmd.twist.angular.z = last_vel_cmd_;
      std::cout << "NO LINE DETECTED, VEL CMD: " << last_vel_cmd_ << std::endl;
    }

    velocity_command_pub_.publish(vel_cmd);
    r.sleep();
  }
}

} // namespace