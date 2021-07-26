#pragma once

#include <ros/ros.h>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TwistStamped.h>

#include "line_event_tracker_msgs/Lines.h"

namespace line_event_tracker_visualizer
{

/**
 * @brief the Visualizer class used to visualize the tracked lines
 */
class Visualizer {
public:
  explicit Visualizer(ros::NodeHandle & nh);
  virtual ~Visualizer();

private:
  /**
   * @brief ros lines callback, draws lines and their ids into the current image
   */
  void linesCallback(const line_event_tracker_msgs::LinesConstPtr & msg);

  /**
   * @brief ros velocity command callback, used for yaw rate visualization
   */
  void velocityCommandCallback(const geometry_msgs::TwistStampedPtr & msg);

  /**
   * @brief ros image callback
   */
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

  /**
   * @brief read camera info
   */
  void readCameraInfo();

  cv::Point2d rotatePoint(cv::Point2d & point, double angle);


private:

  // ros
  ros::NodeHandle nh_;
  ros::Subscriber line_sub_;
  ros::Subscriber vel_cmd_sub_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  std::mutex lines_mutex_;
  std::mutex vel_cmd_mutex_;

  line_event_tracker_msgs::Lines lines_;
  geometry_msgs::TwistStamped vel_cmd_;

  cv::Mat map_x_;
  cv::Mat map_y_;
  std::vector<double> resolution_;

  cv::Mat image_;
  bool store_images_;
  bool show_all_lines_;
  bool show_reference_lines_;
  bool show_only_vertical_lines_;
  bool show_vel_cmd_;
  bool undistort_;
  bool use_dvs_image_;
  std::string images_dir_;
  int image_counter_;

};

} // namespace

