#include "line_event_tracker_visualizer/visualizer.hpp"


#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>

namespace line_event_tracker_visualizer {

Visualizer::Visualizer(ros::NodeHandle & nh) : nh_(nh)
{
  image_counter_ = 0;

  nh_.param<bool>("/line_visualizer/store_images", store_images_, false);
  nh_.param<std::string>("/line_visualizer/images_dir", images_dir_, "");
  nh_.param<bool>("/line_visualizer/show_all_lines", show_all_lines_, true);
  nh_.param<bool>("/line_visualizer/show_reference_lines", show_reference_lines_, true);
  nh_.param<bool>("/line_visualizer/show_only_vertical_lines", show_only_vertical_lines_, true);
  nh_.param<bool>("/line_visualizer/show_vel_cmd", show_vel_cmd_, true);
  nh_.param<bool>("/line_visualizer/undistort", undistort_, false);
  nh_.param<bool>("/line_visualizer/use_dvs_image", use_dvs_image_, false);

  readCameraInfo();

  line_sub_ = nh_.subscribe("lines", 1, &Visualizer::linesCallback, this);
  vel_cmd_sub_ = nh_.subscribe("vel_cmd", 1, &Visualizer::velocityCommandCallback, this);

  image_transport::ImageTransport it_(nh_);
  image_sub_ = it_.subscribe("image", 1, &Visualizer::imageCallback, this);
  image_pub_ = it_.advertise("line_visualization", 1);
}

Visualizer::~Visualizer()
{
  image_pub_.shutdown();
}

void Visualizer::imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{

  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // convert to BGR image
  if (msg->encoding == "rgb8")
  {
    cv::cvtColor(cv_ptr->image, image_, CV_RGB2BGR);
  }

  if (msg->encoding == "mono8")
  {
    cv::cvtColor(cv_ptr->image, image_, CV_GRAY2BGR);
  }

  // undistort
  if (undistort_)
  {
    cv::remap(image_, image_, map_x_, map_y_, cv::INTER_LINEAR);
  }

  {
    std::unique_lock<std::mutex> lock(lines_mutex_);
    std::vector<int> reference_lines;
    double max_length = 0.0;
    int longest_line_id = -1;
    line_event_tracker_msgs::Line longest_line;


    // line selection
    for (auto const &line: lines_.lines)
    {
      if (line.state == line_event_tracker_msgs::Line::INITIALIZING) continue;
      if (line.theta < -0.1396 || line.theta > 0.1396) continue;
//      int mid_point_offset = 260 / 2 - line.pos_y;
//      if (abs(mid_point_offset) > 25) continue;;

      if (line.length > max_length)
      {
        longest_line_id = line.id;
        max_length = line.length;
        longest_line = line;
      }

    }

    if (longest_line_id >= 0)
    {
      reference_lines.push_back(longest_line_id);
    }

    for (auto const &line: lines_.lines) {

      if (show_only_vertical_lines_) {
        if (line.theta <= -0.1396 || line.theta >= 0.1396)
          continue;
      }

      // calculate end points
      auto mid_point = cv::Point2d(line.pos_x, line.pos_y);
      auto end_point_1 = cv::Point2d(line.pos_x + line.length * std::sin(line.theta) / 2,
                                     line.pos_y + line.length * std::cos(line.theta) / 2);
      auto end_point_2 = cv::Point2d(line.pos_x - line.length * std::sin(line.theta) / 2,
                                     line.pos_y - line.length * std::cos(line.theta) / 2);


      if (!use_dvs_image_)
      {
        // correct orientation
        double angle = 0.039;
        mid_point = rotatePoint(mid_point, angle);
        end_point_1 = rotatePoint(end_point_1, angle);
        end_point_2 = rotatePoint(end_point_2, angle);
      }



      cv::Scalar line_color;
      cv::Scalar text_color;
      if (show_all_lines_) {
        switch (line.state) {
          case line_event_tracker_msgs::Line::INITIALIZING:
            line_color = cv::Scalar(160, 160, 160);  // grey
            text_color = cv::Scalar(160, 160, 160);  // grey
            break;
          case line_event_tracker_msgs::Line::HIBERNATING:
            line_color = cv::Scalar(51, 153, 255); // orange
            text_color = cv::Scalar(51, 153, 255); // orange
            break;
          case line_event_tracker_msgs::Line::ACTIVE:
            line_color = cv::Scalar(0, 0, 255); // red
            text_color = cv::Scalar(0, 0, 255); // red
            break;
        }
      } else {
        if (line.state == line_event_tracker_msgs::Line::INITIALIZING) continue;
        line_color = cv::Scalar(0, 0, 255);
        text_color = cv::Scalar(0, 0, 255);

      }

      // reference line override
      if (show_reference_lines_) {
        if (std::find(reference_lines.begin(), reference_lines.end(), line.id) != reference_lines.end()) {
          switch (line.state) {
            case line_event_tracker_msgs::Line::HIBERNATING:
//              line_color = cv::Scalar(204, 102, 0);
//              text_color = cv::Scalar(204, 102, 0);
              line_color = cv::Scalar(102, 204, 0);
              text_color = cv::Scalar(102, 204, 0);
              break;
            case line_event_tracker_msgs::Line::ACTIVE:
              line_color = cv::Scalar(102, 204, 0);
              text_color = cv::Scalar(102, 204, 0);
              break;
          }
        }
      }

      cv::line(image_, end_point_1, end_point_2, line_color, 1, cv::LINE_8, 0);
      cv::putText(image_, std::to_string(line.id), mid_point, cv::FONT_HERSHEY_DUPLEX, 0.45, text_color, 0.5,
                  cv::LINE_AA);

      if ((msg->header.stamp.toSec() - lines_.header.stamp.toSec()) * 1000 > 1000.0 / 10) {
        lines_.lines.clear();
      }
    }

    if (show_vel_cmd_) {
      std::unique_lock<std::mutex> lock(vel_cmd_mutex_);
      double vel_cmd_yaw_rate = vel_cmd_.twist.angular.z;
      double arrow_length_conversion = 120 / 3;
      double arrow_length = -vel_cmd_yaw_rate * arrow_length_conversion;

      int y_pos = 260 - 30;

      cv::Point point_1(346 / 2, y_pos);
      cv::Point point_2(346 / 2 + arrow_length, y_pos);
      cv::Scalar vel_cmd_color(102, 0, 204);
      cv::arrowedLine(image_, point_1, point_2, vel_cmd_color, 1, cv::LINE_8, 0, 0.1);
      cv::putText(image_, std::to_string(vel_cmd_yaw_rate).substr(0, 4), cv::Point(346 / 2 - 15, 260 - 10),
                  cv::FONT_HERSHEY_DUPLEX, 0.4, vel_cmd_color, 0.3, cv::LINE_AA);
    }

    cv_bridge::CvImage cv_image;
    cv_image.encoding = "bgr8";
    image_.copyTo(cv_image.image);
    image_pub_.publish(cv_image.toImageMsg());

    if (store_images_) {
      std::string image_path = images_dir_ + std::to_string(image_counter_) + ".png";
      cv::imwrite(image_path, cv_image.image);
      ++image_counter_;
    }
  }
}

void Visualizer::linesCallback(const line_event_tracker_msgs::LinesConstPtr & msg)
{
  std::unique_lock<std::mutex> lock(lines_mutex_);
  lines_ = *msg;
}

void Visualizer::velocityCommandCallback(const geometry_msgs::TwistStampedPtr &msg)
{
  std::unique_lock<std::mutex> lock(vel_cmd_mutex_);
  vel_cmd_ = *msg;
}

void Visualizer::readCameraInfo()
{
  std::vector<double> intrinsics_new;
  std::vector<double> intrinsics;
  std::vector<double> distortion_coeffs;
  std::string distortion_model;

  cv::Mat K;
  cv::Mat K_new;
  cv::Mat D;

  if (use_dvs_image_)
  {
    nh_.getParam("/line_visualizer/cam0/intrinsics", intrinsics);
    nh_.getParam("/line_visualizer/cam0/distortion_coeffs", distortion_coeffs);
    nh_.getParam("/line_visualizer/cam0/distortion_model", distortion_model);
    nh_.getParam("/line_visualizer/cam0/resolution", resolution_);

    K = (cv::Mat_<double>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);
    D = (cv::Mat_<double>(1, 4) << distortion_coeffs[0], distortion_coeffs[1], distortion_coeffs[2], distortion_coeffs[3]);


    if (distortion_model == "equidistant")
    {
      cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K, cv::Size(resolution_[0], resolution_[1]), CV_32FC1, map_x_, map_y_);
    }
    else if (distortion_model == "radtan")
    {
      cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K, cv::Size(resolution_[0], resolution_[1]), CV_32FC1, map_x_, map_y_);
    }
  }
  else
  {
    nh_.getParam("/line_visualizer/cam0/intrinsics", intrinsics_new);
    nh_.getParam("/line_visualizer/cam1/intrinsics", intrinsics);
    nh_.getParam("/line_visualizer/cam1/distortion_coeffs", distortion_coeffs);
    nh_.getParam("/line_visualizer/cam1/distortion_model", distortion_model);
    nh_.getParam("/line_visualizer/cam0/resolution", resolution_);

    K = (cv::Mat_<double>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);
    K_new = (cv::Mat_<double>(3, 3) << intrinsics_new[0], 0, intrinsics_new[2], 0, intrinsics_new[1], intrinsics_new[3], 0, 0, 1);
    D = (cv::Mat_<double>(1, 4) << distortion_coeffs[0], distortion_coeffs[1], distortion_coeffs[2], distortion_coeffs[3]);


    if (distortion_model == "equidistant")
    {
      cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K_new, cv::Size(resolution_[0], resolution_[1]), CV_32FC1, map_x_, map_y_);
    }
    else if (distortion_model == "radtan")
    {
      cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K_new, cv::Size(resolution_[0], resolution_[1]), CV_32FC1, map_x_, map_y_);
    }
  }
}

cv::Point2d Visualizer::rotatePoint(cv::Point2d &point, double angle)
{
  double x_r = ((point.x - resolution_[0] / 2) * cos(angle)) - ((point.y - resolution_[1] / 2) * sin(angle)) + resolution_[0] / 2;
  double y_r = ((point.x - resolution_[0] / 2) * sin(angle)) + ((point.y - resolution_[1] / 2) * cos(angle)) + resolution_[1] / 2;

  return cv::Point2d(x_r, y_r);
}

} // namespace
