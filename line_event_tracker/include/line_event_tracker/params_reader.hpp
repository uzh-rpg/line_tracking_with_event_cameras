# pragma once

#include <ros/ros.h>

#include <opencv2/core/core.hpp>


#include "line_event_tracker/tracker.hpp"
#include "line_event_tracker/line.hpp"
#include "line_event_tracker/cluster.hpp"
#include "line_event_tracker/refactoryFilter.hpp"
#include "line_event_tracker/neighbouringFilter.hpp"


namespace line_event_tracker {

TrackerOptions readTrackerOptions(const ros::NodeHandle &nh);
LineOptions readLineOptions(const ros::NodeHandle &nh);
ClusterOptions readClusterOptions(const ros::NodeHandle &nh);

void readCameraInfo(const ros::NodeHandle &nh, cv::Mat &K, cv::Mat &D, std::string &distortion_model);

RefactoryFilterOptions readRefactoryFilterOptions(const ros::NodeHandle &nh);
NeighbouringFilterOptions readNeighbouringFilterOptions(const ros::NodeHandle &nh);


}
