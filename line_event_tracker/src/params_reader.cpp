#include "line_event_tracker/params_reader.hpp"
#include <iostream>

namespace line_event_tracker {

TrackerOptions readTrackerOptions(const ros::NodeHandle &nh)
{
  TrackerOptions options;
  nh.param<int>("general/height", options.height_, 260);
  nh.param<int>("general/width", options.width_, 346);
  nh.param<bool>("general/undistort", options.undistort_, false);
  nh.param<int>("general/line_publish_rate", options.line_publish_rate_, 30);
  nh.param<double>("general/periodical_check_frequency", options.periodical_check_frequency_, 100);
  nh.param<bool>("general/write_for_visualization", options.write_for_visualization_, false);

  nh.param<double>("hw_test/mask_height", options.mask_height_, 260);
  nh.param<double>("hw_test/mask_width", options.mask_width_,  346);

  nh.param<double>("filters/rf_period_same_pol", options.filter_rf_period_same_pol_, 20);
  nh.param<double>("filters/rf_period_opp_pol", options.filter_rf_period_opp_pol_, 1);
  nh.param<int>("filters/nf_neighbourhood_size", options.filter_nf_neighbourhood_size_, 5);
  nh.param<int>("filters/nf_min_num_events", options.filter_nf_min_num_events_, 3);
  nh.param<double>("filters/nf_max_age", options.filter_nf_max_age_, 30);
  options.filter_nf_offset_ = (options.filter_nf_neighbourhood_size_ - 1) / 2;

  nh.param<double>("chains/neighbourhood_size", options.chain_neighbourhood_size_, 3);
  options.chain_offset_ = (options.chain_neighbourhood_size_ - 1) / 2;
  nh.param<double>("chains/addition_distance_threshold", options.chain_add_distance_threshold_, 1.0);
  nh.param<int>("chains/max_chain_length", options.chain_max_length_, 10);
  nh.param<double>("chains/deletion_max_event_age", options.chain_max_event_age_, 50);

  return options;

}

LineOptions readLineOptions(const ros::NodeHandle &nh)
{
  LineOptions options;
  nh.param<double>("lines/promotion_threshold", options.prom_threshold_, 1.2);
  nh.param<int>("lines/promotion_num_events", options.prom_num_events_, 25);
  nh.param<int>("lines/init_bin_size", options.init_bin_size_, 2);

  nh.param<int>("lines/init_num_empty_bin", options.init_num_empty_bin_, 2);
  nh.param<int>("lines/init_period", options.init_period_, 70);
  nh.param<double>("lines/init_length", options.init_length_, 40);
  nh.param<double>("lines/addition_threshold", options.add_threshold_, 1.8);
  nh.param<double>("lines/addition_mid_point_distance", options.add_mid_point_dist_, 1.0);
  nh.param<double>("lines/merge_angle_diff", options.merge_angle_diff_, 0.1);
  nh.param<double>("lines/merge_distance_threshold", options.merge_dist_threshold_, 3.5);
  nh.param<double>("lines/merge_distance_multiplier", options.merge_dist_mult_, 1.4);
  nh.param<double>("lines/hiberantion_newest_event_age", options.hib_newest_event_age_, 10);
  nh.param<double>("lines/hibernation_max_hibernation_time", options.hib_max_hib_time_, 80);
  nh.param<double>("lines/hibernation_density_threshold", options.hib_density_threshold_ ,0.1);
  nh.param<double>("lines/update_parameters_periodicity", options.update_param_period_, 20);
  nh.param<double>("lines/update_parameters_new_event_ratio", options.update_param_new_event_ratio_, 0.05);
  nh.param<int>("lines/update_parameters_num_events", options.update_param_num_events_, 8);
  nh.param<double>("lines/update_intersection_periodicity", options.update_intersection_periodicity_, 5);

  nh.param<double>("lines/cleanup_event_age_threshold", options.cleanup_event_age_, 70);
  nh.param<double>("lines/deletion_t_no_events", options.del_t_no_events_, 30);
  nh.param<double>("lines/deletion_t_hibernate_no_events", options.del_t_hib_no_events_, 70);
  nh.param<int>("lines/deletion_out_of_frame_band", options.del_out_of_frame_band_, 10);
  nh.param<double>("lines/deletion_min_length", options.del_min_length_, 25);

  return options;
}

ClusterOptions readClusterOptions(const ros::NodeHandle &nh)
{
  ClusterOptions options;
  nh.param<int>("clusters/creation_num_events", options.creation_num_events, 10);
  nh.param<double>("clusters/addition_distance_NN_threshold", options.add_distance_NN_threshold, 2.0);
  nh.param<double>("clusters/addition_distance_line_treshold", options.add_distance_line_threshold, 1.3);
  nh.param<double>("clusters/addition_mid_point_threshold", options.add_mid_point_threshold, 1.0);
  nh.param<double>("clusters/merge_angle_diff", options.merge_angle_diff, 0.26);
  nh.param<int>("clusters/update_parameters_num_events", options.update_params_num_events, 4);
  nh.param<double>("clusters/cleanup_event_age_threshold", options.cleanup_event_age, 30);
  nh.param<double>("clusters/deletion_t_no_events", options.del_t_no_events, 50);

  return options;
}

void readCameraInfo(const ros::NodeHandle &nh, cv::Mat &K, cv::Mat &D, std::string &distortion_model)
{
  std::vector<double> intrinsics;
  std::vector<double> distortion;
  nh.getParam("cam0/intrinsics", intrinsics);
  nh.getParam("cam0/distortion_coeffs", distortion);
  nh.getParam("cam0/distortion_model", distortion_model);

  K = (cv::Mat_<double>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);
  D = (cv::Mat_<double>(1, 4) << distortion[0], distortion[1], distortion[2], distortion[3]);
  }


RefactoryFilterOptions readRefactoryFilterOptions(const ros::NodeHandle &nh)
{
  RefactoryFilterOptions options;
  nh.param<int>("filters/refactory_period_same_pol", options.refactory_period_same_pol_, 20);
  nh.param<int>("filters/refactory_period_opp_pol", options.refactory_period_opp_pol_, 1);
  nh.param<int>("general/height", options.height_, 260);
  nh.param<int>("general/height", options.height_, 346);

  return options;
}

NeighbouringFilterOptions readNeighbouringFilterOptions(const ros::NodeHandle &nh)
{
  NeighbouringFilterOptions options;
  nh.param<int>("filters/neighbourhood_size", options.neighbourhood_size_, 5);
  nh.param<int>("filters/num_events_neighbourhood", options.min_num_events_neighbourhood_, 3);
  nh.param<int>("filters/time_period", options.time_period_, 30);
  options.offset_ = (options.neighbourhood_size_ - 1) / 2;
  nh.param<int>("general/height", options.height_, 260);
  nh.param<int>("general/height", options.height_, 346);

  return options;
}

}