#pragma once

#include <atomic>
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include <map>
#include <Eigen/Core>

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <dvs_msgs/EventArray.h>

#include <line_event_tracker_msgs/Lines.h>

#include "line_event_tracker/event.hpp"
#include "line_event_tracker/line.hpp"
#include "line_event_tracker/cluster.hpp"
#include "line_event_tracker/chain.hpp"

#include "line_event_tracker/profiler.hpp"

namespace line_event_tracker{

struct TrackerOptions
{
  int height_;
  int width_;
  bool undistort_;
  bool write_for_visualization_;
  int line_publish_rate_;
  double periodical_check_frequency_;

  double mask_width_;
  double mask_height_;

  int filter_nf_neighbourhood_size_;
  int filter_nf_min_num_events_;
  double filter_nf_max_age_;
  int filter_nf_offset_;

  double filter_rf_period_same_pol_;
  double filter_rf_period_opp_pol_;

  double chain_neighbourhood_size_;
  int chain_offset_;
  double chain_add_distance_threshold_;
  double chain_max_event_age_;
  int chain_max_length_;

  LineOptions line_options_;
  ClusterOptions cluster_options_;
};

/**
 * @brief the Tracker class using events to track lines
 */
class Tracker
{
public:
  Tracker(ros::NodeHandle& nh, ros::NodeHandle& pnh);
  ~Tracker();

  /**
   * @brief ros callback for events, stores event in deque
   */
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);


private:

  /**
   * @brief get oldest event from deque and process it
   */
  void processEvents();

  /**
   * @brief get oldest event from events deque
   */
  void waitForEvent(Event &ev);

  /**
   * @brief periodically check lines and clusters
   */
  void periodicalCheck();

  /**
   * @brief publish lines
   */
   void publishLines();

  /**
   * @brief periodically check clusters
   * @param t current time in milliseconds
   */
  void periodicalCheckClusters(double t);

  /**
   * @brief periodically check lines
   * @param t current time in milliseconds
   */
  void periodicalCheckLines(double t);

  /**
   * @brief filter events based of refactory time and number of events in neighbourhood
   * @param ev event
   * @return true if event passed filter
   */
  bool filterEvent(Event &ev);

  /**
   * @brief try adding the current event to an existing line
   * @param ev event
   * @return true if added to line
   */
  bool tryAddToLine(Event &ev);

  /**
   * @brief check if current line with other line can be merged
   * @param curr_line
   * @param other_line
   * @return true if
   */
  bool tryMergingLines(Line &curr_line, Line &other_line) const;

  /**
   * @brief try promoting cluster to line
   * @param cluster_id
   * @param t current time in milliseconds
   * @return true if cluster is promoted
   */
  bool tryPromotingCluster(long cluster_id, double t);

  /**
   * @brief try adding current event to existing cluster
   * @param ev event
   * @return true if added to cluster
   */
  bool tryAddToCluster(Event &ev);

  /**
   * @brief try creating a cluster around current event by chain growth
   * @param ev event
   * @return true if cluster created
   */
  bool tryCreateCluster(Event &ev);

  /**
   * @brief do PCA in 2D on events (only spatial)
   * @param eig_vecs
   * @param eig_vals
   * @param cog
   * @param events
   */
  void PCA2D(Eigen::Matrix<double, 2, 2> &eig_vecs, Eigen::Vector2d &eig_vals, Eigen::Vector2d &cog, std::deque<Event> &events);

  /**
   * @brief do PCA in 3D on events (spatial-temporal)
   * @param eig_vecs
   * @param eig_vals
   * @param cog
   * @param events
   */
  void PCA3D(Eigen::Matrix<double, 3, 3> &eig_vecs, Eigen::Vector3d &eig_vals, Eigen::Vector3d &cog, std::deque<Event> &events);

  /**
   * @brief check whether line is moving out of frame
   * @param line
   * @return out of frame
   */
  bool lineOutOfFrame(const Line & line) const;


  // TESTING
  void writeChain(std::deque<Event> &chain);
  void writeCluster(long cluster_id);
  void writeAllClusters();
  void writeLine(long line_id);
  void writeAllLines();

  void writeLines(double t);
  void writeLineEvents();
  void writeClusters(double t);
  void writeClusterEvents();

private:

  // ros
  ros::NodeHandle& nh_;
  ros::NodeHandle& pnh_;
  ros::Subscriber event_subscriber_;
  ros::Publisher event_publisher_;
  ros::Publisher line_publisher_;

  // options
  TrackerOptions options_;

  // event buffer
  std::deque<Event> events_;

  // testing
  dvs_msgs::EventArray test_events_;
  int write_counter_;

  line_event_tracker_msgs::Lines lines_msg_;

  // camera info
  std::string distortion_model_;
  cv::Mat K_;
  cv::Mat D_;

  // mask
  double mask_x_upper_;
  double mask_x_lower_;
  double mask_y_upper_;
  double mask_y_lower_;

  // mutex
  std::mutex events_mutex_;
  std::mutex lines_mutex_;
  std::mutex clusters_mutex_;

  std::vector<Eigen::MatrixXd> sae_;
  std::vector<Eigen::MatrixXd> sae_unasigned_;

  // lines
  std::map<int, Line> lines_;
  std::map<int, Cluster> clusters_;

  long unique_line_id_;
  long unique_cluster_id_;

  std::atomic<double> curr_time_;

  // profiling
//  Profiler profiler_process_events_;
//  Profiler profiler_undistort_;
//  Profiler profiler_filter_;
//  Profiler profiler_add_to_line_;
//  Profiler profiler_add_to_cluster_;
//  Profiler profiler_create_cluster_;

  // TESTING
//  std::vector<int> num_lines_;
//  std::vector<int> num_clusters_;
  std::vector<double> mean_line_lifetime_;


  std::chrono::steady_clock::time_point start_time_point_;
  bool events_received_ = false;

};

} // end namespace line_event_tracker
