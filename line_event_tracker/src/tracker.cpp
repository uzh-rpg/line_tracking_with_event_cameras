#include <fstream>
#include <sstream>

#include <thread>
#include <cmath>
#include <Eigen/Eigenvalues>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include "line_event_tracker/tracker.hpp"
#include "line_event_tracker/params_reader.hpp"

#include <numeric>

namespace line_event_tracker {

Tracker::Tracker(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh), pnh_(pnh), unique_line_id_(0), unique_cluster_id_(0), write_counter_(0)
{
  // read params
  options_ = readTrackerOptions(pnh_);
  options_.line_options_ = readLineOptions(pnh_);
  options_.cluster_options_ = readClusterOptions(pnh_);
  readCameraInfo(pnh_, K_, D_, distortion_model_);

  // mask bounds
  mask_x_lower_ = double(options_.width_) / 2 - options_.mask_width_ / 2;
  mask_x_upper_ = double(options_.width_) / 2 + options_.mask_width_ / 2;
  mask_y_lower_ = double(options_.height_) / 2 - options_.mask_height_ / 2;
  mask_y_upper_ = double(options_.height_) / 2 + options_.mask_height_ / 2;

  sae_ = std::vector<Eigen::MatrixXd> (2, Eigen::MatrixXd::Constant(options_.width_, options_.height_, -options_.filter_nf_max_age_));
  sae_unasigned_ = std::vector<Eigen::MatrixXd> (2, Eigen::MatrixXd::Constant(options_.width_, options_.height_, -options_.chain_max_event_age_));

  event_subscriber_ = nh_.subscribe("events", 0, &Tracker::eventsCallback, this);

  line_publisher_ = nh_.advertise<line_event_tracker_msgs::Lines>("/line_events_tracker_lines", 5);

  // testing
  event_publisher_ = nh_.advertise<dvs_msgs::EventArray>("/test_events", 10);

  std::thread eventProcessingThread(&Tracker::processEvents, this);
  eventProcessingThread.detach();

  std::thread periodicalCheckThread(&Tracker::periodicalCheck, this);
  periodicalCheckThread.detach();

  std::thread linePublisherThread(&Tracker::publishLines, this);
  linePublisherThread.detach();
}

Tracker::~Tracker()
{

  auto end_time_point = std::chrono::steady_clock::now();
  auto total_time_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time_point - start_time_point_).count();

  event_publisher_.shutdown();
  line_publisher_.shutdown();

//  double sum_lines = std::accumulate(num_lines_.begin(), num_lines_.end(), 0.0);
//  double mean_lines = sum_lines / num_lines_.size();
//
//  double sum_clusters = std::accumulate(num_clusters_.begin(), num_clusters_.end(), 0.0);
//  double mean_clusters = sum_clusters / num_clusters_.size();
//
//  std::cout << "Mean num lines: " << mean_lines << std::endl;
//  std::cout << "Mean num clusters: " << mean_clusters << std::endl;

//  auto process_event_stream_duration = profiler_process_events_.getDurationsStream();
//  auto filter_stream_durations = profiler_filter_.getDurationsStream();
//  auto add_to_line_stream_durations = profiler_add_to_line_.getDurationsStream();
//  auto add_to_cluster_stream_durations = profiler_add_to_cluster_.getDurationsStream();
//  auto create_cluster_stream_durations = profiler_create_cluster_.getDurationsStream();
//
//  auto process_duration = profiler_process_events_.getDurationsLinesClusters();
//  auto filter_durations = profiler_filter_.getDurationsLinesClusters();
//  auto add_to_line_durations = profiler_add_to_line_.getDurationsLinesClusters();
//  auto add_to_cluster_durations = profiler_add_to_cluster_.getDurationsLinesClusters();
//  auto create_cluster_durations = profiler_create_cluster_.getDurationsLinesClusters();
//
//  std::ofstream profiler_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/profile_lines_clusters.txt", std::ofstream::trunc);
//
//  profiler_file << "type" << " " << "number_lines_clusters" << " " << "duration" << "\n";
//
//  for (auto const & e : process_duration)
//  {
//    profiler_file << "0" << " " << e.first << " " <<  e.second << "\n";
//  }
//
//  for (auto const & e : filter_durations)
//  {
//    profiler_file << "1" << " " << e.first << " " <<  e.second << "\n";
//  }
//
//  for (auto const & e : add_to_line_durations)
//  {
//    profiler_file << "2" << " " << e.first << " " <<  e.second << "\n";
//  }
//
//  for (auto const & e : add_to_cluster_durations)
//  {
//    profiler_file << "3" << " " << e.first << " " <<  e.second << "\n";
//  }
//
//  for (auto const & e : create_cluster_durations)
//  {
//    profiler_file << "4" << " " << e.first << " " <<  e.second << "\n";
//  }


//  profiler_file << "type" << " " << "bandwidth" << " " << "duration" << "\n";

//  for (auto const & e : process_event_stream_duration)
//  {
//    for (auto l : e.second)
//    {
//      profiler_file << "0" << " " << e.first << " " << l << "\n";
//    }
//  }
//
//  for (auto const & e : filter_stream_durations)
//  {
//    for (auto l : e.second)
//    {
//      profiler_file << "1" << " " << e.first << " " << l << "\n";
//    }
//  }
//
//  for (auto const & e : add_to_line_stream_durations)
//  {
//    for (auto l : e.second)
//    {
//      profiler_file << "2" << " " << e.first << " " << l << "\n";
//    }
//  }
//
//  for (auto const & e : add_to_cluster_stream_durations)
//  {
//    for (auto l : e.second)
//    {
//      profiler_file << "3" << " " << e.first << " " << l << "\n";
//    }
//  }
//
//  for (auto const & e : create_cluster_stream_durations)
//  {
//    for (auto l : e.second)
//    {
//      profiler_file << "4" << " " << e.first << " " << l << "\n";
//    }
//  }
//
//
//  profiler_file.close();

}

void Tracker::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{

  if (options_.undistort_)
  {
    // profiling
    if (!events_received_)
    {
      start_time_point_ = std::chrono::steady_clock::now();
    }

//    profiler_undistort_.start();
    events_received_ = true;

    cv::Mat events_distorted(msg->events.size(), 1, CV_32FC2);
    for (int i = 0; i < msg->events.size(); ++i)
    {
      events_distorted.at<cv::Vec2f>(i, 0)[0] = msg->events[i].x;
      events_distorted.at<cv::Vec2f>(i, 0)[1] = msg->events[i].y;
    }

    cv::Mat events_undistorted;
    if (distortion_model_ == "equidistant")
    {
      cv::fisheye::undistortPoints(events_distorted, events_undistorted, K_, D_, cv::Mat::eye(3, 3, CV_32FC1), K_);
    }
    else if (distortion_model_ == "radtan")
    {
      cv::undistortPoints(events_distorted, events_undistorted, K_, D_, cv::Mat::eye(3, 3, CV_32FC1), K_);
    }

    // profiling
//    profiler_undistort_.stop();

    Event ev;
    for (int i = 0; i < msg->events.size(); ++i)
    {
      ev.x = events_undistorted.at<cv::Vec2f>(i, 0)[0];
      ev.y = events_undistorted.at<cv::Vec2f>(i, 0)[1];

      // remove out of frame events
      if (ev.x < 0 || ev.x > options_.width_ - 1 || ev.y < 0 || ev.y > options_.height_ - 1)
      {
        continue;
      }

      // mask
//      if (ev.x > mask_x_upper_ || ev.x < mask_x_lower_ || ev.y > mask_y_upper_ || ev.y < mask_y_lower_)
//      {
//        continue;
//      }

      ev.t = msg->events[i].ts.toSec() * 1000;
      ev.p = msg->events[i].polarity;

      std::unique_lock<std::mutex> lock(events_mutex_);
      events_.push_back(ev);
    }
  }
  else
  {
    Event ev;
    for (auto const & event : msg->events)
    {
      ev.x = event.x;
      ev.y = event.y;

      ev.t = event.ts.toSec() * 1000;
      ev.p = event.polarity;

      std::unique_lock<std::mutex> lock(events_mutex_);
      events_.push_back(ev);
    }
  }
}

void Tracker::processEvents()
{
  Event ev;
  while(ros::ok())
  {
    waitForEvent(ev);

//    int num_lines_clusters = lines_.size() + clusters_.size();
//    int num_lines_clusters = lines_.size();
//    int num_lines_clusters = clusters_.size();


//    num_lines_.push_back(lines_.size());
//    num_clusters_.push_back(clusters_.size());

//    profiler_process_events_.start();
//    profiler_filter_.start();
    if (!filterEvent(ev))
    {
//      profiler_filter_.stop(num_lines_clusters);
//      profiler_process_events_.stop(num_lines_clusters);
      continue;
    }
//    profiler_filter_.stop(num_lines_clusters);

//    profiler_add_to_line_.start();
    if (tryAddToLine(ev))
    {
//      profiler_add_to_line_.stop(num_lines_clusters);
//      profiler_process_events_.stop(num_lines_clusters);
      continue;
    }
//    profiler_add_to_line_.stop(num_lines_clusters);

//    profiler_add_to_cluster_.start();
    if (tryAddToCluster(ev))
    {
//      profiler_add_to_cluster_.stop(num_lines_clusters);
//      profiler_process_events_.stop(num_lines_clusters);
      continue;
    }
//    profiler_add_to_cluster_.stop(num_lines_clusters);

//    profiler_create_cluster_.start();
    if (tryCreateCluster(ev))
    {
//      profiler_process_events_.stop(num_lines_clusters);
//      profiler_create_cluster_.stop(num_lines_clusters);
      continue;
    }
//    profiler_process_events_.stop(num_lines_clusters);
//    profiler_create_cluster_.stop(num_lines_clusters);
  }
}

inline void Tracker::waitForEvent(Event &ev)
{
  ros::Rate r(100);

  while(true)
  {
    {
      std::unique_lock<std::mutex> lock(events_mutex_);
      if (!events_.empty()) {
        ev = events_.front();
        events_.pop_front();
        curr_time_ = ev.t;
        return;
      }
    }
    r.sleep();
  }
}

inline void Tracker::periodicalCheck()
{
  ros::Rate r (options_.periodical_check_frequency_);

  while (ros::ok())
  {
    double t = curr_time_.load();
    periodicalCheckClusters(t);
    periodicalCheckLines(t);

    r.sleep();
  }
}

void Tracker::publishLines()
{
  ros::Rate r(options_.line_publish_rate_);

  while(ros::ok())
  {
    double t = curr_time_.load();
    {
      std::unique_lock<std::mutex> lock_lines(lines_mutex_);
      if (!lines_.empty()) {
        lines_msg_.header.stamp.fromSec(t / 1000);
        lines_msg_.height = options_.height_;
        lines_msg_.width = options_.width_;

        for (auto const &entry : lines_) {
          line_event_tracker_msgs::Line line;
          line.id = entry.second.getId();
          line.pos_x = entry.second.getMidPoint()(0);
          line.pos_y = entry.second.getMidPoint()(1);
          line.vel_x = entry.second.getVelocity()(0);
          line.vel_y = entry.second.getVelocity()(1);
          line.theta = std::atan(entry.second.getLineDirection()(0) / entry.second.getLineDirection()(1));
          line.length = entry.second.getLength();
          line.state = static_cast<unsigned int>(entry.second.getState());

          lines_msg_.lines.push_back(line);
        }

        line_publisher_.publish(lines_msg_);
        lines_msg_.lines.clear();
      }
    }

    if (options_.write_for_visualization_)
    {
      {
        std::unique_lock<std::mutex> lock_lines(lines_mutex_);
        writeLines(t);
        writeLineEvents();
      }

      {
        std::unique_lock<std::mutex> lock_clusters(clusters_mutex_);
        writeClusters(t);
        writeClusterEvents();
      }

      ++write_counter_;
    }

    r.sleep();
  }
}

inline void Tracker::periodicalCheckLines(double t)
{
  std::unique_lock<std::mutex> lock(lines_mutex_);

  if (!lines_.empty()) {
    std::vector<int> line_ids_to_delete;

    for (auto &kv : lines_)
    {
      if (!kv.second.periodicalCheck(t) || lineOutOfFrame(kv.second))
      {
        line_ids_to_delete.push_back(kv.first);
      }
    }

    for (auto const line_id : line_ids_to_delete)
    {
      lines_.erase(line_id);
    }

    // merge lines
    std::vector<std::vector<int>> lines_to_be_merged;

    for (auto &line_entry :  lines_)
    {
      auto it = lines_.find(line_entry.first);
      ++it;
      if (it == lines_.end()) break;

      std::vector<int> merge_candidates;
      merge_candidates.push_back(line_entry.first);

      while(it != lines_.end())
      {
        if (tryMergingLines(line_entry.second, it->second))
        {
          merge_candidates.push_back(it->first);
        }

        ++it;
      }

      if (merge_candidates.size() > 1)
      {
        lines_to_be_merged.push_back(merge_candidates);
      }
    }

    for (int i = lines_to_be_merged.size() - 1; i >= 0; --i)
    {
      auto const &merge_candidates = lines_to_be_merged[i];

      for (int j = merge_candidates.size() - 1; j >= 1; --j)
      {
        auto it = lines_.find(merge_candidates[j]);
        auto it_next = lines_.find(merge_candidates[j-1]);
        if (it == lines_.end() || it_next == lines_.end()) continue;

        lines_.at(merge_candidates[j-1]).mergeLines(it->second);
        lines_.erase(it->first);

      }
    }
  }

}

inline bool Tracker::tryMergingLines(Line &curr_line, Line &other_line) const
{
  double angle_diff = acos(abs(curr_line.getLineDirection().dot(other_line.getLineDirection())));
  double distance_curr_to_other = other_line.getDistanceToInferredLine(curr_line.getMidPoint());
  double distance_other_to_curr = curr_line.getDistanceToInferredLine(other_line.getMidPoint());
  double distance_to_line = std::min(distance_curr_to_other, distance_other_to_curr);

  double distance_mid_points = (curr_line.getMidPoint() - other_line.getMidPoint()).norm();
  double distance_mid_points_threshold = options_.line_options_.merge_dist_mult_ * (curr_line.getLength() + other_line.getLength()) / 2;

  if (angle_diff < options_.line_options_.merge_angle_diff_ &&
      distance_to_line < options_.line_options_.merge_dist_threshold_ &&
      distance_mid_points < distance_mid_points_threshold )
  {
    return true;
  }
  else
  {
    return false;
  }
}

inline void Tracker::periodicalCheckClusters(double t)
{

  std::unique_lock<std::mutex> lock(clusters_mutex_);

  if (!clusters_.empty())
  {
    std::vector<int> cluster_ids_to_delete;

    for (auto &kv : clusters_)
    {
      if (!kv.second.periodicalCheck(t))
      {
        cluster_ids_to_delete.push_back(kv.first);
      }
    }

    for (auto const cluster_id: cluster_ids_to_delete)
    {
      clusters_.erase(cluster_id);
    }
  }
}

inline bool Tracker::filterEvent(Event &ev)
{

  double last_timestamp = sae_[ev.p](ev.x, ev.y);
  double last_timestamp_opp = sae_[!ev.p](ev.x, ev.y);

  sae_[ev.p](ev.x, ev.y) = ev.t;

  // refactory filter
  if (ev.t - last_timestamp < options_.filter_rf_period_same_pol_ || ev.t - last_timestamp_opp < options_.filter_rf_period_opp_pol_)
  {
    return false;
  }

  int x_left_bound = (ev.x - options_.filter_nf_offset_ < 0) ? 0 : ev.x - options_.filter_nf_offset_;
  int x_right_bound = (ev.x + options_.filter_nf_offset_ > options_.width_ - 1) ? options_.width_ - 1 : ev.x + options_.filter_nf_offset_;
  int y_upper_bound = (ev.y - options_.filter_nf_offset_ < 0 ) ? 0 : ev.y - options_.filter_nf_offset_;
  int y_lower_bound = (ev.y + options_.filter_nf_offset_ > options_.height_ - 1) ? options_.height_ - 1 : ev.y + options_.filter_nf_offset_;

  auto window = sae_[ev.p].block(x_left_bound, y_upper_bound, x_right_bound - x_left_bound + 1, y_lower_bound - y_upper_bound + 1);
  int num_events = ((Eigen::MatrixXd::Constant(window.rows(), window.cols(), ev.t) - window).array() < options_.filter_nf_max_age_).count();

  // neighbourhood filter
  if (num_events < options_.filter_nf_min_num_events_)
  {
    return false;
  }

  return true;
}

inline bool Tracker::tryAddToLine(Event &ev)
{
  Eigen::Vector2d point;
  point << ev.x, ev.y;

  Line *line_candidate;
  int close_lines_counter = 0;

  std::unique_lock<std::mutex> lock(lines_mutex_);
  for (auto &kv : lines_)
  {
    auto &line = kv.second;
    line.update(ev.t);

    double mid_point_distance = (point - line.getMidPoint()).norm();
    if (mid_point_distance < options_.line_options_.add_mid_point_dist_ * line.getLength())
    {
      if (line.getDistanceToInferredLine(point) < options_.line_options_.add_threshold_)
      {
        line_candidate = &kv.second;
        ++close_lines_counter;
      }
    }
  }

  if (close_lines_counter > 0)
  {
    if (close_lines_counter == 1)
    {
      line_candidate->addEvent(ev);
    }
    return true;
  }

  return false;
}

inline bool Tracker::tryAddToCluster(Event &ev)
{
  Eigen::Vector2d point;
  point << ev.x, ev.y;

  std::vector<long> cluster_ids_close;

  std::unique_lock<std::mutex> lock(clusters_mutex_);

  for (auto const &kv : clusters_)
  {
    auto &cluster = kv.second;
    double distance_line = cluster.getDistanceToInferredLine(point);
    double distance_midpoint = (point - cluster.getCOG()).norm();

    if (distance_line < options_.cluster_options_.add_distance_line_threshold &&
    distance_midpoint < options_.cluster_options_.add_mid_point_threshold * cluster.getLength())
    {
      cluster_ids_close.push_back(kv.first);
    }

  }

  if (!cluster_ids_close.empty())
  {
    auto &curr_cluster = clusters_.at(cluster_ids_close[0]);
    curr_cluster.addEvent(ev);

    // merge clusters
    if (cluster_ids_close.size() > 1)
    {
      for (int i = 1; i < cluster_ids_close.size(); ++i)
      {
        auto &other_cluster = clusters_.at(cluster_ids_close[i]);
        double angle_diff = acos(abs(curr_cluster.getNormal().dot(other_cluster.getNormal())));
        if (angle_diff < options_.cluster_options_.merge_angle_diff)
        {
          curr_cluster.mergeClusters(other_cluster);
          clusters_.erase(cluster_ids_close[i]);
        }
      }
    }

    // try promoting cluster
    long curr_line_id = unique_line_id_;

    std::unique_lock<std::mutex> lock_lines(lines_mutex_);
    if (tryPromotingCluster(cluster_ids_close[0], ev.t))
    {
      // try merging with existing line
      auto &curr_line =  lines_.at(curr_line_id);
      for (auto &kv : lines_ )
      {
        if (kv.first == curr_line_id) continue;
        if (tryMergingLines(curr_line, kv.second))
        {
          kv.second.mergeLines(curr_line);
          lines_.erase(curr_line_id);
          --unique_line_id_;
          break;
        }
      }
    }

    return true;
  }

  return false;
}

inline bool Tracker::tryPromotingCluster(long cluster_id, double t)
{
  auto &cluster = clusters_.at(cluster_id);

  if (cluster.getEvents().size() > options_.line_options_.prom_num_events_)
  {

    Eigen::Vector3d cog;
    cog(0) = cluster.getCOG()(0);
    cog(1) = cluster.getCOG()(1);
    cog(2) = 0;

    for (auto const &ev: cluster.getEvents())
    {
      cog(2) += ev.t;
    }

    cog(2) /= cluster.getEvents().size();

    Eigen::Matrix<double, 3, 3> eig_vecs;
    Eigen::Vector3d eig_vals;

    // solve PCA
    PCA3D(eig_vecs, eig_vals, cog, cluster.getEvents());

    Eigen::Vector3d normal = eig_vecs.col(0);
    Eigen::Vector2d line_dir;
    line_dir << normal(1), -normal(0);
    line_dir.normalize();

    double std_dev_line = eig_vecs.col(1).head<2>().norm() * eig_vals(1) + eig_vecs.col(2).head<2>().norm() * eig_vals(2);
    double length = sqrt(12 * std_dev_line);

    if (eig_vals[0] < options_.line_options_.prom_threshold_ &&
        length > options_.line_options_.del_min_length_)
    {

      lines_.insert(lines_.end(), std::make_pair(unique_line_id_, Line(cluster.getEvents(), normal, eig_vals, eig_vecs, cog, length, options_.line_options_, t, -1)));
      ++unique_line_id_;

      clusters_.erase(cluster_id);
      return true;
    }
    else
    {
      clusters_.erase(cluster_id);
      return false;
    }
  }
  else
  {
    return false;
  }
}

inline bool Tracker::tryCreateCluster(Event &ev)
{

  sae_unasigned_[ev.p](ev.x, ev.y) = -2 * options_.chain_max_event_age_;

  int curr_x, curr_y;
  curr_x = ev.x;
  curr_y = ev.y;

  std::deque<Event> chain;
  chain.push_front(ev);

  Eigen::MatrixXd::Index maxRow, maxCol;
  Eigen::Matrix<double, 3, 3> window;
  double max_time_stamp;

  // ignore events on boarder
  bool on_boarder = (curr_x == 0 || curr_x == options_.width_ - 1 || curr_y == 0 || curr_y == options_.height_ - 1);
  if (on_boarder)
  {
    sae_unasigned_[ev.p](ev.x, ev.y) = ev.t;
    return false;
  }

  // chain growth
  for (int i = 0; i < options_.chain_max_length_; ++i)
  {
    int x_left = curr_x - 1;
    int x_right = curr_x + 1;
    int y_upper = curr_y - 1;
    int y_lower = curr_y + 1;

    // out of frame for a 5x5 neighbourhood
    bool out_of_frame = (x_left <= 1 || x_right >= options_.width_ - 2 || y_upper <= 1 || y_lower >= options_.height_ - 2);
    if (out_of_frame) break;

    if (i == 0)
    {
      window = sae_unasigned_[ev.p].block(x_left, y_upper, x_right - x_left + 1, y_lower - y_upper + 1);
      max_time_stamp = window.maxCoeff(&maxRow, &maxCol);
    }
    else
    {
      max_time_stamp = -options_.chain_max_event_age_;

      for (auto const & pos : Chain::chain_search.at({maxRow, maxCol}))
      {
        double curr_t = sae_unasigned_[ev.p](x_left + pos.first, y_upper + pos.second);

        if (max_time_stamp < curr_t)
        {
          maxRow = pos.first;
          maxCol = pos.second;
          max_time_stamp = curr_t;
        }
      }
    }

    // found event in 3x3
    if (ev.t - max_time_stamp < options_.chain_max_event_age_)
    {
      curr_x = x_left + maxRow;
      curr_y = y_upper + maxCol;
      chain.push_front(Event{curr_x, curr_y, max_time_stamp, ev.p});
    }
    else
    {
      // did not find event in 3x3 neighbourhood, extend search to 5x5
      x_left = x_left - 1 + maxRow;
      y_upper = y_upper - 1 + maxCol;

      for (auto const & pos : Chain::chain_search.at({maxRow, maxCol}))
      {

        double curr_t = sae_unasigned_[ev.p](x_left + pos.first, y_upper + pos.second);

        if (max_time_stamp < curr_t)
        {
          maxRow = pos.first;
          maxCol = pos.second;
          max_time_stamp = curr_t;
        }
      }

      // found event in 5x5
      if (ev.t - max_time_stamp < options_.chain_max_event_age_)
      {
        curr_x = x_left + maxRow;
        curr_y = y_upper + maxCol;
        chain.push_front(Event{curr_x, curr_y, max_time_stamp, ev.p});
      }
      else
      {
        break;
      }
    }
  }

  // cluster generation
  if (chain.size() > options_.cluster_options_.creation_num_events)
  {

    // remove events from sae
    for (auto const &event: chain)
    {
      sae_unasigned_[ev.p].block(event.x-1, event.y-1, 3, 3).setConstant(-options_.chain_max_event_age_);
    }

    Eigen::Vector2d cog;
    cog.setZero();

    for (auto const &chain_ev : chain)
    {
      cog(0) += chain_ev.x;
      cog(1) += chain_ev.y;
    }

    cog /= chain.size();

    // calculate inferred line
    Eigen::Matrix<double, 2, 2> eig_vecs;
    Eigen::Vector2d eig_vals;

    PCA2D(eig_vecs, eig_vals, cog, chain);

    Eigen::Vector2d normal = eig_vecs.col(0);

    // calculate distance to inferred line
    bool is_line = true;
    for (auto const &event : chain)
    {
      double distance_to_line = abs((event.x - cog(0)) * normal(0) + (event.y - cog(1)) * normal(1));

      if (distance_to_line > options_.chain_add_distance_threshold_)
      {
        is_line = false;
        break;
      }
    }

    if (is_line)
    {
      std::unique_lock<std::mutex> lock(clusters_mutex_);
      clusters_.insert(clusters_.end(), std::make_pair(unique_cluster_id_, Cluster(chain, normal, cog, options_.cluster_options_, ev.t)));
      ++unique_cluster_id_;
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    sae_unasigned_[ev.p](ev.x, ev.y) = ev.t;
  }

  return false;
}


inline void Tracker::PCA2D(Eigen::Matrix<double, 2, 2> &eig_vecs, Eigen::Vector2d &eig_vals, Eigen::Vector2d &cog,
                           std::deque<Event> &events)
{
  double xx_var = 0;
  double xy_var = 0;
  double yy_var = 0;

  double x_dif;
  double y_dif;

  for (auto const &ev : events)
  {
    x_dif = ev.x - cog(0);
    y_dif = ev.y - cog(1);

    xx_var += x_dif * x_dif;
    xy_var += x_dif * y_dif;
    yy_var += y_dif * y_dif;
  }

  Eigen::Matrix<double, 2, 2> cov;
  cov(0, 0) = xx_var;
  cov(0, 1) = xy_var;
  cov(1, 0) = xy_var;
  cov(1, 1) = yy_var;
  cov /= events.size();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2>> eig_sol(cov);

  eig_vecs = eig_sol.eigenvectors();
  eig_vals = eig_sol.eigenvalues();
}

inline void Tracker::PCA3D(Eigen::Matrix<double, 3, 3> &eig_vecs, Eigen::Vector3d &eig_vals, Eigen::Vector3d &cog,
                    std::deque<Event> &events)
{
  double xx_var = 0;
  double xy_var = 0;
  double xt_var = 0;
  double yy_var = 0;
  double tt_var = 0;
  double yt_var = 0;

  double x_dif;
  double y_dif;
  double t_dif;


  for (auto const &ev : events)
  {
    x_dif = ev.x - cog(0);
    y_dif = ev.y - cog(1);
    t_dif = ev.t - cog(2);

    xx_var += x_dif * x_dif;
    xy_var += x_dif * y_dif;
    xt_var += x_dif * t_dif;
    yy_var += y_dif * y_dif;
    yt_var += y_dif * t_dif;
    tt_var += t_dif * t_dif;
  }

  Eigen::Matrix<double, 3, 3> cov;
  cov(0, 0) = xx_var;
  cov(0, 1) = xy_var;
  cov(0, 2) = xt_var;
  cov(1, 0) = xy_var;
  cov(1, 1) = yy_var;
  cov(1, 2) = yt_var;
  cov(2, 0) = xt_var;
  cov(2, 1) = yt_var;
  cov(2, 2) = tt_var;
  cov /= events.size();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eig_sol(cov);
  eig_vals = eig_sol.eigenvalues();
  eig_vecs = eig_sol.eigenvectors();
}

bool Tracker::lineOutOfFrame(const Line &line) const
{
  if (line.getMidPoint()(0) > (options_.width_ - options_.line_options_.del_out_of_frame_band_))
  {
    if (line.getVelocity()(0) > 0 || line.getState() == LineState::HIBERNATING)
    {
      return true;
    }
  }

  if (line.getMidPoint()(0) < options_.line_options_.del_out_of_frame_band_)
  {
    if (line.getVelocity()(0) < 0 || line.getState() == LineState::HIBERNATING)
    {
      return true;
    }
  }

  if (line.getMidPoint()(1) > (options_.height_ - options_.line_options_.del_out_of_frame_band_))
  {
    if (line.getVelocity()(1) > 0 || line.getState() == LineState::HIBERNATING)
    {
      return true;
    }
  }

  if (line.getMidPoint()(1) < options_.line_options_.del_out_of_frame_band_)
  {
    if (line.getVelocity()(1) < 0 || line.getState() == LineState::HIBERNATING)
    {
      return true;
    }
  }

  return false;

}


void Tracker::writeChain(std::deque<Event> &chain)
{

  std::ofstream chain_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/chain.txt", std::ofstream::trunc);

  for (auto const & ev : chain)
  {
    chain_file << ev.t << " " << ev.x << " " << ev.y << " " << ev.p << "\n";
  }

  chain_file.close();
}

void Tracker::writeCluster(long cluster_id)
{
  std::ofstream cluster_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/cluster.txt", std::ofstream::trunc);
  auto &cluster = clusters_.at(cluster_id);

  for (auto const & ev : cluster.getEvents())
  {
    cluster_file << ev.t << " " << ev.x << " " << ev.y << " " << ev.p << "\n";
  }

  cluster_file.close();
}

void Tracker::writeAllClusters()
{
  std::ofstream cluster_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/cluster.txt", std::ofstream::trunc);

  for (auto const &entry : clusters_)
  {
    auto &cluster = entry.second;

    for (auto const & ev : cluster.getEvents())
    {
      cluster_file << ev.t << " " << ev.x << " " << ev.y << " " << ev.p << "\n";
    }
  }

  cluster_file.close();
}

void Tracker::writeLine(long line_id)
{
  std::ofstream line_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/line.txt", std::ofstream::trunc);
  std::ofstream line_events_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/line_events.txt", std::ofstream::trunc);

  auto &line = lines_.at(line_id);

  auto end_point_1 = line.getMidPoint() + line.getLineDirection() * line.getLength() / 2;
  auto end_point_2 = line.getMidPoint() - line.getLineDirection() * line.getLength() / 2;

  line_file << line_id << " " << end_point_1(0) << " " << end_point_1(1) << " " << end_point_2(0) << " " << end_point_2(1) << " " << line.getEvents().front().t << " " << line.getState() << "\n";

  // events
  for (auto const &ev : line.getEvents())
  {
    line_events_file << ev.t << " " << ev.x << " " << ev.y << " " << ev.p << "\n";
  }

  line_file.close();
  line_events_file.close();
}

void Tracker::writeAllLines()
{
  std::ofstream lines_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/lines.txt", std::ofstream::trunc);
  std::ofstream lines_events_file("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/lines_events.txt", std::ofstream::trunc);

  for (auto const &entry : lines_)
  {
    auto &line = entry.second;

    auto end_point_1 = line.getMidPoint() + line.getLineDirection() * line.getLength() / 2;
    auto end_point_2 = line.getMidPoint() - line.getLineDirection() * line.getLength() / 2;

    lines_file << entry.first << " " << end_point_1(0) << " " << end_point_1(1) << " " << end_point_2(0) << " " << end_point_2(1) << " " << line.getEvents().front().t << " " << line.getState() << "\n";

    // events
    for (auto const &ev : line.getEvents())
    {
      lines_events_file << ev.t << " " << ev.x << " " << ev.y << " " << ev.p << "\n";
    }

  }

  lines_file.close();
  lines_events_file.close();
}

void Tracker::writeLines(double t)
{
  std::ofstream lines_file;

  if (write_counter_ == 0)
  {
    lines_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/lines.txt", std::ofstream::trunc);
  }
  else
  {
    lines_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/lines.txt", std::ios_base::app);
  }

  std::string lines;
  double curr_time = curr_time_.load();
  for (auto const & l : lines_)
  {
    auto & line = l.second;
    lines.append(std::to_string(write_counter_) + " " + std::to_string(t) + " " + std::to_string(line.getState()) + " " +
    std::to_string(line.getNormal()(0)) + " " + std::to_string(line.getNormal()(1)) + " " + std::to_string(line.getNormal()(2)) + " " +
    std::to_string(line.getCOG()(0)) + " " + std::to_string(line.getCOG()(1)) + " " + std::to_string(line.getCOG()(2)) + " " +
    std::to_string((line.getMidPoint()(0))) + " " + std::to_string((line.getMidPoint()(1))) + " " + std::to_string((line.getLength())) + " " +
    std::to_string(std::atan(line.getLineDirection()(0) / line.getLineDirection()(1))) + " " + std::to_string(line.getId()) + "\n");
  }

  lines_file << lines;
  lines_file.close();
}

void Tracker::writeLineEvents()
{
  std::ofstream events_file;
  std::string events;

  if (write_counter_ == 0)
  {
    events_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/events.txt", std::ofstream::trunc);
  }
  else
  {
    events_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/events.txt", std::ios_base::app);
  }

  // write line events
  for (auto const & l : lines_)
  {
    for (auto const &e : l.second.getEvents())
    {
      events.append(std::to_string(write_counter_) + " " + std::to_string(e.t) + " " + std::to_string(e.x) + " " + std::to_string(e.y) + " " +
                    std::to_string(e.p) + " " + "0" + " " + std::to_string(l.second.getId()) + "\n");
    }
  }

  events_file << events;
  events_file.close();
}

void Tracker::writeClusterEvents()
{
  std::ofstream events_file;
  std::string events;

  if (write_counter_ == 0)
  {
    events_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/events.txt", std::ofstream::trunc);
  }
  else
  {
    events_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/events.txt", std::ios_base::app);
  }

  // write cluster events
  for (auto const & c: clusters_)
  {
    for (auto const & e : c.second.getEvents())
    {
      events.append(std::to_string(write_counter_) + " " + std::to_string(e.t) + " " + std::to_string(e.x) + " " +
                    std::to_string(e.y) + " " + std::to_string(e.p) + " " + "1" + " " + std::to_string(c.first) + "\n");
    }
  }

  events_file << events;
  events_file.close();
}

void Tracker::writeClusters(double t)
{
  std::ofstream clusters_file;

  if (write_counter_ == 0)
  {
    clusters_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/clusters.txt", std::ofstream::trunc);
  }
  else
  {
    clusters_file.open("/home/alexdietsche/git/line_tracking_using_event_cameras/data/temp/clusters.txt", std::ios_base::app);
  }

  std::string clusters;
  double curr_time = curr_time_.load();
  for (auto const & c : clusters_)
  {
    auto & cluster = c.second;
    clusters.append(std::to_string(write_counter_) + " " + std::to_string(t) + " " +
                 std::to_string(cluster.getNormal()(0)) + " " + std::to_string(cluster.getNormal()(1)) + " " +
                 std::to_string(cluster.getCOG()(0)) + " " + std::to_string(cluster.getCOG()(1)) + " " + "\n");
  }

  clusters_file << clusters;
  clusters_file.close();
}

}