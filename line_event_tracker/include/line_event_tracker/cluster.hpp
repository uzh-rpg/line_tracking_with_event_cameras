#pragma once

#include <vector>
#include <deque>
#include <Eigen/Core>

#include "line_event_tracker/event.hpp"

namespace line_event_tracker{

struct ClusterOptions
{
  int creation_num_events;
  double add_distance_NN_threshold;
  double add_distance_line_threshold;
  double add_mid_point_threshold;
  double merge_angle_diff;
  int update_params_num_events;
  double cleanup_event_age;
  double del_t_no_events;
};

/**
 * @brief The cluster class representing line candidates
 */
class Cluster
{
public:

  Cluster() = default;

  Cluster(std::deque<Event> &events, Eigen::Vector2d &normal, Eigen::Vector2d &cog, ClusterOptions & options, double t);

  /**
   * @brief performs a check cluster which includes removing old events, recalculating the line estimate
   * and checking if the cluster can be removed
   * @param t current time in milliseconds
   * @return keep cluster
   */
  bool periodicalCheck(double t);

  /**
   * @brief calculates distance from point to closest event in cluster
   * @param point 2D point
   * @return distance to nearest neighbour
   */
  double getDistanceToNearestNeighbour(Eigen::Vector2d &point) const;

  /**
   * @brief calculates distance from point to inferred line
   * @param point 2D point
   * @return distance to inferred line
   */
  double getDistanceToInferredLine(Eigen::Vector2d &point) const;


  /**
   * @brief add new event
   * @param ev event
   */
  void addEvent(Event &ev);

  /**
   * @brief merge cluster other into current cluster
   * @param other cluster
   */
  void mergeClusters(Cluster &other);

  /**
 * @brief normal getter function
 */
  const std::deque<Event> & getEvents() const;

  /**
   * @brief events getter function
   */
  std::deque<Event> & getEvents();

  /**
   * @brief normal getter function
   */
  const Eigen::Vector2d & getNormal() const;

  /**
   * @brief cog getter function
   */
  const Eigen::Vector2d & getCOG() const;

  /**
   * @brief length getter function
   */
  double getLength() const;

private:

  /**
   * @brief remove old events
   * @param t current time in milliseconds
   */
  void removeOldEvents(double t);

  /**
   * @brief update estimated line
   */
  void updateInferredLine();


private:
  ClusterOptions options_;
  std::deque<Event> events_;

  int num_new_events_;

  Eigen::Vector2d normal_;
  Eigen::Vector2d cog_;
  double length_;

  double t_last_period_check_;

  // intermediate results
  Eigen::Vector2d coordinates_sum_;

};

}
