#pragma once

#include <vector>
#include <deque>
#include <Eigen/Core>

#include "line_event_tracker/event.hpp"

namespace line_event_tracker{


struct LineOptions
{
  double prom_threshold_;
  int prom_num_events_;
  int init_bin_size_;
  int init_num_empty_bin_;
  int init_period_;
  double init_length_;
  double add_threshold_;
  double add_mid_point_dist_;
  double merge_angle_diff_;
  double merge_dist_threshold_;
  double merge_dist_mult_;
  double hib_newest_event_age_;
  double hib_density_threshold_;
  double hib_max_hib_time_;
  double update_param_period_;
  double update_param_new_event_ratio_;
  int update_param_num_events_;
  double update_intersection_periodicity_;
  double cleanup_event_age_;
  double del_t_no_events_;
  double del_t_hib_no_events_;
  int del_out_of_frame_band_;
  double del_min_length_;
};

enum LineState
{
  INITIALIZING,
  HIBERNATING,
  ACTIVE
};

/**
 * @brief The Line class represents the detected lines
 */
class Line
{
public:

  Line() = default;

  Line(std::deque<Event> &events, Eigen::Vector3d &normal, Eigen::Vector3d &eig_vals, Eigen::Matrix<double, 3, 3> &eig_vecs, Eigen::Vector3d &cog,
       double length, LineOptions &options, double t, int id);

  /**
   * @brief adds new event to the events deque
   * @param ev
   */
  void addEvent(Event &ev);

  /**
   * @brief performs a check on the line which includes removing old events, recalculating the plane estimate
   * and then checking if the line can be removed based on its line state
   * @param t current time in milliseconds
   * @return bool keep_line
   */
  bool periodicalCheck(double t);

  /**
   * @brief check for hibernation and update mid point if necessary
   * @param t current time in milliseconds
   */
  void update(double t);

  /**
   * @brief calculate distance to inferred line
   * @param point 2D point whose distance to the line is calculated
   * @return
   */
  double getDistanceToInferredLine(const Eigen::Vector2d &point) const;

  /**
   * @brief merge line other into current line
   * @param other line to be merged into the current line
   */
  void mergeLines(Line &other);

  /**
   * @brief mid point getter function
   */
  const Eigen::Vector2d & getMidPoint() const;

  /**
   * @brief line direction getter function
   */
  const Eigen::Vector2d & getLineDirection() const;

  /**
   * @brief velocity getter function
   */
  const Eigen::Vector2d & getVelocity() const;

  /**
   * @brief normal getter function
   */
  const Eigen::Vector3d & getNormal() const;

  /**
   * @brief cog getter function
   */
  const Eigen::Vector3d & getCOG() const;

  /**
   * @brief events getter function
   */
  const std::deque<Event> & getEvents() const;

  /**
   * @brief state getter function
   */
  const LineState & getState() const;

  /**
   * @brief length getter function
   */
  double getLength() const;

  /**
   * @brief id getter function
   */
  int getId() const;


private:
  /**
   * @brief update mid point estimate based on plane estimate and current time t
   * @param t current time in milliseconds
   */
  void updateMidPoint(double t);

  /**
   * @brief remove old events
   * @param t current time in milliseconds
   */
  void removeOldEvents(double t);

  /**
   * @brief recalculate plane estimate
   */
  void updatePlaneParameters();

  /**
   * @brief get density of events not older than a certain threshold
   * @param t current time in milliseconds
   * @return event density
   */
  double getNewEventDensity(double t) const;

  /**
   * @brief check if line can go into hibernation
   * @param t current time milliseconds
   */
  void checkForHibernation(double t);

  /**
   * @brief calculate length of the longest connected line, events are placed into bins, a line is no longer connected
   * if there is a empty bin gap of more than 1
   * @return connected line length
   */
  double getConnectedLineLength();

private:
  static int initialized_unique_id_;

  int id_;
  LineState state_;
  std::deque<Event> events_;
  LineOptions options_;

  int num_new_events_;
  double length_;

  Eigen::Vector2d mid_point_;
  Eigen::Vector2d line_dir_;
  Eigen::Vector2d vel_;
  Eigen::Vector3d eig_vals_;
  Eigen::Matrix<double, 3, 3> eig_vecs_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d cog_;
  Eigen::Vector3d  c_to_p_;

  double hibernate_start_time_;
  double t_end_init_;

  double t_last_period_check_;
  double t_last_mid_point_update_;

  // intermediate results
  Eigen::Vector3d coordinates_sum_;

};
}

