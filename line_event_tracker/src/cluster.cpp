#include "line_event_tracker/cluster.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <Eigen/Eigenvalues>

namespace line_event_tracker
{
  Cluster::Cluster(std::deque<Event> &events, Eigen::Vector2d &normal, Eigen::Vector2d &cog, ClusterOptions & options, double t):
      events_(events),
      normal_(normal),
      cog_(cog),
      options_(options)
  {
    t_last_period_check_ = t;
    coordinates_sum_ = cog_ * events_.size();
    num_new_events_ = 0;
  }

  bool Cluster::periodicalCheck(double t)
  {
    t_last_period_check_ = t;

    removeOldEvents(t);

    double t_newest_event = events_.front().t;

    if (events_.size() < options_.creation_num_events ||
    t - t_newest_event > options_.cleanup_event_age) return false;

    updateInferredLine();

    return true;
  }

  double Cluster::getDistanceToNearestNeighbour(Eigen::Vector2d &point) const
  {
    double min_distance = std::numeric_limits<double>::max();
    double curr_distance;
    for (auto const &ev : events_)
    {
      curr_distance = pow(point(0) - ev.x, 2) + pow(point(1) - ev.y, 2);
      if (curr_distance < min_distance) min_distance = curr_distance;
    }

    return sqrt(min_distance);
  }

  double Cluster::getDistanceToInferredLine(Eigen::Vector2d &point) const
  {
    return abs(normal_.dot(point - cog_));
  }

  void Cluster::addEvent(Event &ev)
  {
    events_.push_front(ev);
    coordinates_sum_(0) += ev.x;
    coordinates_sum_(1) += ev.y;
    ++num_new_events_;

    if (num_new_events_ > options_.update_params_num_events)
    {
      updateInferredLine();
    }
  }

  const std::deque<Event> & Cluster::getEvents() const
  {
    return events_;
  }

  std::deque<Event> & Cluster::getEvents()
  {
    return events_;
  }

  const Eigen::Vector2d & Cluster::getNormal() const
  {
    return normal_;
  }

  const Eigen::Vector2d & Cluster::getCOG() const
  {
    return cog_;
  }

  double Cluster::getLength() const
  {
      return length_;
  }

  void Cluster::removeOldEvents(double t)
  {
    double t_removal = t - options_.cleanup_event_age;
    int counter = 0;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it)
    {
      if (it->t > t_removal) break;
      coordinates_sum_(0) -= it->x;
      coordinates_sum_(1) -= it->y;
      ++counter;
    }

    events_.resize(events_.size() - counter);
    cog_ = coordinates_sum_ / events_.size();

    updateInferredLine();

  }

  void Cluster::mergeClusters(Cluster &other)
  {
   coordinates_sum_ += other.coordinates_sum_;

    events_.insert(events_.end(), other.events_.begin(), other.events_.end());
    std::sort(events_.begin(), events_.end(), greater_than());

    updateInferredLine();
  }

  void Cluster::updateInferredLine()
  {
    double xx_var = 0;
    double xy_var = 0;
    double yy_var = 0;

    double x_dif;
    double y_dif;

    cog_  = coordinates_sum_ /events_.size();

    for (auto const &ev : events_)
    {
      x_dif = ev.x - cog_(0);
      y_dif = ev.y - cog_(1);

      xx_var += x_dif * x_dif;
      xy_var += x_dif * y_dif;
      yy_var += y_dif * y_dif;
    }

    Eigen::Matrix<double, 2, 2> cov;
    cov(0, 0) = xx_var;
    cov(0, 1) = xy_var;
    cov(1, 0) = xy_var;
    cov(1, 1) = yy_var;
    cov /= events_.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2>> eig_sol(cov);
    normal_ = eig_sol.eigenvectors().col(0);

    length_ = sqrt(12 * eig_sol.eigenvalues()[1]);

  }
}