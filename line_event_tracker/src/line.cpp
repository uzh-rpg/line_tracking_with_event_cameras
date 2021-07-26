#include <algorithm>
#include <numeric>
#include <Eigen/Eigenvalues>

#include "line_event_tracker/line.hpp"

namespace line_event_tracker
{

int Line::initialized_unique_id_ = 0;

Line::Line(std::deque<Event> &events, Eigen::Vector3d &normal, Eigen::Vector3d &eig_vals, Eigen::Matrix<double, 3, 3> &eig_vecs,
           Eigen::Vector3d &cog, double length, LineOptions &options, double t, int id):
           events_(events),
           normal_(normal),
           eig_vals_(eig_vals),
           eig_vecs_(eig_vecs),
           cog_(cog),
           length_(length),
           state_(INITIALIZING),
           options_(options),
           id_(id)
{
  line_dir_ << normal_(1), -normal_(0);
  line_dir_.normalize();
  t_last_period_check_ = t;
  t_last_mid_point_update_ = t;
  c_to_p_ << normal_(0) * normal_(2), normal_(2) * normal_(1), -std::pow(normal_(0), 2) -std::pow(normal_(1), 2);
  vel_ = c_to_p_.head<2>() / (c_to_p_(2) / 1000);
  t_end_init_ = t + options_.init_period_;
  coordinates_sum_ = events_.size() * cog_;

  updateMidPoint(t);
}

inline void Line::updateMidPoint(double t)
{
  cog_ = coordinates_sum_ / events_.size();

  if (state_ == HIBERNATING)
  {
    mid_point_ = cog_.head<2>();
  }
  else
  {
    mid_point_ = (cog_ + ((t - cog_(2)) / (-pow(normal_(0), 2) - pow(normal_(1), 2))) * c_to_p_).head<2>();
  }

  t_last_mid_point_update_ = t;
}

bool Line::periodicalCheck(double t)
{
  t_last_period_check_ = t;
  bool keep_line = true;

  if (state_ != HIBERNATING)
    removeOldEvents(t);

  if (events_.empty())
  {
    return false;
  }

  LineState new_state = state_;
  switch (state_)
  {
    case INITIALIZING:
      updatePlaneParameters();

      if (t > t_end_init_) {
        length_ = getConnectedLineLength();
        if (length_ < options_.init_length_)
        {
          keep_line = false;
        }
        else
        {
          id_ = initialized_unique_id_;
          ++initialized_unique_id_;
          new_state = ACTIVE;
        }
      }
      break;

    case HIBERNATING:
      if (t - events_.front().t > options_.del_t_hib_no_events_)
      {
        keep_line = false;
      }
      if (t - hibernate_start_time_ > options_.hib_max_hib_time_)
      {
        keep_line = false;
      }
      break;

    case ACTIVE:
      updatePlaneParameters();

      // TODO REPORT check if this check can be removed
      if (t - events_.front().t > options_.del_t_no_events_)
      {
        keep_line = false;
      }
      if (length_ < options_.del_min_length_)
      {
        keep_line = false;
      }
      break;
  }

  state_ = new_state;

  return keep_line;
}

void Line::update(double t)
{
  checkForHibernation(t);

  if (double(num_new_events_) / events_.size() > options_.update_param_new_event_ratio_)
  {
    updatePlaneParameters();
  }

  if (t - t_last_mid_point_update_ > options_.update_intersection_periodicity_)
  {
    updateMidPoint(t);
  }

}

void Line::removeOldEvents(double t)
{
  double t_removal = t - options_.cleanup_event_age_;
  int counter = 0;

  for (auto it = events_.rbegin(); it != events_.rend(); ++it)
  {
    if (it->t > t_removal) break;
    coordinates_sum_(0) -= it->x;
    coordinates_sum_(1) -= it->y;
    coordinates_sum_(2) -= it->t;
    ++counter;
  }

  events_.resize(events_.size() - counter);
}

void Line::updatePlaneParameters()
{

  cog_ = coordinates_sum_ / events_.size();

  double xx_var = 0;
  double xy_var = 0;
  double xt_var = 0;
  double yy_var = 0;
  double tt_var = 0;
  double yt_var = 0;

  double x_dif;
  double y_dif;
  double t_dif;

  for (auto const &ev : events_)
  {
    x_dif = ev.x - cog_(0);
    y_dif = ev.y - cog_(1);
    t_dif = ev.t - cog_(2);

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
  cov /= events_.size();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eig_sol(cov);
  eig_vals_ = eig_sol.eigenvalues();
  eig_vecs_ = eig_sol.eigenvectors();

  normal_ = eig_vecs_.col(0);
  line_dir_ << normal_(1), -normal_(0);
  line_dir_.normalize();

  double std_dev_line = abs(line_dir_.dot(eig_vecs_.col(1).head<2>() * eig_vals_(1))) +
                        abs(line_dir_.dot(eig_vecs_.col(2).head<2>() * eig_vals_(2)));
  length_ = sqrt(12 * std_dev_line);

  c_to_p_ << normal_(0) * normal_(2), normal_(2) * normal_(1), -std::pow(normal_(0), 2) -std::pow(normal_(1), 2);

  vel_ = c_to_p_.head<2>() / (c_to_p_(2) / 1000);

  num_new_events_ = 0;

}

double Line::getDistanceToInferredLine(const Eigen::Vector2d &point) const
{
  return abs(normal_.head<2>().dot(point - mid_point_) / normal_.head<2>().norm());
}

double Line::getNewEventDensity(double t) const
{
  double t_volume_lower = t - options_.hib_newest_event_age_;
  if (events_.empty() || length_ == 0) return 0;

  int counter = 0;
  for (auto const  &ev : events_)
  {
    if (ev.t < t_volume_lower) return counter / length_;
    ++counter;
  }

  return events_.size() / length_;
}

void Line::checkForHibernation(double t)
{

  if (state_ == INITIALIZING)
  {
    return;
  }

  if (events_.size() == 0 || t - events_.front().t > options_.hib_newest_event_age_ || getNewEventDensity(t) < options_.hib_density_threshold_)
  {
    if (state_ == ACTIVE)
    {
      hibernate_start_time_ = t;
    }
    state_ = HIBERNATING;
  }
  else
  {
    state_ = ACTIVE;
  }
}

void Line::addEvent(Event &ev)
{
//  new_events_.push_front(ev);

  events_.push_front(ev);
  coordinates_sum_(0) += ev.x;
  coordinates_sum_(1) += ev.y;
  coordinates_sum_(2) += ev.t;
  ++num_new_events_;

  if (double(num_new_events_) / events_.size() > options_.update_param_new_event_ratio_ || ev.t - t_last_period_check_ > options_.update_param_period_)
  {
    updatePlaneParameters();
  }

}

void Line::mergeLines(Line &other)
{
  int size_merged = events_.size() + other.events_.size();

  coordinates_sum_ += other.coordinates_sum_;
  cog_ = coordinates_sum_ / size_merged;

  events_.insert(events_.end(), other.events_.begin(), other.events_.end());
  std::sort(events_.begin(), events_.end(), greater_than());

  updatePlaneParameters();
}

const Eigen::Vector2d & Line::getMidPoint() const
{
  return mid_point_;
}

const Eigen::Vector2d & Line::getLineDirection() const
{
  return line_dir_;
}

const Eigen::Vector2d & Line::getVelocity() const
{
  return vel_;
}

const Eigen::Vector3d & Line::getNormal() const
{
  return normal_;
}

const Eigen::Vector3d & Line::getCOG() const
{
  return cog_;
}

const std::deque<Event> & Line::getEvents() const
{
  return events_;
}

const LineState & Line::getState() const
{
  return state_;
}

double Line::getLength() const
{
  return length_;
}

int Line::getId() const
{
  return id_;
}

double Line::getConnectedLineLength()
{
  std::vector<double> projected_events(events_.size());

  for (int i = 0; i < events_.size(); ++i)
  {
    projected_events[i] = (events_[i].x - mid_point_(0)) * line_dir_(0) + (events_[i].y - mid_point_(1)) * line_dir_(1);
  }

  std::vector<double> projected_events_sorted = projected_events;
  std::sort(projected_events_sorted.begin(), projected_events_sorted.end());

  int bin_start = static_cast<int>(std::floor(projected_events_sorted.front()));
  int bin_end = static_cast<int>(std::ceil(projected_events_sorted.back()));
  int num_bins = bin_end - bin_start + 1;

  std::vector<double> bins(num_bins);
  std::vector<int> bins_counter(num_bins-1);
  std::fill(bins_counter.begin(), bins_counter.end(), 0);
  std::iota(bins.begin(), bins.end(), bin_start);

  int j = 0;
  for (int i = 0; i < num_bins - 1; ++i)
  {
      while (projected_events_sorted[j] >= bins[i] && projected_events_sorted[j] < bins[i+1])
      {
        bins_counter[i] += 1;
        ++j;
      }
  }

  double max_length = 0;
  double curr_length = 0;

  int max_bin_start = 0;
  int max_bin_end = 0;
  int curr_bin_start = 0;
  int curr_bin_end = 0;

  for (int i = 0; i < num_bins-1; ++i)
  {
    if (bins_counter[i] > 0 || bins_counter[i+1] > 0)
    {
      curr_length += 1;
    }
    else
    {
      if (max_length < curr_length)
      {
        max_length = curr_length;
        max_bin_start = curr_bin_start;
        max_bin_end = i;
      }
      curr_length = 0;
      curr_bin_start = i+1;
    }
  }

  if (max_length < curr_length)
  {
    max_length = curr_length;
    max_bin_start = curr_bin_start;
    max_bin_end = num_bins -1;
  }

  for (int i = 0; i < events_.size(); ++i)
  {
    if (projected_events[i] < max_bin_start || projected_events[i] > max_bin_end)
    {
      auto &ev = events_[i];
      coordinates_sum_(0) -= ev.x;
      coordinates_sum_(1) -= ev.y;
      coordinates_sum_(2) -= ev.t;
      events_.erase(events_.begin() + i);
    }
  }

  return max_length;
}

}
