#include "line_event_tracker/profiler.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>

#include <iostream>

namespace line_event_tracker {

Profiler::~Profiler()
{
}

void Profiler::start()
{
  start_time_point_ = std::chrono::steady_clock::now();

  if (start_interval_)
  {
    startInterval();
  }
}

void Profiler::stop()
{
  end_time_point_ = std::chrono::steady_clock::now();
  std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_point_ - start_time_point_);
  durations_.push_back(duration.count());

  if (!start_interval_)
  {
    auto elapsed_time_interval = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_point_ - start_time_point_interval_);
    if (elapsed_time_interval.count() > 150)
    {
      stopInterval();
    }
  }
}

void Profiler::stop(int num_lines_cluster)
{
  end_time_point_ = std::chrono::steady_clock::now();
  std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_point_ - start_time_point_);
  duration_lines_cluster_map_.insert(std::make_pair(num_lines_cluster, duration.count()));
}

void Profiler::startInterval()
{
  start_time_point_interval_ = std::chrono::steady_clock::now();
  start_interval_ = false;
}

void Profiler::stopInterval()
{
  end_time_point_interval_ = std::chrono::steady_clock::now();
  std::chrono::nanoseconds interval_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_point_interval_ - start_time_point_interval_);

  // calculate incoming event stream
  int num_events = durations_.size();
  int event_stream = num_events / (interval_duration.count() / std::pow(10, 9));

  duration_stream_map_.insert(std::make_pair(event_stream, durations_));

  durations_.clear();
  start_interval_ = true;
}


const std::vector<long> & Profiler::getDurations() const
{
  return durations_;
}

const std::map<int, std::vector<long>> & Profiler::getDurationsStream() const
{
  return duration_stream_map_;
}

const std::multimap<int, long> & Profiler::getDurationsLinesClusters() const
{
  return duration_lines_cluster_map_;
}

}
