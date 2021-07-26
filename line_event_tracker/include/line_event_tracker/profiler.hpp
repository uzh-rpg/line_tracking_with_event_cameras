#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <map>

namespace line_event_tracker{

class Profiler
{
public:
  Profiler() = default;
  ~Profiler();

  void start();
  void stop();
  void stop(int num_lines_cluster);

  const std::vector<long> & getDurations() const;
  const std::map<int, std::vector<long>> & getDurationsStream() const;
  const std::multimap<int, long> & getDurationsLinesClusters() const;

  void startInterval();
  void stopInterval();


private:
  std::chrono::steady_clock::time_point start_time_point_;
  std::chrono::steady_clock::time_point end_time_point_;

  std::chrono::steady_clock::time_point start_time_point_interval_;
  std::chrono::steady_clock::time_point end_time_point_interval_;

  std::vector<long> durations_;
  std::map<int, std::vector<long>> duration_stream_map_;
  std::multimap<int, long> duration_lines_cluster_map_;
  bool start_interval_ = true;
};

}