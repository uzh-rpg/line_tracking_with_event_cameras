#pragma once

#include <Eigen/Core>

namespace line_event_tracker{

struct NeighbouringFilterOptions
{
  int neighbourhood_size_;
  int min_num_events_neighbourhood_;
  int time_period_;
  int offset_;
  int height_;
  int width_;

};


class NeighbouringFilter
{
public:
  NeighbouringFilter() = default;
  explicit NeighbouringFilter(NeighbouringFilterOptions options);

private:
  NeighbouringFilterOptions options_;
  Eigen::MatrixXd frame_;

};

}

