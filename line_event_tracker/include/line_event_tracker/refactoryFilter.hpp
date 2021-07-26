#pragma once

#include <Eigen/Core>
#include <dvs_msgs/Event.h>

namespace line_event_tracker{

struct RefactoryFilterOptions
{
  int refactory_period_same_pol_;
  int refactory_period_opp_pol_;
  int height_;
  int width_;
};

class RefactoryFilter {
public:
  RefactoryFilter() = default;
  explicit RefactoryFilter(RefactoryFilterOptions options);

  void update(const dvs_msgs::Event &e);

private:
  RefactoryFilterOptions options_;
  Eigen::MatrixXd frame_;
};

}
