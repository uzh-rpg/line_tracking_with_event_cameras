#include "line_event_tracker/neighbouringFilter.hpp"

namespace line_event_tracker
{

NeighbouringFilter::NeighbouringFilter(NeighbouringFilterOptions options) : options_(options) {
  frame_ = Eigen::MatrixXd::Constant(options_.height_, options_.width_, -1);
}

}
