#pragma once
#include <unordered_map>
#include <utility>
#include <vector>

namespace line_event_tracker
{

/**
 * @brief hash functor for unordered map using std::pair<int, int>
 */
struct pair_hash
{
  std::size_t operator() (const std::pair<int, int> &pair) const
  {
    return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second);
  }
};

/**
 * @brief static class for directed chain search used for cluster generation
 */
class Chain {
public:
  static std::unordered_map<std::pair<int, int>, std::array<std::pair<int, int>, 3>, pair_hash> chain_search;
};
}


