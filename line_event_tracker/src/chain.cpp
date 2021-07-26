#include "line_event_tracker/chain.hpp"

namespace line_event_tracker
{

std::unordered_map<std::pair<int, int>, std::array<std::pair<int, int>, 3>, pair_hash> Chain::chain_search =
    {
        {{0, 0}, {std::make_pair(0, 1), std::make_pair(0, 0), std::make_pair(1, 0)}},
        {{0, 1}, {std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(0, 2)}},
        {{0, 2}, {std::make_pair(0, 1), std::make_pair(0, 2), std::make_pair(1, 2)}},
        {{1, 2}, {std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2)}},
        {{2, 2}, {std::make_pair(1, 2), std::make_pair(2, 2), std::make_pair(2, 1)}},
        {{2, 1}, {std::make_pair(2, 2), std::make_pair(2, 1), std::make_pair(2, 0)}},
        {{2, 0}, {std::make_pair(2, 1), std::make_pair(2, 0), std::make_pair(1, 0)}},
        {{1, 0}, {std::make_pair(0, 2), std::make_pair(0, 1), std::make_pair(0, 0)}}
    };
}

