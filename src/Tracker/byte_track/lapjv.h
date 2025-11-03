#pragma once

#include <cstddef>
#include <vector>

typedef float lapjv_t;

namespace byte_track
{
int lapjv_internal(const size_t n, const std::vector<std::vector<lapjv_t>>& cost, std::vector<int>& x, std::vector<int>& y);
}