#include "cleave/superoptimizer/search/order.h"

#include <iostream>
#include <numeric>

namespace cleave_superoptimizer {
namespace search {

// TODO: there is still issue here:
// example:
//   this.v = {1, 0}, this.type = 1
//   other.v = {0, 2}, other.type = 1
//   (this < other) should return true, but it returns false

// TODO: check if this update fixes the issue
bool Order::operator<(Order const &other) const {
  int max = std::reduce(v.begin(), v.end(), -1,
                        [](int a, int b) { return std::max(a, b); });
  int max_other = std::reduce(other.v.begin(), other.v.end(), -1,
                              [](int a, int b) { return std::max(a, b); });
  if (max != max_other) {
    return max < max_other;
  }
  for (size_t i = 0; i < v.size(); ++i) {
    if (i < other.v.size() && v[i] < other.v[i]) {
      return true;
    }
    if (i < other.v.size() && v[i] > other.v[i]) {
      return false;
    }
    if (i >= other.v.size()) {
      return false;
    }
  }
  if (v.size() < other.v.size()) {
    return true;
  }
  return type < other.type;
}

bool Order::operator<=(Order const &other) const {
  return !(other < *this);
}

Order::Order(std::vector<int> const &v, int type) : v(v), type(type) {}

} // namespace search
} // namespace cleave_superoptimizer
