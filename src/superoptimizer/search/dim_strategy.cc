#include "cleave/superoptimizer/search/dim_strategy.h"
#include "cleave/superoptimizer/utils/containers.h"

#include <algorithm>
#include <random>

namespace cleave_superoptimizer {
namespace search {

namespace {

std::mt19937 &shuffle_engine() {
  static thread_local std::mt19937 engine(std::random_device{}());
  return engine;
}

template <typename T>
void shuffle_vector(std::vector<T> &values) {
  std::shuffle(values.begin(), values.end(), shuffle_engine());
}

} // namespace

DimStrategy::DimStrategy(GeneratorConfig const &config) : config(config) {}

std::vector<type::KNOperatorType> DimStrategy::get_knop_cand() {
  std::vector<type::KNOperatorType> cands = config.knop_to_explore;
  if (config.randomized_branches) {
    shuffle_vector(cands);
  }
  return cands;
}

std::vector<std::vector<int>> DimStrategy::get_unary_input(int num_tensors) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    result.push_back({i});
  }
  if (config.randomized_branches) {
    shuffle_vector(result);
  }
  return result;
}

std::vector<std::vector<int>> DimStrategy::get_binary_input(int num_tensors, bool is_commutative) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    for (int j = is_commutative ? i : 0; j < num_tensors; ++j) {
      result.push_back({i, j});
    }
  }
  if (config.randomized_branches) {
    shuffle_vector(result);
  }
  return result;
}

void get_nary_input(int n,
                    int num_tensors,
                    std::vector<int> &cur,
                    std::vector<std::vector<int>> &result) {
  // TODO (david): support is_commutative
  if ((int)cur.size() == n) {
    result.push_back(cur);
    return;
  }
  for (int i = 0; i < num_tensors; ++i) {
    if (!contains(cur, i)) {
      cur.push_back(i);
      get_nary_input(n, num_tensors, cur, result);
      cur.pop_back();
    }
  }
}

std::vector<std::vector<int>> DimStrategy::get_nary_input(int num_tensors,
                                                          int n) {
  std::vector<std::vector<int>> result;
  std::vector<int> cur;
  cleave_superoptimizer::search::get_nary_input(n, num_tensors, cur, result);
  if (config.randomized_branches) {
    shuffle_vector(result);
  }
  return result;
}

} // namespace search
} // namespace cleave_superoptimizer
