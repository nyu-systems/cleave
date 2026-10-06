#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace cleave_superoptimizer {
namespace tests {
namespace search {

inline bool has_split_dim(std::vector<int> const &can_split) {
  for (int dim_split : can_split) {
    if (dim_split != 0) {
      return true;
    }
  }
  return false;
}

template <typename... SplitVectors>
bool split_enabled(SplitVectors const &...split_vectors) {
  return (has_split_dim(split_vectors) || ...);
}

inline int expect_graph_count(char const *test_name,
                              std::size_t actual_num_graphs,
                              std::size_t expected_num_graphs) {
  if (actual_num_graphs == expected_num_graphs) {
    std::cout << "PASSED: " << test_name << " found "
              << actual_num_graphs << " graph(s)\n";
    return 0;
  }

  std::cerr << "FAILED: " << test_name << " expected "
            << expected_num_graphs << " graph(s), found "
            << actual_num_graphs << "\n";
  return 1;
}

template <typename SearchFn>
int run_search_test(char const *test_name,
                    std::size_t expected_num_graphs,
                    SearchFn &&search_fn) {
  auto st = std::chrono::steady_clock::now();
  auto gen = std::forward<SearchFn>(search_fn)();
  auto et = std::chrono::steady_clock::now();

  std::cout << "Search time = "
            << std::chrono::duration<double>(et - st).count() << "sec\n";
  if (!gen) {
    std::cerr << "FAILED: " << test_name
              << " returned a null graph generator\n";
    return 1;
  }

  return expect_graph_count(test_name,
                            gen->generated_graphs.size(),
                            expected_num_graphs);
}

} // namespace search
} // namespace tests
} // namespace cleave_superoptimizer
