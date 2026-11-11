#pragma once
#include "cleave/superoptimizer/kernel/graph.h"

#include <vector>

namespace cleave_superoptimizer {
namespace search_c {

// Caller owns the returned graphs.
std::vector<cleave_superoptimizer::kernel::Graph *>
    search_graphs(cleave_superoptimizer::kernel::Graph const *input_graph,
                  char const *filename,
                  char const *default_config);

int cython_search(cleave_superoptimizer::kernel::Graph const *input_graph,
                  int max_num_graphs,
                  cleave_superoptimizer::kernel::Graph **new_graphs,
                  char const *filename,
                  char const *default_config);
} // namespace search_c
} // namespace cleave_superoptimizer
