#pragma once
#include "cleave/superoptimizer/kernel/graph.h"

namespace cleave_superoptimizer {
namespace search_c {

int cython_search(cleave_superoptimizer::kernel::Graph const *input_graph,
                  int max_num_graphs,
                  cleave_superoptimizer::kernel::Graph **new_graphs,
                  char const *filename,
                  char const *default_config);
} // namespace search_c
} // namespace cleave_superoptimizer
