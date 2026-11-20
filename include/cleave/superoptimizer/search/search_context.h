#pragma once

#include <memory>
#include <vector>

#include "cleave/superoptimizer/kernel/graph.h"
namespace cleave_superoptimizer {
namespace search {

struct SearchContext {
  std::shared_ptr<kernel::Graph> kn_graph;
  // Where this state sits in the search tree: the index of the branch taken at
  // each level below the root. It does not depend on which thread explores it.
  std::vector<int> path;
};

void from_json(json const &j, SearchContext &c);
void to_json(json &j, SearchContext const &c);

} // namespace search
} // namespace cleave_superoptimizer
