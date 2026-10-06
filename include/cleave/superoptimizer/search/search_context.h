#pragma once

#include <memory>

#include <atomic>

#include "cleave/superoptimizer/kernel/graph.h"
namespace cleave_superoptimizer {
namespace search {

struct SearchContext {
  std::shared_ptr<kernel::Graph> kn_graph;
};

void from_json(json const &j, SearchContext &c);
void to_json(json &j, SearchContext const &c);

class SerializedSearchContext {
public:
  SerializedSearchContext(SearchContext const &c);
  SearchContext deserialize() const;

private:
  json data;
};

} // namespace search
} // namespace cleave_superoptimizer
