#include "cleave/superoptimizer/search/search_context.h"

namespace cleave_superoptimizer {
namespace search {

void from_json(json const &j, SearchContext &c) {
  c.kn_graph = std::make_shared<kernel::Graph>();
  from_json(j.at("kn_graph"), *c.kn_graph);
}

void to_json(json &j, SearchContext const &c) {
  j["kn_graph"] = *c.kn_graph;
}

} // namespace search
} // namespace cleave_superoptimizer
