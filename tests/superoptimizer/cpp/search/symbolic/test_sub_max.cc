// exp(x - max(x, 1)) searches with the default config; results contain
// sub/max and survive JSON + clone.
#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"
#include "cleave/superoptimizer/type.h"

#include <iostream>
#include <memory>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::kernel::Graph;
using cleave_superoptimizer::tile::TileGraph;
namespace ty = cleave_superoptimizer::type;
namespace test = cleave_superoptimizer::tests::search;

bool graphs_valid(std::vector<json> const &graphs) {
  for (json const &j : graphs) {
    Graph g;
    from_json(j, g);
    bool has_sub = false, has_max = false;
    for (auto *op : g.operators) {
      has_sub |= op->op_type == ty::KN_SUB_OP;
      has_max |= ty::op_category(op->op_type) == ty::OpCategory::REDUCE_MAX;
    }
    std::unique_ptr<Graph> c(g.clone());
    if (!has_sub || !has_max || json(*c).size() != j.size()) {
      return false;
    }
  }
  return !graphs.empty();
}

std::unique_ptr<TileGraph> symbolic_graph() {
  TileGraph concrete;
  DTensor X = concrete.create_tiled_input({16, 64});
  DTensor M = concrete.reduce_max(X, 1);
  concrete.create_tiled_output(concrete.exp(concrete.sub(X, M)));
  return std::unique_ptr<TileGraph>(
      TileGraph::create_symbolic(concrete.tile_graph));
}

} // namespace

int main() {
  auto g = symbolic_graph();
  auto config = g->setup_search_config();
  config.search_thread = 8;
  std::vector<json> found;
  int rc = test::run_search_test("symbolic_sub_max", 1, [&] {
    auto gen = g->search_tile_graph("checkpoint_sub_max.json", false, true,
                                    &config);
    found = gen->generated_graphs;
    return gen;
  });
  if (rc == 0 && !graphs_valid(found)) {
    std::cerr << "FAILED: searched graphs lack sub/max or do not round-trip\n";
    return 1;
  }
  return rc;
}
