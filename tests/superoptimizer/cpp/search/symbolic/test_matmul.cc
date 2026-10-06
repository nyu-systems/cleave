#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <memory>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  TileGraph concrete_tile_graph;
  {
    DTensor A = concrete_tile_graph.create_tiled_input({32, 1024});
    DTensor B = concrete_tile_graph.create_tiled_input({1024, 2048});
    DTensor C = concrete_tile_graph.matmul(A, B);
    concrete_tile_graph.create_tiled_output(C);
  }

  std::unique_ptr<TileGraph> symbolic_tile_graph(
      TileGraph::create_symbolic(concrete_tile_graph.tile_graph));
  auto config = symbolic_tile_graph->setup_search_config();
  return symbolic_tile_graph->search_tile_graph(
      "checkpoint_symbolic_matmul.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("symbolic_matmul", 2, run_search);
}
