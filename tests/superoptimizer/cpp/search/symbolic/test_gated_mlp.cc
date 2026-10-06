#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <memory>
#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> x_can_split = {0, 1};
  std::vector<int> w1_can_split = {1, 0};
  std::vector<int> w3_can_split = {1, 0};
  TileGraph concrete_tile_graph(
      test::split_enabled(x_can_split, w1_can_split, w3_can_split));
  {
    DTensor X = concrete_tile_graph.create_tiled_input(
        {16, 4096}, x_can_split);
    DTensor W1 = concrete_tile_graph.create_tiled_input(
        {4096, 4096}, w1_can_split);
    DTensor W3 = concrete_tile_graph.create_tiled_input(
        {4096, 4096}, w3_can_split);

    DTensor D1 = concrete_tile_graph.matmul(X, W1);
    DTensor D2 = concrete_tile_graph.matmul(X, W3);
    D1 = concrete_tile_graph.silu(D1);
    DTensor O = concrete_tile_graph.mul(D1, D2);
    concrete_tile_graph.create_tiled_output(O);
  }

  std::unique_ptr<TileGraph> symbolic_tile_graph(
      TileGraph::create_symbolic(concrete_tile_graph.tile_graph));
  auto config = symbolic_tile_graph->setup_search_config();
  return symbolic_tile_graph->search_tile_graph(
      "checkpoint_symbolic_gated_mlp.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("symbolic_gated_mlp", 3, run_search);
}
