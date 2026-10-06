#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> x_can_split = {0, 0};
  std::vector<int> w1_can_split = {0, 0};
  std::vector<int> w2_can_split = {0, 0};
  TileGraph input_graph(
      test::split_enabled(x_can_split, w1_can_split, w2_can_split),
      /*symbolic=*/true);
  {
    int m = 4, k = 6, n = 8;
    DTensor X = input_graph.create_tiled_input(
        {m, k}, x_can_split);
    DTensor W1 = input_graph.create_tiled_input(
        {k, n}, w1_can_split);
    DTensor W2 = input_graph.create_tiled_input(
        {k, n}, w2_can_split);

    DTensor D1 = input_graph.matmul(X, W1);
    DTensor D2 = input_graph.matmul(X, W2);
    D1 = input_graph.silu(D1);
    DTensor O = input_graph.mul(D1, D2);

    input_graph.create_tiled_output(O);
  }

  return input_graph.search_tile_graph("checkpoint_gated_mlp.json");
}

} // namespace

int main() {
  return test::run_search_test("gated_mlp", 1, run_search);
}
