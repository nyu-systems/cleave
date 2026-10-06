#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> x_can_split = {0, 0};
  std::vector<int> w_can_split = {0, 0};
  std::vector<int> a_can_split = {0, 0};
  std::vector<int> b_can_split = {0, 0};
  TileGraph input_graph(
      test::split_enabled(x_can_split, w_can_split, a_can_split, b_can_split),
      /*symbolic=*/true);
  {
    int m = 4, k = 6, r = 8, n = 10;
    DTensor X = input_graph.create_tiled_input(
        {m, k}, x_can_split);
    DTensor W = input_graph.create_tiled_input(
        {k, n}, w_can_split);
    DTensor A = input_graph.create_tiled_input(
        {k, r}, a_can_split);
    DTensor B = input_graph.create_tiled_input(
        {r, n}, b_can_split);

    DTensor D = input_graph.matmul(X, A);
    DTensor E = input_graph.matmul(D, B);
    DTensor C = input_graph.matmul(X, W);
    DTensor O = input_graph.add(C, E);

    input_graph.create_tiled_output(O);
  }

  return input_graph.search_tile_graph(
      "checkpoint_lora.json",
      /*enable_concat_matmul_transformation=*/true);
}

} // namespace

int main() {
  return test::run_search_test("lora", 5, run_search);
}
