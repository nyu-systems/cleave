#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> a_can_split = {0, 0, 0, 1};
  std::vector<int> b_can_split = {0, 0, 1, 0};
  TileGraph input_graph(
      test::split_enabled(a_can_split, b_can_split),
      /*symbolic=*/true);
  {
    DTensor A = input_graph.create_tiled_input(
        {2, 1, 16, 32}, a_can_split);
    DTensor B = input_graph.create_tiled_input(
        {1, 4, 32, 24}, b_can_split);
    DTensor C = input_graph.matmul(A, B);
    input_graph.create_tiled_output(C);
  }

  return input_graph.search_tile_graph(
      "checkpoint_bmm_bcast.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true);
}

} // namespace

int main() {
  return test::run_search_test("bmm_bcast", 2, run_search);
}
