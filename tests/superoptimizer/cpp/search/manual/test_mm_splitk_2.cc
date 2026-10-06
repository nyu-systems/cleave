#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> q_can_split = {0, 0, 0};
  std::vector<int> k_can_split = {0, 0, 0};
  std::vector<int> v_can_split = {0, 0, 0};
  TileGraph input_graph(
      test::split_enabled(q_can_split, k_can_split, v_can_split),
      /*symbolic=*/true);
  {
    DTensor Q = input_graph.create_tiled_input(
        {5, 2, 8}, q_can_split);
    DTensor K = input_graph.create_tiled_input(
        {5, 8, 16}, k_can_split);
    DTensor V = input_graph.create_tiled_input(
        {5, 16, 9}, v_can_split);
    DTensor A = input_graph.matmul(Q, K);
    DTensor B = input_graph.matmul(A, V);
    input_graph.create_tiled_output(B);
  }

  auto config = input_graph.setup_search_config();
  config.search_thread = 16;
  return input_graph.search_tile_graph(
      "checkpoint_mm_splitk_2.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("mm_splitk_2", 4, run_search);
}
