#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  std::vector<int> q_can_split = {0, 0, 0, 0};
  std::vector<int> k_can_split = {0, 0, 0, 0};
  std::vector<int> v_can_split = {0, 0, 0, 0};
  TileGraph input_graph(
      test::split_enabled(q_can_split, k_can_split, v_can_split),
      /*symbolic=*/true);
  {
    int b = 3, h = 5, s_q = 16, s_kv = 18, d = 8;
    DTensor Q = input_graph.create_tiled_input(
        {b, h, s_q, d}, q_can_split);
    DTensor K = input_graph.create_tiled_input(
        {b, h, s_kv, d}, k_can_split);
    DTensor V = input_graph.create_tiled_input(
        {b, h, s_kv, d}, v_can_split);

    DTensor S = input_graph.matmul(Q, K, false, true);
    DTensor O = input_graph.matmul(S, V);
    input_graph.create_tiled_output(O);
  }

  auto config = input_graph.setup_search_config();
  config.search_thread = 16;
  return input_graph.search_tile_graph(
      "checkpoint_matmul_matmul_bhsd.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("matmul_matmul_bhsd", 2, run_search);
}
