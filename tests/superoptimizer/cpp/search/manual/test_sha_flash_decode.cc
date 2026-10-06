#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  bool split_qk = true;
  bool split_pv = true;
  std::vector<int> q_can_split = {0, split_qk};
  std::vector<int> k_can_split = {split_qk, split_pv};
  std::vector<int> v_can_split = {split_pv, 0};
  TileGraph input_graph(
      test::split_enabled(q_can_split, k_can_split, v_can_split),
      /*symbolic=*/true);
  {
    int s = 16, d = 8;
    DTensor Q = input_graph.create_tiled_input(
        {s, d}, q_can_split);
    DTensor K = input_graph.create_tiled_input(
        {d, s}, k_can_split);
    DTensor V = input_graph.create_tiled_input(
        {s, d}, v_can_split);

    DTensor A = input_graph.matmul(Q, K);
    DTensor E = input_graph.exp(A);
    DTensor S = input_graph.reduction(E, 1);
    DTensor D = input_graph.div(E, S);
    DTensor O = input_graph.matmul(D, V);

    input_graph.create_tiled_output(O);
  }

  auto config = input_graph.setup_search_config();
  config.search_thread = 16;
  return input_graph.search_tile_graph(
      "checkpoint_sha.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("sha_flash_decode", 20, run_search);
}
