#include "search_test_utils.h"
#include "cleave/superoptimizer/tile/graph.h"

#include <memory>
#include <vector>

namespace {

using cleave_superoptimizer::kernel::DTensor;
using cleave_superoptimizer::tile::TileGraph;
namespace test = cleave_superoptimizer::tests::search;

auto run_search() {
  bool split_qk = true;
  bool split_pv = true;
  std::vector<int> q_can_split = {0, 0, 0, split_qk};
  std::vector<int> k_can_split = {0, 0, split_qk, split_pv};
  std::vector<int> v_can_split = {0, 0, split_pv, 0};
  TileGraph concrete_tile_graph(
      test::split_enabled(q_can_split, k_can_split, v_can_split));
  {
    int ho = 1, hi_q = 32, hi_kv = 8, s = 4096, d = 64;
    DTensor Q = concrete_tile_graph.create_tiled_input(
        {ho, hi_q, s, d}, q_can_split);
    DTensor K = concrete_tile_graph.create_tiled_input(
        {ho, hi_kv, d, s}, k_can_split);
    DTensor V = concrete_tile_graph.create_tiled_input(
        {ho, hi_kv, s, d}, v_can_split);

    DTensor A = concrete_tile_graph.matmul(Q, K);
    DTensor E = concrete_tile_graph.exp(A);
    DTensor S = concrete_tile_graph.reduction(E, 3);
    DTensor D = concrete_tile_graph.div(E, S);
    DTensor O = concrete_tile_graph.matmul(D, V);
    concrete_tile_graph.create_tiled_output(O);
  }

  std::unique_ptr<TileGraph> symbolic_tile_graph(
      TileGraph::create_symbolic(concrete_tile_graph.tile_graph));
  auto config = symbolic_tile_graph->setup_search_config();
  config.search_thread = 16;
  return symbolic_tile_graph->search_tile_graph(
      "checkpoint_symbolic_gqa.json",
      /*enable_concat_matmul_transformation=*/false,
      /*show_config=*/true,
      &config);
}

} // namespace

int main() {
  return test::run_search_test("symbolic_gqa_flash_decode", 29, run_search);
}
