#pragma once

#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/kernel/graph.h"

#include "cleave/superoptimizer/search/search.h"
#include "cleave/superoptimizer/tile/config.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace cleave_superoptimizer {
namespace tile {

class TileGraph {
public:
  explicit TileGraph(
      bool split_enabled = false,
      bool symbolic = false);
  ~TileGraph();

  /**
   * @brief Create a symbolic tile graph from a concrete kernel graph
   *
   * Automatically converts a computation graph with concrete dimensions
   * (e.g., [16, 4096]) into a symbolic tile graph with distinct small values
   * (e.g., [6, 10]) for efficient graph search.
   *
   * Example:
   *   kernel::Graph concrete_graph;
   *   // ... build concrete graph with real dimensions ...
   *   TileGraph *symbolic_tg = TileGraph::create_symbolic(concrete_graph);
   *   symbolic_tg->search_tile_graph("checkpoint.json");
   *   delete symbolic_tg;
   *
   * @param concrete_graph The computation graph with concrete dimensions
   * @param symbol_map [out] Optional map from symbolic dimension to
   * concrete dimension
   * @return Pointer to TileGraph with symbolic dimensions ready for search (caller owns)
   */
  static TileGraph *create_symbolic(
      cleave_superoptimizer::kernel::Graph const &concrete_graph,
      std::unordered_map<int, int> *symbol_map = nullptr);

  DTensor create_tiled_input(std::vector<int> const &dims);
  DTensor create_tiled_input(std::vector<int> const &dims,
                             std::vector<int> const &can_split);
  void create_tiled_output(DTensor const &O);
  std::unique_ptr<cleave_superoptimizer::search::KernelGraphGenerator> search_tile_graph(std::string const &checkpoint_file_name = "checkpoint.json",
                         bool const &enable_concat_matmul_transformation = false,
                         bool const &show_config = false,
                         cleave_superoptimizer::search::GeneratorConfig* external_config = nullptr);
  int search_tile_graph(std::string const &checkpoint_file_name,
                        bool const &enable_concat_matmul_transformation,
                        bool const &show_config,
                        cleave_superoptimizer::search::GeneratorConfig* external_config,
                        int max_num_graphs,
                        cleave_superoptimizer::kernel::Graph **new_graphs);
  cleave_superoptimizer::search::GeneratorConfig setup_search_config();
                         
  // ops
  DTensor matmul(DTensor const &A,
                 DTensor const &B,
                 bool trans_A = false,
                 bool trans_B = false);
  DTensor add(DTensor const &A, DTensor const &B);
  DTensor add(DTensor const &A, float scalar);
  DTensor mul(DTensor const &A, DTensor const &B);
  DTensor mul(DTensor const &A, float scalar);
  DTensor div(DTensor const &A, DTensor const &B);
  DTensor div(DTensor const &A, float scalar);
  DTensor div(float scalar, DTensor const &A);
  DTensor exp(DTensor const &I);
  DTensor silu(DTensor const &I);
  DTensor sqrt(DTensor const &I);
  DTensor reduction(DTensor const &I, int const &dim);

public:
  cleave_superoptimizer::kernel::Graph tile_graph;
  cleave_superoptimizer::tile::TileSearchConfig tile_config;
  bool symbolic;
};

} // namespace tile
} // namespace cleave_superoptimizer
