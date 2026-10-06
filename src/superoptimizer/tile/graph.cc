#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/search/search.h"

#include "cleave/superoptimizer/tile/graph.h"
#include "cleave/superoptimizer/utils/logging.h"

#include <cassert>
#include <fstream>
#include <sstream>

namespace cleave_superoptimizer {
namespace tile {

namespace {

void log_symbolic_input_mapping(
    cleave_superoptimizer::kernel::Graph const &concrete_graph,
    std::unordered_map<size_t, std::vector<int>> const
        &symbolic_dims_by_guid) {
  if (!CSO_VLOG_IS_ON(1)) {
    return;
  }

  auto append_shape = [](std::ostream &os, auto const &dims, int num_dims) {
    os << "[";
    for (int i = 0; i < num_dims; i++) {
      os << dims[i];
      if (i < num_dims - 1) {
        os << ", ";
      }
    }
    os << "]";
  };

  std::ostringstream os;
  os << "Symbolic input dimension mapping:";
  for (auto const &op : concrete_graph.operators) {
    if (op->op_type != cleave_superoptimizer::type::KN_INPUT_OP) {
      continue;
    }

    kernel::DTensor const &tensor = op->output_tensors[0];
    os << "\n  Input tensor " << tensor.guid << ": ";
    append_shape(os, tensor.dim, tensor.num_dims);
    os << " -> ";

    auto it = symbolic_dims_by_guid.find(tensor.guid);
    if (it == symbolic_dims_by_guid.end()) {
      os << "<missing>";
    } else {
      append_shape(os, it->second, static_cast<int>(it->second.size()));
    }
  }

  CSO_VLOG(1) << os.str();
}

void require_symbolized_search_graph(bool symbolic) {
  if (!symbolic) {
    CSO_LOG(FATAL)
        << "TileGraph search requires a symbolized graph. "
        << "Use TileGraph::create_symbolic(...) before search instead of "
        << "searching a concrete TileGraph constructed with symbolic=false.";
  }
}

void record_symbol_mapping_entry(
    std::unordered_map<int, int> *symbol_map,
    int symbolic_dim,
    int concrete_dim) {
  if (symbol_map == nullptr) {
    return;
  }

  auto it = symbol_map->find(symbolic_dim);
  CSO_CHECK(it == symbol_map->end() || it->second == concrete_dim)
      << "Conflicting concrete dimensions for symbolic dim " << symbolic_dim
      << ": " << it->second << " vs " << concrete_dim;
  (*symbol_map)[symbolic_dim] = concrete_dim;
}

void populate_symbol_mapping(
    cleave_superoptimizer::kernel::Graph const &concrete_graph,
    std::unordered_map<size_t, std::vector<int>> const &symbolic_dims_by_guid,
    std::unordered_map<int, int> *symbol_map) {
  if (symbol_map == nullptr) {
    return;
  }

  symbol_map->clear();
  for (auto const &op : concrete_graph.operators) {
    if (op->op_type != cleave_superoptimizer::type::KN_INPUT_OP) {
      continue;
    }

    CSO_CHECK(op->output_tensors.size() == 1);
    kernel::DTensor const &tensor = op->output_tensors[0];
    auto it = symbolic_dims_by_guid.find(tensor.guid);
    CSO_CHECK(it != symbolic_dims_by_guid.end())
        << "Missing symbolic dimensions for input tensor " << tensor.guid;

    std::vector<int> const &symbolic_dims = it->second;
    CSO_CHECK(static_cast<int>(symbolic_dims.size()) == tensor.num_dims)
        << "Symbolic dimension count mismatch for input tensor " << tensor.guid
        << ": " << symbolic_dims.size() << " vs " << tensor.num_dims;

    for (int i = 0; i < tensor.num_dims; ++i) {
      record_symbol_mapping_entry(symbol_map, symbolic_dims[i], tensor.dim[i]);
    }
  }
}

} // namespace

TileGraph::TileGraph(
    bool split_enabled,
    bool symbolic)
    : tile_graph(),
      tile_config(/*infer_num_ops=*/false, split_enabled),
      symbolic(symbolic) {}

TileGraph::~TileGraph() {}

DTensor TileGraph::create_tiled_input(std::vector<int> const &dims) {
  tile_config.update_config(cleave_superoptimizer::type::KN_INPUT_OP);

  std::vector<int> can_split(dims.size(), tile_config.split_enabled ? 1 : 0);
  std::vector<bool> can_reduce_b(dims.size(), false);
  std::vector<bool> can_split_b(can_split.size(), false);
  for (size_t i = 0; i < can_split.size(); ++i) {
    can_split_b[i] = (can_split[i] != 0);
  }

  DTensor output = tile_graph.new_input(dims, can_reduce_b, can_split_b);
  auto *input_op =
      dynamic_cast<cleave_superoptimizer::kernel::KNInputOp *>(
          tile_graph.operators.back());
  assert(input_op != nullptr);
  input_op->infer_can_split = true;
  return output;
}

DTensor TileGraph::create_tiled_input(std::vector<int> const &dims,
                                      std::vector<int> const &can_split) {
  tile_config.update_config(cleave_superoptimizer::type::KN_INPUT_OP);

  bool has_split_dim = false;
  for (int split_flag : can_split) {
    if (split_flag != 0) {
      has_split_dim = true;
      break;
    }
  }
  assert((tile_config.split_enabled || !has_split_dim) &&
         "TileGraph::create_tiled_input: can_split must be all false when "
         "split_enabled=false. Either set all can_split entries to 0/false, "
         "or create TileGraph with split_enabled=true.");

  std::vector<bool> can_reduce_b(dims.size(), false);
  std::vector<bool> can_split_b(can_split.size(), false);
  for (size_t i = 0; i < can_split.size(); ++i) {
    can_split_b[i] = (can_split[i] != 0);
  }

  return tile_graph.new_input(dims, can_reduce_b, can_split_b);
}

void TileGraph::create_tiled_output(DTensor const &O) {
  tile_config.update_config(cleave_superoptimizer::type::KN_OUTPUT_OP);
  tile_graph.mark_output(O);
}

cleave_superoptimizer::search::GeneratorConfig TileGraph::setup_search_config() {

  cleave_superoptimizer::search::GeneratorConfig config = cleave_superoptimizer::search::GeneratorConfig::get_default_config();
  
  // override the default config with tile-specific settings
  config.search_thread = 16;
  config.max_num_kernel_graph_op = tile_config.max_num_kernel_graph_op;

  config.knop_to_explore = std::vector<cleave_superoptimizer::type::KNOperatorType>(tile_config.knop_to_explore.begin(), tile_config.knop_to_explore.end());
  std::sort(config.knop_to_explore.begin(), config.knop_to_explore.end(),
            [](const cleave_superoptimizer::type::KNOperatorType &a, const cleave_superoptimizer::type::KNOperatorType &b) {
              return static_cast<int>(a) < static_cast<int>(b);
            });
  config.verifier_type = cleave_superoptimizer::search::VerifierType::PROBABILISTIC_VERIFIER;

  return config;
}

std::unique_ptr<cleave_superoptimizer::search::KernelGraphGenerator> TileGraph::search_tile_graph(std::string const &checkpoint_file_name,
                                  bool const &enable_concat_matmul_transformation,
                                  bool const &show_config,
                                  cleave_superoptimizer::search::GeneratorConfig* external_config) {
  require_symbolized_search_graph(symbolic);

  cleave_superoptimizer::search::GeneratorConfig config;
  if (!external_config) {
    config = setup_search_config();
  } else {
    config = *external_config;
  }

  if (enable_concat_matmul_transformation) {
    config.enable_concat_matmul_transformation();
  }

  if (show_config) {
    config.show();
  }
  auto gen = std::make_unique<cleave_superoptimizer::search::KernelGraphGenerator>(tile_graph, config, checkpoint_file_name.data());
  gen->generate_kernel_graphs();
  return gen;
}

int TileGraph::search_tile_graph(std::string const &checkpoint_file_name,
                                 bool const &enable_concat_matmul_transformation,
                                 bool const &show_config,
                                 cleave_superoptimizer::search::GeneratorConfig* external_config,
                                 int max_num_graphs,
                                 cleave_superoptimizer::kernel::Graph **new_graphs) {
  require_symbolized_search_graph(symbolic);

  if (!checkpoint_file_name.empty()) {
    std::ifstream generated_graphs_file(checkpoint_file_name,
                                        std::ifstream::binary);
    if (generated_graphs_file) {
      CSO_LOG(INFO) << "Loading generated graphs from checkpoint file: "
                    << checkpoint_file_name;
      json j;
      generated_graphs_file >> j;
      int num = 0;
      for (json const &graph : j) {
        assert(num < max_num_graphs);
        new_graphs[num] = new cleave_superoptimizer::kernel::Graph();
        from_json(graph, *new_graphs[num]);
        num++;
      }
      return num;
    }
  }

  cleave_superoptimizer::search::GeneratorConfig config;
  if (!external_config) {
    config = setup_search_config();
  } else {
    config = *external_config;
  }

  if (enable_concat_matmul_transformation) {
    config.enable_concat_matmul_transformation();
  }

  if (show_config) {
    config.show();
  }
  cleave_superoptimizer::search::KernelGraphGenerator gen(
      tile_graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs();
  int num = 0;
  for (json const &j : gen.generated_graphs) {
    assert(num < max_num_graphs);
    new_graphs[num] = new cleave_superoptimizer::kernel::Graph();
    from_json(j, *new_graphs[num]);
    num++;
  }
  return num;
}


TileGraph *TileGraph::create_symbolic(
    cleave_superoptimizer::kernel::Graph const &concrete_graph,
    std::unordered_map<int, int> *symbol_map) {

  // Create a symbolic version of the kernel graph
  std::unordered_map<size_t, std::vector<int>> symbolic_dims_by_guid;
  cleave_superoptimizer::kernel::Graph *symbolic_kg =
      concrete_graph.create_symbolic_graph(&symbolic_dims_by_guid);
  log_symbolic_input_mapping(concrete_graph, symbolic_dims_by_guid);
  populate_symbol_mapping(
      concrete_graph, symbolic_dims_by_guid, symbol_map);

  // Create a TileGraph with the symbolic kernel graph
  TileGraph *symbolic_tile_graph =
      new TileGraph(concrete_graph.has_split_input(), /*symbolic=*/true);

  // Copy the symbolic kernel graph into the tile graph
  // Note: We need to reconstruct the tile graph by copying operators
  // since tile_graph is not a pointer but a value member

  // For each operator in the symbolic kernel graph, update tile config
  for (auto const &op : symbolic_kg->operators) {
    symbolic_tile_graph->tile_config.update_config(op->op_type);
  }

  // Replace the tile_graph with the symbolic one
  // We need to manually copy operators and update their graph pointers
  for (auto const &op : symbolic_kg->operators) {
    // Transfer ownership to the tile_graph
    symbolic_tile_graph->tile_graph.operators.push_back(op);
    // Update the operator's graph pointer to the new graph
    op->kgraph = &(symbolic_tile_graph->tile_graph);
  }

  // Clear everything from symbolic_kg so it doesn't try to free/delete anything
  symbolic_kg->operators.clear();
  delete symbolic_kg;

  return symbolic_tile_graph;
}

} // namespace tile
} // namespace cleave_superoptimizer
