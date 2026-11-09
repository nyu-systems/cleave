#include "cleave/superoptimizer/search/search_c.h"
#include "cleave/superoptimizer/search/search.h"
#include "cleave/superoptimizer/utils/logging.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace cleave_superoptimizer {
namespace search_c {

std::vector<kernel::Graph *> search_graphs(kernel::Graph const *input_graph,
                                           char const *filename,
                                           char const *default_config) {
  CSO_REQUIRE(input_graph != nullptr, std::invalid_argument,
              "search: input graph is null");
  std::vector<json> graphs_json;
  bool loaded = false;
  if (filename) {
    std::ifstream generated_graphs_file(filename, std::ifstream::binary);
    if (generated_graphs_file) {
      json j;
      generated_graphs_file >> j;
      for (json const &graph : j) {
        graphs_json.push_back(graph);
      }
      loaded = true;
    }
  }

  if (!loaded) {
    search::GeneratorConfig config =
        search::GeneratorConfig::get_default_config();
    if (default_config != nullptr) {
      if (!strcmp(default_config, "attention")) {
        config.enable_attention_specific_optimization();
      } else if (!strcmp(default_config, "lora")) {
        config.enable_concat_matmul_transformation();
      } else if (!strcmp(default_config, "mlp")) {
      }
    }
    char const *result_filename =
        filename ? filename : "cleave_superoptimizer_search_checkpoint.json";
    search::KernelGraphGenerator gen(*input_graph, config, result_filename);
    gen.config.show();
    gen.generate_kernel_graphs();
    graphs_json = std::move(gen.generated_graphs);
  }

  std::vector<std::unique_ptr<kernel::Graph>> owned;
  for (json const &j : graphs_json) {
    owned.emplace_back(new kernel::Graph());
    from_json(j, *owned.back());
  }
  std::vector<kernel::Graph *> result;
  for (auto &g : owned) {
    result.push_back(g.release());
  }
  return result;
}

int cython_search(kernel::Graph const *input_graph,
                  int max_num_graphs,
                  kernel::Graph **new_graphs,
                  char const *filename,
                  char const *default_config) {
  std::vector<kernel::Graph *> graphs =
      search_graphs(input_graph, filename, default_config);
  if (static_cast<int>(graphs.size()) > max_num_graphs) {
    for (auto *g : graphs) {
      delete g;
    }
    CSO_THROW(std::runtime_error,
              "Search produced " << graphs.size()
                                 << " graphs, more than max_num_graphs="
                                 << max_num_graphs);
  }
  std::copy(graphs.begin(), graphs.end(), new_graphs);
  return static_cast<int>(graphs.size());
}

} // namespace search_c
} // namespace cleave_superoptimizer
