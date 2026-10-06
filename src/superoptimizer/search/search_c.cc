#include "cleave/superoptimizer/search/search_c.h"
#include "cleave/superoptimizer/search/search.h"

#include <fstream>

namespace cleave_superoptimizer {
namespace search_c {

int cython_search(cleave_superoptimizer::kernel::Graph const *input_graph,
                  int max_num_graphs,
                  cleave_superoptimizer::kernel::Graph **new_graphs,
                  char const *filename,
                  char const *default_config) {
  if (filename) {
    std::ifstream generated_graphs_file(filename, std::ifstream::binary);
    if (generated_graphs_file) {
      json j;
      generated_graphs_file >> j;
      int num = 0;
      for (json const &graph : j) {
        assert(num < max_num_graphs);
        new_graphs[num] = new kernel::Graph();
        from_json(graph, *new_graphs[num]);
        num++;
      }
      return num;
    }
  }

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
  search::KernelGraphGenerator gen(
      *input_graph, config, result_filename);
  gen.config.show();
  gen.generate_kernel_graphs();
  int num = 0;
  for (json const &j : gen.generated_graphs) {
    assert(num < max_num_graphs);
    new_graphs[num] = new kernel::Graph();
    from_json(j, *new_graphs[num]);
    num++;
  }
  return num;
}

} // namespace search_c
} // namespace cleave_superoptimizer
