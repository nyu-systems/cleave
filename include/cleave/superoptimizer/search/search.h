#pragma once

#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/search/dim_strategy.h"
#include "cleave/superoptimizer/search/order.h"
#include "cleave/superoptimizer/search/range_propagation/irange.h"
#include "cleave/superoptimizer/search/search_context.h"
#include "cleave/superoptimizer/search/search_state_manager.h"
#include "cleave/superoptimizer/search/verification/verifier.h"
#include "cleave/superoptimizer/utils/json_utils.h"

namespace cleave_superoptimizer {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *filename,
                       std::function<void(std::vector<SerializedSearchContext>)>
                           call_back_on_valid_graph = nullptr);

  void generate_kernel_graphs();

  GeneratorConfig config;
  DimStrategy dim_strategy;

  char const *filename;
  std::vector<json> generated_graphs;
  int num_thread;
  std::function<void(std::vector<SerializedSearchContext>)> call_back_on_valid_graph;
  bool eager_save_viz_graphs;

private:
  struct ScalarBinarySpec {
    type::KNOperatorType op_type;
    float scalar;
    bool scalar_on_left;
    std::shared_ptr<AbstractExpr> input_pattern;
  };

  // Computation graph-related fields
  std::vector<std::shared_ptr<AbstractExpr>> computation_graph_output_patterns;
  std::vector<std::tuple<std::vector<int>,
                         std::vector<bool>,
                         std::vector<bool>>>
      computation_graph_input_attrs;
  std::vector<ScalarBinarySpec> scalar_binary_specs;

  // Statistics-related fields
  std::atomic<int> num_total_random_tests;
  std::atomic<int> num_valid_kernel_graphs;
  std::atomic<int> num_total_states;

  // Time
  std::chrono::time_point<std::chrono::steady_clock> start_time;

  // count number of tasks
  std::atomic<int> num_tasks;
  size_t max_depth;

  //
  std::unordered_map<std::string, bool> seen_patterns;

  // Ranges-related fields
  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  std::vector<std::vector<IKNRange>> target_ranges;

  // Verifier
  std::shared_ptr<Verifier> verifier;

  void generate_next_operator(
      SearchContext &c,
      std::function<bool(SearchContext const &)> const &verify,
      std::vector<SerializedSearchContext> &verified,
      size_t depth);

  void preprocess(kernel::Graph const &computation_graph);
  bool check_pattern(std::shared_ptr<AbstractExpr> pattern);
  bool verify(kernel::Graph &g);

  void save_results() const;
  double get_elapsed_time_in_sec() const;
  void show_statistics() const;
  void _on_valid_graph(std::vector<SerializedSearchContext> verified_graphs) const;

};

} // namespace search
} // namespace cleave_superoptimizer
