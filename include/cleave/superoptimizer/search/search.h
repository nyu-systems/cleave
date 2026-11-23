#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/search/dim_strategy.h"
#include "cleave/superoptimizer/search/order.h"
#include "cleave/superoptimizer/search/search_context.h"
#include "cleave/superoptimizer/search/verification/verifier.h"
#include "cleave/superoptimizer/utils/json_utils.h"

namespace cleave_superoptimizer {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *filename);

  void generate_kernel_graphs();

  GeneratorConfig config;
  DimStrategy dim_strategy;

  char const *filename;
  // In search-tree order once generate_kernel_graphs() returns.
  std::vector<json> generated_graphs;
  int num_thread;
  bool eager_save_viz_graphs;

private:
  struct ScalarBinarySpec {
    type::KNOperatorType op_type;
    float scalar;
    bool scalar_on_left;
    std::shared_ptr<AbstractExpr> input_pattern;
    // ac_canonical_string(input_pattern), for programs with declared inputs
    std::string input_canonical;
  };

  // Computation graph-related fields
  std::vector<std::shared_ptr<AbstractExpr>> computation_graph_output_patterns;
  std::vector<std::tuple<std::vector<int>,
                         std::vector<bool>,
                         std::vector<bool>,
                         type::DataType,
                         std::shared_ptr<kernel::InputDeclaration const>>>
      computation_graph_input_attrs;
  // The target declares inputs: unread declared inputs are no outputs, and
  // scalar ops compare patterns in AC-canonical form.
  bool has_declared_inputs = false;
  // the output patterns' canonical subterms (has_declared_inputs only)
  std::vector<CanonicalSubterms> computation_graph_output_subterms;
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

  // Shared across OpenMP tasks; guarded by seen_patterns_mutex.
  std::unordered_map<std::string, bool> seen_patterns;
  std::shared_mutex seen_patterns_mutex;

  // Verifier
  std::shared_ptr<Verifier> verifier;

  // Search-tree position of each entry of generated_graphs; appended in the
  // same critical section.
  std::vector<std::vector<int>> generated_graph_paths;
  void sort_generated_graphs();

  // First exception thrown inside an OpenMP region; rethrown after the
  // region ends (an exception escaping a task/parallel region terminates).
  std::exception_ptr search_error;
  std::mutex search_error_mutex;
  std::atomic<bool> search_failed{false};
  void record_search_error(std::exception_ptr e);

  void generate_next_operator(
      SearchContext &c,
      std::function<bool(SearchContext const &)> const &verify,
      size_t depth);

  void preprocess(kernel::Graph const &computation_graph);
  bool check_pattern(std::shared_ptr<AbstractExpr> pattern);
  bool verify(kernel::Graph &g, std::vector<int> const &path);

  void save_results() const;
  double get_elapsed_time_in_sec() const;
  void show_statistics() const;
  void _on_valid_graph() const;

};

} // namespace search
} // namespace cleave_superoptimizer
