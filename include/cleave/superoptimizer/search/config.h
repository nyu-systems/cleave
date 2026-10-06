#pragma once

#include <vector>

#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/type.h"

namespace cleave_superoptimizer {
namespace search {

using kernel::DTensor;
using kernel::KNOperator;

enum class VerifierType {
  PROBABILISTIC_VERIFIER,
  FORMAL_VERIFIER,
};

struct GeneratorConfig {
  size_t max_num_kernel_graph_op;
  size_t search_thread;

  VerifierType verifier_type;

  std::vector<type::KNOperatorType> knop_to_explore;
  bool
      randomized_branches; // Only for developers to tune the search performance
  bool _enable_attention_specific_optimization;
  bool _enable_concat_matmul_transformation;

  void show() const;
  void enable_attention_specific_optimization();
  void enable_concat_matmul_transformation();

  static GeneratorConfig get_default_config();
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneratorConfig,
                                   max_num_kernel_graph_op,
                                   search_thread,
                                   knop_to_explore,
                                   _enable_attention_specific_optimization,
                                   _enable_concat_matmul_transformation);

} // namespace search
} // namespace cleave_superoptimizer
