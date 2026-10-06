#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/utils/logging.h"

#include <sstream>

namespace cleave_superoptimizer {
namespace search {

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      5 /* max_num_kernel_graph_op */,
      8 /* search_thread */,
      VerifierType::PROBABILISTIC_VERIFIER,
      {
          type::KN_MATMUL_OP,
          type::KN_EXP_OP,
          type::KN_SILU_OP,
          type::KN_GELU_OP,
          type::KN_RELU_OP,
          type::KN_CLAMP_OP,
          type::KN_SPLIT_0_OP,
          type::KN_SPLIT_1_OP,
          type::KN_SPLIT_2_OP,
          type::KN_SPLIT_3_OP,
          type::KN_SPLIT_4_OP,
          type::KN_ADD_OP,
          type::KN_MUL_OP,
          type::KN_DIV_OP,
          // type::KN_REDUCTION_2_OP,
      } /* knop_to_explore */,
      false /* randomized_branches */,
      false /* enable_attention_specific_optimization */,
      false /* enable_concat_matmul_transformation */,
  };
}

void GeneratorConfig::enable_attention_specific_optimization() {
  _enable_attention_specific_optimization = true;
}

void GeneratorConfig::enable_concat_matmul_transformation() {
  _enable_concat_matmul_transformation = true;
}

void GeneratorConfig::show() const {
  std::ostringstream os;
  os << "Search configuration: max_num_kernel_graph_op="
     << max_num_kernel_graph_op << ", search_thread=" << search_thread
     << ", knop_to_explore=[";
  bool first = true;
  for (auto const &knop : knop_to_explore) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << static_cast<int>(knop);
  }
  os << "]";
  CSO_LOG(INFO) << os.str();
}

} // namespace search
} // namespace cleave_superoptimizer
