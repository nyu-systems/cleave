#include "cleave/superoptimizer/tile/graph.h"
#include "cleave/superoptimizer/tile/config.h"

namespace cleave_superoptimizer {
namespace tile {

TileSearchConfig::TileSearchConfig(bool infer_num_ops, bool split_enabled)
    : infer_num_ops(infer_num_ops), split_enabled(split_enabled) {
  if (!infer_num_ops) {
    max_num_kernel_graph_op = 13; // TODO: pick a better default value
  } else {
    max_num_kernel_graph_op = 1;
  }
}

TileSearchConfig::~TileSearchConfig() {
  knop_to_explore.clear();
}

void TileSearchConfig::update_config(cleave_superoptimizer::type::KNOperatorType const op_type) {
  switch (op_type) {
    case cleave_superoptimizer::type::KN_INPUT_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      break;

    case cleave_superoptimizer::type::KN_MATMUL_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_MATMUL_OP);
      if (split_enabled) {
        knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_0_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_1_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_2_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_3_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_4_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_0_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_1_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_2_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_3_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_4_OP);
      }
      break;

    case cleave_superoptimizer::type::KN_ADD_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_ADD_OP);
      break;

    case cleave_superoptimizer::type::KN_MUL_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_MUL_OP);
      break;

    case cleave_superoptimizer::type::KN_DIV_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_DIV_OP);
      break;

    case cleave_superoptimizer::type::KN_EXP_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_EXP_OP);
      break;

    case cleave_superoptimizer::type::KN_SILU_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_SILU_OP);
      break;

    case cleave_superoptimizer::type::KN_SQRT_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_SQRT_OP);
      break;

    case cleave_superoptimizer::type::KN_REDUCTION_0_OP:
    case cleave_superoptimizer::type::KN_REDUCTION_1_OP:
    case cleave_superoptimizer::type::KN_REDUCTION_2_OP:
    case cleave_superoptimizer::type::KN_REDUCTION_3_OP:
    case cleave_superoptimizer::type::KN_REDUCTION_4_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_0_OP);
      knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_1_OP);
      knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_2_OP);
      knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_3_OP);
      knop_to_explore.insert(cleave_superoptimizer::type::KN_REDUCTION_4_OP);
      if (split_enabled) {
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_0_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_1_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_2_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_3_OP);
        knop_to_explore.insert(cleave_superoptimizer::type::KN_SPLIT_4_OP);
      }
      break;

    case cleave_superoptimizer::type::KN_OUTPUT_OP:
      if (infer_num_ops) max_num_kernel_graph_op++;
      break;

    default:
      assert(false && "Unsupported operator type for tile search config");
  }
}

} // namespace tile
} // namespace cleave_superoptimizer
