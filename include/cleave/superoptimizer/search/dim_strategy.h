#pragma once

#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/search/op_utils.h"

namespace cleave_superoptimizer {
namespace search {

struct DimStrategy {
  DimStrategy() = default;
  DimStrategy(GeneratorConfig const &config);

  std::vector<type::KNOperatorType> get_knop_cand();
  std::vector<std::vector<int>> get_unary_input(int num_tensors);
  std::vector<std::vector<int>> get_binary_input(int num_tensors, bool is_commutative);
  std::vector<std::vector<int>> get_nary_input(int num_tensors, int n);

  template <typename OpType, typename TensorType>
  std::vector<std::vector<int>>
      get_input_cand_idx(OpType op_type,
                         std::vector<TensorType> const &all_inputs) {
    if (is_unary(op_type)) {
      return get_unary_input(all_inputs.size());
    }
    if (is_binary(op_type)) {
      return get_binary_input(all_inputs.size(), is_commutative(op_type));
    }
    return get_nary_input(all_inputs.size(), get_input_number(op_type));
    assert(false && "Unsupported operator");
  }
  GeneratorConfig config;
};

} // namespace search
} // namespace cleave_superoptimizer
