#include "cleave/superoptimizer/kernel/element_unary.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

DTensor Graph::exp(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_EXP_OP);
}

DTensor *Graph::exp(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_EXP_OP);
}

DTensor Graph::square(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SQUARE_OP);
}

DTensor *Graph::square(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SQUARE_OP);
}

DTensor Graph::sqrt(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SQRT_OP);
}

DTensor *Graph::sqrt(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SQRT_OP);
}

DTensor Graph::silu(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SILU_OP);
}

DTensor *Graph::silu(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_SILU_OP);
}

DTensor Graph::gelu(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_GELU_OP);
}

DTensor *Graph::gelu(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_GELU_OP);
}

DTensor Graph::relu(DTensor const &input) {
  return elementunary(input, cleave_superoptimizer::type::KN_RELU_OP);
}

DTensor *Graph::relu(DTensor const *input) {
  return elementunary(input, cleave_superoptimizer::type::KN_RELU_OP);
}

DTensor Graph::clamp(DTensor const &input,
                     float const &min_val,
                     float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor *Graph::clamp(DTensor const *input,
                      float const &min_val,
                      float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor Graph::elementunary_clamp(DTensor const &input,
                                  float const &min_val, 
                                  float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary_clamp(DTensor const *input,
                                   float const &min_val, 
                                   float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(*input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}


KNOperator *Graph::create_elementunary_clamp_op(DTensor const &input,
                                                float const &min_val, 
                                                float const &max_val) {
  KNElementUnaryOp *op = new KNClampUnaryOp(this, input, min_val, max_val);

  return op;
}

DTensor Graph::elementunary(DTensor const &input,
                            cleave_superoptimizer::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary(DTensor const *input,
                             cleave_superoptimizer::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_op(DTensor const &input,
                                          cleave_superoptimizer::type::KNOperatorType type) {
  KNElementUnaryOp *op = new KNElementUnaryOp(this, input, type);

  return op;
}

KNClampUnaryOp::KNClampUnaryOp(Graph *_kgraph,
               DTensor const &input,
               float min_val,
               float max_val)
    : KNElementUnaryOp(_kgraph, input, cleave_superoptimizer::type::KN_CLAMP_OP),
      min_val(min_val), max_val(max_val) {}

KNElementUnaryOp::KNElementUnaryOp(Graph *_kgraph,
                                   DTensor const &input,
                                   cleave_superoptimizer::type::KNOperatorType type)
    : cleave_superoptimizer::kernel::KNOperator(_kgraph, type, input) {
  DTensor output = input;
  for (int i = 0; i < output.num_dims; i++) {
    output.can_reduce[i] = input.can_reduce[i];
    output.can_split[i] = input.can_split[i];
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementUnaryOp::~KNElementUnaryOp() {}

KNElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
