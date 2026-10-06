#include "cleave/superoptimizer/kernel/rms_norm.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

using namespace cleave_superoptimizer::type;

DTensor Graph::rms_norm(DTensor const &input,
                        std::vector<int> const &normalized_shape) {
  KNOperator *op = create_rms_norm_op(input, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::rms_norm(DTensor const *input,
                         std::vector<int> const &normalized_shape) {
  KNOperator *op = create_rms_norm_op(*input, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

DTensor Graph::rms_norm(DTensor const &input,
                        DTensor const &elementwise_affine,
                        std::vector<int> const &normalized_shape) {
  KNOperator *op =
      create_rms_norm_op(input, elementwise_affine, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::rms_norm(DTensor const *input,
                         DTensor const *elementwise_affine,
                         std::vector<int> const &normalized_shape) {
  KNOperator *op =
      create_rms_norm_op(*input, *elementwise_affine, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_rms_norm_op(DTensor const &input,
                              std::vector<int> const &normalized_shape) {
  int num_norm_dims = normalized_shape.size();
  if (num_norm_dims > input.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < num_norm_dims; i++) {
    if (normalized_shape[i] != input.dim[input.num_dims - num_norm_dims + i]) {
      return nullptr;
    }
  }

  KNRMSNormOp *op = new KNRMSNormOp(this, input, normalized_shape);
  return op;
}

KNOperator *
    Graph::create_rms_norm_op(DTensor const &input,
                              DTensor const &elementwise_affine,
                              std::vector<int> const &normalized_shape) {
  // We currently only allow normalizing the last dimension
  assert(false && "To be implemented");
}

KNRMSNormOp::KNRMSNormOp(Graph *_kgraph,
                         DTensor const &input,
                         std::vector<int> const &normalized_shape)
    : KNOperator(_kgraph, KN_RMS_NORM_OP, input) {
  normalized_size = 1;
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    normalized_size *= normalized_shape[i];
  }
  DTensor output = input;
  for (int i = 0; i < output.num_dims; i++) {
    output.can_reduce[i] = input.can_reduce[i];
    output.can_split[i] = input.can_split[i];
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  output_tensors.push_back(output);
}

KNRMSNormOp::~KNRMSNormOp() {}

KNRMSNormOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
