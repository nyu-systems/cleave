#include "cleave/superoptimizer/kernel/split.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

using namespace cleave_superoptimizer::type;

DTensor Graph::split(DTensor const &input, int dim) {
  KNOperator *op = create_split_op(input, dim);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::split(DTensor const *input, int dim) {
  KNOperator *op = create_split_op(*input, dim);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_split_op(DTensor const &input, int dim) {
  if (input.num_dims + 1 > MAX_TENSOR_DIMS) {
    return nullptr;
  }
  if (dim < 0 || dim >= input.num_dims) {
    return nullptr;
  }
  if (input.dim[dim] % NUM_SPLITS != 0) {
    return nullptr;
  }
  KNSplitOp *op = new KNSplitOp(this, input, dim);
  return op;
}

KNSplitOp::KNSplitOp(Graph *_kgraph, DTensor const &input, int split_dim)
    : KNOperator(_kgraph, (KNOperatorType)(KN_SPLIT_0_OP + split_dim), input),
      split_dim_idx(split_dim) {
  DTensor output = input;
  bool input_can_reduce[MAX_TENSOR_DIMS];
  bool input_can_split[MAX_TENSOR_DIMS];
  for (int i = 0; i < input.num_dims; i++) {
    input_can_reduce[i] = input.can_reduce[i];
    input_can_split[i] = input.can_split[i];
  }
  output.num_dims = input.num_dims + 1;
  output.dim[0] = NUM_SPLITS;
  output.can_reduce[0] = true;
  output.can_split[0] = false;
  for (int i = 0; i < input.num_dims; i++) {
    int out_idx = i + 1;
    output.dim[out_idx] = input.dim[i];
    output.can_reduce[out_idx] = input_can_reduce[i];
    output.can_split[out_idx] = input_can_split[i];
    if (i == split_dim_idx) {
      output.dim[out_idx] = input.dim[i] / NUM_SPLITS;
      output.can_split[out_idx] = false; // prevent further splits on this dim
    }
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNSplitOp::~KNSplitOp() {}

KNSplitOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
