#include "cleave/superoptimizer/kernel/reduction.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

using namespace cleave_superoptimizer::type;

DTensor Graph::reduction(DTensor const &input, int dim, int size) {
  KNOperator *op = create_reduction_op(input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::reduction(DTensor const *input, int dim, int size) {
  KNOperator *op = create_reduction_op(*input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_reduction_op(DTensor const &input, int dim, int size) {
  if (input.num_dims <= dim) {
    return nullptr;
  }
  if (input.dim[dim] % size != 0) {
    return nullptr;
  }

  KNReductionOp *op = new KNReductionOp(this, input, dim, size);
  return op;
}

KNReductionOp::KNReductionOp(Graph *_kgraph,
                             DTensor const &input,
                             int dim,
                             int size)
    : KNOperator(_kgraph, (KNOperatorType)(KN_REDUCTION_0_OP + dim), input),
      reduction_dim_idx(dim), reduction_dim_size(size) {
  DTensor output = input;
  assert(dim < output.num_dims);
  assert(output.dim[dim] % size == 0);
  output.dim[dim] = size;
  for (int i = 0; i < reduction_dim_idx; i++) {
    output.can_reduce[i] = input.can_reduce[i];
    output.can_split[i] = input.can_split[i];
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNReductionOp::~KNReductionOp() {}

KNReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
