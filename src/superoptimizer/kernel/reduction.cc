#include "cleave/superoptimizer/kernel/reduction.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

using namespace cleave_superoptimizer::type;

DTensor Graph::reduce_sum(DTensor const &input, int dim, int size) {
  KNOperator *op = create_reduce_sum_op(input, dim, size);
  CSO_REQUIRE(op != nullptr, std::invalid_argument,
              __func__ << ": incompatible operand shapes or attributes");
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::reduce_sum(DTensor const *input, int dim, int size) {
  KNOperator *op = create_reduce_sum_op(*input, dim, size);
  CSO_REQUIRE(op != nullptr, std::invalid_argument,
              __func__ << ": incompatible operand shapes or attributes");
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_reduce_sum_op(DTensor const &input, int dim, int size) {
  if (dim < 0 || size <= 0 || input.num_dims <= dim) {
    return nullptr;
  }
  if (input.dim[dim] % size != 0) {
    return nullptr;
  }

  KNReduceSumOp *op = new KNReduceSumOp(this, input, dim, size);
  return op;
}

KNReduceSumOp::KNReduceSumOp(Graph *_kgraph,
                             DTensor const &input,
                             int dim,
                             int size)
    : KNOperator(_kgraph, (KNOperatorType)(KN_REDUCE_SUM_0_OP + dim), input),
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

KNReduceSumOp::~KNReduceSumOp() {}

KNReduceSumOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

DTensor Graph::reduce_max(DTensor const &input, int dim) {
  KNOperator *op = create_reduce_max_op(input, dim);
  CSO_REQUIRE(op != nullptr, std::invalid_argument,
              __func__ << ": incompatible operand shapes or attributes");
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor *Graph::reduce_max(DTensor const *input, int dim) {
  KNOperator *op = create_reduce_max_op(*input, dim);
  CSO_REQUIRE(op != nullptr, std::invalid_argument,
              __func__ << ": incompatible operand shapes or attributes");
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_reduce_max_op(DTensor const &input, int dim) {
  if (dim < 0 || dim >= input.num_dims || dim >= MAX_TENSOR_DIMS) {
    return nullptr;
  }
  return new KNReduceMaxOp(this, input, dim);
}

KNReduceMaxOp::KNReduceMaxOp(Graph *_kgraph,
                                   DTensor const &input,
                                   int dim)
    : KNOperator(_kgraph, (KNOperatorType)(KN_REDUCE_MAX_0_OP + dim), input),
      reduction_dim_idx(dim) {
  DTensor output = input;
  output.dim[dim] = 1;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  output_tensors.push_back(output);
}

KNReduceMaxOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
