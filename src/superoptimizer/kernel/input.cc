#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/kernel/operator.h"
#include "cleave/superoptimizer/utils/logging.h"

#include <stdexcept>

namespace cleave_superoptimizer {
namespace kernel {

namespace {

void check_input_attrs(std::vector<int> const &dims,
                       std::vector<bool> const &can_reduce,
                       std::vector<bool> const &can_split) {
  CSO_REQUIRE(dims.size() <= static_cast<size_t>(MAX_TENSOR_DIMS),
              std::invalid_argument,
              "Input tensors support at most " << MAX_TENSOR_DIMS
                                               << " dims, got "
                                               << dims.size());
  CSO_REQUIRE((can_reduce.empty() || can_reduce.size() == dims.size()) &&
                  (can_split.empty() || can_split.size() == dims.size()),
              std::invalid_argument,
              "can_reduce/can_split length must match dims length "
                  << dims.size());
}

} // namespace

DTensor Graph::new_input(std::vector<int> const &dims,
                         std::vector<bool> const &can_reduce,
                         std::vector<bool> const &can_split,
                         type::DataType dtype) {
  KNOperator *op = create_input_op(dims, can_reduce, can_split, dtype);
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor *Graph::new_input_ptr(std::vector<int> const &dims,
                              std::vector<bool> const &can_reduce,
                              std::vector<bool> const &can_split,
                              type::DataType dtype) {
  KNOperator *op = create_input_op(dims, can_reduce, can_split, dtype);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_input_op(std::vector<int> const &dims,
                                   std::vector<bool> const &can_reduce,
                                   std::vector<bool> const &can_split,
                                   type::DataType dtype) {
  KNInputOp *op = new KNInputOp(this, dims, can_reduce, can_split);
  op->output_tensors[0].dtype = dtype;
  return op;
}

DTensor Graph::new_input(std::vector<int> const &dims) {
  return new_input(dims, {}, {});
}

DTensor *Graph::new_input_ptr(std::vector<int> const &dims) {
  return new_input_ptr(dims, {}, {});
}

KNInputOp::KNInputOp(Graph *_graph,
                     std::vector<int> const &dims,
                     std::vector<bool> const &can_reduce,
                     std::vector<bool> const &can_split,
                     int3 _input_map)
    : KNOperator(_graph, cleave_superoptimizer::type::KN_INPUT_OP),
      input_map(_input_map) {
  check_input_attrs(dims, can_reduce, can_split);
  // Allow empty vectors - will be filled with defaults
  std::vector<bool> final_reduce = can_reduce;
  std::vector<bool> final_split = can_split;

  if (final_reduce.empty()) {
    final_reduce.assign(dims.size(), false);
  }
  if (final_split.empty()) {
    final_split.assign(dims.size(), false);
  }

  DTensor tensor;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
    tensor.can_reduce[i] = final_reduce[i];
    tensor.can_split[i] = final_split[i];
  }
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.guid = DTensor::next_guid++;
  output_tensors.push_back(tensor);
}

KNInputOp::~KNInputOp() {}

KNInputOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_map", input_map},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
