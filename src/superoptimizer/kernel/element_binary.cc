#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <cassert>

namespace cleave_superoptimizer {
namespace kernel {

DTensor Graph::add(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_ADD_OP);
}

DTensor Graph::add(DTensor const &input, float scalar) {
  return elementbinary(input, scalar, false, cleave_superoptimizer::type::KN_ADD_OP);
}

DTensor *Graph::add(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_ADD_OP);
}

DTensor Graph::mul(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_MUL_OP);
}

DTensor Graph::mul(DTensor const &input, float scalar) {
  return elementbinary(input, scalar, false, cleave_superoptimizer::type::KN_MUL_OP);
}

DTensor *Graph::mul(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_MUL_OP);
}

DTensor Graph::div(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_DIV_OP);
}

DTensor Graph::div(DTensor const &input, float scalar) {
  return elementbinary(input, scalar, false, cleave_superoptimizer::type::KN_DIV_OP);
}

DTensor Graph::div(float scalar, DTensor const &input) {
  return elementbinary(input, scalar, true, cleave_superoptimizer::type::KN_DIV_OP);
}

DTensor *Graph::div(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, cleave_superoptimizer::type::KN_DIV_OP);
}

DTensor Graph::elementbinary(DTensor const &input1,
                             DTensor const &input2,
                             cleave_superoptimizer::type::KNOperatorType type) {
  KNOperator *op = create_elementbinary_op(input1, input2, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor Graph::elementbinary(DTensor const &input,
                             float scalar,
                             bool scalar_on_left,
                             cleave_superoptimizer::type::KNOperatorType type) {
  KNOperator *op =
      create_elementbinary_op(input, scalar, scalar_on_left, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementbinary(DTensor const *input1,
                              DTensor const *input2,
                              cleave_superoptimizer::type::KNOperatorType type) {
  DTensor output = elementbinary(*input1, *input2, type);
  return &(output.owner_op->output_tensors[0]);
}

KNOperator *Graph::create_elementbinary_op(DTensor const &input1,
                                           DTensor const &input2,
                                           cleave_superoptimizer::type::KNOperatorType type) {
  // DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (input1.num_dims != input2.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i] && input1.dim[i] > 1 &&
        input2.dim[i] > 1) {
      return nullptr;
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }

  KNElementBinaryOp *op = new KNElementBinaryOp(this, input1, input2, type);
  return op;
}

KNOperator *Graph::create_elementbinary_op(DTensor const &input,
                                           float scalar,
                                           bool scalar_on_left,
                                           cleave_superoptimizer::type::KNOperatorType type) {
  KNElementBinaryOp *op =
      new KNElementBinaryOp(this, input, scalar, scalar_on_left, type);
  return op;
}

KNElementBinaryOp::KNElementBinaryOp(Graph *_kgraph,
                                     DTensor const &input1,
                                     DTensor const &input2,
                                     cleave_superoptimizer::type::KNOperatorType type)
    : cleave_superoptimizer::kernel::KNOperator(_kgraph, type, input1, input2) {
  assert(input1.num_dims == input2.num_dims);
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i]) {
      assert(input1.dim[i] == 1 || input2.dim[i] == 1);
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
    if (input1.dim[i] == 1 && input2.dim[i] != 1) {
      output.can_reduce[i] = input2.can_reduce[i];
      output.can_split[i] = input2.can_split[i];
    } else if (input2.dim[i] == 1 && input1.dim[i] != 1) {
      output.can_reduce[i] = input1.can_reduce[i];
      output.can_split[i] = input1.can_split[i];
    } else {
      output.can_reduce[i] = input1.can_reduce[i] && input2.can_reduce[i];
      output.can_split[i] = input1.can_split[i] && input2.can_split[i];
    }
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementBinaryOp::KNElementBinaryOp(Graph *_kgraph,
                                     DTensor const &input,
                                     float scalar,
                                     bool scalar_on_left,
                                     cleave_superoptimizer::type::KNOperatorType type)
    : cleave_superoptimizer::kernel::KNOperator(_kgraph, type, input) {
  has_scalar = true;
  this->scalar = scalar;
  this->scalar_on_left = scalar_on_left;
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementBinaryOp::~KNElementBinaryOp() {}

KNElementBinaryOp::operator json() const {
  json j{{"op_type", op_type},
         {"input_tensors", input_tensors},
         {"output_tensors", output_tensors}};
  if (has_scalar) {
    j["scalar"] = scalar;
    j["scalar_on_left"] = scalar_on_left;
  }
  return j;
}

} // namespace kernel
} // namespace cleave_superoptimizer
