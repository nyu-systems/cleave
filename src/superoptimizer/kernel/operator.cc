#include "cleave/superoptimizer/kernel/operator.h"
#include "cleave/superoptimizer/kernel/graph.h"

namespace cleave_superoptimizer {
namespace kernel {

KNOperator::KNOperator(Graph *_graph, cleave_superoptimizer::type::KNOperatorType _type)
    : kgraph(_graph), op_type(_type) {}

KNOperator::KNOperator(Graph *_graph,
                       cleave_superoptimizer::type::KNOperatorType _type,
                       DTensor const &A)
    : kgraph(_graph), op_type(_type) {
  input_tensors.push_back(A);
}

KNOperator::KNOperator(Graph *_graph,
                       cleave_superoptimizer::type::KNOperatorType _type,
                       DTensor const &A,
                       DTensor const &B)
    : kgraph(_graph), op_type(_type) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

KNOperator::KNOperator(Graph *_graph,
                       cleave_superoptimizer::type::KNOperatorType _type,
                       std::vector<DTensor> const &inputs)
    : kgraph(_graph), op_type(_type) {
  for (auto const &i : inputs) {
    input_tensors.push_back(i);
  }
}

KNOperator::~KNOperator() {}

int KNOperator::get_input_dtensors(DTensor **inputs) {
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    inputs[i] = &input_tensors[i];
  }
  return input_tensors.size();
}

int KNOperator::get_output_dtensors(DTensor **outputs) {
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    outputs[i] = &output_tensors[i];
  }
  return output_tensors.size();
}
} // namespace kernel
} // namespace cleave_superoptimizer
