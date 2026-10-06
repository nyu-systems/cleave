#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

void Graph::mark_output(DTensor const &A, bool infer_output_reducibility) {
  KNOperator *op = create_output_op(A);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 0);

  if (infer_output_reducibility) {
    infer_reducibility();
  }
}

KNOperator *Graph::create_output_op(DTensor const &A) {
  KNOutputOp *op = new KNOutputOp(this, A);
  return op;
}

KNOutputOp::KNOutputOp(Graph *_kgraph,
                       DTensor const &A,
                       int3 _output_map)
    : KNOperator(_kgraph, cleave_superoptimizer::type::KN_OUTPUT_OP, A),
      output_map(_output_map) {}

KNOutputOp::~KNOutputOp() {}

KNOutputOp::operator json() const {
  return json{{"op_type", op_type},
              {"output_map", output_map},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace cleave_superoptimizer
