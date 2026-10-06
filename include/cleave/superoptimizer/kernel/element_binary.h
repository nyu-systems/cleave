#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

class KNElementBinaryOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNElementBinaryOp(Graph *_graph,
                    DTensor const &input1,
                    DTensor const &input2,
                    cleave_superoptimizer::type::KNOperatorType type);
  KNElementBinaryOp(Graph *_graph,
                    DTensor const &input,
                    float scalar,
                    bool scalar_on_left,
                    cleave_superoptimizer::type::KNOperatorType type);
  ~KNElementBinaryOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  bool has_scalar = false;
  float scalar = 0.0f;
  bool scalar_on_left = false;
};

} // namespace kernel
} // namespace cleave_superoptimizer
