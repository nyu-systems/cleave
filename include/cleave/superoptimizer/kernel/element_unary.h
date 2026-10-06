#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

class KNElementUnaryOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNElementUnaryOp(Graph *_graph,
                   DTensor const &input,
                   cleave_superoptimizer::type::KNOperatorType type);
  ~KNElementUnaryOp();
  bool fingerprint(void) override;

  operator json() const override;
};

class KNClampUnaryOp : public KNElementUnaryOp {
public:
  KNClampUnaryOp(Graph *_graph,
                 DTensor const &input,
                 float min_val,
                 float max_val);
  
public:
  float min_val;
  float max_val;
};

} // namespace kernel
} // namespace cleave_superoptimizer
