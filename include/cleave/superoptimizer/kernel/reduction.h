#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

class KNReductionOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNReductionOp(Graph *_graph, DTensor const &input, int dim, int size);
  ~KNReductionOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int reduction_dim_idx, reduction_dim_size;
};

} // namespace kernel
} // namespace cleave_superoptimizer
