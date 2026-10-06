#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

class KNRMSNormOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNRMSNormOp(Graph *_graph,
              DTensor const &input,
              std::vector<int> const &normalized_shape);
  ~KNRMSNormOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int normalized_size;
};

} // namespace kernel
} // namespace cleave_superoptimizer
