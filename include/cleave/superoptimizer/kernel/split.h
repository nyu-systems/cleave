#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

constexpr int NUM_SPLITS = 2;

class KNSplitOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNSplitOp(Graph *_graph, DTensor const &input, int split_dim);
  ~KNSplitOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int split_dim_idx;
};

} // namespace kernel
} // namespace cleave_superoptimizer
