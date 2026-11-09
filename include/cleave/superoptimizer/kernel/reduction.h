#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

class KNReduceSumOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNReduceSumOp(Graph *_graph, DTensor const &input, int dim, int size);
  ~KNReduceSumOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int reduction_dim_idx, reduction_dim_size;
};

// Opaque: fingerprinted by hashing each reduced slice, never rewritten.
class KNReduceMaxOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNReduceMaxOp(Graph *_graph, DTensor const &input, int dim);
  bool fingerprint(void) override;

  operator json() const override;

public:
  int reduction_dim_idx;
};

} // namespace kernel
} // namespace cleave_superoptimizer
