#pragma once

#include "cleave/superoptimizer/kernel/operator.h"

namespace cleave_superoptimizer {
namespace kernel {

struct MatmulDims {
  int a_row_idx;
  int a_reduce_idx;
  int b_col_idx;
  int b_reduce_idx;
};

MatmulDims get_matmul_dims(DTensor const &A,
                           DTensor const &B,
                           bool trans_A,
                           bool trans_B);

class KNMatmulOp : public cleave_superoptimizer::kernel::KNOperator {
public:
  KNMatmulOp(Graph *_graph,
             DTensor const &A,
             DTensor const &B,
             bool trans_A = false,
             bool trans_B = false);
  ~KNMatmulOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  bool trans_A;
  bool trans_B;
};

void from_json(json const &j, KNMatmulOp &op);

class MatmulKey {
public:
  MatmulKey(DTensor const &A, DTensor const &B, bool trans_A, bool trans_B);
  bool operator==(MatmulKey const &b) const;
  DTensor operand_a;
  DTensor operand_b;
  bool trans_A;
  bool trans_B;
};

} // namespace kernel
} // namespace cleave_superoptimizer

namespace std {
template <>
struct hash<cleave_superoptimizer::kernel::MatmulKey> {
  size_t operator()(cleave_superoptimizer::kernel::MatmulKey const &) const;
};
} // namespace std
