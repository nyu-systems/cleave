#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <algorithm>
#include <cassert>
#include <iostream>

namespace cleave_superoptimizer {
namespace kernel {

namespace {

bool has_compatible_batch_dim(int a_dim, int b_dim) {
  if (a_dim <= 0 || b_dim <= 0) {
    return false;
  }
  int divisor_dim = std::min(a_dim, b_dim);
  int dividend_dim = std::max(a_dim, b_dim);
  return dividend_dim % divisor_dim == 0;
}

} // namespace

MatmulDims get_matmul_dims(DTensor const &A,
                           DTensor const &B,
                           bool trans_A,
                           bool trans_B) {
  MatmulDims dims;
  dims.a_row_idx = trans_A ? A.num_dims - 1 : A.num_dims - 2;
  dims.a_reduce_idx = trans_A ? A.num_dims - 2 : A.num_dims - 1;
  dims.b_col_idx = trans_B ? B.num_dims - 2 : B.num_dims - 1;
  dims.b_reduce_idx = trans_B ? B.num_dims - 1 : B.num_dims - 2;
  return dims;
}

DTensor Graph::matmul(DTensor const &A,
                      DTensor const &B,
                      bool trans_A,
                      bool trans_B) {
  KNOperator *op = create_matmul_op(A, B, trans_A, trans_B);
  assert(op != nullptr);
  operators.push_back(op);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::matmul(DTensor const *A,
                       DTensor const *B,
                       bool trans_A,
                       bool trans_B) {
  KNOperator *op = create_matmul_op(*A, *B, trans_A, trans_B);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_matmul_op(DTensor const &A,
                                    DTensor const &B,
                                    bool trans_A,
                                    bool trans_B) {
  if (A.num_dims < 2) {
    return nullptr;
  }
  if (B.num_dims < 2) {
    return nullptr;
  }
  int num_dims = std::max(A.num_dims, B.num_dims);
  if (num_dims > MAX_TENSOR_DIMS) {
    return nullptr;
  }
  MatmulDims dims = get_matmul_dims(A, B, trans_A, trans_B);
  if (A.dim[dims.a_reduce_idx] != B.dim[dims.b_reduce_idx]) {
    return nullptr;
  }
  int a_offset = num_dims - A.num_dims;
  int b_offset = num_dims - B.num_dims;
  for (int i = 0; i < num_dims - 2; i++) {
    int a_dim = i < a_offset ? 1 : A.dim[i - a_offset];
    int b_dim = i < b_offset ? 1 : B.dim[i - b_offset];
    if (!has_compatible_batch_dim(a_dim, b_dim)) {
      return nullptr;
    }
  }

  DTensor C;
  C.num_dims = num_dims;
  for (int i = 0; i < C.num_dims - 2; i++) {
    int a_dim = i < a_offset ? 1 : A.dim[i - a_offset];
    int b_dim = i < b_offset ? 1 : B.dim[i - b_offset];
    C.dim[i] = std::max(a_dim, b_dim);
  }
  C.dim[C.num_dims - 2] = A.dim[dims.a_row_idx];
  C.dim[C.num_dims - 1] = B.dim[dims.b_col_idx];

  KNMatmulOp *op = new KNMatmulOp(this, A, B, trans_A, trans_B);
  return op;
}

KNMatmulOp::KNMatmulOp(Graph *_kgraph,
                       DTensor const &A,
                       DTensor const &B,
                       bool trans_A_,
                       bool trans_B_)
    : KNOperator(_kgraph, cleave_superoptimizer::type::KN_MATMUL_OP, A, B),
      trans_A(trans_A_),
      trans_B(trans_B_) {
  DTensor C;
  assert(A.num_dims >= 2);
  assert(B.num_dims >= 2);
  int num_dims = std::max(A.num_dims, B.num_dims);
  assert(num_dims <= MAX_TENSOR_DIMS);
  MatmulDims dims = get_matmul_dims(A, B, trans_A, trans_B);
  assert(A.dim[dims.a_reduce_idx] == B.dim[dims.b_reduce_idx]);
  int a_offset = num_dims - A.num_dims;
  int b_offset = num_dims - B.num_dims;
  auto get_dim = [](DTensor const &tensor, int offset, int idx) {
    return idx < offset ? 1 : tensor.dim[idx - offset];
  };
  for (int i = 0; i < num_dims - 2; i++) {
    int a_dim = get_dim(A, a_offset, i);
    int b_dim = get_dim(B, b_offset, i);
    assert(has_compatible_batch_dim(a_dim, b_dim));
  }

  auto get_default = [](const bool *flags, int idx, bool default_value) {
    if (idx < 0) {
      return default_value;
    }
    return flags[idx];
  };
  C.num_dims = num_dims;
  for (int i = 0; i < C.num_dims - 2; i++) {
    int a_dim = get_dim(A, a_offset, i);
    int b_dim = get_dim(B, b_offset, i);
    int a_dim_idx = i - a_offset;
    int b_dim_idx = i - b_offset;
    C.dim[i] = std::max(a_dim, b_dim);
    if (a_dim < b_dim) {
      assert(b_dim_idx >= 0);
      C.can_reduce[i] = B.can_reduce[b_dim_idx];
      C.can_split[i] = B.can_split[b_dim_idx];
    } else if (b_dim < a_dim) {
      assert(a_dim_idx >= 0);
      C.can_reduce[i] = A.can_reduce[a_dim_idx];
      C.can_split[i] = A.can_split[a_dim_idx];
    } else {
      C.can_reduce[i] = get_default(A.can_reduce, a_dim_idx, true) && get_default(B.can_reduce, b_dim_idx, true);
      C.can_split[i] = get_default(A.can_split, a_dim_idx, true) && get_default(B.can_split, b_dim_idx, true);
    }
  }
  C.dim[C.num_dims - 2] = A.dim[dims.a_row_idx];
  C.dim[C.num_dims - 1] = B.dim[dims.b_col_idx];
  C.can_reduce[C.num_dims - 2] = A.can_reduce[dims.a_row_idx];
  C.can_split[C.num_dims - 2] = A.can_split[dims.a_row_idx];
  C.can_reduce[C.num_dims - 1] = B.can_reduce[dims.b_col_idx];
  C.can_split[C.num_dims - 1] = B.can_split[dims.b_col_idx];
  C.owner_op = this;
  C.owner_ts_idx = 0;
  C.guid = DTensor::next_guid++;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

KNMatmulOp::~KNMatmulOp() {}

KNMatmulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"trans_A", trans_A},
              {"trans_B", trans_B}};
}

void from_json(json const &j, KNMatmulOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  if (j.contains("trans_A")) {
    j.at("trans_A").get_to(op.trans_A);
  } else {
    op.trans_A = false;
  }
  if (j.contains("trans_B")) {
    j.at("trans_B").get_to(op.trans_B);
  } else {
    op.trans_B = false;
  }
}

MatmulKey::MatmulKey(DTensor const &A,
                     DTensor const &B,
                     bool trans_A_,
                     bool trans_B_)
    : operand_a(A), operand_b(B), trans_A(trans_A_), trans_B(trans_B_) {}

bool MatmulKey::operator==(MatmulKey const &b) const {
  if (b.operand_a != operand_a) {
    return false;
  }
  if (b.operand_b != operand_b) {
    return false;
  }
  if (b.trans_A != trans_A) {
    return false;
  }
  if (b.trans_B != trans_B) {
    return false;
  }
  return true;
}

} // namespace kernel
} // namespace cleave_superoptimizer

namespace std {
size_t hash<cleave_superoptimizer::kernel::MatmulKey>::operator()(
    cleave_superoptimizer::kernel::MatmulKey const &key) const {
  size_t ret = 0;
  hash_combine(ret, key.operand_a);
  hash_combine(ret, key.operand_b);
  hash_combine(ret, key.trans_A);
  hash_combine(ret, key.trans_B);
  return ret;
}
}; // namespace std
