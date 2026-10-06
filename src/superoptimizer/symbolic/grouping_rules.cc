#include "cleave/superoptimizer/symbolic/grouping_rules.h"

#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/element_unary.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/reduction.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"
#include "cleave/superoptimizer/type.h"

#include <algorithm>
#include <cassert>

namespace cleave_superoptimizer {
namespace symbolic {

namespace {

int get_aligned_batch_idx(int batch_idx,
                          int num_batch_dims,
                          int input_batch_dims) {
  int offset = num_batch_dims - input_batch_dims;
  return batch_idx < offset ? -1 : batch_idx - offset;
}

} // namespace

void GroupingRules::apply_grouping(kernel::KNOperator const *op,
                                   DimensionGrouper &grouper) {
  using namespace cleave_superoptimizer::type;

  switch (op->op_type) {
    // Input and output operators don't create grouping constraints
    case KN_INPUT_OP:
    case KN_OUTPUT_OP:
      break;

    // Matmul operator
    case KN_MATMUL_OP:
      group_matmul(op, grouper);
      break;

    // Element-wise unary operators
    case KN_EXP_OP:
    case KN_SQUARE_OP:
    case KN_SQRT_OP:
    case KN_SILU_OP:
    case KN_SIGMOID_OP:
    case KN_GELU_OP:
    case KN_RELU_OP:
    case KN_CLAMP_OP:
    case KN_LOG_OP:
      group_element_unary(op, grouper);
      break;

    // Element-wise binary operators
    case KN_ADD_OP:
    case KN_MUL_OP:
    case KN_DIV_OP:
      group_element_binary(op, grouper);
      break;

    // Reduction operators
    case KN_REDUCTION_0_OP:
    case KN_REDUCTION_1_OP:
    case KN_REDUCTION_2_OP:
    case KN_REDUCTION_3_OP:
    case KN_REDUCTION_4_OP:
      group_reduction(op, grouper);
      break;

    // RMS normalization
    case KN_RMS_NORM_OP:
      group_rms_norm(op, grouper);
      break;

    // Split operators
    case KN_SPLIT_0_OP:
    case KN_SPLIT_1_OP:
    case KN_SPLIT_2_OP:
    case KN_SPLIT_3_OP:
    case KN_SPLIT_4_OP:
      assert(false); // split is an internal operator during serarch, should not appear here
      break;

    default:
      // Unknown operator type - skip grouping
      break;
  }
}

void GroupingRules::group_matmul(kernel::KNOperator const *op,
                                 DimensionGrouper &grouper) {
  assert(op->input_tensors.size() == 2);
  assert(op->output_tensors.size() == 1);

  kernel::DTensor const &A = op->input_tensors[0];
  kernel::DTensor const &B = op->input_tensors[1];
  kernel::DTensor const &C = op->output_tensors[0];

  assert(A.num_dims >= 2);
  assert(B.num_dims >= 2);

  auto const *matmul_op = dynamic_cast<kernel::KNMatmulOp const *>(op);
  assert(matmul_op != nullptr);

  kernel::MatmulDims dims =
      kernel::get_matmul_dims(A, B, matmul_op->trans_A, matmul_op->trans_B);

  DimensionID A_k{A.guid, dims.a_reduce_idx};
  DimensionID B_k{B.guid, dims.b_reduce_idx};
  group_if_matching_non_broadcast(grouper, A_k, A.dim[dims.a_reduce_idx],
                                  B_k, B.dim[dims.b_reduce_idx]);

  // Group batch dimensions
  int C_m_idx = C.num_dims - 2;
  int C_n_idx = C.num_dims - 1;

  // Number of batch dimensions
  int num_batch_dims_A = A.num_dims - 2;
  int num_batch_dims_B = B.num_dims - 2;
  int num_batch_dims_C = C.num_dims - 2;
  int num_batch_dims = std::max({num_batch_dims_A, num_batch_dims_B, num_batch_dims_C});

  // Group batch dimensions pairwise
  for (int i = 0; i < num_batch_dims; i++) {
    int A_idx = get_aligned_batch_idx(i, num_batch_dims, num_batch_dims_A);
    int B_idx = get_aligned_batch_idx(i, num_batch_dims, num_batch_dims_B);
    int C_idx = get_aligned_batch_idx(i, num_batch_dims, num_batch_dims_C);

    // Group A and C batch dimensions
    if (A_idx >= 0 && C_idx >= 0) {
      DimensionID A_dim{A.guid, A_idx};
      DimensionID C_dim{C.guid, C_idx};
      group_if_compatible_batch(grouper, A_dim, A.dim[A_idx], C_dim,
                                C.dim[C_idx]);
    }

    // Group B and C batch dimensions
    if (B_idx >= 0 && C_idx >= 0) {
      DimensionID B_dim{B.guid, B_idx};
      DimensionID C_dim{C.guid, C_idx};
      group_if_compatible_batch(grouper, B_dim, B.dim[B_idx], C_dim,
                                C.dim[C_idx]);
    }

    // Group A and B batch dimensions
    if (A_idx >= 0 && B_idx >= 0) {
      DimensionID A_dim{A.guid, A_idx};
      DimensionID B_dim{B.guid, B_idx};
      group_if_compatible_batch(grouper, A_dim, A.dim[A_idx], B_dim,
                                B.dim[B_idx]);
    }
  }

  // Group m dimensions (logical A rows with C's second-to-last)
  DimensionID A_m{A.guid, dims.a_row_idx};
  DimensionID C_m{C.guid, C_m_idx};
  group_if_matching_non_broadcast(grouper, A_m, A.dim[dims.a_row_idx], C_m,
                                  C.dim[C_m_idx]);

  // Group n dimensions (logical B columns with C's last)
  DimensionID B_n{B.guid, dims.b_col_idx};
  DimensionID C_n{C.guid, C_n_idx};
  group_if_matching_non_broadcast(grouper, B_n, B.dim[dims.b_col_idx], C_n,
                                  C.dim[C_n_idx]);
}

void GroupingRules::group_element_binary(kernel::KNOperator const *op,
                                         DimensionGrouper &grouper) {
  auto const *bin_op =
      dynamic_cast<kernel::KNElementBinaryOp const *>(op);
  if (bin_op && bin_op->has_scalar) {
    group_element_unary(op, grouper);
    return;
  }

  assert(op->input_tensors.size() == 2);
  assert(op->output_tensors.size() == 1);

  kernel::DTensor const &A = op->input_tensors[0];
  kernel::DTensor const &B = op->input_tensors[1];
  kernel::DTensor const &C = op->output_tensors[0];

  assert(A.num_dims == B.num_dims);
  assert(A.num_dims == C.num_dims);

  // For each dimension, group if both inputs have same size > 1
  for (int i = 0; i < A.num_dims; i++) {
    DimensionID A_dim{A.guid, i};
    DimensionID B_dim{B.guid, i};
    DimensionID C_dim{C.guid, i};

    int A_size = A.dim[i];
    int B_size = B.dim[i];
    int C_size = C.dim[i];

    // Group A and C
    group_if_matching_non_broadcast(grouper, A_dim, A_size, C_dim, C_size);

    // Group B and C
    group_if_matching_non_broadcast(grouper, B_dim, B_size, C_dim, C_size);

    // Group A and B
    group_if_matching_non_broadcast(grouper, A_dim, A_size, B_dim, B_size);
  }
}

void GroupingRules::group_element_unary(kernel::KNOperator const *op,
                                        DimensionGrouper &grouper) {
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  kernel::DTensor const &input = op->input_tensors[0];
  kernel::DTensor const &output = op->output_tensors[0];

  assert(input.num_dims == output.num_dims);

  // Group each dimension pairwise (input and output must match)
  for (int i = 0; i < input.num_dims; i++) {
    DimensionID input_dim{input.guid, i};
    DimensionID output_dim{output.guid, i};

    int input_size = input.dim[i];
    int output_size = output.dim[i];

    // For unary ops, input and output dimensions should always match
    if (input_size == output_size) {
      if (input_size == 1) {
        grouper.mark_broadcast(input_dim);
        grouper.mark_broadcast(output_dim);
      } else {
        grouper.merge(input_dim, output_dim);
      }
    }
  }
}

void GroupingRules::group_reduction(kernel::KNOperator const *op,
                                    DimensionGrouper &grouper) {
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  kernel::DTensor const &input = op->input_tensors[0];
  kernel::DTensor const &output = op->output_tensors[0];

  assert(input.num_dims == output.num_dims);

  // Determine reduction dimension from operator type
  int reduction_dim = op->op_type - type::KN_REDUCTION_0_OP;
  assert(reduction_dim >= 0 && reduction_dim <= 4 && "Invalid reduction operator type");

  assert(reduction_dim >= 0 && reduction_dim < input.num_dims);

  // Group all non-reduced dimensions
  for (int i = 0; i < input.num_dims; i++) {
    DimensionID input_dim{input.guid, i};
    DimensionID output_dim{output.guid, i};

    int input_size = input.dim[i];
    int output_size = output.dim[i];

    if (i == reduction_dim) {
      // The reduced dimension might have different size
      // If output size is 1, mark as broadcast
      assert(output_size == 1);
      grouper.mark_broadcast(output_dim);
      // Input dimension on reduction axis is not broadcast unless size 1
      if (input_size == 1) {
        grouper.mark_broadcast(input_dim);
      }
    } else {
      // Non-reduced dimensions should match
      group_if_matching_non_broadcast(grouper, input_dim, input_size,
                                      output_dim, output_size);
    }
  }
}

void GroupingRules::group_rms_norm(kernel::KNOperator const *op,
                                   DimensionGrouper &grouper) {
  // RMS norm has similar structure to element-wise unary
  // Input and output have same shape
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  kernel::DTensor const &input = op->input_tensors[0];
  kernel::DTensor const &output = op->output_tensors[0];

  assert(input.num_dims == output.num_dims);

  // Group each dimension pairwise
  for (int i = 0; i < input.num_dims; i++) {
    DimensionID input_dim{input.guid, i};
    DimensionID output_dim{output.guid, i};

    int input_size = input.dim[i];
    int output_size = output.dim[i];

    group_if_matching_non_broadcast(grouper, input_dim, input_size, output_dim,
                                    output_size);
  }
}

void GroupingRules::group_if_matching_non_broadcast(DimensionGrouper &grouper,
                                                    DimensionID dim1,
                                                    int size1,
                                                    DimensionID dim2,
                                                    int size2) {
  grouper.touch(dim1);
  grouper.touch(dim2);

  // If both have size 1, mark both as broadcast
  if (size1 == 1 && size2 == 1) {
    grouper.mark_broadcast(dim1);
    grouper.mark_broadcast(dim2);
  }
  // If only one has size 1, mark that one as broadcast
  else if (size1 == 1) {
    grouper.mark_broadcast(dim1);
  } else if (size2 == 1) {
    grouper.mark_broadcast(dim2);
  }
  // If both have same size > 1, merge them
  else if (size1 == size2) {
    grouper.merge(dim1, dim2);
  }
  // If sizes differ and neither is 1, they're in different groups
  // (no action needed - they'll be assigned different symbols)
}

void GroupingRules::group_if_compatible_batch(DimensionGrouper &grouper,
                                              DimensionID dim1,
                                              int size1,
                                              DimensionID dim2,
                                              int size2) {
  group_if_matching_non_broadcast(grouper, dim1, size1, dim2, size2);

  if (size1 <= 1 || size2 <= 1 || size1 == size2) {
    return;
  }
  if (size1 % size2 == 0) {
    grouper.mark_divisible(dim1, dim2);
  } else if (size2 % size1 == 0) {
    grouper.mark_divisible(dim2, dim1);
  }
}

} // namespace symbolic
} // namespace cleave_superoptimizer
