#include "cleave/superoptimizer/kernel/fingerprint_cpu.h"
#include "cleave/superoptimizer/config.h"
#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/element_unary.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/operator.h"
#include "cleave/superoptimizer/kernel/reduction.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"
#include "cleave/superoptimizer/kernel/split.h"
#include "cleave/superoptimizer/type.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cleave_superoptimizer {
namespace kernel {

namespace {

using cleave_superoptimizer::type::FingerprintType;

FingerprintType fingerprint_from_scalar(float scalar) {
  uint32_t bits = 0;
  std::memcpy(&bits, &scalar, sizeof(bits));
  return static_cast<FingerprintType>(
      bits % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

struct CpuFingerprintTables {
  CpuFingerprintTables() {
    exp_table.resize(cleave_superoptimizer::config::FINGERPRINT_Q);
    exp_table[0] = 1;
    for (int i = 1; i < cleave_superoptimizer::config::FINGERPRINT_Q; i++) {
      exp_table[i] = (exp_table[i - 1] *
                      cleave_superoptimizer::config::FINGERPRINT_EXP_BASE) %
                     cleave_superoptimizer::config::FINGERPRINT_P;
    }

    div_p_table.resize(cleave_superoptimizer::config::FINGERPRINT_P);
    for (uint32_t i = 0; i < cleave_superoptimizer::config::FINGERPRINT_P;
         i++) {
      div_p_table[i] = 1;
      for (uint32_t j = 1; j < cleave_superoptimizer::config::FINGERPRINT_P;
           j++) {
        if ((i * j) % cleave_superoptimizer::config::FINGERPRINT_P == 1) {
          div_p_table[i] = static_cast<FingerprintType>(j);
        }
      }
    }

    div_q_table.resize(cleave_superoptimizer::config::FINGERPRINT_Q);
    for (uint32_t i = 0; i < cleave_superoptimizer::config::FINGERPRINT_Q;
         i++) {
      div_q_table[i] = 1;
      for (uint32_t j = 1; j < cleave_superoptimizer::config::FINGERPRINT_Q;
           j++) {
        if ((i * j) % cleave_superoptimizer::config::FINGERPRINT_Q == 1) {
          div_q_table[i] = static_cast<FingerprintType>(j);
        }
      }
    }

    sqrt_p_table.resize(cleave_superoptimizer::config::FINGERPRINT_P);
    for (uint32_t i = 0; i < cleave_superoptimizer::config::FINGERPRINT_P;
         i++) {
      FingerprintType value = 1;
      for (uint32_t j = 0;
           j < (cleave_superoptimizer::config::FINGERPRINT_P + 1) / 4;
           j++) {
        value = (value * i) % cleave_superoptimizer::config::FINGERPRINT_P;
      }
      sqrt_p_table[i] = value;
    }

    sqrt_q_table.resize(cleave_superoptimizer::config::FINGERPRINT_Q);
    for (uint32_t i = 0; i < cleave_superoptimizer::config::FINGERPRINT_Q;
         i++) {
      FingerprintType value = 1;
      for (uint32_t j = 0;
           j < (cleave_superoptimizer::config::FINGERPRINT_Q + 1) / 4;
           j++) {
        value = (value * i) % cleave_superoptimizer::config::FINGERPRINT_Q;
      }
      sqrt_q_table[i] = value;
    }
  }

  std::vector<FingerprintType> exp_table;
  std::vector<FingerprintType> div_p_table;
  std::vector<FingerprintType> div_q_table;
  std::vector<FingerprintType> sqrt_p_table;
  std::vector<FingerprintType> sqrt_q_table;
};

CpuFingerprintTables &get_tables() {
  static CpuFingerprintTables tables;
  return tables;
}

using Store = CpuFingerprintEngine::Store;

Store &get_store() {
  static Store store;
  return store;
}

inline FingerprintType fingerprint_add(FingerprintType a, FingerprintType b) {
  uint32_t x = a;
  uint32_t y = b;
  return static_cast<FingerprintType>(
      (x + y) % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline FingerprintType fingerprint_mul(FingerprintType a, FingerprintType b) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual = ((x % cleave_superoptimizer::config::FINGERPRINT_P) *
                         (y % cleave_superoptimizer::config::FINGERPRINT_P)) %
                        cleave_superoptimizer::config::FINGERPRINT_P;
  uint32_t q_residual = ((x % cleave_superoptimizer::config::FINGERPRINT_Q) *
                         (y % cleave_superoptimizer::config::FINGERPRINT_Q)) %
                        cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t z =
      p_residual * cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1 +
      q_residual * cleave_superoptimizer::config::FINGERPRINT_P_MUL_Q_MOD_1;
  return static_cast<FingerprintType>(
      z % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline FingerprintType fingerprint_div(FingerprintType a,
                                       FingerprintType b,
                                       CpuFingerprintTables const &tables) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual =
      ((x % cleave_superoptimizer::config::FINGERPRINT_P) *
       tables.div_p_table[y % cleave_superoptimizer::config::FINGERPRINT_P]) %
      cleave_superoptimizer::config::FINGERPRINT_P;
  uint32_t q_residual =
      ((x % cleave_superoptimizer::config::FINGERPRINT_Q) *
       tables.div_q_table[y % cleave_superoptimizer::config::FINGERPRINT_Q]) %
      cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t z =
      p_residual * cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1 +
      q_residual * cleave_superoptimizer::config::FINGERPRINT_P_MUL_Q_MOD_1;
  return static_cast<FingerprintType>(
      z % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline FingerprintType fingerprint_exp(FingerprintType input,
                                       CpuFingerprintTables const &tables) {
  FingerprintType q_residual =
      input % cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t result = tables.exp_table[q_residual];
  result = (result * cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1) %
           cleave_superoptimizer::config::FINGERPRINT_PQ;
  return static_cast<FingerprintType>(result);
}

inline FingerprintType fingerprint_silu(FingerprintType input,
                                        CpuFingerprintTables const &tables) {
  FingerprintType q_residual =
      input % cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t result = tables.exp_table[q_residual];
  result = (result * q_residual *
            cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1) %
           cleave_superoptimizer::config::FINGERPRINT_PQ;
  return static_cast<FingerprintType>(result);
}

inline FingerprintType fingerprint_gelu(FingerprintType input,
                                        CpuFingerprintTables const &tables) {
  FingerprintType q_residual =
      input % cleave_superoptimizer::config::FINGERPRINT_Q;
  FingerprintType scaling_factor = 2;
  FingerprintType scaled_q_residual = static_cast<FingerprintType>(
      (scaling_factor * q_residual) %
      cleave_superoptimizer::config::FINGERPRINT_Q);
  uint32_t result = tables.exp_table[scaled_q_residual];
  result = (result * q_residual *
            cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1) %
           cleave_superoptimizer::config::FINGERPRINT_PQ;
  return static_cast<FingerprintType>(result);
}

inline FingerprintType fingerprint_sqrt(FingerprintType input,
                                        CpuFingerprintTables const &tables) {
  uint32_t x = input;
  x = tables.sqrt_p_table[x % cleave_superoptimizer::config::FINGERPRINT_P] *
          cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1 +
      tables.sqrt_q_table[x % cleave_superoptimizer::config::FINGERPRINT_Q] *
          cleave_superoptimizer::config::FINGERPRINT_P_MUL_Q_MOD_1;
  return static_cast<FingerprintType>(
      x % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline FingerprintType fingerprint_clamp(FingerprintType input) {
  uint32_t q_residual = input % cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t p_residual = input % cleave_superoptimizer::config::FINGERPRINT_P;
  q_residual = std::min<uint32_t>(
      2 * cleave_superoptimizer::config::FINGERPRINT_Q / 3,
      std::max<uint32_t>(cleave_superoptimizer::config::FINGERPRINT_Q / 3,
                         q_residual));
  p_residual = std::min<uint32_t>(
      2 * cleave_superoptimizer::config::FINGERPRINT_P / 3,
      std::max<uint32_t>(cleave_superoptimizer::config::FINGERPRINT_P / 3,
                         p_residual));
  uint32_t z =
      p_residual * cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1 +
      q_residual * cleave_superoptimizer::config::FINGERPRINT_P_MUL_Q_MOD_1;
  return static_cast<FingerprintType>(
      z % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline FingerprintType fingerprint_relu(FingerprintType input) {
  uint32_t q_residual = input % cleave_superoptimizer::config::FINGERPRINT_Q;
  uint32_t p_residual = input % cleave_superoptimizer::config::FINGERPRINT_P;
  q_residual = std::max<uint32_t>(
      cleave_superoptimizer::config::FINGERPRINT_Q / 2, q_residual);
  p_residual = std::max<uint32_t>(
      cleave_superoptimizer::config::FINGERPRINT_P / 2, p_residual);
  uint32_t z =
      p_residual * cleave_superoptimizer::config::FINGERPRINT_Q_MUL_P_MOD_1 +
      q_residual * cleave_superoptimizer::config::FINGERPRINT_P_MUL_Q_MOD_1;
  return static_cast<FingerprintType>(
      z % cleave_superoptimizer::config::FINGERPRINT_PQ);
}

inline uint32_t fingerprint_rand(uint32_t seed) {
  uint32_t x = seed + 0x9e3779b9;
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  x ^= x >> 16;
  return x;
}

void compute_strides(DTensor const &tensor, int *strides) {
  int stride = 1;
  for (int i = tensor.num_dims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= tensor.dim[i];
  }
}

int map_repeat_index(int out_idx, int input_dim, int output_dim) {
  assert(input_dim > 0);
  assert(output_dim > 0);
  assert(output_dim % input_dim == 0);
  int repeat_factor = output_dim / input_dim;
  assert(repeat_factor > 0);
  return out_idx / repeat_factor;
}

int compute_matmul_batch_offsets(DTensor const &A,
                                 DTensor const &B,
                                 DTensor const &C,
                                 std::vector<int64_t> &a_offsets,
                                 std::vector<int64_t> &b_offsets,
                                 std::vector<int64_t> &c_offsets) {
  int num_dims = C.num_dims;
  int batch_dims = num_dims - 2;
  int num_batches = 1;
  for (int i = 0; i < batch_dims; i++) {
    num_batches *= C.dim[i];
  }

  int a_strides[MAX_TENSOR_DIMS];
  int b_strides[MAX_TENSOR_DIMS];
  int c_strides[MAX_TENSOR_DIMS];
  compute_strides(A, a_strides);
  compute_strides(B, b_strides);
  compute_strides(C, c_strides);

  int a_dim_offset = num_dims - A.num_dims;
  int b_dim_offset = num_dims - B.num_dims;
  a_offsets.assign(num_batches, 0);
  b_offsets.assign(num_batches, 0);
  c_offsets.assign(num_batches, 0);
  for (int b = 0; b < num_batches; b++) {
    int tmp = b;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    int64_t c_offset = 0;
    for (int i = batch_dims - 1; i >= 0; --i) {
      int out_idx = tmp % C.dim[i];
      tmp /= C.dim[i];
      int a_dim_idx = i - a_dim_offset;
      int b_dim_idx = i - b_dim_offset;
      if (i >= a_dim_offset) {
        assert(a_dim_idx >= 0);
        int a_dim = A.dim[a_dim_idx];
        assert(C.dim[i] % a_dim == 0);
        int a_idx = map_repeat_index(out_idx, a_dim, C.dim[i]);
        a_offset += static_cast<int64_t>(a_idx) * a_strides[a_dim_idx];
      }
      if (i >= b_dim_offset) {
        assert(b_dim_idx >= 0);
        int b_dim = B.dim[b_dim_idx];
        assert(C.dim[i] % b_dim == 0);
        int b_idx = map_repeat_index(out_idx, b_dim, C.dim[i]);
        b_offset += static_cast<int64_t>(b_idx) * b_strides[b_dim_idx];
      }
      c_offset += static_cast<int64_t>(out_idx) * c_strides[i];
    }
    a_offsets[b] = a_offset;
    b_offsets[b] = b_offset;
    c_offsets[b] = c_offset;
  }
  return num_batches;
}

void compute_input_fingerprint(Store &store,
                               DTensor const &output,
                               int input_id) {
  int num_elements = output.num_elements();
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int idx = 0; idx < num_elements; idx++) {
    uint32_t seed = static_cast<uint32_t>(
        (input_id + idx) % cleave_superoptimizer::config::FINGERPRINT_PQ);
    output_fingerprint_ptr[idx] =
        static_cast<FingerprintType>(fingerprint_rand(seed) & 0xFFFF);
  }
}

void compute_matmul_fingerprint(Store &store,
                                DTensor const &A,
                                DTensor const &B,
                                DTensor const &C,
                                bool trans_A,
                                bool trans_B) {
  int a_num_dims = A.num_dims;
  int b_num_dims = B.num_dims;
  int c_num_dims = C.num_dims;
  int row_a_orig = A.dim[a_num_dims - 2];
  int col_a_orig = A.dim[a_num_dims - 1];
  int row_b_orig = B.dim[b_num_dims - 2];
  int col_b_orig = B.dim[b_num_dims - 1];
  int row_a = trans_A ? col_a_orig : row_a_orig;
  int column_a = trans_A ? row_a_orig : col_a_orig;
  int row_b = trans_B ? col_b_orig : row_b_orig;
  int column_b = trans_B ? row_b_orig : col_b_orig;
  int row_c = C.dim[c_num_dims - 2];
  int column_c = C.dim[c_num_dims - 1];
  assert(column_a == row_b);
  assert(row_c == row_a);
  assert(column_c == column_b);
  int a_strides[MAX_TENSOR_DIMS];
  int b_strides[MAX_TENSOR_DIMS];
  compute_strides(A, a_strides);
  compute_strides(B, b_strides);
  std::vector<int64_t> a_offsets;
  std::vector<int64_t> b_offsets;
  std::vector<int64_t> c_offsets;
  int num_batches =
      compute_matmul_batch_offsets(A, B, C, a_offsets, b_offsets, c_offsets);
  FingerprintType *a_ptr = store.get_buffer(A);
  FingerprintType *b_ptr = store.get_buffer(B);
  FingerprintType *c_ptr = store.get_buffer(C);
  for (int b = 0; b < num_batches; b++) {
    FingerprintType *a_batch = a_ptr + a_offsets[b];
    FingerprintType *b_batch = b_ptr + b_offsets[b];
    FingerprintType *c_batch = c_ptr + c_offsets[b];
    for (int r = 0; r < row_c; r++) {
      for (int c = 0; c < column_c; c++) {
        FingerprintType result = 0;
        for (int k = 0; k < column_a; k++) {
          int a_row_idx = trans_A ? k : r;
          int a_col_idx = trans_A ? r : k;
          int b_row_idx = trans_B ? c : k;
          int b_col_idx = trans_B ? k : c;
          FingerprintType x = a_batch[a_row_idx * a_strides[a_num_dims - 2] +
                                      a_col_idx * a_strides[a_num_dims - 1]];
          FingerprintType y = b_batch[b_row_idx * b_strides[b_num_dims - 2] +
                                      b_col_idx * b_strides[b_num_dims - 1]];
          result = fingerprint_add(result, fingerprint_mul(x, y));
        }
        c_batch[r * column_c + c] = result;
      }
    }
  }
}

void compute_elementunary_fingerprint(
    Store &store,
    cleave_superoptimizer::type::KNOperatorType op_type,
    DTensor const &input,
    DTensor const &output) {
  assert(input.num_elements() == output.num_elements());
  int num_elements = input.num_elements();
  CpuFingerprintTables &tables = get_tables();
  FingerprintType *input_fingerprint_ptr = store.get_buffer(input);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int i = 0; i < num_elements; i++) {
    if (op_type == cleave_superoptimizer::type::KN_EXP_OP) {
      output_fingerprint_ptr[i] =
          fingerprint_exp(input_fingerprint_ptr[i], tables);
    } else if (op_type == cleave_superoptimizer::type::KN_SQRT_OP) {
      output_fingerprint_ptr[i] =
          fingerprint_sqrt(input_fingerprint_ptr[i], tables);
    } else if (op_type == cleave_superoptimizer::type::KN_SILU_OP) {
      output_fingerprint_ptr[i] =
          fingerprint_silu(input_fingerprint_ptr[i], tables);
    } else if (op_type == cleave_superoptimizer::type::KN_GELU_OP) {
      output_fingerprint_ptr[i] =
          fingerprint_gelu(input_fingerprint_ptr[i], tables);
    } else if (op_type == cleave_superoptimizer::type::KN_RELU_OP) {
      output_fingerprint_ptr[i] = fingerprint_relu(input_fingerprint_ptr[i]);
    } else if (op_type == cleave_superoptimizer::type::KN_CLAMP_OP) {
      output_fingerprint_ptr[i] = fingerprint_clamp(input_fingerprint_ptr[i]);
    } else {
      assert(false && "Unimplemented");
    }
  }
}

void compute_elementbinary_fingerprint(
    Store &store,
    cleave_superoptimizer::type::KNOperatorType op_type,
    DTensor const &input1,
    DTensor const &input2,
    DTensor const &output) {
  int num_elements = output.num_elements();
  CpuFingerprintTables &tables = get_tables();
  FingerprintType *input1_fingerprint_ptr = store.get_buffer(input1);
  FingerprintType *input2_fingerprint_ptr = store.get_buffer(input2);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int idx = 0; idx < num_elements; idx++) {
    int i = idx;
    int input1_stride = 1, input1_idx = 0;
    int input2_stride = 1, input2_idx = 0;
    for (int d = output.num_dims - 1; d >= 0; d--) {
      input1_idx += (i % input1.dim[d]) * input1_stride;
      input2_idx += (i % input2.dim[d]) * input2_stride;
      input1_stride *= input1.dim[d];
      input2_stride *= input2.dim[d];
      i /= output.dim[d];
    }
    FingerprintType x = input1_fingerprint_ptr[input1_idx];
    FingerprintType y = input2_fingerprint_ptr[input2_idx];
    if (op_type == cleave_superoptimizer::type::KN_ADD_OP) {
      output_fingerprint_ptr[idx] = fingerprint_add(x, y);
    } else if (op_type == cleave_superoptimizer::type::KN_MUL_OP) {
      output_fingerprint_ptr[idx] = fingerprint_mul(x, y);
    } else if (op_type == cleave_superoptimizer::type::KN_DIV_OP) {
      output_fingerprint_ptr[idx] = fingerprint_div(x, y, tables);
    } else {
      assert(false && "Unimplemented");
    }
  }
}

void compute_elementbinary_fingerprint_scalar(
    Store &store,
    cleave_superoptimizer::type::KNOperatorType op_type,
    DTensor const &input,
    float scalar,
    bool scalar_on_left,
    DTensor const &output) {
  int num_elements = output.num_elements();
  CpuFingerprintTables &tables = get_tables();
  FingerprintType *input_fingerprint_ptr = store.get_buffer(input);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  FingerprintType scalar_fingerprint = fingerprint_from_scalar(scalar);
  for (int idx = 0; idx < num_elements; idx++) {
    FingerprintType x = input_fingerprint_ptr[idx];
    FingerprintType lhs = scalar_on_left ? scalar_fingerprint : x;
    FingerprintType rhs = scalar_on_left ? x : scalar_fingerprint;
    if (op_type == cleave_superoptimizer::type::KN_ADD_OP) {
      output_fingerprint_ptr[idx] = fingerprint_add(lhs, rhs);
    } else if (op_type == cleave_superoptimizer::type::KN_MUL_OP) {
      output_fingerprint_ptr[idx] = fingerprint_mul(lhs, rhs);
    } else if (op_type == cleave_superoptimizer::type::KN_DIV_OP) {
      output_fingerprint_ptr[idx] = fingerprint_div(lhs, rhs, tables);
    } else {
      assert(false && "Unimplemented");
    }
  }
}
void compute_reduction_fingerprint(Store &store,
                                   DTensor const &input,
                                   DTensor const &output,
                                   int reduction_dim_idx) {
  int num_elements = output.num_elements();
  int output_stride = 1;
  int input_stride = 1;
  for (int i = reduction_dim_idx; i < output.num_dims; i++) {
    output_stride *= output.dim[i];
    input_stride *= input.dim[i];
  }
  int reduction_factor =
      input.dim[reduction_dim_idx] / output.dim[reduction_dim_idx];
  assert(output_stride * reduction_factor == input_stride);
  FingerprintType *input_fingerprint_ptr = store.get_buffer(input);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int i = 0; i < num_elements; i++) {
    FingerprintType result = 0;
    int n = i / output_stride;
    int m = i % output_stride;
    for (int k = 0; k < reduction_factor; k++) {
      FingerprintType value =
          input_fingerprint_ptr[n * input_stride + m + k * output_stride];
      result = fingerprint_add(result, value);
    }
    output_fingerprint_ptr[i] = result;
  }
}

void compute_split_fingerprint(Store &store,
                               DTensor const &input,
                               DTensor const &output,
                               int split_dim_idx) {
  assert(output.num_dims == input.num_dims + 1);
  assert(output.dim[0] == NUM_SPLITS);
  int input_strides[MAX_TENSOR_DIMS];
  int output_strides[MAX_TENSOR_DIMS];
  compute_strides(input, input_strides);
  compute_strides(output, output_strides);
  int split_dim_size = input.dim[split_dim_idx] / NUM_SPLITS;
  int num_elements = output.num_elements();
  FingerprintType *input_fingerprint_ptr = store.get_buffer(input);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int idx = 0; idx < num_elements; idx++) {
    int remaining = idx;
    int output_idx[MAX_TENSOR_DIMS];
    for (int d = 0; d < output.num_dims; d++) {
      output_idx[d] = remaining / output_strides[d];
      remaining %= output_strides[d];
    }
    int input_linear = 0;
    for (int d = 0; d < input.num_dims; d++) {
      int output_dim = d + 1;
      int input_idx = output_idx[output_dim];
      if (d == split_dim_idx) {
        int split_idx = output_idx[0];
        input_idx = split_idx * split_dim_size + output_idx[output_dim];
      }
      input_linear += input_idx * input_strides[d];
    }
    output_fingerprint_ptr[idx] = input_fingerprint_ptr[input_linear];
  }
}

void compute_rms_norm_fingerprint(Store &store,
                                  DTensor const &input,
                                  DTensor const &output,
                                  int normalized_size) {
  int num_samples = output.num_elements() / normalized_size;
  CpuFingerprintTables &tables = get_tables();
  FingerprintType *input_fingerprint_ptr = store.get_buffer(input);
  FingerprintType *output_fingerprint_ptr = store.get_buffer(output);
  for (int i = 0; i < num_samples; i++) {
    FingerprintType square_sum = 0;
    for (int k = 0; k < normalized_size; k++) {
      FingerprintType x = input_fingerprint_ptr[i * normalized_size + k];
      x = fingerprint_mul(x, x);
      square_sum = fingerprint_add(square_sum, x);
    }
    FingerprintType n = static_cast<FingerprintType>(
        normalized_size % cleave_superoptimizer::config::FINGERPRINT_PQ);
    FingerprintType z = fingerprint_div(square_sum, n, tables);
    FingerprintType rms = fingerprint_sqrt(z, tables);
    for (int k = 0; k < normalized_size; k++) {
      FingerprintType x = input_fingerprint_ptr[i * normalized_size + k];
      output_fingerprint_ptr[i * normalized_size + k] =
          fingerprint_div(x, rms, tables);
    }
  }
}

} // namespace

cleave_superoptimizer::type::FingerprintType *
    CpuFingerprintEngine::Store::get_buffer(DTensor const &tensor) {
  auto &buffer = fingerprints[tensor.guid];
  size_t num_elements = tensor.num_elements();
  if (buffer.size() != num_elements) {
    assert(buffer.size() == 0 &&
           "Different sizes detected on buffers of the same guid");
    buffer.assign(num_elements, 0);
  }
  return buffer.data();
}

void CpuFingerprintEngine::load_fingerprint_from_ctensor(
    DTensor const &tensor, cleave_superoptimizer::cpu::CTensor const &ctensor) {
  size_t bytes = tensor.fingerprint_size();
  FingerprintType *dst = store_.get_buffer(tensor);
  std::memcpy(dst, ctensor.fingerprint_ptr, bytes);
}

void CpuFingerprintEngine::load_fingerprint_from_buf(DTensor const &tensor,
                                                     std::byte const *buf,
                                                     size_t buf_size) {
  FingerprintType *dst = store_.get_buffer(tensor);
  std::memcpy(dst, buf, buf_size);
}

void CpuFingerprintEngine::compute(KNOperator const *op) {
  if (op->op_type == cleave_superoptimizer::type::KN_INPUT_OP) {
    return;
  }
  if (op->op_type == cleave_superoptimizer::type::KN_OUTPUT_OP) {
    return;
  }
  if (op->op_type == cleave_superoptimizer::type::KN_MATMUL_OP) {
    auto const *matmul = dynamic_cast<KNMatmulOp const *>(op);
    bool trans_A = matmul ? matmul->trans_A : false;
    bool trans_B = matmul ? matmul->trans_B : false;
    compute_matmul_fingerprint(store_,
                               op->input_tensors[0],
                               op->input_tensors[1],
                               op->output_tensors[0],
                               trans_A,
                               trans_B);
    return;
  }
  if (op->op_type == cleave_superoptimizer::type::KN_ADD_OP ||
      op->op_type == cleave_superoptimizer::type::KN_MUL_OP ||
      op->op_type == cleave_superoptimizer::type::KN_DIV_OP) {
    auto const *bin_op = dynamic_cast<KNElementBinaryOp const *>(op);
    if (bin_op && bin_op->has_scalar) {
      compute_elementbinary_fingerprint_scalar(store_,
                                               op->op_type,
                                               op->input_tensors[0],
                                               bin_op->scalar,
                                               bin_op->scalar_on_left,
                                               op->output_tensors[0]);
    } else {
      compute_elementbinary_fingerprint(store_,
                                        op->op_type,
                                        op->input_tensors[0],
                                        op->input_tensors[1],
                                        op->output_tensors[0]);
    }
    return;
  }
  if (op->op_type == cleave_superoptimizer::type::KN_REDUCTION_0_OP ||
      op->op_type == cleave_superoptimizer::type::KN_REDUCTION_1_OP ||
      op->op_type == cleave_superoptimizer::type::KN_REDUCTION_2_OP ||
      op->op_type == cleave_superoptimizer::type::KN_REDUCTION_3_OP ||
      op->op_type == cleave_superoptimizer::type::KN_REDUCTION_4_OP) {
    auto const *red = static_cast<KNReductionOp const *>(op);
    compute_reduction_fingerprint(store_,
                                  op->input_tensors[0],
                                  op->output_tensors[0],
                                  red->reduction_dim_idx);
    return;
  }
  if (op->op_type >= cleave_superoptimizer::type::KN_SPLIT_FIRST_OP_ID &&
      op->op_type <= cleave_superoptimizer::type::KN_SPLIT_LAST_OP_ID) {
    auto const *split = static_cast<KNSplitOp const *>(op);
    compute_split_fingerprint(store_,
                              op->input_tensors[0],
                              op->output_tensors[0],
                              split->split_dim_idx);
    return;
  }
  if (op->op_type == cleave_superoptimizer::type::KN_RMS_NORM_OP) {
    auto const *rms = static_cast<KNRMSNormOp const *>(op);
    compute_rms_norm_fingerprint(store_,
                                 op->input_tensors[0],
                                 op->output_tensors[0],
                                 rms->normalized_size);
    return;
  }
  compute_elementunary_fingerprint(
      store_, op->op_type, op->input_tensors[0], op->output_tensors[0]);
}

bool CpuFingerprintEngine::matches(
    DTensor const &tensor, cleave_superoptimizer::cpu::CTensor const &ctensor) {
  int num_elements = static_cast<int>(tensor.num_elements());
  FingerprintType *src = store_.get_buffer(tensor);
  FingerprintType const *ref = ctensor.fingerprint_ptr;
  for (int i = 0; i < num_elements; i++) {
    if (src[i] != ref[i]) {
      return false;
    }
  }
  return true;
}

cleave_superoptimizer::cpu::CTensor
    DTensor::copy_fingerprint_to_ctensor() const {
  cleave_superoptimizer::cpu::CTensor ctensor;
  ctensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    ctensor.dim[i] = dim[i];
  }
  Store &store = get_store();
  size_t bytes = fingerprint_size();
  ctensor.fingerprint_ptr = static_cast<FingerprintType *>(malloc(bytes));
  FingerprintType *src = store.get_buffer(*this);
  std::memcpy(ctensor.fingerprint_ptr, src, bytes);
  return ctensor;
}

bool DTensor::has_same_fingerprint(
    cleave_superoptimizer::cpu::CTensor const &ref) const {
  // if (num_dims != ref.num_dims) {
  //   return false;
  // }
  // for (int i = 0; i < num_dims; i++) {
  //   if (dim[i] != ref.dim[i]) {
  //     return false;
  //   }
  // }
  if (num_elements() != ref.num_elements()) {
    return false;
  }
  Store &store = get_store();
  int num_elements = static_cast<int>(this->num_elements());
  FingerprintType *src = store.get_buffer(*this);
  FingerprintType const *ref_ptr = ref.fingerprint_ptr;
  for (int i = 0; i < num_elements; i++) {
    if (src[i] != ref_ptr[i]) {
      return false;
    }
  }
  return true;
}

bool KNInputOp::fingerprint() {
  return this->fingerprint(0);
}

bool KNInputOp::fingerprint(int input_id) {
  compute_input_fingerprint(get_store(), output_tensors[0], input_id);
  return true;
}

bool KNMatmulOp::fingerprint(void) {
  compute_matmul_fingerprint(get_store(),
                             input_tensors[0],
                             input_tensors[1],
                             output_tensors[0],
                             trans_A,
                             trans_B);
  return true;
}

bool KNElementUnaryOp::fingerprint(void) {
  compute_elementunary_fingerprint(
      get_store(), op_type, input_tensors[0], output_tensors[0]);
  return true;
}

bool KNElementBinaryOp::fingerprint(void) {
  if (has_scalar) {
    compute_elementbinary_fingerprint_scalar(get_store(),
                                             op_type,
                                             input_tensors[0],
                                             scalar,
                                             scalar_on_left,
                                             output_tensors[0]);
  } else {
    compute_elementbinary_fingerprint(get_store(),
                                      op_type,
                                      input_tensors[0],
                                      input_tensors[1],
                                      output_tensors[0]);
  }
  return true;
}

bool KNReductionOp::fingerprint(void) {
  compute_reduction_fingerprint(
      get_store(), input_tensors[0], output_tensors[0], reduction_dim_idx);
  return true;
}

bool KNSplitOp::fingerprint(void) {
  compute_split_fingerprint(
      get_store(), input_tensors[0], output_tensors[0], split_dim_idx);
  return true;
}

bool KNRMSNormOp::fingerprint(void) {
  compute_rms_norm_fingerprint(
      get_store(), input_tensors[0], output_tensors[0], normalized_size);
  return true;
}

bool KNOutputOp::fingerprint(void) {
  return true;
}

} // namespace kernel
} // namespace cleave_superoptimizer
