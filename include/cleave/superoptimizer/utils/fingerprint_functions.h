#pragma once

#include "cleave/superoptimizer/config.h"
#include "cleave/superoptimizer/type.h"
#include <algorithm>

namespace cleave_superoptimizer {
namespace utils {

using namespace cleave_superoptimizer::type;
using namespace cleave_superoptimizer::config;

inline __device__ FingerprintType compute_add_fingerprint(FingerprintType a,
                                                          FingerprintType b) {
  uint32_t x = a;
  uint32_t y = b;
  return (x + y) % FINGERPRINT_PQ;
}

inline __device__ FingerprintType compute_mul_fingerprint(FingerprintType a,
                                                          FingerprintType b) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual =
      ((x % FINGERPRINT_P) * (y % FINGERPRINT_P)) % FINGERPRINT_P;
  uint32_t q_residual =
      ((x % FINGERPRINT_Q) * (y % FINGERPRINT_Q)) % FINGERPRINT_Q;
  uint32_t z = p_residual * FINGERPRINT_Q_MUL_P_MOD_1 +
               q_residual * FINGERPRINT_P_MUL_Q_MOD_1;
  return (z % FINGERPRINT_PQ);
}

inline __device__ FingerprintType
    compute_div_fingerprint(FingerprintType a,
                            FingerprintType b,
                            FingerprintType *div_p_lookup_table,
                            FingerprintType *div_q_lookup_table) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual =
      ((x % FINGERPRINT_P) * div_p_lookup_table[y % FINGERPRINT_P]) %
      FINGERPRINT_P;
  uint32_t q_residual =
      ((x % FINGERPRINT_Q) * div_q_lookup_table[y % FINGERPRINT_Q]) %
      FINGERPRINT_Q;
  uint32_t z = p_residual * FINGERPRINT_Q_MUL_P_MOD_1 +
               q_residual * FINGERPRINT_P_MUL_Q_MOD_1;
  return (z % FINGERPRINT_PQ);
}

inline __device__ FingerprintType compute_exp_fingerprint(
    FingerprintType input, FingerprintType *exp_lookup_table) {
  FingerprintType q_residual = input % FINGERPRINT_Q;
  uint32_t result = exp_lookup_table[q_residual];
  result = (result * FINGERPRINT_Q_MUL_P_MOD_1) % FINGERPRINT_PQ;
  return result;
}

inline __device__ FingerprintType
    compute_sqrt_fingerprint(FingerprintType input,
                             FingerprintType *sqrt_p_lookup_table,
                             FingerprintType *sqrt_q_lookup_table) {
  uint32_t x = input;
  x = sqrt_p_lookup_table[x % FINGERPRINT_P] * FINGERPRINT_Q_MUL_P_MOD_1 +
      sqrt_q_lookup_table[x % FINGERPRINT_Q] * FINGERPRINT_P_MUL_Q_MOD_1;
  return x % FINGERPRINT_PQ;
}

inline __device__ FingerprintType compute_silu_fingerprint(
    FingerprintType input, FingerprintType *exp_lookup_table) {
  // Note that we use $x * e^x$ as the fingerprint for SILU
  // (i.e., $x / (1+e^{-x})$, since plus one can be easier
  // implemented at any level of the GPU compute hierarchy
  // and $x / e^{-x} = x * e^x$.
  FingerprintType q_residual = input % FINGERPRINT_Q;
  uint32_t result = exp_lookup_table[q_residual];
  result = (result * q_residual * FINGERPRINT_Q_MUL_P_MOD_1) % FINGERPRINT_PQ;
  return result;
}

inline __device__ FingerprintType compute_gelu_fingerprint(
    FingerprintType input, FingerprintType *exp_lookup_table) {
  // Approximating GeLU as x*sigmoid(1.702x)
  FingerprintType q_residual = input % FINGERPRINT_Q;
  FingerprintType scaling_factor = 2;
  FingerprintType scaled_q_residual =
      (scaling_factor * q_residual) % FINGERPRINT_Q;
  uint32_t result = exp_lookup_table[scaled_q_residual];
  result = (result * q_residual * FINGERPRINT_Q_MUL_P_MOD_1) % FINGERPRINT_PQ;
  return result;
}

inline __device__ FingerprintType
    compute_clamp_fingerprint(FingerprintType input) {
  // We use min(max(FINGERPRINT_Q/3, input), FINGERPRINT_Q*2/3) to approximate
  // clamp Note that we ignore the input arguments to clamp
  // https://pytorch.org/docs/main/generated/torch.clamp.html
  uint32_t q_residual = input % FINGERPRINT_Q;
  uint32_t p_residual = input % FINGERPRINT_P;
  q_residual =
      min(2 * FINGERPRINT_Q / 3, max(FINGERPRINT_Q / 3, (int)q_residual));
  p_residual =
      min(2 * FINGERPRINT_P / 3, max(FINGERPRINT_P / 3, (int)p_residual));
  uint32_t z = p_residual * FINGERPRINT_Q_MUL_P_MOD_1 +
               q_residual * FINGERPRINT_P_MUL_Q_MOD_1;
  return z % FINGERPRINT_PQ;
}

inline __device__ FingerprintType
    compute_relu_fingerprint(FingerprintType input) {
  // We use max(FINGERPRINT_Q/2, input) to approximate relu
  uint32_t q_residual = input % FINGERPRINT_Q;
  uint32_t p_residual = input % FINGERPRINT_P;
  q_residual = max(FINGERPRINT_Q / 2, (int)q_residual);
  p_residual = max(FINGERPRINT_P / 2, (int)p_residual);
  uint32_t z = p_residual * FINGERPRINT_Q_MUL_P_MOD_1 +
               q_residual * FINGERPRINT_P_MUL_Q_MOD_1;
  return z % FINGERPRINT_PQ;
}

} // namespace utils
} // namespace cleave_superoptimizer
