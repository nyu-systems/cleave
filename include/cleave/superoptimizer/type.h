#pragma once

#include "cleave/superoptimizer/utils/json_utils.h"
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <string>

namespace cleave_superoptimizer {
namespace type {

typedef uint16_t FingerprintType;
typedef int64_t GuidType;

// only to be used in create_op in search.cc
inline std::unordered_map<std::string, float> CLAMP_MIN_MAX;

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_MATMUL_OP = 1003,
  // ElementUnary
  KN_EXP_OP = 1100,
  KN_SQUARE_OP = 1101,
  KN_SQRT_OP = 1102,
  KN_SILU_OP = 1104,
  KN_SIGMOID_OP = 1105,
  KN_GELU_OP = 1106,
  // non-lax elementunary ops
  KN_RELU_OP = 1150,
  KN_CLAMP_OP = 1151,
  KN_LOG_OP = 1160,
  // ElementBinary
  KN_ADD_OP = 1200,
  KN_MUL_OP = 1201,
  KN_DIV_OP = 1202,
  // Reduction & Normalization
  KN_REDUCTION_0_OP = 1300,
  KN_REDUCTION_1_OP = 1301,
  KN_REDUCTION_2_OP = 1302,
  KN_REDUCTION_3_OP = 1303,
  KN_REDUCTION_4_OP = 1304,
  KN_RMS_NORM_OP = 1350,
  // Concat & Split
  KN_CONCAT_FIRST_OP_ID = 1400,
  KN_CONCAT_0_OP = 1400,
  KN_CONCAT_1_OP = 1401,
  KN_CONCAT_2_OP = 1402,
  KN_CONCAT_LAST_OP_ID = 1409,
  KN_SPLIT_FIRST_OP_ID = 1420,
  KN_SPLIT_0_OP = 1420,
  KN_SPLIT_1_OP = 1421,
  KN_SPLIT_2_OP = 1422,
  KN_SPLIT_3_OP = 1423,
  KN_SPLIT_4_OP = 1424,
  KN_SPLIT_LAST_OP_ID = 1424,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_OUTPUT_OP, "kn_output_op"},
                                 {KN_MATMUL_OP, "kn_matmul_op"},
                                 {KN_REDUCTION_0_OP, "kn_reduction_0_op"},
                                 {KN_REDUCTION_1_OP, "kn_reduction_1_op"},
                                 {KN_REDUCTION_2_OP, "kn_reduction_2_op"},
                                 {KN_REDUCTION_3_OP, "kn_reduction_3_op"},
                                 {KN_REDUCTION_4_OP, "kn_reduction_4_op"},
                                 {KN_RMS_NORM_OP, "kn_rms_norm_op"},
                                 {KN_EXP_OP, "kn_exp_op"},
                                 {KN_SQUARE_OP, "kn_square_op"},
                                 {KN_SQRT_OP, "kn_sqrt_op"},
                                 {KN_SILU_OP, "kn_silu_op"},
                                 {KN_GELU_OP, "kn_gelu_op"},
                                 {KN_RELU_OP, "kn_relu_op"},
                                 {KN_CLAMP_OP, "kn_clamp_op"},
                                 {KN_LOG_OP, "kn_log_op"},
                                 {KN_ADD_OP, "kn_add_op"},
                                 {KN_MUL_OP, "kn_mul_op"},
                                 {KN_DIV_OP, "kn_div_op"},
                                 {KN_SPLIT_0_OP, "kn_split_0_op"},
                                 {KN_SPLIT_1_OP, "kn_split_1_op"},
                                 {KN_SPLIT_2_OP, "kn_split_2_op"},
                                 {KN_SPLIT_3_OP, "kn_split_3_op"},
                                 {KN_SPLIT_4_OP, "kn_split_4_op"},
                             })

enum ActivationType {
  ACT_UNKOWN = 3000,
  ACT_EXP = 3001,
  ACT_RELU = 3002,
  ACT_GELU = 3003,
  ACT_SILU = 3004,
  ACT_NONE = 3099,
};

} // namespace type
} // namespace cleave_superoptimizer
