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
#define CSO_OP(name, value, str, cat) name = value,
#include "cleave/superoptimizer/op_types.def"
#undef CSO_OP
  KN_CONCAT_FIRST_OP_ID = KN_CONCAT_0_OP,
  KN_CONCAT_LAST_OP_ID = 1409,
  KN_SPLIT_FIRST_OP_ID = KN_SPLIT_0_OP,
  KN_SPLIT_LAST_OP_ID = KN_SPLIT_4_OP,
};

enum class OpCategory {
  META,
  INPUT,
  OUTPUT,
  MATMUL,
  UNARY,
  CLAMP,
  BINARY,
  REDUCE_SUM,
  REDUCE_MAX,
  RMS_NORM,
  CONCAT,
  SPLIT,
};

struct OpTypeInfo {
  KNOperatorType type;
  char const *name;
  OpCategory category;
};

inline constexpr OpTypeInfo kOpTypes[] = {
#define CSO_OP(name, value, str, cat) {name, str, OpCategory::cat},
#include "cleave/superoptimizer/op_types.def"
#undef CSO_OP
};

inline constexpr int kNumOpTypes = sizeof(kOpTypes) / sizeof(kOpTypes[0]);

inline OpTypeInfo const *find_op_type(KNOperatorType t) {
  for (auto const &info : kOpTypes) {
    if (info.type == t) {
      return &info;
    }
  }
  return nullptr;
}

// nullptr for values not in op_types.def.
inline char const *op_type_name(KNOperatorType t) {
  OpTypeInfo const *info = find_op_type(t);
  return info ? info->name : nullptr;
}

inline OpCategory op_category(KNOperatorType t) {
  OpTypeInfo const *info = find_op_type(t);
  return info ? info->category : OpCategory::META;
}

// Same semantics as NLOHMANN_JSON_SERIALIZE_ENUM: unknown values and names
// map to the first entry (KN_UNKOWN).
inline void to_json(json &j, KNOperatorType const &t) {
  char const *name = op_type_name(t);
  j = name ? name : kOpTypes[0].name;
}

inline void from_json(json const &j, KNOperatorType &t) {
  t = kOpTypes[0].type;
  if (!j.is_string()) {
    return;
  }
  std::string const &name = j.get_ref<std::string const &>();
  for (auto const &info : kOpTypes) {
    if (name == info.name) {
      t = info.type;
      return;
    }
  }
}

// Storage dtype of a tensor. Metadata only: excluded from DTensor equality,
// hashing and fingerprints, so it never changes what the search explores.
enum DataType {
  DT_UNKNOWN = 0,
  DT_INT8,
  DT_INT16,
  DT_INT32,
  DT_INT64,
  DT_UINT8,
  DT_UINT16,
  DT_UINT32,
  DT_UINT64,
  DT_FLOAT16,
  DT_BFLOAT16,
  DT_FLOAT32,
  DT_FLOAT64,
};

// "" <-> DT_UNKNOWN; data_type_from_string throws std::invalid_argument.
DataType data_type_from_string(std::string const &name);
std::string data_type_to_string(DataType dtype);

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
