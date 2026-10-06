#pragma once

#include "cleave/superoptimizer/cpu/cmem_tensor.h"
#include "cleave/superoptimizer/type.h"
#include "cleave/superoptimizer/utils/json_utils.h"
#include <atomic>
#include <cstddef>
#include <functional>

namespace cleave_superoptimizer {
namespace kernel {

constexpr int MAX_TENSOR_DIMS = 5;

class KNOperator;

struct alignas(16) DTensor {
  DTensor(void);
  inline bool operator==(DTensor const &b) const {
    if (num_dims != b.num_dims) {
      return false;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return false;
      }
      if (can_reduce[i] != b.can_reduce[i]) {
        return false;
      }
      if (can_split[i] != b.can_split[i]) {
        return false;
      }
    }
    if (owner_op != b.owner_op) {
      return false;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return false;
    }
    return true;
  }
  inline bool operator!=(DTensor const &b) const {
    if (num_dims != b.num_dims) {
      return true;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return true;
      }
      if (can_reduce[i] != b.can_reduce[i]) {
        return true;
      }
      if (can_split[i] != b.can_split[i]) {
        return true;
      }
    }
    if (owner_op != b.owner_op) {
      return true;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return true;
    }
    return false;
  }

  inline size_t num_elements() const {
    size_t num = 1;
    for (int i = 0; i < num_dims; i++) {
      num *= dim[i];
    }
    return num;
  }

  inline size_t fingerprint_size() const {
    using namespace cleave_superoptimizer::type;
    size_t fingerprint_type_size = sizeof(FingerprintType);
    return num_elements() * fingerprint_type_size;
  }

  // bool has_same_fingerprint(DTensor const &ref) const;
  bool has_same_fingerprint(
      cleave_superoptimizer::cpu::CTensor const &ref) const;
  cleave_superoptimizer::cpu::CTensor copy_fingerprint_to_ctensor() const;

public:
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  bool can_reduce[MAX_TENSOR_DIMS];
  bool can_split[MAX_TENSOR_DIMS];
  type::GuidType guid;
  //  DTensor fields
  KNOperator *owner_op;
  int owner_ts_idx;
  static std::atomic<int64_t> next_guid;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    DTensor, num_dims, dim, can_reduce, can_split, guid)

} // namespace kernel
} // namespace cleave_superoptimizer

namespace std {
template <>
struct hash<cleave_superoptimizer::kernel::DTensor> {
  size_t operator()(cleave_superoptimizer::kernel::DTensor const &) const;
};
}; // namespace std
