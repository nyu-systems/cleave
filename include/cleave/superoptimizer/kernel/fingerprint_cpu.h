#pragma once

#include "cleave/superoptimizer/cpu/cmem_tensor.h"
#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/kernel/operator.h"
#include "cleave/superoptimizer/type.h"
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace cleave_superoptimizer {
namespace kernel {

class CpuFingerprintEngine {
public:
  struct Store {
    cleave_superoptimizer::type::FingerprintType *get_buffer(DTensor const &tensor);
    std::unordered_map<cleave_superoptimizer::type::GuidType,
                       std::vector<cleave_superoptimizer::type::FingerprintType>>
        fingerprints;
  };

  CpuFingerprintEngine() = default;
  void load_fingerprint_from_ctensor(DTensor const &tensor,
                                     cleave_superoptimizer::cpu::CTensor const &ctensor);
  void load_fingerprint_from_buf(DTensor const &tensor,
                                 std::byte const *buf,
                                 size_t buf_size);
  void compute(KNOperator const *op);
  bool matches(DTensor const &tensor, cleave_superoptimizer::cpu::CTensor const &ctensor);

private:
  Store store_;
};

} // namespace kernel
} // namespace cleave_superoptimizer
