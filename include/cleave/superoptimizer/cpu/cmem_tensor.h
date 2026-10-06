#pragma once

#include "cleave/superoptimizer/type.h"
#include <cstddef>

namespace cleave_superoptimizer {
namespace cpu {

constexpr int MAX_TENSOR_DIMS = 5;

struct CTensor {
  CTensor(void) {
    num_dims = 0;
    for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
      dim[i] = 0;
    }
    fingerprint_ptr = nullptr;
  }

  size_t size() const {
    size_t num_elements = 1;
    for (int i = 0; i < num_dims; i++) {
      num_elements *= dim[i];
    }
    return num_elements * sizeof(cleave_superoptimizer::type::FingerprintType);
  }

  size_t num_elements() const {
    size_t num = 1;
    for (int i = 0; i < num_dims; i++) {
      num *= dim[i];
    }
    return num;
  }

  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  cleave_superoptimizer::type::FingerprintType *fingerprint_ptr;
};

} // namespace cpu
} // namespace cleave_superoptimizer
