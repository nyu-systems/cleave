#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/utils/hash_utils.h"
#include <functional>

namespace cleave_superoptimizer {
namespace kernel {

DTensor::DTensor() {
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    can_reduce[i] = false;
    can_split[i] = false;
  }
  owner_op = nullptr;
  owner_ts_idx = -1000;
}

std::atomic<int64_t> DTensor::next_guid = 10000000;

} // namespace kernel
} // namespace cleave_superoptimizer

namespace std {

size_t hash<cleave_superoptimizer::kernel::DTensor>::operator()(
    cleave_superoptimizer::kernel::DTensor const &tensor) const {
  size_t ret = hash<int>()(tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    hash_combine(ret, tensor.can_reduce[i]);
    hash_combine(ret, tensor.can_split[i]);
  }
  hash_combine(ret, tensor.owner_op);
  hash_combine(ret, tensor.owner_ts_idx);
  return ret;
}

} // namespace std
