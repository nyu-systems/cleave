#pragma once

#include "cleave/superoptimizer/type.h"
#include "cleave/superoptimizer/kernel/device_tensor.h"

#include <unordered_set>
// #include <cstddef>
#include <vector>

namespace cleave_superoptimizer {
namespace tile {

using cleave_superoptimizer::kernel::DTensor;

class TileSearchConfig {
public:
  TileSearchConfig(bool infer_num_ops = false, bool split_enabled = false);
  ~TileSearchConfig();

  void update_config(cleave_superoptimizer::type::KNOperatorType const op_type);

public:
  size_t max_num_kernel_graph_op;

  std::unordered_set<cleave_superoptimizer::type::KNOperatorType> knop_to_explore;

  bool infer_num_ops = false;
  bool split_enabled = false;
};

} // namespace tile
} // namespace cleave_superoptimizer
