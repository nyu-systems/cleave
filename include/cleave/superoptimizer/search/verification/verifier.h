#pragma once

#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/search/verification/output_match.h"

namespace cleave_superoptimizer {
namespace search {

class Verifier {
public:
  Verifier() = default;
  virtual OutputMatch verify(kernel::Graph const &graph) = 0;
  virtual ~Verifier() = default;
};

} // namespace search
} // namespace cleave_superoptimizer
