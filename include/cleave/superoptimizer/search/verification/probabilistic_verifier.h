#pragma once

#include "cleave/superoptimizer/search/verification/verifier.h"

#include <mutex>

namespace cleave_superoptimizer {
namespace search {

class ProbabilisticVerifier : public Verifier {
public:
  ProbabilisticVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;

  static std::mutex fingerprint_mutex;

private:
  std::vector<cpu::CTensor> input_graph_fingerprints;
};

} // namespace search
} // namespace cleave_superoptimizer