#pragma once

#include "cleave/superoptimizer/search/verification/verifier.h"


namespace cleave_superoptimizer {
namespace search {

class ProbabilisticVerifier : public Verifier {
public:
  ProbabilisticVerifier(kernel::Graph const &input_graph);
  ~ProbabilisticVerifier() override;
  ProbabilisticVerifier(ProbabilisticVerifier const &) = delete;
  ProbabilisticVerifier &operator=(ProbabilisticVerifier const &) = delete;
  virtual OutputMatch verify(kernel::Graph const &graph) override;

private:
  std::vector<cpu::CTensor> input_graph_fingerprints;
};

} // namespace search
} // namespace cleave_superoptimizer