#include "cleave/superoptimizer/search/verification/probabilistic_verifier.h"
#include "cleave/superoptimizer/kernel/fingerprint_cpu.h"
#include "cleave/superoptimizer/search/op_utils.h"

#include <cstdlib>

namespace cleave_superoptimizer {
namespace search {

namespace {
// Fingerprints live in a thread-local store; drop them when a pass ends so
// nothing outlives one verification.
struct ThreadFingerprintScope {
  ~ThreadFingerprintScope() {
    kernel::clear_thread_fingerprints();
  }
};
} // namespace

ProbabilisticVerifier::ProbabilisticVerifier(kernel::Graph const &input_graph) {
  ThreadFingerprintScope scope;
  int input_id = 0;
  for (auto const &op : input_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      static_cast<kernel::KNInputOp *>(op)->fingerprint(input_id++);
    } else {
      op->fingerprint();
    }
  }

  for (kernel::KNOperator *op : input_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      input_graph_fingerprints.push_back(
          op->input_tensors[0].copy_fingerprint_to_ctensor());
    }
  }
}

ProbabilisticVerifier::~ProbabilisticVerifier() {
  for (auto &t : input_graph_fingerprints) {
    free(t.fingerprint_ptr);
  }
}

OutputMatch ProbabilisticVerifier::verify(kernel::Graph const &graph) {
  ThreadFingerprintScope scope;

  std::vector<kernel::DTensor> fingerprints;

  int input_id = 0;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      static_cast<kernel::KNInputOp *>(op)->fingerprint(input_id++);
    } else {
      op->fingerprint();
    }
  }

  for (auto const &op : graph.operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(graph, tensor) == 0) {
        fingerprints.push_back(tensor);
      }
    }
  }

  assert(fingerprints.size() == input_graph_fingerprints.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (!fingerprints[match[i]].has_same_fingerprint(
              input_graph_fingerprints[i])) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(fingerprints.size());
  do {
    if (verify_with_match(match)) {
      assert(match.is_valid());
      return match;
    }
  } while (match.next());
  return OutputMatch::invalid_match();
}

} // namespace search
} // namespace cleave_superoptimizer