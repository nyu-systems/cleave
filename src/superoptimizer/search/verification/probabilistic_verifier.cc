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
// Input k's fingerprint is a function of k; a declared input's is its
// definition's over its sources' (the same in the target and every
// candidate).
void fingerprint_graph(kernel::Graph const &g) {
  int input_id = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      auto *input = static_cast<kernel::KNInputOp *>(op);
      if (input->declaration) {
        kernel::compute_declared_input_fingerprint(g, *input);
      } else {
        input->fingerprint(input_id);
      }
      input_id++;
    } else {
      op->fingerprint();
    }
  }
}

} // namespace

ProbabilisticVerifier::ProbabilisticVerifier(kernel::Graph const &input_graph) {
  ThreadFingerprintScope scope;
  fingerprint_graph(input_graph);

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

  fingerprint_graph(graph);

  for (auto const &op : graph.operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(graph, tensor) == 0 && !is_declared_input(op)) {
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