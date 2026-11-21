// The symbolic dims of a graph do not depend on the guids of its tensors.
//
// Guids come from one counter for the whole process, and the threads of a
// running search draw from it too, so the tensors of one graph need not have
// consecutive guids. The symbols must be the ones the graph gets when it is
// the first thing a process builds.
#include "cleave/superoptimizer/kernel/graph.h"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

namespace kn = cleave_superoptimizer::kernel;
using kn::DTensor;

using Dims = std::vector<std::vector<int>>;

// e = exp(q @ k^T / 8 - 6.25); (e @ v) / rowsum(e). gaps[i] tensors of
// another graph are created right before the i-th tensor of this one.
// Returns the symbolic dims of q, k and v.
Dims symbolic_inputs(std::vector<int> const &gaps) {
  kn::Graph g, other;
  size_t next = 0;
  auto gap = [&]() {
    int n = next < gaps.size() ? gaps[next] : 0;
    next++;
    for (int i = 0; i < n; i++) {
      other.new_input({2, 2});
    }
  };
  gap();
  DTensor q = g.new_input({32, 4096, 64});
  gap();
  DTensor k = g.new_input({32, 2048, 64});
  gap();
  DTensor v = g.new_input({32, 2048, 64});
  gap();
  DTensor s = g.matmul(q, k, /*trans_A=*/false, /*trans_B=*/true);
  gap();
  DTensor d = g.div(s, 8.0f);
  gap();
  DTensor a = g.add(d, -6.25f);
  gap();
  DTensor e = g.exp(a);
  gap();
  DTensor o = g.matmul(e, v);
  gap();
  DTensor l = g.reduce_sum(e, 2);
  gap();
  DTensor r = g.div(o, l);
  g.mark_output(r, /*infer_output_reducibility=*/false);

  std::unordered_map<size_t, std::vector<int>> dims;
  std::unique_ptr<kn::Graph> symbolic(g.create_symbolic_graph(&dims));
  return {dims.at(q.guid), dims.at(k.guid), dims.at(v.guid)};
}

void show(char const *what, Dims const &dims) {
  std::cerr << what << ":";
  for (auto const &tensor : dims) {
    std::cerr << " [";
    for (int d : tensor) {
      std::cerr << " " << d;
    }
    std::cerr << " ]";
  }
  std::cerr << std::endl;
}

} // namespace

int main() {
  int const n = 10; // tensors of the graph
  Dims const fresh = symbolic_inputs({});
  // five groups: heads, S_Q, S_KV, the contracted D, and v's D
  bool ok = fresh[0][0] == fresh[1][0] && fresh[1][0] == fresh[2][0] &&
            fresh[1][1] == fresh[2][1] && fresh[0][2] == fresh[1][2] &&
            fresh[0][1] != fresh[1][1] && fresh[0][2] != fresh[2][2];
  if (!ok) {
    show("unexpected groups", fresh);
  }

  std::vector<std::vector<int>> patterns;
  for (int at = 1; at < n; at++) { // one foreign tensor, at each place
    std::vector<int> gaps(n, 0);
    gaps[at] = 1;
    patterns.push_back(gaps);
  }
  patterns.push_back(std::vector<int>(n, 1));
  patterns.push_back({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  unsigned state = 12345; // fixed pseudo-random gaps of 0..39 tensors
  for (int p = 0; p < 8; p++) {
    std::vector<int> gaps;
    for (int i = 0; i < n; i++) {
      state = state * 1103515245u + 12345u;
      gaps.push_back(static_cast<int>((state >> 16) % 40));
    }
    patterns.push_back(gaps);
  }
  // ... and the graph as a later graph of the process, without gaps
  patterns.push_back(std::vector<int>(n, 0));

  for (size_t p = 0; p < patterns.size(); p++) {
    Dims const got = symbolic_inputs(patterns[p]);
    if (got != fresh) {
      ok = false;
      std::cerr << "gap pattern " << p << ": other symbols" << std::endl;
      show("  first graph of the process", fresh);
      show("  this graph", got);
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << ": symbolic dims of "
            << patterns.size() << " builds against the first" << std::endl;
  return ok ? 0 : 1;
}
