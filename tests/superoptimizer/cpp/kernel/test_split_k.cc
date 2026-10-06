#include "cleave/superoptimizer/cpu/cmem_tensor.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/kernel/split.h"
#include "cleave/superoptimizer/type.h"
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

cleave_superoptimizer::kernel::DTensor
    new_auto_split_input(cleave_superoptimizer::kernel::Graph &graph,
                         std::vector<int> const &dims) {
  auto input = graph.new_input(dims);
  static_cast<cleave_superoptimizer::kernel::KNInputOp *>(
      graph.operators.back())
      ->infer_can_split = true;
  return input;
}

void free_ctensor_fingerprint(cleave_superoptimizer::cpu::CTensor &ctensor) {
  free(ctensor.fingerprint_ptr);
  ctensor.fingerprint_ptr = nullptr;
}

void eval_graph(cleave_superoptimizer::kernel::Graph &graph) {
  for (auto *op : graph.operators) {
    if (op->op_type == cleave_superoptimizer::type::KN_INPUT_OP) {
      static_cast<cleave_superoptimizer::kernel::KNInputOp *>(op)
          ->fingerprint();
    } else {
      op->fingerprint();
    }
  }
}

bool run_split_cpu_test(int split_dim_A, int split_dim_B) {
  cleave_superoptimizer::kernel::Graph graph0;
  int const M = 8, K = 8, N = 8;
  auto A0 = new_auto_split_input(graph0, {M, K});
  auto B0 = new_auto_split_input(graph0, {K, N});

  auto C0 = graph0.matmul(A0, B0);
  graph0.mark_output(C0);

  cleave_superoptimizer::kernel::Graph graph1; // split-k
  auto A1 = new_auto_split_input(graph1, {M, K});
  auto B1 = new_auto_split_input(graph1, {K, N});

  auto AS = graph1.split(A1, split_dim_A);
  auto BS = graph1.split(B1, split_dim_B);
  if (AS.dim[AS.num_dims - 1] != BS.dim[BS.num_dims - 2]) {
    // incompatible split
    return false;
  }
  auto CS = graph1.matmul(AS, BS);
  auto C1 = graph1.reduction(CS, 0);
  graph1.mark_output(C1);

  eval_graph(graph0);
  eval_graph(graph1);
  // auto C0_ctensor = C0.copy_fingerprint_to_ctensor();
  auto C1_ctensor = C1.copy_fingerprint_to_ctensor();

  cleave_superoptimizer::cpu::CTensor output_ref;

  bool matches = C0.has_same_fingerprint(C1_ctensor);
  free_ctensor_fingerprint(C1_ctensor);
  return matches;
}

cleave_superoptimizer::kernel::DTensor
    build_attn_ref(cleave_superoptimizer::kernel::Graph &graph, int s, int d) {
  auto Q = new_auto_split_input(graph, {s, d});
  auto K = new_auto_split_input(graph, {d, s});
  auto V = new_auto_split_input(graph, {s, d});

  auto A = graph.matmul(Q, K);
  auto E = graph.exp(A);
  auto S = graph.reduction(E, 1);
  auto D = graph.div(E, S);
  auto O = graph.matmul(D, V);
  graph.mark_output(O);
  return O;
}

bool run_attn_splitk_qk_test() {
  int const s = 8, d = 8;

  cleave_superoptimizer::kernel::Graph graph_ref;
  auto O0 = build_attn_ref(graph_ref, s, d);

  cleave_superoptimizer::kernel::Graph graph_split;
  auto Q1 = new_auto_split_input(graph_split, {s, d});
  auto K1 = new_auto_split_input(graph_split, {d, s});
  auto V1 = new_auto_split_input(graph_split, {s, d});

  auto QS = graph_split.split(Q1, 1);
  auto KS = graph_split.split(K1, 0);
  if (QS.dim[QS.num_dims - 1] != KS.dim[KS.num_dims - 2]) {
    return false;
  }
  auto AS = graph_split.matmul(QS, KS);
  auto A1 = graph_split.reduction(AS, 0);
  auto E1 = graph_split.exp(A1);
  auto S1 = graph_split.reduction(E1, 2);
  auto D1 = graph_split.div(E1, S1);
  auto O1 = graph_split.matmul(D1, V1);
  graph_split.mark_output(O1);

  eval_graph(graph_ref);
  eval_graph(graph_split);
  auto O1_ctensor = O1.copy_fingerprint_to_ctensor();
  bool matches = O0.has_same_fingerprint(O1_ctensor);
  free_ctensor_fingerprint(O1_ctensor);
  return matches;
}

bool run_attn_splitk_av_test() {
  int const s = 8, d = 8;

  cleave_superoptimizer::kernel::Graph graph_ref;
  auto O0 = build_attn_ref(graph_ref, s, d);

  cleave_superoptimizer::kernel::Graph graph_split;
  auto Q1 = new_auto_split_input(graph_split, {s, d});
  auto K1 = new_auto_split_input(graph_split, {d, s});
  auto V1 = new_auto_split_input(graph_split, {s, d});

  auto A1 = graph_split.matmul(Q1, K1);
  auto E1 = graph_split.exp(A1);
  auto S1 = graph_split.reduction(E1, 1);
  auto D1 = graph_split.div(E1, S1);
  auto DS = graph_split.split(D1, 1);
  auto VS = graph_split.split(V1, 0);
  if (DS.dim[DS.num_dims - 1] != VS.dim[VS.num_dims - 2]) {
    return false;
  }
  auto OS = graph_split.matmul(DS, VS);
  auto O1 = graph_split.reduction(OS, 0);
  graph_split.mark_output(O1);

  eval_graph(graph_ref);
  eval_graph(graph_split);
  auto O1_ctensor = O1.copy_fingerprint_to_ctensor();
  bool matches = O0.has_same_fingerprint(O1_ctensor);
  free_ctensor_fingerprint(O1_ctensor);
  return matches;
}

bool run_attn_splitk_flashdecode_test() {
  int const s = 8, d = 8;

  cleave_superoptimizer::kernel::Graph graph_ref;
  auto O0 = build_attn_ref(graph_ref, s, d);

  cleave_superoptimizer::kernel::Graph graph_split;
  auto Q1 = new_auto_split_input(graph_split, {s, d});
  auto K1 = new_auto_split_input(graph_split, {d, s});
  auto V1 = new_auto_split_input(graph_split, {s, d});

  auto A1 = graph_split.matmul(Q1, K1);
  auto AS = graph_split.split(A1, 1);
  auto ES = graph_split.exp(AS);
  auto SS = graph_split.reduction(ES, 2);
  auto S1 = graph_split.reduction(SS, 0);
  auto VS = graph_split.split(V1, 0);
  auto OS = graph_split.matmul(ES, VS);
  auto O1 = graph_split.reduction(OS, 0);
  auto OD = graph_split.div(O1, S1);
  graph_split.mark_output(OD);

  eval_graph(graph_ref);
  eval_graph(graph_split);
  auto OD_ctensor = OD.copy_fingerprint_to_ctensor();
  bool matches = O0.has_same_fingerprint(OD_ctensor);
  free_ctensor_fingerprint(OD_ctensor);
  return matches;
}

bool run_test(char const *name, bool (*test_fn)(), bool expected) {
  bool ok = test_fn() == expected;
  if (!ok) {
    std::cerr << "FAILED: " << name << "\n";
  } else {
    std::cout << "PASSED: " << name << "\n";
  }
  return ok;
}

} // namespace

int main() {
  int failures = 0;
  failures += !run_test(
      "split_dim_0_0", []() { return run_split_cpu_test(0, 0); }, false);
  failures += !run_test(
      "split_dim_0_1", []() { return run_split_cpu_test(0, 1); }, false);
  failures += !run_test(
      "split_dim_1_0", []() { return run_split_cpu_test(1, 0); }, true);
  failures += !run_test(
      "split_dim_1_1", []() { return run_split_cpu_test(1, 1); }, false);
  failures += !run_test(
      "attn_splitk_qk", []() { return run_attn_splitk_qk_test(); }, true);
  failures += !run_test(
      "attn_splitk_av", []() { return run_attn_splitk_av_test(); }, true);
  failures += !run_test(
      "attn_splitk_flashdecode",
      []() { return run_attn_splitk_flashdecode_test(); },
      true);

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
  }
  return failures == 0 ? 0 : 1;
}
