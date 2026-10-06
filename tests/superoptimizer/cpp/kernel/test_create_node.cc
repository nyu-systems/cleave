#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/type.h"
#include <iostream>
#include <vector>

namespace {

std::vector<bool> make_all_true_flags(size_t n) {
  return std::vector<bool>(n, true);
}

bool check_dims(cleave_superoptimizer::kernel::DTensor const &tensor,
                std::vector<int> const &expected) {
  if (tensor.num_dims != static_cast<int>(expected.size())) {
    return false;
  }
  for (int i = 0; i < tensor.num_dims; i++) {
    if (tensor.dim[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

bool run_matmul_create_test(std::vector<int> const &a_dims,
                            std::vector<int> const &b_dims,
                            std::vector<int> const &expected_dims) {
  cleave_superoptimizer::kernel::Graph graph;
  auto A = graph.new_input(a_dims,
                           make_all_true_flags(a_dims.size()),
                           make_all_true_flags(a_dims.size()));
  auto B = graph.new_input(b_dims,
                           make_all_true_flags(b_dims.size()),
                           make_all_true_flags(b_dims.size()));
  auto *op = graph.create_matmul_op(A, B);
  if (expected_dims.empty()) {
    if (op != nullptr) {
      delete op;
      return false;
    }
    return true;
  }
  if (op == nullptr) {
    return false;
  }
  bool ok = check_dims(op->output_tensors[0], expected_dims);
  delete op;
  return ok;
}

bool run_matmul_prepend_lhs_test() {
  return run_matmul_create_test({3, 4}, {2, 4, 5}, {2, 3, 5});
}

bool run_matmul_prepend_rhs_test() {
  return run_matmul_create_test({2, 3, 4}, {4, 5}, {2, 3, 5});
}

bool run_matmul_incompatible_batch_test() {
  return run_matmul_create_test({2, 3, 4}, {3, 4, 5}, {});
}

bool run_matmul_incompatible_k_test() {
  return run_matmul_create_test({2, 3}, {4, 5}, {});
}

bool run_matmul_insufficient_dims_lhs_test() {
  return run_matmul_create_test({4}, {4, 5}, {});
}

bool run_matmul_insufficient_dims_rhs_test() {
  return run_matmul_create_test({2, 3}, {3}, {});
}

bool run_test(char const *name, bool (*test_fn)()) {
  bool ok = test_fn();
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
  failures += !run_test("matmul_prepend_lhs", run_matmul_prepend_lhs_test);
  failures += !run_test("matmul_prepend_rhs", run_matmul_prepend_rhs_test);
  failures +=
      !run_test("matmul_incompatible_batch", run_matmul_incompatible_batch_test);
  failures += !run_test("matmul_incompatible_k", run_matmul_incompatible_k_test);
  failures += !run_test("matmul_insufficient_dims_lhs",
                        run_matmul_insufficient_dims_lhs_test);
  failures += !run_test("matmul_insufficient_dims_rhs",
                        run_matmul_insufficient_dims_rhs_test);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
  }
  return failures == 0 ? 0 : 1;
}
