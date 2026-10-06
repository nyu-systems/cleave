#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/type.h"
#include <iostream>
#include <memory>
#include <vector>

namespace {

// Helper to get input tensor by index
cleave_superoptimizer::kernel::DTensor* get_input_tensor(cleave_superoptimizer::kernel::Graph& graph, int idx) {
  int count = 0;
  for (auto op : graph.operators) {
    if (op->op_type == cleave_superoptimizer::type::KN_INPUT_OP && op->output_tensors.size() > 0) {
      if (count == idx) {
        return &op->output_tensors[0];
      }
      count++;
    }
  }
  return nullptr;
}

// Test 1: Simple matmul should mark K dimension as reducible
bool test_matmul_k_dimension() {
  cleave_superoptimizer::kernel::Graph graph;

  // Create inputs: A [M, K], B [K, N]
  // Using empty vectors for can_reduce/can_split - should default to false
  auto A = graph.new_input({128, 64});
  auto B = graph.new_input({64, 128});

  // Matmul C = A @ B, result is [M, N]
  auto C = graph.matmul(A, B);

  // Mark output - this should trigger automatic inference
  graph.mark_output(C);

  // Get the actual input tensors
  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);
  cleave_superoptimizer::kernel::DTensor* input_B = get_input_tensor(graph, 1);

  if (!input_A || !input_B) {
    std::cerr << "Failed to retrieve input tensors\n";
    return false;
  }

  // Check expectations: K dimension should be marked as reducible
  // For A [M, K]: can_reduce should be [false, true] (K is reducible)
  // For B [K, N]: can_reduce should be [true, false] (K is reducible)
  bool a_m_reducible = input_A->can_reduce[0]; // M in A
  bool a_k_reducible = input_A->can_reduce[1]; // K in A
  bool b_k_reducible = input_B->can_reduce[0]; // K in B
  bool b_n_reducible = input_B->can_reduce[1]; // N in B

  if (!a_k_reducible || !b_k_reducible || b_n_reducible || a_m_reducible) {
    std::cerr << "K dimension not marked as reducible: "
              << "A[0]=" << a_m_reducible << " (expected false), "
              << "A[1]=" << a_k_reducible << " (expected true), "
              << "B[0]=" << b_k_reducible << " (expected true), "
              << "B[1]=" << b_n_reducible << " (expected false),\n";
    return false;
  }

  return true;
}

// Test 2: Matmul followed by reduction
bool test_matmul_with_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64});
  auto B = graph.new_input({64, 128});

  // Matmul C = A @ B, result is [M, N]
  auto C = graph.matmul(A, B);

  // Reduce over dimension 0 (M dimension)
  auto D = graph.reduction(C, 0);

  // Mark output
  graph.mark_output(D);

  // Get inputs
  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);
  cleave_superoptimizer::kernel::DTensor* input_B = get_input_tensor(graph, 1);

  if (!input_A || !input_B) {
    std::cerr << "Failed to retrieve input tensors\n";
    return false;
  }

  // Both K dimension and M dimension should be reducible
  // A [M, K]: both should be true
  // B [K, N]: K should be true
  bool a_m_reducible = input_A->can_reduce[0]; // M in A (reduced)
  bool a_k_reducible = input_A->can_reduce[1]; // K in A (contracted)
  bool b_k_reducible = input_B->can_reduce[0]; // K in B (contracted)
  bool b_n_reducible = input_B->can_reduce[1]; // N in B (not reduced)

  if (!a_m_reducible || !a_k_reducible || !b_k_reducible || b_n_reducible) {
    std::cerr << "Expected dimensions not marked as reducible: "
              << "A[0]=" << a_m_reducible << " (expected true), "
              << "A[1]=" << a_k_reducible << " (expected true), "
              << "B[0]=" << b_k_reducible << " (expected true), "
              << "B[1]=" << b_n_reducible << " (expected false)\n";
    return false;
  }

  return true;
}

// Test 3: Element-wise operations don't introduce reducibility
bool test_elementwise_no_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64});
  auto B = graph.exp(A);
  auto C = graph.square(B);

  graph.mark_output(C);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  // Element-wise ops don't reduce, so all should be false
  for (int i = 0; i < input_A->num_dims; i++) {
    if (input_A->can_reduce[i]) {
      std::cerr << "Unexpected reducibility in dimension " << i
                << " for pure element-wise graph\n";
      return false;
    }
  }

  return true;
}

// Test 4: Reduction operator marks the reduced dimension
bool test_direct_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64, 32});

  // Reduce over dimension 1
  auto B = graph.reduction(A, 1);

  graph.mark_output(B);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  // Only dimension 1 should be marked as reducible
  if (!input_A->can_reduce[1]) {
    std::cerr << "Reduced dimension not marked as reducible\n";
    return false;
  }

  if (input_A->can_reduce[0] || input_A->can_reduce[2]) {
    std::cerr << "Non-reduced dimensions incorrectly marked as reducible\n";
    return false;
  }

  return true;
}

// Test 5: Element-wise operations followed by reduction
bool test_elementwise_then_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64, 32});

  // Apply element-wise operations
  // A [128, 64, 32] -> B [128, 64, 32]
  auto B = graph.exp(A);
  // B [128, 64, 32] -> C [128, 64, 32]
  auto C = graph.square(B);
  // C [128, 64, 32] -> D [128, 64, 32]
  auto D = graph.silu(C);

  // Reduce over dimension 1
  // D [128, 64, 32] -> E [128, 32]
  auto E = graph.reduction(D, 1);

  graph.mark_output(E);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  // Dimension 1 should be marked as reducible (it's reduced at the end)
  if (!input_A->can_reduce[1]) {
    std::cerr << "Expected dimension 1 to be reducible after element-wise ops and reduction\n";
    return false;
  }

  // Dimensions 0 and 2 should not be reducible
  if (input_A->can_reduce[0] || input_A->can_reduce[2]) {
    std::cerr << "Dimensions 0 and 2 incorrectly marked as reducible: "
              << "A[0]=" << input_A->can_reduce[0] << " (expected false), "
              << "A[2]=" << input_A->can_reduce[2] << " (expected false)\n";
    return false;
  }

  return true;
}

// Test 6: auto-mode inputs infer can_split from can_reduce
bool test_auto_can_split_inference() {
  cleave_superoptimizer::kernel::Graph graph;

  // Create inputs: A [M, K], B [K, N]
  auto A = graph.new_input({128, 64});
  auto B = graph.new_input({64, 128});

  static_cast<cleave_superoptimizer::kernel::KNInputOp *>(graph.operators[0])
      ->infer_can_split = true;
  static_cast<cleave_superoptimizer::kernel::KNInputOp *>(graph.operators[1])
      ->infer_can_split = true;

  // Matmul C = A @ B, result is [M, N]
  auto C = graph.matmul(A, B);

  // Mark output - this should trigger automatic inference
  graph.mark_output(C);

  // Get the actual input tensors
  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);
  cleave_superoptimizer::kernel::DTensor* input_B = get_input_tensor(graph, 1);

  if (!input_A || !input_B) {
    std::cerr << "Failed to retrieve input tensors\n";
    return false;
  }

  // Verify can_split matches can_reduce for input A
  for (int i = 0; i < input_A->num_dims; i++) {
    if (input_A->can_split[i] != input_A->can_reduce[i]) {
      std::cerr << "Input A dimension " << i << ": can_split="
                << input_A->can_split[i] << " but can_reduce="
                << input_A->can_reduce[i] << " (should match)\n";
      return false;
    }
  }

  // Verify can_split matches can_reduce for input B
  for (int i = 0; i < input_B->num_dims; i++) {
    if (input_B->can_split[i] != input_B->can_reduce[i]) {
      std::cerr << "Input B dimension " << i << ": can_split="
                << input_B->can_split[i] << " but can_reduce="
                << input_B->can_reduce[i] << " (should match)\n";
      return false;
    }
  }

  // Specifically verify K dimension is both reducible and splittable
  if (!input_A->can_split[1] || !input_B->can_split[0]) {
    std::cerr << "K dimension not marked as splittable: "
              << "A[1]=" << input_A->can_split[1] << " (expected true), "
              << "B[0]=" << input_B->can_split[0] << " (expected true)\n";
    return false;
  }

  return true;
}

// Test 7: explicit can_split stays user-controlled when valid
bool test_explicit_can_split_validation() {
  cleave_superoptimizer::kernel::Graph graph;

  std::vector<bool> can_reduce = {false, false};
  std::vector<bool> a_can_split = {false, true};
  std::vector<bool> b_can_split = {true, false};

  auto A = graph.new_input({128, 64},
                           can_reduce, a_can_split);
  auto B = graph.new_input({64, 128},
                           can_reduce, b_can_split);

  auto C = graph.matmul(A, B);
  graph.mark_output(C);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);
  cleave_superoptimizer::kernel::DTensor* input_B = get_input_tensor(graph, 1);

  if (!input_A || !input_B) {
    std::cerr << "Failed to retrieve input tensors\n";
    return false;
  }

  if (!input_A->can_reduce[1] || !input_B->can_reduce[0]) {
    std::cerr << "Expected K dimensions to be inferred as reducible\n";
    return false;
  }

  if (!input_A->can_split[1] || !input_B->can_split[0]) {
    std::cerr << "Explicit can_split flags were not preserved: "
              << "A[1]=" << input_A->can_split[1] << " (expected true), "
              << "B[0]=" << input_B->can_split[0] << " (expected false)\n";
    return false;
  }

  if (input_A->can_split[0] || input_B->can_split[1]) {
    std::cerr << "Unexpected explicit can_split on non-requested dimensions\n";
    return false;
  }

  return true;
}

bool test_scalar_elementwise_no_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64});
  auto B = graph.div(A, 2.0f);
  auto C = graph.add(B, -3.0f);

  graph.mark_output(C);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  for (int i = 0; i < input_A->num_dims; i++) {
    if (input_A->can_reduce[i]) {
      std::cerr << "Unexpected reducibility in dimension " << i
                << " for scalar element-wise graph\n";
      return false;
    }
  }

  return true;
}

bool test_scalar_elementwise_then_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64, 32});
  auto B = graph.div(A, 2.0f);
  auto C = graph.add(B, -3.0f);
  auto D = graph.reduction(C, 1);

  graph.mark_output(D);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  if (!input_A->can_reduce[1]) {
    std::cerr << "Expected scalar element-wise chain to propagate reduced "
                 "dimension 1\n";
    return false;
  }

  if (input_A->can_reduce[0] || input_A->can_reduce[2]) {
    std::cerr << "Scalar element-wise chain incorrectly marked non-reduced "
              << "dimensions: A[0]=" << input_A->can_reduce[0]
              << " (expected false), A[2]=" << input_A->can_reduce[2]
              << " (expected false)\n";
    return false;
  }

  return true;
}

bool test_scalar_left_div_then_reduction() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64, 32});
  auto B = graph.div(2.0f, A);
  auto C = graph.reduction(B, 2);

  graph.mark_output(C);

  cleave_superoptimizer::kernel::DTensor* input_A = get_input_tensor(graph, 0);

  if (!input_A) {
    std::cerr << "Failed to retrieve input tensor\n";
    return false;
  }

  if (!input_A->can_reduce[2]) {
    std::cerr << "Expected scalar-left div to propagate reduced dimension 2\n";
    return false;
  }

  if (input_A->can_reduce[0] || input_A->can_reduce[1]) {
    std::cerr << "Scalar-left div incorrectly marked non-reduced dimensions: "
              << "A[0]=" << input_A->can_reduce[0] << " (expected false), "
              << "A[1]=" << input_A->can_reduce[1] << " (expected false)\n";
    return false;
  }

  return true;
}

bool test_symbolic_scalar_binary_conversion() {
  cleave_superoptimizer::kernel::Graph graph;

  auto A = graph.new_input({128, 64});
  auto B = graph.div(2.0f, A);
  auto C = graph.add(B, -3.0f);

  graph.mark_output(C);

  std::unique_ptr<cleave_superoptimizer::kernel::Graph> symbolic_graph(
      graph.create_symbolic_graph());

  int scalar_div_count = 0;
  int scalar_add_count = 0;

  for (auto *op : symbolic_graph->operators) {
    if (op->op_type != cleave_superoptimizer::type::KN_DIV_OP &&
        op->op_type != cleave_superoptimizer::type::KN_ADD_OP) {
      continue;
    }

    auto *bin_op =
        dynamic_cast<cleave_superoptimizer::kernel::KNElementBinaryOp *>(op);
    if (!bin_op || !bin_op->has_scalar) {
      std::cerr << "Expected symbolic binary op to preserve scalar metadata\n";
      return false;
    }

    if (op->input_tensors.size() != 1 || op->output_tensors.size() != 1) {
      std::cerr << "Expected symbolic scalar binary op to have one input and "
                   "one output\n";
      return false;
    }

    if (op->op_type == cleave_superoptimizer::type::KN_DIV_OP) {
      scalar_div_count++;
      if (bin_op->scalar != 2.0f || !bin_op->scalar_on_left) {
        std::cerr << "Symbolic scalar-left div metadata mismatch\n";
        return false;
      }
    } else if (op->op_type == cleave_superoptimizer::type::KN_ADD_OP) {
      scalar_add_count++;
      if (bin_op->scalar != -3.0f || bin_op->scalar_on_left) {
        std::cerr << "Symbolic scalar add metadata mismatch\n";
        return false;
      }
    }
  }

  if (scalar_div_count != 1 || scalar_add_count != 1) {
    std::cerr << "Expected one scalar div and one scalar add in symbolic graph, "
              << "found div=" << scalar_div_count
              << ", add=" << scalar_add_count << "\n";
    return false;
  }

  return true;
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
  std::cout << "Testing automatic reducibility inference...\n\n";

  int failures = 0;
  failures += !run_test("matmul_k_dimension", test_matmul_k_dimension);
  failures += !run_test("matmul_with_reduction", test_matmul_with_reduction);
  failures += !run_test("elementwise_no_reduction", test_elementwise_no_reduction);
  failures += !run_test("direct_reduction", test_direct_reduction);
  failures += !run_test("elementwise_then_reduction", test_elementwise_then_reduction);
  failures += !run_test("auto_can_split_inference", test_auto_can_split_inference);
  failures += !run_test("explicit_can_split_validation", test_explicit_can_split_validation);
  failures += !run_test("scalar_elementwise_no_reduction", test_scalar_elementwise_no_reduction);
  failures += !run_test("scalar_elementwise_then_reduction", test_scalar_elementwise_then_reduction);
  failures += !run_test("scalar_left_div_then_reduction", test_scalar_left_div_then_reduction);
  failures += !run_test("symbolic_scalar_binary_conversion", test_symbolic_scalar_binary_conversion);

  if (failures != 0) {
    std::cerr << "\n" << failures << " test(s) failed\n";
  } else {
    std::cout << "\nAll tests passed!\n";
  }

  return failures == 0 ? 0 : 1;
}
