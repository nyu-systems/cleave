#include "cleave/superoptimizer/config.h"
#include "cleave/superoptimizer/cpu/cmem_tensor.h"
#include "cleave/superoptimizer/kernel/fingerprint_cpu.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/kernel/split.h"
#include "cleave/superoptimizer/type.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<size_t> make_strides(std::vector<int> const &dims) {
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

std::vector<bool> make_all_false_flags(size_t n) {
  return std::vector<bool>(n, false);
}

void free_ctensor_fingerprint(cleave_superoptimizer::cpu::CTensor &ctensor) {
  free(ctensor.fingerprint_ptr);
  ctensor.fingerprint_ptr = nullptr;
}

bool compare_fingerprints(
    cleave_superoptimizer::kernel::Graph &graph,
    std::vector<cleave_superoptimizer::kernel::DTensor> const &outputs) {
  cleave_superoptimizer::kernel::CpuFingerprintEngine engine;
  int input_id = 0;
  for (auto *op : graph.operators) {
    if (op->op_type == cleave_superoptimizer::type::KN_INPUT_OP) {
      auto *input_op =
          static_cast<cleave_superoptimizer::kernel::KNInputOp *>(op);
      input_op->fingerprint(input_id++);
      cleave_superoptimizer::cpu::CTensor input_ctensor =
          input_op->output_tensors[0].copy_fingerprint_to_ctensor();
      engine.load_fingerprint_from_ctensor(input_op->output_tensors[0],
                                           input_ctensor);
      free_ctensor_fingerprint(input_ctensor);
    } else {
      op->fingerprint();
      engine.compute(op);
    }
  }

  for (auto const &output : outputs) {
    cleave_superoptimizer::cpu::CTensor output_ctensor =
        output.copy_fingerprint_to_ctensor();
    bool matches = engine.matches(output, output_ctensor);
    free_ctensor_fingerprint(output_ctensor);
    if (!matches) {
      return false;
    }
  }
  return true;
}

bool run_matmul_test() {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> a_dims{2, 3};
  std::vector<int> b_dims{3, 4};
  auto A = graph.new_input(a_dims,
                           make_all_false_flags(a_dims.size()),
                           make_all_false_flags(a_dims.size()));
  auto B = graph.new_input(b_dims,
                           make_all_false_flags(b_dims.size()),
                           make_all_false_flags(b_dims.size()));
  auto C = graph.matmul(A, B);
  graph.mark_output(C);
  return compare_fingerprints(graph, {C});
}

bool run_matmul_broadcast(std::vector<int> a_dims, std::vector<int> b_dims) {
  cleave_superoptimizer::kernel::Graph graph;
  auto A = graph.new_input(a_dims,
                           make_all_false_flags(a_dims.size()),
                           make_all_false_flags(a_dims.size()));
  auto B = graph.new_input(b_dims,
                           make_all_false_flags(b_dims.size()),
                           make_all_false_flags(b_dims.size()));
  auto C = graph.matmul(A, B);
  graph.mark_output(C);

  cleave_superoptimizer::kernel::CpuFingerprintEngine engine;
  for (auto *op : graph.operators) {
    engine.compute(op);
  }
  auto A_ctensor = A.copy_fingerprint_to_ctensor();
  auto B_ctensor = B.copy_fingerprint_to_ctensor();
  auto C_ctensor_ref = C.copy_fingerprint_to_ctensor();
  std::memset(C_ctensor_ref.fingerprint_ptr,
              0,
              C.fingerprint_size()); // create an empty C to store ref

  size_t dtype_size = sizeof(cleave_superoptimizer::type::FingerprintType);
  int num_dims = C.num_dims;
  int batch_dims = num_dims - 2;
  int num_batches = 1, m = C.dim[num_dims - 2], k = A.dim[num_dims - 1],
      n = C.dim[num_dims - 1];
  for (int i = 0; i < batch_dims; i++) {
    num_batches *= C.dim[i];
  }

  auto a_strides = make_strides({A.dim, A.dim + A.num_dims});
  auto b_strides = make_strides({B.dim, B.dim + B.num_dims});
  auto c_strides = make_strides({C.dim, C.dim + C.num_dims});

  for (int b = 0; b < num_batches; ++b) {
    int tmp = b;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    int64_t c_offset = 0;
    for (int i = batch_dims - 1; i >= 0; --i) {
      int c_idx = tmp % C.dim[i];
      tmp /= C.dim[i];
      int a_idx = A.dim[i] == 1 ? 0 : c_idx;
      int b_idx = B.dim[i] == 1 ? 0 : c_idx;
      a_offset += static_cast<int64_t>(a_idx) * a_strides[i];
      b_offset += static_cast<int64_t>(b_idx) * b_strides[i];
      c_offset += static_cast<int64_t>(c_idx) * c_strides[i];
    }
    // construct a matmul graph
    cleave_superoptimizer::kernel::Graph graph_ref;
    auto a_ref_dims =
        std::vector<int>(a_dims.begin() + batch_dims, a_dims.end());
    auto b_ref_dims =
        std::vector<int>(b_dims.begin() + batch_dims, b_dims.end());
    auto A_ref = graph_ref.new_input(a_ref_dims,
                                     make_all_false_flags(a_ref_dims.size()),
                                     make_all_false_flags(a_ref_dims.size()));
    auto B_ref = graph_ref.new_input(b_ref_dims,
                                     make_all_false_flags(b_ref_dims.size()),
                                     make_all_false_flags(b_ref_dims.size()));
    auto C_ref = graph_ref.matmul(A_ref, B_ref);
    graph_ref.mark_output(C_ref);
    engine.load_fingerprint_from_buf(
        A_ref,
        (std::byte *)(A_ctensor.fingerprint_ptr + a_offset),
        dtype_size * m * k);
    engine.load_fingerprint_from_buf(
        B_ref,
        (std::byte *)(B_ctensor.fingerprint_ptr + b_offset),
        dtype_size * k * n);
    assert(graph_ref.operators[2]->op_type ==
           cleave_superoptimizer::type::KN_MATMUL_OP);
    engine.compute(graph_ref.operators[2]); // matmul op
    auto C_ref_ctensor_b = C_ref.copy_fingerprint_to_ctensor();

    assert(C_ref_ctensor_b.size() == dtype_size * m * n);
    std::memcpy(C_ctensor_ref.fingerprint_ptr + c_offset,
                C_ref_ctensor_b.fingerprint_ptr,
                dtype_size * m * n);
    free_ctensor_fingerprint(C_ref_ctensor_b);
  }
  auto matches = engine.matches(C, C_ctensor_ref);
  free_ctensor_fingerprint(A_ctensor);
  free_ctensor_fingerprint(B_ctensor);
  free_ctensor_fingerprint(C_ctensor_ref);
  return matches;
}

bool run_matmul_broadcast_rhs_test() {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> a_dims{2, 3, 4};
  std::vector<int> b_dims{1, 4, 5};
  return run_matmul_broadcast(a_dims, b_dims);
}

bool run_matmul_broadcast_lhs_test() {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> a_dims{1, 3, 4};
  std::vector<int> b_dims{2, 4, 5};
  return run_matmul_broadcast(a_dims, b_dims);
}

bool run_matmul_broadcast_multi_dim_test() {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> a_dims{2, 1, 3, 4};
  std::vector<int> b_dims{1, 4, 4, 5};
  return run_matmul_broadcast(a_dims, b_dims);
}

bool run_elementunary_test(cleave_superoptimizer::type::KNOperatorType op_type,
                           bool clamp) {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> dims{2, 4};
  auto input = graph.new_input(dims,
                               make_all_false_flags(dims.size()),
                               make_all_false_flags(dims.size()));
  cleave_superoptimizer::kernel::DTensor output;
  if (clamp) {
    output = graph.clamp(input, -1.0f, 1.0f);
  } else {
    output = graph.elementunary(input, op_type);
  }
  graph.mark_output(output);
  return compare_fingerprints(graph, {output});
}

bool run_elementbinary_test(
    cleave_superoptimizer::type::KNOperatorType op_type) {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> dims{2, 4};
  auto input1 = graph.new_input(dims,
                                make_all_false_flags(dims.size()),
                                make_all_false_flags(dims.size()));
  auto input2 = graph.new_input(dims,
                                make_all_false_flags(dims.size()),
                                make_all_false_flags(dims.size()));
  auto output = graph.elementbinary(input1, input2, op_type);
  graph.mark_output(output);
  return compare_fingerprints(graph, {output});
}

bool run_reduction_test(int dim) {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> dims{2, 3, 4};
  auto input = graph.new_input(dims,
                               make_all_false_flags(dims.size()),
                               make_all_false_flags(dims.size()));
  auto output = graph.reduction(input, dim, 1);
  graph.mark_output(output);
  return compare_fingerprints(graph, {output});
}

bool run_rms_norm_test() {
  cleave_superoptimizer::kernel::Graph graph;
  std::vector<int> dims{2, 4};
  auto input = graph.new_input(dims,
                               make_all_false_flags(dims.size()),
                               make_all_false_flags(dims.size()));
  auto output = graph.rms_norm(input, std::vector<int>{4});
  graph.mark_output(output);
  return compare_fingerprints(graph, {output});
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
  failures += !run_test("matmul", run_matmul_test);
  failures += !run_test("matmul_broadcast_rhs", run_matmul_broadcast_rhs_test);
  failures += !run_test("matmul_broadcast_lhs", run_matmul_broadcast_lhs_test);
  failures += !run_test("matmul_broadcast_multi_dim",
                        run_matmul_broadcast_multi_dim_test);
  failures += !run_test("elementunary_exp", []() {
    return run_elementunary_test(cleave_superoptimizer::type::KN_EXP_OP, false);
  });
  failures += !run_test("elementunary_silu", []() {
    return run_elementunary_test(cleave_superoptimizer::type::KN_SILU_OP,
                                 false);
  });
  failures += !run_test("elementunary_gelu", []() {
    return run_elementunary_test(cleave_superoptimizer::type::KN_GELU_OP,
                                 false);
  });
  failures += !run_test("elementunary_relu", []() {
    return run_elementunary_test(cleave_superoptimizer::type::KN_RELU_OP,
                                 false);
  });
  failures += !run_test("elementunary_clamp", []() {
    return run_elementunary_test(cleave_superoptimizer::type::KN_CLAMP_OP,
                                 true);
  });
  failures += !run_test("elementbinary_add", []() {
    return run_elementbinary_test(cleave_superoptimizer::type::KN_ADD_OP);
  });
  failures += !run_test("elementbinary_mul", []() {
    return run_elementbinary_test(cleave_superoptimizer::type::KN_MUL_OP);
  });
  failures += !run_test("elementbinary_div", []() {
    return run_elementbinary_test(cleave_superoptimizer::type::KN_DIV_OP);
  });
  failures +=
      !run_test("reduction_dim0", []() { return run_reduction_test(0); });
  failures +=
      !run_test("reduction_dim1", []() { return run_reduction_test(1); });
  failures +=
      !run_test("reduction_dim2", []() { return run_reduction_test(2); });
  failures += !run_test("rms_norm", run_rms_norm_test);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
  }
  return failures == 0 ? 0 : 1;
}
