#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/utils/hash_utils.h"

// Symbolic graph construction
#include "cleave/superoptimizer/symbolic/dimension_group.h"
#include "cleave/superoptimizer/symbolic/grouping_rules.h"
#include "cleave/superoptimizer/utils/logging.h"

// Operator headers for dynamic_cast in symbolic graph creation
#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/element_unary.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"

#include <algorithm>
#include <iostream>

namespace cleave_superoptimizer {
namespace kernel {

Graph::Graph() {}

Graph::~Graph() {
  while (!operators.empty()) {
    delete operators.back();
    operators.pop_back();
  }
}

size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
  size_t h1 = std::hash<int>{}(p.first);
  size_t h2 = std::hash<int>{}(p.second);
  hash_combine(h1, h2);
  return h1;
}

int Graph::get_input_dtensors(DTensor **inputs) {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == cleave_superoptimizer::type::KN_INPUT_OP) {
      assert(op->output_tensors.size() == 1);
      inputs[num_inputs++] = &op->output_tensors[0];
    }
  }
  return num_inputs;
}

bool Graph::has_split_input() const {
  for (auto const &op : operators) {
    if (op->op_type != cleave_superoptimizer::type::KN_INPUT_OP) {
      continue;
    }
    for (auto const &tensor : op->output_tensors) {
      for (int i = 0; i < tensor.num_dims; i++) {
        if (tensor.can_split[i]) {
          return true;
        }
      }
    }
  }
  return false;
}

Graph *Graph::create_symbolic_graph(
    std::unordered_map<size_t, std::vector<int>>
        *symbolic_dims_by_guid) const {
  // Import symbolic namespace classes
  using namespace cleave_superoptimizer::symbolic;

  // Step 1: Grouping pass - traverse all operators and group dimensions
  DimensionGrouper grouper;
  for (auto const &op : operators) {
    GroupingRules::apply_grouping(op, grouper);
  }

  // Step 2: Symbolization pass - assign distinct symbols to groups
  SymbolicGraphConfig config;
  config.base_symbol = 6;
  config.use_prime_numbers = true;
  config.symbol_increment = 4;

  DimensionSymbolizer symbolizer(grouper, config);

  // Step 3: Create new symbolic graph
  Graph *symbolic_graph = new Graph();

  // Map from original tensor GUID to symbolic tensor
  std::unordered_map<size_t, DTensor> guid_to_symbolic_tensor;

  // Step 4: Traverse operators in topological order and recreate with symbolic dims
  for (auto const &op : operators) {
    using namespace cleave_superoptimizer::type;

    switch (op->op_type) {
      case KN_INPUT_OP: {
        // Get original input tensor
        assert(op->output_tensors.size() == 1);
        DTensor const &orig_tensor = op->output_tensors[0];

        // Get symbolic dimensions
        std::vector<int> sym_dims =
            symbolizer.get_symbolic_dims(orig_tensor.guid, orig_tensor.num_dims);

        // Create symbolic input with same attributes but symbolic dimensions
        std::vector<bool> can_reduce_vec(
            orig_tensor.can_reduce,
            orig_tensor.can_reduce + orig_tensor.num_dims);
        std::vector<bool> can_split_vec(
            orig_tensor.can_split,
            orig_tensor.can_split + orig_tensor.num_dims);

        DTensor sym_tensor =
            symbolic_graph->new_input(sym_dims, can_reduce_vec, can_split_vec);

        // Store mapping
        guid_to_symbolic_tensor[orig_tensor.guid] = sym_tensor;

        if (symbolic_dims_by_guid) {
          (*symbolic_dims_by_guid)[orig_tensor.guid] = sym_dims;
        }
        break;
      }

      case KN_MATMUL_OP: {
        assert(op->input_tensors.size() == 2);
        DTensor const &A = op->input_tensors[0];
        DTensor const &B = op->input_tensors[1];

        DTensor sym_A = guid_to_symbolic_tensor[A.guid];
        DTensor sym_B = guid_to_symbolic_tensor[B.guid];
        auto const *matmul_op = dynamic_cast<KNMatmulOp const *>(op);
        assert(matmul_op != nullptr);
        DTensor sym_C = symbolic_graph->matmul(
            sym_A, sym_B, matmul_op->trans_A, matmul_op->trans_B);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_C;
        break;
      }

      case KN_ADD_OP:
      case KN_MUL_OP:
      case KN_DIV_OP: {
        auto const *bin_op = dynamic_cast<KNElementBinaryOp const *>(op);
        if (bin_op && bin_op->has_scalar) {
          assert(op->input_tensors.size() == 1);
          DTensor const &input = op->input_tensors[0];

          DTensor sym_input = guid_to_symbolic_tensor[input.guid];
          DTensor sym_output =
              symbolic_graph->elementbinary(sym_input,
                                            bin_op->scalar,
                                            bin_op->scalar_on_left,
                                            op->op_type);

          guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
          break;
        }

        assert(op->input_tensors.size() == 2);
        DTensor const &A = op->input_tensors[0];
        DTensor const &B = op->input_tensors[1];

        DTensor sym_A = guid_to_symbolic_tensor[A.guid];
        DTensor sym_B = guid_to_symbolic_tensor[B.guid];
        DTensor sym_C = symbolic_graph->elementbinary(sym_A, sym_B, op->op_type);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_C;
        break;
      }

      case KN_EXP_OP:
      case KN_SQUARE_OP:
      case KN_SQRT_OP:
      case KN_SILU_OP:
      case KN_SIGMOID_OP:
      case KN_GELU_OP:
      case KN_RELU_OP:
      case KN_LOG_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &input = op->input_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];
        DTensor sym_output = symbolic_graph->elementunary(sym_input, op->op_type);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
        break;
      }

      case KN_CLAMP_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &input = op->input_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];

        // Get min/max values from the clamp operator
        auto *clamp_op = dynamic_cast<kernel::KNClampUnaryOp *>(op);
        float min_val = clamp_op->min_val;
        float max_val = clamp_op->max_val;

        DTensor sym_output = symbolic_graph->clamp(sym_input, min_val, max_val);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
        break;
      }

      case KN_REDUCTION_0_OP:
      case KN_REDUCTION_1_OP:
      case KN_REDUCTION_2_OP:
      case KN_REDUCTION_3_OP:
      case KN_REDUCTION_4_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &input = op->input_tensors[0];
        DTensor const &output = op->output_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];

        // Determine reduction dimension from operator type
        int reduction_dim = op->op_type - KN_REDUCTION_0_OP;

        // Get reduction size from output dimension
        int reduction_size = output.dim[reduction_dim];

        DTensor sym_output =
            symbolic_graph->reduction(sym_input, reduction_dim, reduction_size);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
        break;
      }

      case KN_RMS_NORM_OP: {
        assert(false); // TODO: disable for safety because normalized_shape is not stored
        assert(op->input_tensors.size() >= 1);
        DTensor const &input = op->input_tensors[0];
        DTensor const &output = op->output_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];

        // RMS norm normalizes over the last N dimensions
        // We need to reconstruct the normalized_shape from the input dimensions
        // The KNRMSNormOp only stores normalized_size (product), not the shape
        // Since output has same shape as input, we can use the output's dimensions
        // For now, we'll normalize over the last dimension as a common case
        // A more complete solution would store normalized_shape in KNRMSNormOp
        std::vector<int> normalized_shape;
        // Assume normalization over last dimension (most common case)
        normalized_shape.push_back(sym_input.dim[sym_input.num_dims - 1]);

        DTensor sym_output;
        if (op->input_tensors.size() == 1) {
          sym_output = symbolic_graph->rms_norm(sym_input, normalized_shape);
        } else {
          // Has elementwise affine parameter
          DTensor const &affine = op->input_tensors[1];
          DTensor sym_affine = guid_to_symbolic_tensor[affine.guid];
          sym_output =
              symbolic_graph->rms_norm(sym_input, sym_affine, normalized_shape);
        }

        guid_to_symbolic_tensor[output.guid] = sym_output;
        break;
      }

      case KN_SPLIT_0_OP:
      case KN_SPLIT_1_OP:
      case KN_SPLIT_2_OP:
      case KN_SPLIT_3_OP:
      case KN_SPLIT_4_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &input = op->input_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];

        // Determine split dimension from operator type
        int split_dim = op->op_type - KN_SPLIT_0_OP;

        DTensor sym_output = symbolic_graph->split(sym_input, split_dim);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
        break;
      }


      case KN_OUTPUT_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &orig_output = op->input_tensors[0];
        DTensor sym_output = guid_to_symbolic_tensor[orig_output.guid];
        symbolic_graph->mark_output(sym_output);
        break;
      }

      default:
        // Unknown operator - skip
        break;
    }
  }

  return symbolic_graph;
}

// Helper function to get actual input tensor (not the copy)
// Returns a reference to the original tensor from the producer operator
inline DTensor &get_input_tensor(KNOperator *op, int idx) {
  DTensor const &input_copy = op->input_tensors[idx];
  assert(input_copy.owner_op != nullptr && "Input tensor must have owner");
  return input_copy.owner_op->output_tensors[input_copy.owner_ts_idx];
}

void Graph::infer_reducibility() {
  // Phase 1: Initialize all tensors' can_reduce to false
  // This includes the graph output (the input to KN_OUTPUT_OP)
  for (KNOperator *op : operators) {
    for (DTensor &tensor : op->output_tensors) {
      for (int i = 0; i < tensor.num_dims; i++) {
        tensor.can_reduce[i] = false;
      }
    }
  }

  // Phase 2: Reverse topological traversal
  // Start from the end (output op) and work backward to inputs
  // The output op won't propagate anything (its inputs remain false)
  // Other operators propagate true values backward based on their semantics
  for (int op_idx = operators.size() - 1; op_idx >= 0; op_idx--) {
    KNOperator *op = operators[op_idx];
    propagate_reducibility_backward(op);
  }

  // Phase 3: Finalize input can_split. Auto-mode inputs infer it from
  // can_reduce; explicit inputs are validated against inferred can_reduce.
  for (KNOperator *op : operators) {
    if (op->op_type == type::KN_INPUT_OP) {
      auto *input_op = static_cast<KNInputOp *>(op);
      for (DTensor &tensor : op->output_tensors) {
        for (int i = 0; i < tensor.num_dims; i++) {
          if (input_op->infer_can_split) {
            tensor.can_split[i] = tensor.can_reduce[i];
          } else if (tensor.can_split[i] && !tensor.can_reduce[i]) {
            CSO_LOG(FATAL)
                << "Input tensor " << tensor.guid << " dimension " << i
                << " was marked can_split=true, but inferred can_reduce=false. "
                   "Requested input split dimensions must be reducible.";
          }
        }
      }
    }
  }
}

void Graph::propagate_reducibility_backward(KNOperator *op) {
  switch (op->op_type) {
    case type::KN_INPUT_OP:
      // Terminal node - no propagation needed
      break;

    case type::KN_OUTPUT_OP:
      // Output dimensions remain false (not reduced in output)
      break;

    case type::KN_REDUCTION_0_OP:
    case type::KN_REDUCTION_1_OP:
    case type::KN_REDUCTION_2_OP:
    case type::KN_REDUCTION_3_OP:
    case type::KN_REDUCTION_4_OP:
      propagate_reduction_op(op);
      break;

    case type::KN_SPLIT_0_OP:
    case type::KN_SPLIT_1_OP:
    case type::KN_SPLIT_2_OP:
    case type::KN_SPLIT_3_OP:
    case type::KN_SPLIT_4_OP:
      // input graph will not contain split op
      break;

    case type::KN_MATMUL_OP:
      propagate_matmul_op(op);
      break;

    case type::KN_EXP_OP:
    case type::KN_SILU_OP:
    case type::KN_SQUARE_OP:
    case type::KN_SQRT_OP:
    case type::KN_SIGMOID_OP:
    case type::KN_GELU_OP:
    case type::KN_RELU_OP:
    case type::KN_LOG_OP:
      propagate_elementwise_unary_op(op);
      break;

    case type::KN_CLAMP_OP:
      propagate_elementwise_unary_op(op);
      break;

    case type::KN_ADD_OP:
    case type::KN_MUL_OP:
    case type::KN_DIV_OP:
      if (auto *bin_op = dynamic_cast<KNElementBinaryOp *>(op);
          bin_op && bin_op->has_scalar) {
        propagate_elementwise_unary_op(op);
      } else {
        propagate_elementwise_binary_op(op);
      }
      break;

    case type::KN_RMS_NORM_OP:
      propagate_rms_norm_op(op);
      break;

    default:
      // Unknown operator - conservative: don't propagate
      break;
  }
}

void Graph::propagate_reduction_op(KNOperator *op) {
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  DTensor const &output = op->output_tensors[0];
  DTensor &input = get_input_tensor(op, 0);

  int reduction_dim = op->op_type - type::KN_REDUCTION_0_OP;

  // Rule 1: Mark the reduced dimension as can_reduce=true
  if (reduction_dim < input.num_dims) {
    input.can_reduce[reduction_dim] = true;
  }

  // Rule 2: Propagate output dimensions to matching input dimensions
  // Output has same dimensions as input, just reduced_dim has different size
  for (int i = 0; i < output.num_dims; i++) {
    if (output.can_reduce[i]) {
      input.can_reduce[i] = true;
    }
  }
}

void Graph::propagate_matmul_op(KNOperator *op) {
  assert(op->input_tensors.size() == 2);
  assert(op->output_tensors.size() == 1);

  DTensor const &output = op->output_tensors[0]; // C [..., M, N]
  DTensor &input1 = get_input_tensor(op, 0);     // A [..., M, K]
  DTensor &input2 = get_input_tensor(op, 1);     // B [..., K, N]

  int num_dims = output.num_dims;

  // Rule 1: Propagate batch dimensions (all dims except last 2)
  for (int i = 0; i < num_dims - 2; i++) {
    if (output.can_reduce[i]) {
      if (i < input1.num_dims - 2) {
        input1.can_reduce[i] = true;
      }
      if (i < input2.num_dims - 2) {
        input2.can_reduce[i] = true;
      }
    }
  }

  // Rule 2: Output dim M (num_dims-2) maps to input1 dim M (input1.num_dims-2)
  if (output.can_reduce[num_dims - 2]) {
    input1.can_reduce[input1.num_dims - 2] = true;
  }

  // Rule 3: Output dim N (num_dims-1) maps to input2 dim N (input2.num_dims-1)
  if (output.can_reduce[num_dims - 1]) {
    input2.can_reduce[input2.num_dims - 1] = true;
  }

  // Rule 4: K dimension is reduced (always set to true)
  input1.can_reduce[input1.num_dims - 1] = true; // K in A
  input2.can_reduce[input2.num_dims - 2] = true; // K in B
}

void Graph::propagate_elementwise_unary_op(KNOperator *op) {
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  DTensor const &output = op->output_tensors[0];
  DTensor &input = get_input_tensor(op, 0);

  // Identity mapping: output dim i → input dim i
  for (int i = 0; i < output.num_dims; i++) {
    if (output.can_reduce[i]) {
      input.can_reduce[i] = true;
    }
  }
}

void Graph::propagate_elementwise_binary_op(KNOperator *op) {
  assert(op->input_tensors.size() == 2);
  assert(op->output_tensors.size() == 1);

  DTensor const &output = op->output_tensors[0];
  DTensor &input1 = get_input_tensor(op, 0);
  DTensor &input2 = get_input_tensor(op, 1);

  // Propagate to both inputs (handle broadcasting)
  for (int i = 0; i < output.num_dims; i++) {
    if (output.can_reduce[i]) {
      if (i < input1.num_dims) {
        input1.can_reduce[i] = true;
      }
      if (i < input2.num_dims) {
        input2.can_reduce[i] = true;
      }
    }
  }
}

void Graph::propagate_rms_norm_op(KNOperator *op) {
  // RMS norm normalizes over certain dimensions
  // For now, treat like element-wise unary
  propagate_elementwise_unary_op(op);
}

void to_json(json &j, Graph const &g) {
  for (KNOperator *const op : g.operators) {
    j.push_back(json(*op));
  }
}

void from_json(json const &j, Graph &g) {
  std::unordered_map<size_t, size_t>
      guid_mapping; // from deseralized guid to json guid

  auto get_tensor_from_guid = [&](size_t guid) {
    for (auto const &op : g.operators) {
      for (DTensor const &dtensor : op->output_tensors) {
        if (guid_mapping.at(dtensor.guid) == guid) {
          return dtensor;
        }
      }
    }
    assert(false);
  };

  for (json const &jop : j) {
    type::KNOperatorType op_type;
    jop.at("op_type").get_to(op_type);
    switch (op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        int num_dim, dim[MAX_TENSOR_DIMS];
        bool can_reduce_arr[MAX_TENSOR_DIMS];
        bool can_split_arr[MAX_TENSOR_DIMS];
        size_t guidO;
        jop.at("output_tensors")[0].at("num_dims").get_to(num_dim);
        jop.at("output_tensors")[0].at("dim").get_to(dim);
        jop.at("output_tensors")[0].at("can_reduce").get_to(can_reduce_arr);
        jop.at("output_tensors")[0].at("can_split").get_to(can_split_arr);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        std::vector<int> dims = to_vector(num_dim, dim);
        std::vector<bool> can_reduce = to_vector(num_dim, can_reduce_arr);
        std::vector<bool> can_split = to_vector(num_dim, can_split_arr);
        DTensor const &output = g.new_input(dims, can_reduce, can_split);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_OUTPUT_OP: {
        size_t guid;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        g.mark_output(get_tensor_from_guid(guid),
                      /*infer_output_reducibility=*/false);
        break;
      }
      case type::KNOperatorType::KN_MATMUL_OP: {
        size_t guidA, guidB, guidO;
        bool trans_A = false;
        bool trans_B = false;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        if (jop.contains("trans_A")) {
          jop.at("trans_A").get_to(trans_A);
        }
        if (jop.contains("trans_B")) {
          jop.at("trans_B").get_to(trans_B);
        }
        DTensor const &output =
            g.matmul(get_tensor_from_guid(guidA),
                     get_tensor_from_guid(guidB),
                     trans_A,
                     trans_B);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_EXP_OP:
      case type::KNOperatorType::KN_SQUARE_OP:
      case type::KNOperatorType::KN_SQRT_OP:
      case type::KNOperatorType::KN_SILU_OP:
      case type::KNOperatorType::KN_GELU_OP:
      case type::KNOperatorType::KN_RELU_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.elementunary(get_tensor_from_guid(guid), op_type);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_CLAMP_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.elementunary_clamp(get_tensor_from_guid(guid), type::CLAMP_MIN_MAX["min_val"], type::CLAMP_MIN_MAX["max_val"]);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_DIV_OP:
      case type::KNOperatorType::KN_ADD_OP:
      case type::KNOperatorType::KN_MUL_OP: {
        size_t guidA, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &input = get_tensor_from_guid(guidA);
        if (jop.contains("scalar")) {
          float scalar = jop.at("scalar").get<float>();
          bool scalar_on_left = jop.value("scalar_on_left", false);
          DTensor const &output =
              g.elementbinary(input, scalar, scalar_on_left, op_type);
          guid_mapping[output.guid] = guidO;
        } else {
          size_t guidB;
          jop.at("input_tensors")[1].at("guid").get_to(guidB);
          DTensor const &output = g.elementbinary(
              input, get_tensor_from_guid(guidB), op_type);
          guid_mapping[output.guid] = guidO;
        }
        break;
      }
      case type::KNOperatorType::KN_REDUCTION_0_OP:
      case type::KNOperatorType::KN_REDUCTION_1_OP:
      case type::KNOperatorType::KN_REDUCTION_2_OP:
      case type::KNOperatorType::KN_REDUCTION_3_OP:
      case type::KNOperatorType::KN_REDUCTION_4_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.reduction(get_tensor_from_guid(guid),
                        op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_SPLIT_0_OP:
      case type::KNOperatorType::KN_SPLIT_1_OP:
      case type::KNOperatorType::KN_SPLIT_2_OP:
      case type::KNOperatorType::KN_SPLIT_3_OP:
      case type::KNOperatorType::KN_SPLIT_4_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.split(get_tensor_from_guid(guid),
                    op_type - type::KNOperatorType::KN_SPLIT_0_OP);
        guid_mapping[output.guid] = guidO;
        break;
      }
      default:
        assert(false && "Cannot deserialize this operator");
    }
  }
}

} // namespace kernel
} // namespace cleave_superoptimizer
