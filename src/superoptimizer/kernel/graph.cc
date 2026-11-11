#include "cleave/superoptimizer/kernel/graph.h"

// Symbolic graph construction
#include "cleave/superoptimizer/symbolic/dimension_group.h"
#include "cleave/superoptimizer/symbolic/grouping_rules.h"
#include "cleave/superoptimizer/utils/logging.h"

// Operator headers for dynamic_cast in symbolic graph creation
#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/element_unary.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/reduction.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
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
  std::unique_ptr<Graph> symbolic_graph(new Graph());

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
            symbolic_graph->new_input(sym_dims, can_reduce_vec, can_split_vec,
                                      orig_tensor.dtype);

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
      case KN_DIV_OP:
      case KN_SUB_OP: {
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

      case KN_REDUCE_SUM_0_OP:
      case KN_REDUCE_SUM_1_OP:
      case KN_REDUCE_SUM_2_OP:
      case KN_REDUCE_SUM_3_OP:
      case KN_REDUCE_SUM_4_OP: {
        assert(op->input_tensors.size() == 1);
        DTensor const &input = op->input_tensors[0];
        DTensor const &output = op->output_tensors[0];

        DTensor sym_input = guid_to_symbolic_tensor[input.guid];

        // Determine reduction dimension from operator type
        int reduction_dim = op->op_type - KN_REDUCE_SUM_0_OP;

        // Get reduction size from output dimension
        int reduction_size = output.dim[reduction_dim];

        DTensor sym_output =
            symbolic_graph->reduce_sum(sym_input, reduction_dim, reduction_size);

        guid_to_symbolic_tensor[op->output_tensors[0].guid] = sym_output;
        break;
      }

      case KN_REDUCE_MAX_0_OP:
      case KN_REDUCE_MAX_1_OP:
      case KN_REDUCE_MAX_2_OP:
      case KN_REDUCE_MAX_3_OP:
      case KN_REDUCE_MAX_4_OP: {
        DTensor sym_input = guid_to_symbolic_tensor[op->input_tensors[0].guid];
        guid_to_symbolic_tensor[op->output_tensors[0].guid] =
            symbolic_graph->reduce_max(sym_input,
                                          op->op_type - KN_REDUCE_MAX_0_OP);
        break;
      }

      case KN_RMS_NORM_OP: {
        // TODO: disabled because normalized_shape is not stored
        CSO_THROW(std::runtime_error,
                  "Symbolizing rms_norm is not supported");
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

  return symbolic_graph.release();
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
            CSO_THROW(std::invalid_argument,
                      "Input tensor "
                          << tensor.guid << " dimension " << i
                          << " was marked can_split=true, but inferred "
                             "can_reduce=false. Requested input split "
                             "dimensions must be reducible.");
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

    case type::KN_REDUCE_SUM_0_OP:
    case type::KN_REDUCE_SUM_1_OP:
    case type::KN_REDUCE_SUM_2_OP:
    case type::KN_REDUCE_SUM_3_OP:
    case type::KN_REDUCE_SUM_4_OP:
      propagate_reduce_sum_op(op);
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
    case type::KN_SUB_OP:
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

    case type::KN_REDUCE_MAX_0_OP:
    case type::KN_REDUCE_MAX_1_OP:
    case type::KN_REDUCE_MAX_2_OP:
    case type::KN_REDUCE_MAX_3_OP:
    case type::KN_REDUCE_MAX_4_OP:
      propagate_reduce_max_op(op);
      break;

    default:
      // Unknown operator - conservative: don't propagate
      break;
  }
}

void Graph::propagate_reduce_sum_op(KNOperator *op) {
  assert(op->input_tensors.size() == 1);
  assert(op->output_tensors.size() == 1);

  DTensor const &output = op->output_tensors[0];
  DTensor &input = get_input_tensor(op, 0);

  int reduction_dim = op->op_type - type::KN_REDUCE_SUM_0_OP;

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

// Opaque max cannot be split-and-combined by sum, so the reduced dim is never
// marked reducible; other dims map through unchanged.
void Graph::propagate_reduce_max_op(KNOperator *op) {
  DTensor const &output = op->output_tensors[0];
  DTensor &input = get_input_tensor(op, 0);
  int reduced = op->op_type - type::KN_REDUCE_MAX_0_OP;
  for (int i = 0; i < output.num_dims; i++) {
    if (i != reduced && output.can_reduce[i]) {
      input.can_reduce[i] = true;
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

namespace {

// Everything needed to re-create one operator; filled from JSON or from a
// live operator so from_json and clone() share one construction path.
struct OpSpec {
  type::KNOperatorType op_type = type::KN_UNKOWN;
  std::vector<DTensor> inputs;
  std::vector<int> dims;
  std::vector<bool> can_reduce, can_split;
  type::DataType dtype = type::DT_UNKNOWN;
  bool infer_can_split = false;
  bool trans_A = false, trans_B = false;
  bool has_scalar = false, scalar_on_left = false;
  float scalar = 0.0f;
  float min_val = 0.0f, max_val = 0.0f;
  int reduction_size = 1;
  int normalized_size = 0;
};

// Appends the op described by `s` to `g`; returns its output (nullptr for
// output ops).
DTensor const *append_op(Graph &g, OpSpec const &s) {
  using type::OpCategory;
  KNOperator *op = nullptr;
  switch (type::op_category(s.op_type)) {
    case OpCategory::INPUT:
      op = g.create_input_op(s.dims, s.can_reduce, s.can_split, s.dtype);
      static_cast<KNInputOp *>(op)->infer_can_split = s.infer_can_split;
      break;
    case OpCategory::OUTPUT:
      g.mark_output(s.inputs.at(0), /*infer_output_reducibility=*/false);
      return nullptr;
    case OpCategory::MATMUL:
      op = g.create_matmul_op(s.inputs.at(0), s.inputs.at(1), s.trans_A,
                              s.trans_B);
      break;
    case OpCategory::UNARY:
      op = g.create_elementunary_op(s.inputs.at(0), s.op_type);
      break;
    case OpCategory::CLAMP:
      op = g.create_elementunary_clamp_op(s.inputs.at(0), s.min_val,
                                          s.max_val);
      break;
    case OpCategory::BINARY:
      op = s.has_scalar
               ? g.create_elementbinary_op(s.inputs.at(0), s.scalar,
                                           s.scalar_on_left, s.op_type)
               : g.create_elementbinary_op(s.inputs.at(0), s.inputs.at(1),
                                           s.op_type);
      break;
    case OpCategory::REDUCE_SUM:
      op = g.create_reduce_sum_op(s.inputs.at(0),
                                 s.op_type - type::KN_REDUCE_SUM_0_OP,
                                 s.reduction_size);
      break;
    case OpCategory::REDUCE_MAX:
      op = g.create_reduce_max_op(s.inputs.at(0),
                                     s.op_type - type::KN_REDUCE_MAX_0_OP);
      break;
    case OpCategory::SPLIT:
      op = g.create_split_op(s.inputs.at(0), s.op_type - type::KN_SPLIT_0_OP);
      break;
    case OpCategory::RMS_NORM: {
      // Only normalized_size is stored: recover the trailing dims it spans.
      DTensor const &in = s.inputs.at(0);
      std::vector<int> shape;
      int prod = 1;
      for (int d = in.num_dims - 1; d >= 0 && prod < s.normalized_size; d--) {
        prod *= in.dim[d];
        shape.insert(shape.begin(), in.dim[d]);
      }
      op = s.inputs.size() > 1
               ? g.create_rms_norm_op(in, s.inputs[1], shape)
               : g.create_rms_norm_op(in, shape);
      break;
    }
    default:
      break;
  }
  CSO_REQUIRE(op != nullptr, std::runtime_error,
              "Cannot rebuild operator type " << static_cast<int>(s.op_type));
  g.operators.push_back(op);
  return &op->output_tensors[0];
}

} // namespace

Graph *Graph::clone() const {
  std::unique_ptr<Graph> g(new Graph());
  std::unordered_map<type::GuidType, DTensor> mapped;
  for (KNOperator const *op : operators) {
    OpSpec s;
    s.op_type = op->op_type;
    for (DTensor const &t : op->input_tensors) {
      s.inputs.push_back(mapped.at(t.guid));
    }
    switch (type::op_category(op->op_type)) {
      case type::OpCategory::INPUT: {
        DTensor const &t = op->output_tensors[0];
        s.dims.assign(t.dim, t.dim + t.num_dims);
        s.can_reduce.assign(t.can_reduce, t.can_reduce + t.num_dims);
        s.can_split.assign(t.can_split, t.can_split + t.num_dims);
        s.dtype = t.dtype;
        s.infer_can_split = static_cast<KNInputOp const *>(op)->infer_can_split;
        break;
      }
      case type::OpCategory::MATMUL: {
        auto const *m = static_cast<KNMatmulOp const *>(op);
        s.trans_A = m->trans_A;
        s.trans_B = m->trans_B;
        break;
      }
      case type::OpCategory::CLAMP: {
        auto const *c = static_cast<KNClampUnaryOp const *>(op);
        s.min_val = c->min_val;
        s.max_val = c->max_val;
        break;
      }
      case type::OpCategory::BINARY: {
        auto const *b = static_cast<KNElementBinaryOp const *>(op);
        s.has_scalar = b->has_scalar;
        s.scalar = b->scalar;
        s.scalar_on_left = b->scalar_on_left;
        break;
      }
      case type::OpCategory::REDUCE_SUM:
        s.reduction_size =
            static_cast<KNReduceSumOp const *>(op)->reduction_dim_size;
        break;
      case type::OpCategory::RMS_NORM:
        s.normalized_size =
            static_cast<KNRMSNormOp const *>(op)->normalized_size;
        break;
      default:
        break;
    }
    DTensor const *out = append_op(*g, s);
    if (out != nullptr) {
      mapped[op->output_tensors[0].guid] = *out;
    }
  }
  return g.release();
}

void from_json(json const &j, Graph &g) {
  // json guid -> deserialized tensor
  std::unordered_map<size_t, DTensor> mapped;
  auto tensor_at = [&](json const &jt) {
    size_t guid = jt.at("guid").get<size_t>();
    auto it = mapped.find(guid);
    CSO_REQUIRE(it != mapped.end(), std::runtime_error,
                "Graph JSON references unknown tensor guid " << guid);
    return it->second;
  };

  for (json const &jop : j) {
    OpSpec s;
    jop.at("op_type").get_to(s.op_type);
    CSO_REQUIRE(type::op_category(s.op_type) != type::OpCategory::META &&
                    type::op_category(s.op_type) != type::OpCategory::CONCAT,
                std::runtime_error,
                "Cannot deserialize operator type "
                    << static_cast<int>(s.op_type));
    if (s.op_type == type::KN_INPUT_OP) {
      json const &jout = jop.at("output_tensors")[0];
      int num_dim = jout.at("num_dims").get<int>();
      CSO_REQUIRE(num_dim <= MAX_TENSOR_DIMS, std::runtime_error,
                  "Graph JSON input has more than " << MAX_TENSOR_DIMS
                                                    << " dims");
      int dim[MAX_TENSOR_DIMS];
      bool can_reduce_arr[MAX_TENSOR_DIMS];
      bool can_split_arr[MAX_TENSOR_DIMS];
      jout.at("dim").get_to(dim);
      jout.at("can_reduce").get_to(can_reduce_arr);
      jout.at("can_split").get_to(can_split_arr);
      s.dims = to_vector(num_dim, dim);
      s.can_reduce = to_vector(num_dim, can_reduce_arr);
      s.can_split = to_vector(num_dim, can_split_arr);
      s.dtype = type::data_type_from_string(jout.value("dtype", std::string()));
    } else {
      // Scalar binary ops serialize their single tensor operand only.
      size_t n = jop.contains("scalar") ? 1 : jop.at("input_tensors").size();
      for (size_t i = 0; i < n; i++) {
        s.inputs.push_back(tensor_at(jop.at("input_tensors")[i]));
      }
    }
    s.trans_A = jop.value("trans_A", false);
    s.trans_B = jop.value("trans_B", false);
    if (jop.contains("scalar")) {
      s.has_scalar = true;
      s.scalar = jop.at("scalar").get<float>();
      s.scalar_on_left = jop.value("scalar_on_left", false);
    }
    if (s.op_type == type::KN_CLAMP_OP) {
      s.min_val = type::CLAMP_MIN_MAX["min_val"];
      s.max_val = type::CLAMP_MIN_MAX["max_val"];
    }
    if (s.op_type == type::KN_RMS_NORM_OP) {
      // Not serialized; assume the last dim (as create_symbolic_graph does).
      DTensor const &in = s.inputs.at(0);
      s.normalized_size = in.dim[in.num_dims - 1];
    }
    DTensor const *out = append_op(g, s);
    if (out != nullptr) {
      mapped[jop.at("output_tensors")[0].at("guid").get<size_t>()] = *out;
    }
  }
}

} // namespace kernel
} // namespace cleave_superoptimizer
