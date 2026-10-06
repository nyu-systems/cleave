#include "cleave/superoptimizer/search/op_utils.h"
#include "cleave/superoptimizer/utils/containers.h"
#include "cleave/superoptimizer/utils/logging.h"
#include <cctype>
#include <sstream>

namespace cleave_superoptimizer {
namespace search {

bool is_binary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_ADD_OP,
      type::KNOperatorType::KN_MUL_OP,
      type::KNOperatorType::KN_MATMUL_OP,
      type::KNOperatorType::KN_DIV_OP,
  };
  return contains(true_values, op);
}

bool is_unary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_REDUCTION_3_OP,
      type::KNOperatorType::KN_REDUCTION_4_OP,
      type::KNOperatorType::KN_SPLIT_0_OP,
      type::KNOperatorType::KN_SPLIT_1_OP,
      type::KNOperatorType::KN_SPLIT_2_OP,
      type::KNOperatorType::KN_SPLIT_3_OP,
      type::KNOperatorType::KN_SPLIT_4_OP,
      type::KNOperatorType::KN_EXP_OP,
      type::KNOperatorType::KN_SQRT_OP,
      type::KNOperatorType::KN_SILU_OP,
      type::KNOperatorType::KN_GELU_OP,
      type::KNOperatorType::KN_RELU_OP,
      type::KNOperatorType::KN_CLAMP_OP,
      type::KNOperatorType::KN_RMS_NORM_OP,
      type::KNOperatorType::KN_OUTPUT_OP,
  };
  return contains(true_values, op);
}

bool is_commutative(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_ADD_OP,
      type::KNOperatorType::KN_MUL_OP,
  };
  return contains(true_values, op);
}

int get_input_number(type::KNOperatorType op) {
  if (is_unary(op)) {
    return 1;
  }
  if (is_binary(op)) {
    return 2;
  }
  CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                 << static_cast<int>(op);
  return 0;
}

std::shared_ptr<AbstractExpr>
    make_reduction_pattern(int dim, std::shared_ptr<AbstractExpr> opd) {
  if (dim == 1) {
    return opd;
  }
  return std::make_shared<Red>(dim, opd);
}

std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &tensor,
                                          std::shared_ptr<AbstractExpr> opd) {
  assert(opd != nullptr);
  if (op >= type::KNOperatorType::KN_REDUCTION_0_OP &&
      op <= type::KNOperatorType::KN_REDUCTION_4_OP) {
    int dim = op - type::KNOperatorType::KN_REDUCTION_0_OP;
    if (tensor.num_dims <= dim) {
      return nullptr;
    }
    return make_reduction_pattern(tensor.dim[dim], opd);
  }
  switch (op) {
    case type::KNOperatorType::KN_SPLIT_0_OP:
    case type::KNOperatorType::KN_SPLIT_1_OP:
    case type::KNOperatorType::KN_SPLIT_2_OP:
    case type::KNOperatorType::KN_SPLIT_3_OP:
    case type::KNOperatorType::KN_SPLIT_4_OP:
      return std::make_shared<Split>(op - type::KNOperatorType::KN_SPLIT_0_OP,
                                     opd);
    case type::KNOperatorType::KN_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::KNOperatorType::KN_SQRT_OP:
      return std::make_shared<Sqrt>(opd);
    case type::KNOperatorType::KN_SILU_OP:
      return std::make_shared<Silu>(opd);
    case type::KNOperatorType::KN_GELU_OP:
      return std::make_shared<Gelu>(opd);
    case type::KNOperatorType::KN_RELU_OP:
      return std::make_shared<Relu>(opd);
    case type::KNOperatorType::KN_CLAMP_OP:
      return std::make_shared<Clamp>(type::CLAMP_MIN_MAX["min_val"], type::CLAMP_MIN_MAX["max_val"], opd);
    case type::KNOperatorType::KN_OUTPUT_OP:
      return opd;
    default:
      CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                     << static_cast<int>(op);
      return nullptr;
  }
}

std::shared_ptr<AbstractExpr>
    get_pattern(type::KNOperatorType op,
                DTensor const &tensor,
                std::shared_ptr<AbstractExpr> opd,
                float scalar,
                bool scalar_on_left) {
  assert(opd != nullptr);
  auto make_scalar_name = [](float scalar_val) {
    std::ostringstream oss;
    oss << "c_" << scalar_val;
    std::string name = oss.str();
    for (char &ch : name) {
      if (!std::isalnum(static_cast<unsigned char>(ch))) {
        ch = '_';
      }
    }
    return name;
  };
  std::shared_ptr<AbstractExpr> scalar_pattern =
      std::make_shared<Var>(make_scalar_name(scalar));
  std::shared_ptr<AbstractExpr> lhs =
      scalar_on_left ? scalar_pattern : opd;
  std::shared_ptr<AbstractExpr> rhs =
      scalar_on_left ? opd : scalar_pattern;
  switch (op) {
    case type::KNOperatorType::KN_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::KNOperatorType::KN_MUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    case type::KNOperatorType::KN_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    default:
      CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                     << static_cast<int>(op);
      return nullptr;
  }
}

std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &tensor_l,
                                          DTensor const &tensor_r,
                                          std::shared_ptr<AbstractExpr> lhs,
                                          std::shared_ptr<AbstractExpr> rhs) {

  assert(lhs != nullptr);
  assert(rhs != nullptr);
  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      if (tensor_l.num_dims < 2 || tensor_r.num_dims < 2) {
        return nullptr;
      }
      {
        int a_last = tensor_l.dim[tensor_l.num_dims - 1];
        int a_second_last = tensor_l.dim[tensor_l.num_dims - 2];
        int b_last = tensor_r.dim[tensor_r.num_dims - 1];
        int b_second_last = tensor_r.dim[tensor_r.num_dims - 2];
        int reduction_dim = -1;
        if (a_last == b_second_last) {
          reduction_dim = a_last;
        } else if (a_last == b_last) {
          reduction_dim = a_last;
        } else if (a_second_last == b_second_last) {
          reduction_dim = a_second_last;
        } else if (a_second_last == b_last) {
          reduction_dim = a_second_last;
        } else {
          return nullptr;
        }
        return make_reduction_pattern(reduction_dim,
                                      std::make_shared<Mul>(lhs, rhs));
      }
    case type::KNOperatorType::KN_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::KNOperatorType::KN_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    case type::KNOperatorType::KN_MUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    default:
      CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                     << static_cast<int>(op);
      return nullptr;
  }
}

std::shared_ptr<AbstractExpr>
    get_pattern(type::KNOperatorType op,
                std::vector<DTensor> const &tensors,
                std::vector<std::shared_ptr<AbstractExpr>> const &opds) {
  for (auto const &expr : opds) {
    if (!expr) {
      return nullptr;
    }
  }
  if (tensors.size() == 1) {
    return get_pattern(op, tensors[0], opds[0]);
  }
  if (tensors.size() == 2) {
    return get_pattern(op, tensors[0], tensors[1], opds[0], opds[1]);
  }
  CSO_LOG(FATAL) << "Unsupported operator arity " << tensors.size()
                 << " for KNOperatorType " << static_cast<int>(op);
  return nullptr;
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input) {
  if (type >= type::KNOperatorType::KN_REDUCTION_0_OP &&
      type <= type::KNOperatorType::KN_REDUCTION_4_OP) {
    return g.create_reduction_op(
        input, type - type::KNOperatorType::KN_REDUCTION_0_OP, 1);
  }
  switch (type) {
    case type::KNOperatorType::KN_SPLIT_0_OP:
    case type::KNOperatorType::KN_SPLIT_1_OP:
    case type::KNOperatorType::KN_SPLIT_2_OP:
    case type::KNOperatorType::KN_SPLIT_3_OP:
    case type::KNOperatorType::KN_SPLIT_4_OP:
      return g.create_split_op(
          input, type - type::KNOperatorType::KN_SPLIT_0_OP);
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
      return g.create_elementunary_op(input, type);
    case type::KNOperatorType::KN_CLAMP_OP:
      assert((!type::CLAMP_MIN_MAX.empty()) && "CLAMP_MIN_MAX not assigned");
      return g.create_elementunary_clamp_op(input, type::CLAMP_MIN_MAX["min_val"], type::CLAMP_MIN_MAX["max_val"]);
    default:
      CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                     << static_cast<int>(type);
      return nullptr;
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input1,
                      DTensor const &input2) {
  switch (type) {
    case type::KNOperatorType::KN_MATMUL_OP:
      // (david) it is safe to return the first that works because we use symbols
      // then one should avoid different dimensions with same symbols
      // so the first that matches will be valid since other symbols will be different
      if (auto *op = g.create_matmul_op(input1, input2)) {
        return op;
      }
      if (auto *op = g.create_matmul_op(input1, input2, false, true)) {
        return op;
      }
      if (auto *op = g.create_matmul_op(input1, input2, true, false)) {
        return op;
      }
      return g.create_matmul_op(input1, input2, true, true);
    case type::KNOperatorType::KN_DIV_OP:
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP:
      return g.create_elementbinary_op(input1, input2, type);
    default:
      CSO_LOG(FATAL) << "Unsupported KNOperatorType "
                     << static_cast<int>(type);
      return nullptr;
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      std::vector<DTensor> const &inputs) {
  if (inputs.size() == 1) {
    return create_op(g, type, inputs[0]);
  }
  if (inputs.size() == 2) {
    return create_op(g, type, inputs[0], inputs[1]);
  }
  return nullptr;
}

size_t count_op_of_type(type::KNOperatorType op_type, kernel::Graph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

} // namespace search
} // namespace cleave_superoptimizer
