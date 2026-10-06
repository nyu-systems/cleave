#include "cleave/superoptimizer/search/abstract_expr/abstract_expr_eval.h"
#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"
#include "cleave/superoptimizer/search/op_utils.h"
#include <cctype>
#include <sstream>

namespace cleave_superoptimizer {
namespace search {

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns) {
  auto make_scalar_name = [](float scalar) {
    std::ostringstream oss;
    oss << "c_" << scalar;
    std::string name = oss.str();
    for (char &ch : name) {
      if (!std::isalnum(static_cast<unsigned char>(ch))) {
        ch = '_';
      }
    }
    return name;
  };
  int input_id = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    } else if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      patterns.insert({op->output_tensors[0].guid,
                       std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type == type::KNOperatorType::KN_RMS_NORM_OP) {
      std::shared_ptr<AbstractExpr> input_pattern =
          patterns.at(op->input_tensors[0].guid);
      std::shared_ptr<AbstractExpr> denominator_pattern = std::make_shared<RMS>(
          static_cast<kernel::KNRMSNormOp *>(op)->normalized_size,
          input_pattern);
      std::shared_ptr<AbstractExpr> output_pattern =
          std::make_shared<Div>(input_pattern, denominator_pattern);
      patterns.insert({op->output_tensors[0].guid, output_pattern});
    } else {
      if (op->op_type == type::KNOperatorType::KN_ADD_OP ||
          op->op_type == type::KNOperatorType::KN_MUL_OP ||
          op->op_type == type::KNOperatorType::KN_DIV_OP) {
        auto const *bin_op = dynamic_cast<kernel::KNElementBinaryOp *>(op);
        if (bin_op && bin_op->has_scalar) {
          std::shared_ptr<AbstractExpr> tensor_pattern =
              patterns.at(op->input_tensors[0].guid);
          std::shared_ptr<AbstractExpr> scalar_pattern =
              std::make_shared<Var>(make_scalar_name(bin_op->scalar));
          std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
          std::vector<kernel::DTensor> input_tensors;
          input_tensors.push_back(op->input_tensors[0]);
          input_tensors.push_back(op->input_tensors[0]);
          if (bin_op->scalar_on_left) {
            input_patterns.push_back(scalar_pattern);
            input_patterns.push_back(tensor_pattern);
          } else {
            input_patterns.push_back(tensor_pattern);
            input_patterns.push_back(scalar_pattern);
          }
          patterns.insert(
              {op->output_tensors[0].guid,
               get_pattern(op->op_type, input_tensors, input_patterns)});
          continue;
        }
      }
      std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
      for (auto const &input_tensor : op->input_tensors) {
        assert(contains_key(patterns, input_tensor.guid));
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      if (op->op_type == type::KNOperatorType::KN_MATMUL_OP) {
        auto const *matmul = dynamic_cast<kernel::KNMatmulOp *>(op);
        bool trans_A = matmul ? matmul->trans_A : false;
        int reduction_dim = trans_A ? op->input_tensors[0].num_dims - 2
                                    : op->input_tensors[0].num_dims - 1;
        int reduction_size = op->input_tensors[0].dim[reduction_dim];
        std::shared_ptr<AbstractExpr> mul_expr =
            std::make_shared<Mul>(input_patterns[0], input_patterns[1]);
        std::shared_ptr<AbstractExpr> out_expr =
            reduction_size == 1 ? mul_expr
                                : std::make_shared<Red>(reduction_size,
                                                        mul_expr);
        patterns.insert({op->output_tensors[0].guid, out_expr});
      } else {
        patterns.insert(
            {op->output_tensors[0].guid,
             get_pattern(op->op_type, op->input_tensors, input_patterns)});
      }
    }
  }
}

} // namespace search
} // namespace cleave_superoptimizer
