#pragma once

#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/kernel/operator.h"
#include <unordered_map>
#include <vector>

namespace cleave_superoptimizer {
namespace kernel {

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph();
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    std::vector<bool> const &can_reduce,
                    std::vector<bool> const &can_split);
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         std::vector<bool> const &can_reduce,
                         std::vector<bool> const &can_split);
  KNOperator *create_input_op(std::vector<int> const &dims,
                              std::vector<bool> const &can_reduce,
                              std::vector<bool> const &can_split);
  DTensor new_input(std::vector<int> const &dims);
  DTensor *new_input_ptr(std::vector<int> const &dims);
  // output operator
  void mark_output(DTensor const &A, bool infer_output_reducibility = true);
  KNOperator *create_output_op(DTensor const &A);
  // matmul operator
  DTensor matmul(DTensor const &A,
                 DTensor const &B,
                 bool trans_A = false,
                 bool trans_B = false);
  DTensor *matmul(DTensor const *A,
                  DTensor const *B,
                  bool trans_A = false,
                  bool trans_B = false);
  KNOperator *create_matmul_op(DTensor const &A,
                               DTensor const &B,
                               bool trans_A = false,
                               bool trans_B = false);
  // elementunary operator
  DTensor exp(DTensor const &input);
  DTensor *exp(DTensor const *input);
  DTensor square(DTensor const &input);
  DTensor *square(DTensor const *input);
  DTensor sqrt(DTensor const &input);
  DTensor *sqrt(DTensor const *input);
  DTensor silu(DTensor const &input);
  DTensor *silu(DTensor const *input);
  DTensor gelu(DTensor const &input);
  DTensor *gelu(DTensor const *input);
  DTensor relu(DTensor const &input);
  DTensor *relu(DTensor const *input);
  DTensor
      clamp(DTensor const &input, float const &min_val, float const &max_val);
  DTensor *
      clamp(DTensor const *input, float const &min_val, float const &max_val);
  DTensor elementunary(DTensor const &input,
                       cleave_superoptimizer::type::KNOperatorType _type);
  DTensor *elementunary(DTensor const *input,
                        cleave_superoptimizer::type::KNOperatorType _type);

  KNOperator *create_elementunary_op(DTensor const &input,
                                     cleave_superoptimizer::type::KNOperatorType _type);
  
  DTensor elementunary_clamp(DTensor const &input,
                             float const &min_val,
                             float const &max_val);
  DTensor *elementunary_clamp(DTensor const *input,
                              float const &min_val,
                              float const &max_val);

  KNOperator *create_elementunary_clamp_op(DTensor const &input,
                                           float const &min_val,
                                           float const &max_val);
  
  // elementunary operator
  DTensor add(DTensor const &input1, DTensor const &input2);
  DTensor add(DTensor const &input, float scalar);
  DTensor mul(DTensor const &input1, DTensor const &input2);
  DTensor mul(DTensor const &input, float scalar);
  DTensor div(DTensor const &input1, DTensor const &input2);
  DTensor div(DTensor const &input, float scalar);
  DTensor div(float scalar, DTensor const &input);
  DTensor *add(DTensor const *input1, DTensor const *input2);
  DTensor *mul(DTensor const *input1, DTensor const *input2);
  DTensor *div(DTensor const *input1, DTensor const *input2);
  DTensor elementbinary(DTensor const &input1,
                        DTensor const &input2,
                        cleave_superoptimizer::type::KNOperatorType _type);
  DTensor elementbinary(DTensor const &input,
                        float scalar,
                        bool scalar_on_left,
                        cleave_superoptimizer::type::KNOperatorType _type);
  DTensor *elementbinary(DTensor const *input1,
                         DTensor const *input2,
                         cleave_superoptimizer::type::KNOperatorType _type);
  KNOperator *create_elementbinary_op(DTensor const &input1,
                                      DTensor const &input2,
                                      cleave_superoptimizer::type::KNOperatorType _type);
  KNOperator *create_elementbinary_op(DTensor const &input,
                                      float scalar,
                                      bool scalar_on_left,
                                      cleave_superoptimizer::type::KNOperatorType _type);
  // reduction operator
  DTensor reduction(DTensor const &input, int dim, int size = 1);
  DTensor *reduction(DTensor const *input, int dim, int size = 1);
  KNOperator *create_reduction_op(DTensor const &input, int dim, int factor);
  // split operator
  DTensor split(DTensor const &input, int dim);
  DTensor *split(DTensor const *input, int dim);
  KNOperator *create_split_op(DTensor const &input, int dim);
  // normalization operator
  DTensor rms_norm(DTensor const &input,
                   std::vector<int> const &normalized_shape);
  DTensor *rms_norm(DTensor const *input,
                    std::vector<int> const &normalized_shape);
  KNOperator *create_rms_norm_op(DTensor const &input,
                                 std::vector<int> const &normalized_shape);
  DTensor rms_norm(DTensor const &input,
                   DTensor const &elementwise_afffine,
                   std::vector<int> const &normalized_shape);
  DTensor *rms_norm(DTensor const *input,
                    DTensor const *elementwise_affine,
                    std::vector<int> const &normalized_shape);
  KNOperator *create_rms_norm_op(DTensor const &input,
                                 DTensor const &elementwise_affine,
                                 std::vector<int> const &normalized_shape);
  // helper functions
  int get_input_dtensors(DTensor **inputs);
  bool has_split_input() const;

  // Automatically infer can_reduce properties via reverse propagation
  void infer_reducibility();

private:
  // Helper methods for reducibility inference
  void propagate_reducibility_backward(KNOperator *op);
  void propagate_reduction_op(KNOperator *op);
  void propagate_split_op(KNOperator *op);
  void propagate_matmul_op(KNOperator *op);
  void propagate_elementwise_unary_op(KNOperator *op);
  void propagate_elementwise_binary_op(KNOperator *op);
  void propagate_rms_norm_op(KNOperator *op);

public:
  Graph *create_symbolic_graph(
      std::unordered_map<size_t, std::vector<int>>
          *symbolic_dims_by_guid = nullptr) const;

public:
  std::vector<cleave_superoptimizer::kernel::KNOperator *> operators;

  using OpType = KNOperator;
  using TensorType = DTensor;
};

void to_json(json &j, Graph const &g);
void from_json(json const &j, Graph &g);

} // namespace kernel
} // namespace cleave_superoptimizer
