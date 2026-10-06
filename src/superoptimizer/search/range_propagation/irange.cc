#include "cleave/superoptimizer/search/range_propagation/irange.h"
#include "cleave/superoptimizer/kernel/matmul.h"
#include "cleave/superoptimizer/kernel/rms_norm.h"
#include "cleave/superoptimizer/utils/containers.h"

#include <iostream>

namespace cleave_superoptimizer {
namespace search {

namespace {

struct MatmulDimInfo {
  int a_reduce_idx;
  int a_row_idx;
  int b_reduce_idx;
  int b_col_idx;
};

MatmulDimInfo get_matmul_dim_info(kernel::KNOperator const &op) {
  auto const *matmul = dynamic_cast<kernel::KNMatmulOp const *>(&op);
  bool trans_A = matmul ? matmul->trans_A : false;
  bool trans_B = matmul ? matmul->trans_B : false;
  MatmulDimInfo info;
  info.a_reduce_idx =
      trans_A ? op.input_tensors[0].num_dims - 2
              : op.input_tensors[0].num_dims - 1;
  info.a_row_idx =
      trans_A ? op.input_tensors[0].num_dims - 1
              : op.input_tensors[0].num_dims - 2;
  info.b_reduce_idx =
      trans_B ? op.input_tensors[1].num_dims - 1
              : op.input_tensors[1].num_dims - 2;
  info.b_col_idx =
      trans_B ? op.input_tensors[1].num_dims - 2
              : op.input_tensors[1].num_dims - 1;
  return info;
}

} // namespace

IKNRange::IKNRange(RangeSet<KNRange, size_t> const &range_set)
    : range_set(range_set) {}

void IKNRange::combine(IKNRange const &range, bool simplify) {
  range_set = range_set.combine(range.range_set);
  if (simplify) {
    this->simplify();
  }
}

bool IKNRange::is_subrange(IKNRange const &knrange) const {
  if (knrange.is_empty()) {
    return this->is_empty();
  }
  Range range = knrange.range_set.get_only();
  return is_subrange(range);
}

bool IKNRange::is_subrange(Range const &range) const {
  for (auto const &r : range_set.ranges) {
    if (!r.is_subrange(range)) {
      return false;
    }
  }
  return true;
}

void IKNRange::simplify() {
  range_set.simplify();
}

bool IKNRange::is_empty() const {
  return range_set.is_empty();
}

bool IKNRange::is_valid() const {
  return range_set.is_valid();
}

IKNRange IKNRange::point_range(std::vector<int> const &point) {
  std::vector<KNRange> ranges{KNRange::point_range(point)};
  std::vector<PropagationPath<size_t>> paths{PropagationPath<size_t>()};
  return IKNRange(RangeSet(ranges, paths));
}

IKNRange forward_propagate(IKNRange const &range,
                           kernel::KNOperator const &op,
                           size_t opd_idx) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
    case type::KNOperatorType::KN_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? range : IKNRange();
      break;
    }
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP: {
      ret = range;
      break;
    }
    case type::KNOperatorType::KN_DIV_OP: {
      if (opd_idx == 0) {
        ret = range;
      } else {
        assert(opd_idx == 1);
        ret = IKNRange(
            range.range_set.extend_dim(op.output_tensors[0].num_dims - 1)
                .truncate(op.output_tensors[0]));
      }
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      int dim_to_extend = opd_idx == 0 ? op.output_tensors[0].num_dims - 1
                                       : op.output_tensors[0].num_dims - 2;
      ret = IKNRange(range.range_set.extend_dim(dim_to_extend)
                         .truncate(op.output_tensors[0]));
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP:
    case type::KNOperatorType::KN_REDUCTION_3_OP:
    case type::KNOperatorType::KN_REDUCTION_4_OP: {
      int dim = op.op_type - type::KNOperatorType::KN_REDUCTION_0_OP;
      ret = IKNRange(
          range.range_set.extend_dim(dim).truncate(op.output_tensors[0]));
      break;
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      int num_norm_dims = 0, s = 1;
      ret = range;
      while (s < static_cast<kernel::KNRMSNormOp const &>(op).normalized_size) {
        num_norm_dims++;
        s *= op.output_tensors[0]
                 .dim[op.output_tensors[0].num_dims - num_norm_dims];
        ret = IKNRange(ret.range_set.extend_dim(op.output_tensors[0].num_dims -
                                                num_norm_dims));
      }
      ret = IKNRange(ret.range_set.truncate(op.output_tensors[0]));
      break;
    }
    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.output_tensors[0].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

IKNRange backward_propagate(IKNRange const &knrange,
                            kernel::KNOperator const &op,
                            size_t opd_idx) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
    case type::KNOperatorType::KN_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? knrange : IKNRange();
      break;
    }
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP: {
      ret = knrange;
      break;
    }
    case type::KNOperatorType::KN_DIV_OP: {
      ret = IKNRange(
          knrange.range_set.extend_dim(op.input_tensors[opd_idx].num_dims - 1)
              .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      MatmulDimInfo dims = get_matmul_dim_info(op);
      int dim_to_extend =
          opd_idx == 0 ? dims.a_reduce_idx : dims.b_reduce_idx;
      ret = IKNRange(knrange.range_set.extend_dim(dim_to_extend)
                         .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP:
    case type::KNOperatorType::KN_REDUCTION_3_OP:
    case type::KNOperatorType::KN_REDUCTION_4_OP: {
      int dim = op.op_type - type::KNOperatorType::KN_REDUCTION_0_OP;
      ret = IKNRange(knrange.range_set.extend_dim(dim).truncate(
          op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      int num_norm_dims = 0, s = 1;
      ret = knrange;
      while (s < static_cast<kernel::KNRMSNormOp const &>(op).normalized_size) {
        num_norm_dims++;
        s *= op.output_tensors[0]
                 .dim[op.output_tensors[0].num_dims - num_norm_dims];
        ret = IKNRange(ret.range_set.extend_dim(op.output_tensors[0].num_dims -
                                                num_norm_dims));
      }
      ret = IKNRange(ret.range_set.truncate(op.output_tensors[0]));
      break;
    }

    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

IKNRange multiplicative_interact(IKNRange const &knrange,
                                 kernel::KNOperator const &op,
                                 size_t opd_idx_from,
                                 size_t opd_idx_to) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_MUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        ret = knrange;
      }
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        MatmulDimInfo dims = get_matmul_dim_info(op);
        int from_reduce_idx =
            opd_idx_from == 0 ? dims.a_reduce_idx : dims.b_reduce_idx;
        int to_reduce_idx =
            opd_idx_to == 0 ? dims.a_reduce_idx : dims.b_reduce_idx;
        int to_nonreduce_idx =
            opd_idx_to == 0 ? dims.a_row_idx : dims.b_col_idx;
        int from_num_dims = op.input_tensors[opd_idx_from].num_dims;
        RangeSet<KNRange, size_t> mapped = knrange.range_set;
        if (from_reduce_idx != to_reduce_idx) {
          mapped = mapped.transpose(from_num_dims - 2, from_num_dims - 1);
        }
        ret = IKNRange(
            mapped.extend_dim(to_nonreduce_idx).truncate(
                op.input_tensors[opd_idx_to]));
      }
      break;
    }
    default:
      return IKNRange();
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx_to].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

std::vector<IKNRange>
    forward_propagate(std::vector<IKNRange> const &input_ranges,
                      kernel::KNOperator const &op) {
  if (op.op_type == type::KNOperatorType::KN_OUTPUT_OP) {
    return {};
  }
  IKNRange output_range;
  for (size_t i = 0; i < input_ranges.size(); ++i) {
    IKNRange output_range_i = forward_propagate(input_ranges[i], op, i);
    output_range.combine(output_range_i);
  }
  return {output_range};
}

std::vector<IKNRange>
    backward_propagate(std::vector<IKNRange> const &output_ranges,
                       kernel::KNOperator const &op) {
  if (op.op_type == type::KNOperatorType::KN_OUTPUT_OP) {
    return {};
  }
  assert(output_ranges.size() == 1);
  std::vector<IKNRange> input_ranges;
  for (size_t i = 0; i < op.input_tensors.size(); ++i) {
    input_ranges.push_back(backward_propagate(output_ranges[0], op, i));
  }
  return input_ranges;
}

std::vector<IKNRange>
    multiplicative_interact(std::vector<IKNRange> const &input_ranges,
                            kernel::KNOperator const &op) {
  std::vector<IKNRange> interacted_ranges(input_ranges.size());
  for (size_t from = 0; from < op.input_tensors.size(); ++from) {
    for (size_t to = 0; to < op.input_tensors.size(); ++to) {
      interacted_ranges[to].combine(
          multiplicative_interact(input_ranges[from], op, from, to));
    }
  }
  return interacted_ranges;
}

std::ostream &operator<<(std::ostream &os, IKNRange const &knrange) {
  os << "IKNRange(" << knrange.range_set << ")";
  return os;
}

void range_propagate_forward(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    std::vector<IKNRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<IKNRange> output_ranges = forward_propagate(input_ranges, *op);
    assert(output_ranges.size() == op->output_tensors.size());
    for (size_t i = 0; i < op->output_tensors.size(); ++i) {
      forward_ranges[op->output_tensors[i].guid].combine(output_ranges[i]);
    }
  }
}

void range_propagate_backward(
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : reversed(graph.operators)) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    }
    std::vector<IKNRange> output_ranges;
    for (auto const &output_tensor : op->output_tensors) {
      output_ranges.push_back(backward_ranges[output_tensor.guid]);
    }
    std::vector<IKNRange> input_ranges = backward_propagate(output_ranges, *op);
    assert(input_ranges.size() == op->input_tensors.size());
    for (size_t j = 0; j < op->input_tensors.size(); ++j) {
      backward_ranges[op->input_tensors[j].guid].combine(input_ranges[j]);
    }
  }
}

void interact_range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : graph.operators) {
    std::vector<IKNRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<IKNRange> interacted_ranges =
        multiplicative_interact(input_ranges, *op);
    assert(interacted_ranges.size() == op->input_tensors.size());
    for (size_t i = 0; i < interacted_ranges.size(); ++i) {
      backward_ranges[op->input_tensors[i].guid].combine(interacted_ranges[i]);
    }
  }
}

std::unordered_map<size_t, IKNRange>
    range_propagate(std::unordered_map<size_t, IKNRange> &forward_ranges,
                    kernel::Graph const &graph) {
  range_propagate_forward(forward_ranges, graph);
  std::unordered_map<size_t, IKNRange> backward_ranges;
  interact_range_propagate(forward_ranges, backward_ranges, graph);
  range_propagate_backward(backward_ranges, graph);
  return backward_ranges;
}

bool check_range(std::pair<size_t, IKNRange> const &init_range,
                 std::vector<IKNRange> const &target_ranges,
                 kernel::Graph const &graph) {
  std::vector<IKNRange> interact_ranges =
      get_interact_ranges(init_range, graph);
  assert(interact_ranges.size() == target_ranges.size());
  for (size_t i = 0; i < interact_ranges.size(); ++i) {
    if (!interact_ranges[i].is_subrange(target_ranges[i])) {
      return false;
    }
  }
  return true;
}

bool check_range(std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
                 std::vector<std::vector<IKNRange>> const &target_ranges,
                 kernel::Graph const &graph) {
  for (size_t i = 0; i < init_ranges.size(); ++i) {
    if (!check_range(init_ranges[i], target_ranges[i], graph)) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<size_t, IKNRange>>
    get_init_ranges(kernel::Graph const &graph) {

  auto get_points = [](kernel::DTensor const &dtensor) {
    std::vector<std::vector<int>> points;
    points.push_back(std::vector<int>(dtensor.num_dims, 0));
    {
      std::vector<int> random_point;
      for (int i = 0; i < dtensor.num_dims; ++i) {
        random_point.push_back(rand() % dtensor.dim[i]);
      }
      // points.push_back(random_point);
    }
    return points;
  };

  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  for (size_t i = 0; i < graph.operators.size(); ++i) {
    if (graph.operators[i]->op_type == type::KNOperatorType::KN_INPUT_OP) {
      for (auto const &point :
           get_points(graph.operators[i]->output_tensors[0])) {
        init_ranges.push_back(std::make_pair(i, IKNRange::point_range(point)));
      }
    }
  }
  for (auto &[idx, range] : init_ranges) {
    range.range_set =
        range.range_set.truncate(graph.operators[idx]->output_tensors[0]);
  }
  return init_ranges;
}

std::vector<IKNRange>
    get_interact_ranges(std::pair<size_t, IKNRange> const &init_range,
                        kernel::Graph const &graph) {
  std::unordered_map<size_t, IKNRange> forward_ranges;
  forward_ranges[graph.operators[init_range.first]->output_tensors[0].guid] =
      init_range.second;

  std::unordered_map<size_t, IKNRange> backward_ranges =
      range_propagate(forward_ranges, graph);

  std::vector<IKNRange> interact_ranges;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      interact_ranges.push_back(backward_ranges[op->output_tensors[0].guid]);
    }
  }
  return interact_ranges;
}

std::vector<std::vector<IKNRange>> get_interact_ranges(
    std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
    kernel::Graph const &graph) {
  return vector_map(init_ranges,
                    [&](std::pair<size_t, IKNRange> const &init_range) {
                      return get_interact_ranges(init_range, graph);
                    });
}

} // namespace search
} // namespace cleave_superoptimizer
