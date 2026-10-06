#include "cleave/superoptimizer/search/search.h"
#include "cleave/superoptimizer/kernel/element_binary.h"
#include "cleave/superoptimizer/kernel/split.h"
#include "cleave/superoptimizer/search/abstract_expr/abstract_expr_eval.h"
#include "cleave/superoptimizer/search/config.h"
#include "cleave/superoptimizer/search/dim_strategy.h"
#include "cleave/superoptimizer/search/op_utils.h"
#include "cleave/superoptimizer/search/verification/formal_verifier.h"
#include "cleave/superoptimizer/search/verification/probabilistic_verifier.h"
#include "cleave/superoptimizer/type.h"
#include "cleave/superoptimizer/utils/containers.h"
#include "cleave/superoptimizer/utils/json_utils.h"
#include "cleave/superoptimizer/utils/logging.h"

#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>

namespace cleave_superoptimizer {
namespace search {

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    char const *filename,
    std::function<void(std::vector<SerializedSearchContext>)> call_back_on_valid_graph /* = nullptr*/)
    : config(config), dim_strategy(DimStrategy(config)), filename(filename),
      num_thread(config.search_thread), call_back_on_valid_graph(call_back_on_valid_graph),
      num_total_random_tests(0), num_valid_kernel_graphs(0),
      num_total_states(0), num_tasks(0), max_depth(5) {
  preprocess(computation_graph);
  char *p = std::getenv("CLEAVE_SUPEROPTIMIZER_EAGER_DUMP");
  eager_save_viz_graphs = (p != nullptr && std::string(p) == "1");
  if (eager_save_viz_graphs) {
    CSO_LOG(INFO) << "Eager save viz graphs enabled";
  }
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_all_tensors(GraphType const &g) {
  std::vector<typename GraphType::TensorType> tensors;
  for (auto const &op : g.operators) {
    for (auto const &tensor : op->output_tensors) {
      tensors.push_back(tensor);
    }
  }
  return tensors;
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_tensors_from_idx(GraphType const &g, std::vector<int> idx) {
  std::vector<typename GraphType::TensorType> tensors,
      all_tensors = get_all_tensors(g);
  for (auto i : idx) {
    tensors.push_back(all_tensors[i]);
  }
  return tensors;
}

template <typename GraphType>
Order get_max_op_order(GraphType const &g) {
  std::vector<typename GraphType::TensorType> all_tensors = get_all_tensors(g);
  std::unordered_map<int, int> guid2index;
  for (size_t i = 0; i < all_tensors.size(); ++i) {
    guid2index[all_tensors[i].guid] = i;
  }
  std::vector<int> input_idx;
  for (auto const &input : g.operators.back()->input_tensors) {
    assert(contains_key(guid2index, input.guid));
    input_idx.push_back(guid2index.at(input.guid));
  }
  return Order(input_idx, static_cast<int>(g.operators.back()->op_type));
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_output_tensors(GraphType const &g) {
  std::vector<typename GraphType::TensorType> output_tensors;
  for (auto const &op : g.operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(g, tensor) == 0) {
        output_tensors.push_back(tensor);
      }
    }
  }
  return output_tensors;
}

void KernelGraphGenerator::_on_valid_graph(std::vector<SerializedSearchContext> verified_graphs) const {  
  if (this->eager_save_viz_graphs) {
    std::vector<json> json_graphs = this->generated_graphs;
    std::string filename = this->filename;
    std::string fname = filename.substr(0, filename.size() - 4) + "_in_progress.json";
    std::ofstream ofs(fname);
    ofs << json(json_graphs);
    ofs.close();
    CSO_LOG(INFO) << "Saved valid kernel graph to " << fname;
    std::string cmd = "python $CLEAVE_SUPEROPTIMIZER_ROOT/python/cleave/superoptimizer/visualizer.py " + fname;
    int status = std::system(cmd.c_str());
    if (status != 0) {
      CSO_LOG(WARNING) << "Visualizer command failed with status " << status
                       << ": " << cmd;
    }
  }
  if (this->call_back_on_valid_graph) {
    this->call_back_on_valid_graph(verified_graphs);
  }
}

bool valid_reduction(cleave_superoptimizer::type::KNOperatorType op_type,
                     std::vector<cleave_superoptimizer::kernel::DTensor> const &input_tensors) {
  assert(input_tensors.size() == 1);
  const auto &input = input_tensors[0];
  if (op_type < type::KN_REDUCTION_0_OP ||
      op_type > type::KN_REDUCTION_4_OP) {
    assert(false && "unreachable");
  }
  int dim = op_type - type::KN_REDUCTION_0_OP;
  if (input.num_dims <= dim || input.dim[dim] == 1) {
    return false;
  }
  if (!input.can_reduce[dim]) {
    return false;
  }
  // it cannot immediately connect to a split op and reduce the leading dim
  // get producer op
  auto producer_op = input.owner_op;
  if (producer_op->op_type >= type::KN_SPLIT_FIRST_OP_ID &&
      producer_op->op_type <= type::KN_SPLIT_LAST_OP_ID &&
      dim == 0) {
    return false;
  }

  return true;
}

bool valid_split(cleave_superoptimizer::type::KNOperatorType op_type,
                     std::vector<cleave_superoptimizer::kernel::DTensor> const &input_tensors,
                     SearchContext const &c,
                     GeneratorConfig const &config) {
  assert(input_tensors.size() == 1);
  const auto &input = input_tensors[0];
  int dim = static_cast<int>(op_type) - static_cast<int>(type::KN_SPLIT_FIRST_OP_ID);
  if (input.num_dims <= dim || input.dim[dim] % kernel::NUM_SPLITS != 0) {
    return false;
  }
  if (!input.can_split[dim]) {
    return false;
  }
  // check num ops budget. Heuristic: 2 splits need at least 1 reduction op.
  int num_split_ops = 0, num_reduction_ops = 0;
  for (auto const &op : c.kn_graph->operators) {
    if (op->op_type >= type::KN_SPLIT_FIRST_OP_ID &&
        op->op_type <= type::KN_SPLIT_LAST_OP_ID) {
      num_split_ops++;
    }
    if (op->op_type >= type::KN_REDUCTION_0_OP &&
        op->op_type <= type::KN_REDUCTION_4_OP) {
      num_reduction_ops++;
    }
  }
  int least_needed_reduction_ops = (num_split_ops + 1) / 2;
  int additional_reduction_ops_needed = std::max(0, least_needed_reduction_ops - num_reduction_ops);
  if (c.kn_graph->operators.size() + additional_reduction_ops_needed + 1 > config.max_num_kernel_graph_op) {
    return false;
  }
  return true;
}

void KernelGraphGenerator::generate_next_operator(
    SearchContext &c,
    std::function<bool(SearchContext const &)> const &verify,
    std::vector<SerializedSearchContext> &verified,
    size_t depth) {
  int n_states = ++num_total_states;
  CSO_VLOG(1) << std::string(depth * 2, ' ')
              << "depth=" << depth
              << ", total states=" << n_states
              << ", total tasks=" << num_tasks.load();
  if (num_total_states % 100 == 1) {
    show_statistics();
  }
  if (verify(c)) {
    verified.push_back(SerializedSearchContext(c));
    _on_valid_graph(verified);
    return;
  }

  std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> algebraic_pattern;
  abstract_expr_eval(*c.kn_graph, algebraic_pattern);
  if (c.kn_graph->operators.size() >= config.max_num_kernel_graph_op) {
    return;
  }
  std::vector<DTensor> all_tensors = get_all_tensors(*c.kn_graph);
  for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
    CSO_VLOG(2) << std::string(depth * 2, ' ')
                << "trying knop " << static_cast<int>(op_type)
                << " at depth " << depth;
    for (auto const &input_idx :
         dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
      Order order(input_idx, static_cast<int>(op_type));
      if (order <= get_max_op_order(*c.kn_graph)) {
        continue;
      }
      std::vector<DTensor> input_tensors =
          get_tensors_from_idx(*c.kn_graph, input_idx);

      // reduction must reduce on a dim of length > 1
      if (op_type >= type::KN_REDUCTION_0_OP &&
          op_type <= type::KN_REDUCTION_4_OP) {
        if (!valid_reduction(op_type, input_tensors)) {
          continue;
        }
      }
      if (op_type >= type::KN_SPLIT_FIRST_OP_ID &&
          op_type <= type::KN_SPLIT_LAST_OP_ID) {
        if (!valid_split(op_type, input_tensors, c, config)) {
          continue;
        }
      }

      std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
      for (auto const &t : input_tensors) {
        assert(contains_key(algebraic_pattern, t.guid));
        input_patterns.push_back(algebraic_pattern.at(t.guid));
      }
      std::shared_ptr<AbstractExpr> pattern =
          get_pattern(op_type, input_tensors, input_patterns);
      if (!check_pattern(pattern)) {
        continue;
      }

      KNOperator *new_op = create_op(*c.kn_graph, op_type, input_tensors);

      if (new_op) {
        c.kn_graph->operators.push_back(new_op);
        // if (check_range(init_ranges, target_ranges, *c.kn_graph)) {
        if (depth < max_depth) {
          num_tasks++;
          SearchContext c_tmp = SerializedSearchContext(c).deserialize();
#pragma omp task
          { generate_next_operator(c_tmp, verify, verified, depth + 1); }
        } else {
          generate_next_operator(c, verify, verified, depth + 1);
        }
        // }
        delete c.kn_graph->operators.back();
        c.kn_graph->operators.pop_back();
      }
    }

    // try scalar-based binary ops
    if (op_type == type::KNOperatorType::KN_ADD_OP ||
        op_type == type::KNOperatorType::KN_MUL_OP ||
        op_type == type::KNOperatorType::KN_DIV_OP) {
      auto unary_input_idx = dim_strategy.get_unary_input(all_tensors.size());
      for (auto const &input_idx : unary_input_idx) {
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= get_max_op_order(*c.kn_graph)) {
          continue;
        }
        std::vector<DTensor> input_tensors =
            get_tensors_from_idx(*c.kn_graph, input_idx);
        std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
        for (auto const &t : input_tensors) {
          assert(contains_key(algebraic_pattern, t.guid));
          input_patterns.push_back(algebraic_pattern.at(t.guid));
        }
        for (auto const &spec : scalar_binary_specs) {
          if (spec.op_type != op_type) {
            continue;
          }
          if (spec.input_pattern &&
              input_patterns[0]->to_string() !=
                  spec.input_pattern->to_string()) {
            continue;
          }
          std::shared_ptr<AbstractExpr> pattern =
              get_pattern(op_type,
                          input_tensors[0],
                          input_patterns[0],
                          spec.scalar,
                          spec.scalar_on_left);
          if (!check_pattern(pattern)) {
            continue;
          }
          KNOperator *scalar_op = c.kn_graph->create_elementbinary_op(
              input_tensors[0], spec.scalar, spec.scalar_on_left, op_type);
          if (!scalar_op) {
            continue;
          }
          c.kn_graph->operators.push_back(scalar_op);
          if (depth < max_depth) {
            num_tasks++;
            SearchContext c_tmp = SerializedSearchContext(c).deserialize();
#pragma omp task
            { generate_next_operator(c_tmp, verify, verified, depth + 1); }
          } else {
            generate_next_operator(c, verify, verified, depth + 1);
          }
          delete c.kn_graph->operators.back();
          c.kn_graph->operators.pop_back();
        }
      }
    }
  }
}

void KernelGraphGenerator::generate_kernel_graphs() {
  start_time = std::chrono::steady_clock::now();
  SearchContext c;
  c.kn_graph = std::make_shared<kernel::Graph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, can_reduce, can_split] = input_attr;
    c.kn_graph->new_input(dim, can_reduce, can_split);
  }

  std::vector<SerializedSearchContext> verified;

  CSO_LOG(INFO) << "[Search] Starting kernel graph generation with "
                << num_thread << " threads";
#pragma omp parallel num_threads(num_thread)
  {
#pragma omp single
    {
      generate_next_operator(
          c,
          [this](SearchContext const &c) {
            return this->verify(*c.kn_graph);
          },
          verified,
          0);
    }
  }

  CSO_LOG(INFO) << "[Search] Generated " << num_tasks.load() << " tasks";

  save_results();

  CSO_LOG(INFO) << "[Search] Finished kernel graph generation in "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start_time)
                       .count()
                << " sec";
  CSO_LOG(INFO) << "[Search] Total states explored: "
                << num_total_states.load();
  CSO_LOG(INFO) << "[Search] Random tests performed: "
                << num_total_random_tests.load();
  CSO_LOG(INFO) << "[Search] Valid kernel graphs explored: "
                << num_valid_kernel_graphs.load();
}

void KernelGraphGenerator::preprocess(kernel::Graph const &computation_graph) {
  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      computation_graph_input_attrs.push_back(
          {to_vector(op->output_tensors[0].num_dims, op->output_tensors[0].dim),
           to_vector(op->output_tensors[0].num_dims,
                     op->output_tensors[0].can_reduce),
           to_vector(op->output_tensors[0].num_dims,
                     op->output_tensors[0].can_split)});
    }
  }

  std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>>
      computation_graph_patterns;
  abstract_expr_eval(computation_graph, computation_graph_patterns);

  init_ranges = get_init_ranges(computation_graph);
  target_ranges = get_interact_ranges(init_ranges, computation_graph);
  assert(init_ranges.size() == target_ranges.size());

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      computation_graph_output_patterns.push_back(
          computation_graph_patterns.at(op->input_tensors[0].guid));
    }
  }

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_ADD_OP ||
        op->op_type == type::KNOperatorType::KN_MUL_OP ||
        op->op_type == type::KNOperatorType::KN_DIV_OP) {
      auto const *bin_op = dynamic_cast<kernel::KNElementBinaryOp *>(op);
      if (bin_op && bin_op->has_scalar) {
        std::shared_ptr<AbstractExpr> input_pattern =
            computation_graph_patterns.at(op->input_tensors[0].guid);
        bool seen = false;
        for (auto const &spec : scalar_binary_specs) {
          if (spec.op_type == op->op_type &&
              spec.scalar == bin_op->scalar &&
              spec.scalar_on_left == bin_op->scalar_on_left &&
              spec.input_pattern &&
              spec.input_pattern->to_string() == input_pattern->to_string()) {
            seen = true;
            break;
          }
        }
        if (!seen) {
          scalar_binary_specs.push_back({op->op_type,
                                         bin_op->scalar,
                                         bin_op->scalar_on_left,
                                         input_pattern});
        }
      }
    }
  }

  if (config.verifier_type == VerifierType::PROBABILISTIC_VERIFIER) {
    this->verifier = std::make_shared<ProbabilisticVerifier>(computation_graph);
  } else {
    this->verifier = std::make_shared<FormalVerifier>(computation_graph);
  }
}

bool KernelGraphGenerator::check_pattern(
    std::shared_ptr<AbstractExpr> pattern) {
  if (!pattern) {
    return false;
  }

  if (seen_patterns.find(pattern->to_string()) != seen_patterns.end()) {
    return seen_patterns[pattern->to_string()];
  }

  for (auto const &final_pattern : computation_graph_output_patterns) {
    if (pattern->subpattern_to(*final_pattern)) {

#pragma omp critical
      { seen_patterns[pattern->to_string()] = true; }
      return true;
    }
  }
#pragma omp critical
  { seen_patterns[pattern->to_string()] = false; }
  return false;
}

bool KernelGraphGenerator::verify(kernel::Graph &g) {
  std::vector<DTensor> outputs = get_output_tensors(g);

    auto save_graph = [&]() {
#pragma omp critical
      { 
        generated_graphs.push_back(json(g)); 
        _on_valid_graph({});
      }
    };
  char *env = std::getenv("CLEAVE_SUPEROPTIMIZER_DEBUG_SAVE");
  int save_level = 0; // 0 save valid, 1 save match outputs, 2 save all
  if (env && std::string(env) == "ALL") save_level = 2;
  else if (env && std::string(env) == "MO") save_level = 1;

  if (save_level == 2) {
      save_graph();
  }

  if (outputs.size() != computation_graph_output_patterns.size()) {
    return false;
  }
  
  if (save_level == 1) {
    save_graph();
  }

  {
    ++num_total_random_tests;
    auto mark_outputs = [&](OutputMatch const &match) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        g.mark_output(outputs[match[i]], /*infer_output_reducibility=*/false);
      }
    };

    auto unmark_outputs = [&]() {
      while (g.operators.back()->op_type ==
             type::KNOperatorType::KN_OUTPUT_OP) {
        delete g.operators.back();
        g.operators.pop_back();
      }
    };


    {
      std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>>
          computation_graph_patterns;
      abstract_expr_eval(g, computation_graph_patterns); // to update the patterns after marking outputs
      for (kernel::KNOperator *op : g.operators) {
        if (get_num_consumers(g, op->output_tensors[0]) == 0) {
          CSO_VLOG(3) << "[Rand test #" << num_total_random_tests.load()
                      << "] mugraph output pattern: "
                      << computation_graph_patterns.at(op->output_tensors[0].guid)
                             ->to_string();
        }
      }
    }
    OutputMatch match = verifier->verify(g);
    if (match.is_valid()) {
      ++num_valid_kernel_graphs;
      mark_outputs(match);
      if (save_level == 0) {
        save_graph();
      }
      unmark_outputs();
      return true;
    }
  }

  return false;
}

void KernelGraphGenerator::save_results() const {
  std::ofstream ofs(filename);
  ofs << json(generated_graphs);
}

double KernelGraphGenerator::get_elapsed_time_in_sec() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_time)
      .count();
}

void KernelGraphGenerator::show_statistics() const {
  CSO_VLOG(1) << "[Search] States: " << num_total_states.load()
              << ", Random tests: " << num_total_random_tests.load()
              << ", Valid mugraphs: " << num_valid_kernel_graphs.load()
              << ", Time: " << get_elapsed_time_in_sec();
}

} // namespace search
} // namespace cleave_superoptimizer
