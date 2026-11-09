// Op table, Graph::clone, sub / max fingerprints, and sub / max search config.
#include "cleave/superoptimizer/kernel/fingerprint_cpu.h"
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/search/search.h"
#include "cleave/superoptimizer/search/verification/probabilistic_verifier.h"
#include "cleave/superoptimizer/tile/graph.h"
#include "cleave/superoptimizer/type.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace kn = cleave_superoptimizer::kernel;
namespace ty = cleave_superoptimizer::type;
namespace se = cleave_superoptimizer::search;
using kn::DTensor;

std::vector<bool> flags(size_t n, bool v = false) {
  return std::vector<bool>(n, v);
}

// Graph JSON with guids renumbered in order of appearance.
json canonical(kn::Graph const &g) {
  json j = g;
  std::unordered_map<int64_t, int64_t> ids;
  for (json &op : j) {
    for (char const *key : {"input_tensors", "output_tensors"}) {
      for (json &t : op[key]) {
        int64_t guid = t["guid"].get<int64_t>();
        auto it = ids.emplace(guid, static_cast<int64_t>(ids.size())).first;
        t["guid"] = it->second;
      }
    }
  }
  return j;
}

bool test_op_table() {
  std::set<int> values;
  std::set<std::string> names;
  for (auto const &info : ty::kOpTypes) {
    values.insert(info.type);
    names.insert(info.name);
    json j = info.type;
    ty::KNOperatorType back = ty::KN_INPUT_OP;
    j.get_to(back);
    if (j.get<std::string>() != info.name || back != info.type) {
      return false;
    }
  }
  // Checkpoint names that must never change.
  json add = ty::KN_ADD_OP, red = ty::KN_REDUCE_SUM_2_OP, unk = ty::KN_UNKOWN;
  json bogus = static_cast<ty::KNOperatorType>(4242);
  ty::KNOperatorType parsed = ty::KN_ADD_OP;
  json("no_such_op").get_to(parsed);
  return values.size() == static_cast<size_t>(ty::kNumOpTypes) &&
         names.size() == values.size() && add == "kn_add_op" &&
         red == "kn_reduce_sum_2_op" && unk == "kn_unkown" &&
         bogus == "kn_unkown" && parsed == ty::KN_UNKOWN &&
         json(ty::KN_SUB_OP) == "kn_sub_op" &&
         json(ty::KN_REDUCE_MAX_1_OP) == "kn_reduce_max_1_op";
}

kn::Graph *build_mixed(kn::Graph &g) {
  DTensor A = g.new_input({4, 8}, flags(2), flags(2, true), ty::DT_BFLOAT16);
  DTensor B = g.new_input({8, 6}, flags(2), flags(2));
  DTensor C = g.matmul(A, B);
  DTensor D = g.div(g.add(C, 2.0f), 3.0f);
  DTensor M = g.reduce_max(D, 1);
  DTensor S = g.sub(D, M);
  DTensor E = g.exp(S);
  DTensor R = g.reduce_sum(E, 1);
  g.mark_output(g.div(E, R), /*infer_output_reducibility=*/false);
  return &g;
}

bool test_clone_and_json_round_trip() {
  kn::Graph g;
  build_mixed(g);
  std::unique_ptr<kn::Graph> c(g.clone());
  kn::Graph from;
  from_json(json(g), from);
  json ref = canonical(g);
  if (canonical(*c) != ref || canonical(from) != ref) {
    return false;
  }
  // Fresh guids, rewired owners.
  for (size_t i = 0; i < g.operators.size(); i++) {
    for (auto const &t : c->operators[i]->output_tensors) {
      if (t.guid == g.operators[i]->output_tensors[0].guid ||
          t.owner_op != c->operators[i]) {
        return false;
      }
    }
  }
  return c->operators[0]->output_tensors[0].dtype == ty::DT_BFLOAT16;
}

std::vector<uint16_t> fp(DTensor const &t) {
  auto ct = t.copy_fingerprint_to_ctensor();
  std::vector<uint16_t> v(ct.fingerprint_ptr,
                          ct.fingerprint_ptr + ct.num_elements());
  free(ct.fingerprint_ptr);
  return v;
}

void run_fingerprints(kn::Graph &g) {
  int input_id = 0;
  for (auto *op : g.operators) {
    if (op->op_type == ty::KN_INPUT_OP) {
      static_cast<kn::KNInputOp *>(op)->fingerprint(input_id++);
    } else {
      op->fingerprint();
    }
  }
}

bool test_fingerprints() {
  kn::Graph g;
  DTensor A = g.new_input({3, 5}, flags(2), flags(2));
  DTensor B = g.new_input({3, 5}, flags(2), flags(2));
  DTensor S = g.sub(A, B);
  DTensor back = g.add(S, B);   // (A - B) + B == A (mod PQ)
  DTensor A_mod = g.add(A, 0.0f);
  DTensor M0 = g.reduce_max(A, 1);
  DTensor M1 = g.reduce_max(A, 1);
  DTensor MB = g.reduce_max(B, 1);
  run_fingerprints(g);
  // The CPU engine agrees with the per-op path.
  kn::CpuFingerprintEngine engine;
  for (auto *op : g.operators) {
    if (op->op_type == ty::KN_INPUT_OP) {
      auto ct = op->output_tensors[0].copy_fingerprint_to_ctensor();
      engine.load_fingerprint_from_ctensor(op->output_tensors[0], ct);
      free(ct.fingerprint_ptr);
    } else {
      engine.compute(op);
    }
  }
  bool engine_ok = true;
  for (DTensor const &t : {S, M0, MB}) {
    auto ct = t.copy_fingerprint_to_ctensor();
    engine_ok = engine_ok && engine.matches(t, ct);
    free(ct.fingerprint_ptr);
  }
  bool ok = engine_ok && fp(back) == fp(A_mod) && fp(M0) == fp(M1) &&
            fp(M0) != fp(MB) && M0.dim[1] == 1 && M0.num_dims == 2;
  kn::clear_thread_fingerprints();
  return ok;
}

// Each thread fingerprints into its own store: concurrent verifiers agree
// with a serial run and need no lock.
bool test_thread_local_store() {
  kn::Graph ref;
  build_mixed(ref);
  se::ProbabilisticVerifier verifier(ref);
  std::vector<int> ok(8, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; t++) {
    threads.emplace_back([&, t] {
      int good = 1;
      for (int it = 0; it < 20; it++) {
        kn::Graph g;
        build_mixed(g);
        // verify() expects outputs unmarked.
        delete g.operators.back();
        g.operators.pop_back();
        good &= verifier.verify(g).is_valid() ? 1 : 0;
      }
      ok[t] = good;
    });
  }
  for (auto &th : threads) {
    th.join();
  }
  for (int v : ok) {
    if (!v) {
      return false;
    }
  }
  return true;
}

bool is_sub_max(ty::KNOperatorType t) {
  return t == ty::KN_SUB_OP ||
         ty::op_category(t) == ty::OpCategory::REDUCE_MAX;
}

// Graphs with sub / max are accepted and knop_to_explore is used as given.
bool test_generator_keeps_config() {
  kn::Graph g;
  DTensor A = g.new_input({4, 8}, flags(2), flags(2));
  g.mark_output(g.reduce_max(A, 1), false);
  se::GeneratorConfig cfg = se::GeneratorConfig::get_default_config();
  for (auto op : cfg.knop_to_explore) {
    if (is_sub_max(op)) {
      return false;
    }
  }
  cfg.knop_to_explore.push_back(ty::KN_SUB_OP);
  cfg.knop_to_explore.push_back(ty::KN_REDUCE_MAX_1_OP);
  se::KernelGraphGenerator gen(g, cfg, "unused.json");
  return gen.config.knop_to_explore == cfg.knop_to_explore;
}

// sub with a scalar is sugar: same graph as the add / mul it expands to.
bool test_sub_scalar_sugar() {
  kn::Graph g, ref;
  DTensor A = g.new_input({4, 8}, flags(2), flags(2));
  g.mark_output(g.sub(A, 2.0f), false);
  g.mark_output(g.sub(2.0f, A), false);
  DTensor R = ref.new_input({4, 8}, flags(2), flags(2));
  ref.mark_output(ref.add(R, -2.0f), false);
  ref.mark_output(ref.add(ref.mul(R, -1.0f), 2.0f), false);
  return canonical(g) == canonical(ref);
}

} // namespace

int main() {
  struct Case {
    char const *name;
    bool (*fn)();
  };
  Case cases[] = {
      {"op_table", test_op_table},
      {"clone_and_json_round_trip", test_clone_and_json_round_trip},
      {"sub_and_max_fingerprints", test_fingerprints},
      {"thread_local_store", test_thread_local_store},
      {"generator_keeps_config", test_generator_keeps_config},
      {"sub_scalar_sugar", test_sub_scalar_sugar},
  };
  int failed = 0;
  for (auto const &c : cases) {
    bool ok = false;
    try {
      ok = c.fn();
    } catch (std::exception const &e) {
      std::cerr << c.name << " threw: " << e.what() << "\n";
    }
    std::cout << (ok ? "PASSED: " : "FAILED: ") << c.name << "\n";
    failed += ok ? 0 : 1;
  }
  return failed;
}
