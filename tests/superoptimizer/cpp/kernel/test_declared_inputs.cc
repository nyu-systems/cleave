// Declared inputs: an input that IS a function of earlier inputs.
//
// o = exp(q @ k^T * c - L) @ v is declared over (q, k, v, L). The declaration
// survives clone and a JSON round trip, gives the unread o symbolic dims (it
// used to abort: "Dimension not found in symbol map"), makes o reducible where
// its definition is, gives o the fingerprint of its definition (and so a wrong
// declaration does not make o equal to P @ v), and an unread declared input
// is not an output.
#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/search/verification/probabilistic_verifier.h"
#include "cleave/superoptimizer/type.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace kn = cleave_superoptimizer::kernel;
namespace type = cleave_superoptimizer::type;
using kn::DTensor;

constexpr int H = 4, SQ = 128, SKV = 256, D = 64;
constexpr float C = 0.125f;

int failures = 0;

void check(bool ok, std::string const &what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << std::endl;
    failures++;
  }
}

struct Inputs {
  DTensor q, k, v, L, o;
};

DTensor probabilities(kn::Graph &g, Inputs const &in) {
  DTensor s = g.mul(g.matmul(in.q, in.k, false, true), C);
  return g.exp(g.sub(s, in.L));
}

// The definition of o over (q, k, v, L); `wrong`: P @ k instead of P @ v.
std::unique_ptr<kn::Graph> definition(bool wrong = false) {
  auto d = std::make_unique<kn::Graph>();
  Inputs in;
  in.q = d->new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  in.k = d->new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  in.v = d->new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  in.L = d->new_input({H, SQ, 1}, {}, {}, type::DT_FLOAT32);
  DTensor p = probabilities(*d, in);
  d->mark_output(d->matmul(p, wrong ? in.k : in.v));
  return d;
}

Inputs inputs(kn::Graph &g, bool declare, bool wrong = false) {
  Inputs in;
  in.q = g.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  in.k = g.new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  in.v = g.new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  in.L = g.new_input({H, SQ, 1}, {}, {}, type::DT_FLOAT32);
  in.o = g.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  if (declare) {
    g.declare_input(in.o, *definition(wrong), {in.q, in.k, in.v, in.L});
  }
  return in;
}

DTensor input_tensor(kn::Graph const &g, int position) {
  return g.input_ops().at(position)->output_tensors[0];
}

bool declared(kn::Graph const &g, int position) {
  return g.input_ops().at(position)->declaration != nullptr;
}

std::vector<int> dims(DTensor const &t) {
  return std::vector<int>(t.dim, t.dim + t.num_dims);
}

std::vector<bool> reducible(DTensor const &t) {
  return std::vector<bool>(t.can_reduce, t.can_reduce + t.num_dims);
}

void test_clone_and_json_keep_the_declaration() {
  kn::Graph g;
  Inputs in = inputs(g, true);
  g.mark_output(g.mul(probabilities(g, in), 2.0f));
  std::unique_ptr<kn::Graph> copy(g.clone());
  check(declared(*copy, 4) && !declared(*copy, 3), "clone keeps the declaration");
  check(copy->input_ops()[4]->declaration == g.input_ops()[4]->declaration,
        "clones share the declaration");
  json j;
  to_json(j, g);
  kn::Graph back;
  from_json(j, back);
  check(declared(back, 4), "JSON keeps the declaration");
  auto const &decl = *back.input_ops()[4]->declaration;
  check(decl.sources == std::vector<int>({0, 1, 2, 3}), "JSON keeps the sources");
  check(dims(decl.result()) == std::vector<int>({H, SQ, D}),
        "JSON keeps the definition");
  check(kn::declaration_key(back.input_ops()[4]) ==
            kn::declaration_key(g.input_ops()[4]),
        "the declaration key is guid-free");
  check(kn::declaration_key(g.input_ops()[3]).empty(),
        "an undeclared input has no declaration key");
}

void test_bad_declarations_are_refused() {
  auto refused = [](auto &&declare) {
    try {
      declare();
    } catch (std::invalid_argument const &) {
      return true;
    }
    return false;
  };
  {
    kn::Graph g;
    Inputs in = inputs(g, false);
    check(refused([&] { g.declare_input(in.q, *definition(), {in.q, in.k, in.v, in.L}); }),
          "a source after the declared input");
  }
  {
    kn::Graph g;
    Inputs in = inputs(g, false);
    check(refused([&] { g.declare_input(in.o, *definition(), {in.q, in.k, in.v}); }),
          "fewer sources than definition inputs");
  }
  {
    kn::Graph g;
    Inputs in = inputs(g, false);
    check(refused([&] { g.declare_input(in.o, *definition(), {in.k, in.q, in.v, in.L}); }),
          "a source of another shape (SKV vs SQ)");
  }
  {
    kn::Graph g;
    Inputs in = inputs(g, false);
    DTensor x = g.exp(in.q);
    check(refused([&] { g.declare_input(x, *definition(), {in.q, in.k, in.v, in.L}); }),
          "a tensor that is not an input");
  }
}

void test_an_unread_declared_input_gets_symbols() {
  kn::Graph g;
  Inputs in = inputs(g, true);
  g.mark_output(g.matmul(probabilities(g, in), in.v));  // o unread
  std::unordered_map<size_t, std::vector<int>> sym;
  std::unique_ptr<kn::Graph> s(g.create_symbolic_graph(&sym));
  std::vector<int> q = sym.at(in.q.guid), v = sym.at(in.v.guid), o = sym.at(in.o.guid);
  check(o == std::vector<int>({q[0], q[1], v[2]}), "o has the symbols of [H, SQ, D]");
  check(declared(*s, 4), "the symbolic graph keeps the declaration");
  check(dims(s->input_ops()[4]->declaration->result()) == o,
        "the symbolic definition has symbolic dims");
}

void test_reducibility_follows_the_definition() {
  // dP = dO @ v^T; delta = rowsum(P * dP); the target does not read o, yet o
  // is reducible over D (P @ v is, from v): sum_D(dO * o) can be built.
  kn::Graph g;
  Inputs in;
  in.q = g.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  in.k = g.new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  in.v = g.new_input({H, SKV, D}, {}, {}, type::DT_FLOAT16);
  DTensor dO = g.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  in.L = g.new_input({H, SQ, 1}, {}, {}, type::DT_FLOAT32);
  in.o = g.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  g.declare_input(in.o, *definition(), {in.q, in.k, in.v, in.L});
  DTensor p = probabilities(g, in);
  DTensor dp = g.matmul(dO, in.v, false, true);
  DTensor delta = g.reduce_sum(g.mul(p, dp), 2);
  DTensor ds = g.mul(g.mul(p, g.sub(dp, delta)), C);
  g.mark_output(g.matmul(ds, in.k));
  check(reducible(input_tensor(g, 5)) == std::vector<bool>({false, false, true}),
        "o is reducible over D only");
  // the same graph without the declaration: o has no reader, no flag
  kn::Graph plain;
  DTensor x = plain.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  DTensor y = plain.new_input({H, SQ, D}, {}, {}, type::DT_FLOAT16);
  (void)y;
  plain.mark_output(plain.exp(x));
  check(reducible(input_tensor(plain, 1)) == std::vector<bool>({false, false, false}),
        "an undeclared unread input stays irreducible");
}

// o * 2 against (P @ v) * 2: the same when o is declared as P @ v.
bool matches(bool wrong_declaration) {
  kn::Graph target;
  Inputs t = inputs(target, true, wrong_declaration);
  target.mark_output(target.mul(t.o, 2.0f));
  cleave_superoptimizer::search::ProbabilisticVerifier verifier(target);
  kn::Graph candidate;
  Inputs c = inputs(candidate, true, wrong_declaration);
  DTensor out = candidate.mul(candidate.matmul(probabilities(candidate, c), c.v), 2.0f);
  (void)out;
  bool direct = verifier.verify(candidate).is_valid();
  json j;
  to_json(j, candidate);
  kn::Graph back;
  from_json(j, back);
  check(verifier.verify(back).is_valid() == direct, "a JSON round trip verifies alike");
  return direct;
}

void test_fingerprint_is_the_definitions() {
  check(matches(false), "o declared as P @ v verifies against P @ v");
  check(!matches(true), "o declared as P @ k does not verify against P @ v");
}

} // namespace

int main() {
  test_clone_and_json_keep_the_declaration();
  test_bad_declarations_are_refused();
  test_an_unread_declared_input_gets_symbols();
  test_reducibility_follows_the_definition();
  test_fingerprint_is_the_definitions();
  if (failures) {
    std::cerr << failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "test_declared_inputs: all passed" << std::endl;
  return 0;
}
