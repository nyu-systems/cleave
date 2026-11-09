#include "cleave/superoptimizer/search/abstract_expr/abstract_expr.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "cleave/superoptimizer/utils/containers.h"

namespace cleave_superoptimizer {
namespace search {

z3::expr_vector to_expr_vector(z3::context &c,
                               std::vector<z3::expr> const &_vec) {
  z3::expr_vector vec(c);
  for (auto const &e : _vec) {
    vec.push_back(e);
  }
  return vec;
}

namespace {

std::vector<std::shared_ptr<AbstractExpr>> children(AbstractExpr const &e) {
  if (auto const *n = dynamic_cast<Add const *>(&e)) {
    return {n->lhs, n->rhs};
  }
  if (auto const *n = dynamic_cast<Mul const *>(&e)) {
    return {n->lhs, n->rhs};
  }
  if (auto const *n = dynamic_cast<Div const *>(&e)) {
    return {n->lhs, n->rhs};
  }
  if (auto const *n = dynamic_cast<Sub const *>(&e)) {
    return {n->lhs, n->rhs};
  }
  if (auto const *n = dynamic_cast<Exp const *>(&e)) {
    return {n->exponent};
  }
  if (auto const *n = dynamic_cast<Sqrt const *>(&e)) {
    return {n->a};
  }
  if (auto const *n = dynamic_cast<Silu const *>(&e)) {
    return {n->a};
  }
  if (auto const *n = dynamic_cast<Gelu const *>(&e)) {
    return {n->a};
  }
  if (auto const *n = dynamic_cast<Relu const *>(&e)) {
    return {n->a};
  }
  if (auto const *n = dynamic_cast<Clamp const *>(&e)) {
    return {n->elems};
  }
  if (auto const *n = dynamic_cast<RMS const *>(&e)) {
    return {n->elems};
  }
  if (auto const *n = dynamic_cast<Sum const *>(&e)) {
    return {n->summand};
  }
  if (auto const *n = dynamic_cast<Split const *>(&e)) {
    return {n->input};
  }
  if (auto const *n = dynamic_cast<Opaque const *>(&e)) {
    return n->args;
  }
  return {};
}

// Collects Sub/Opaque nodes so their axioms are added only when used: the
// default search space must see the exact same solver queries as before.
void collect_extensions(AbstractExpr const &e,
                        bool &has_sub,
                        std::unordered_map<std::string, size_t> &opaque) {
  if (dynamic_cast<Sub const *>(&e)) {
    has_sub = true;
  }
  if (auto const *n = dynamic_cast<Opaque const *>(&e)) {
    opaque[n->name] = n->args.size();
  }
  for (auto const &child : children(e)) {
    if (child) {
      collect_extensions(*child, has_sub, opaque);
    }
  }
}

} // namespace

bool AbstractExpr::subpattern_to(AbstractExpr const &other) const {
  z3::context c;

  z3::sort P = c.uninterpreted_sort("P");
  z3::sort I = c.int_sort();
  z3::sort F = c.fpa_sort(8, 24);

  z3::func_decl add = c.function("add", P, P, P);
  z3::func_decl mul = c.function("mul", P, P, P);
  z3::func_decl div = c.function("div", P, P, P);
  z3::func_decl exp = c.function("exp", P, P);
  z3::func_decl sqrt = c.function("sqrt", P, P);
  z3::func_decl silu = c.function("silu", P, P);
  z3::func_decl gelu = c.function("gelu", P, P);
  z3::func_decl relu = c.function("relu", P, P);
  z3::func_decl clamp = c.function("clamp", F, F, P, P);
  z3::func_decl rms = c.function("rms", I, P, P);
  z3::func_decl sum = c.function("sum", I, P, P);

  z3::func_decl subpattern = c.function("subpattern", P, P, c.bool_sort());

  z3::solver s(c);

  z3::params p(c);
  p.set("mbqi", true);
  p.set("rlimit", 80000u);
  s.set(p);

  z3::expr x = c.constant("x", P);
  z3::expr y = c.constant("y", P);
  z3::expr z = c.constant("z", P);
  z3::expr i = c.int_const("i");
  z3::expr i1 = c.int_const("i1");
  z3::expr i2 = c.int_const("i2");
  z3::expr f = c.fpa_const("f", 8, 24);
  z3::expr f1 = c.fpa_const("f", 8, 24);

  std::unordered_set<std::string> all_variables;
  z3::expr pattern1 = to_z3(c, all_variables),
           pattern2 = other.to_z3(c, all_variables);

  for (std::string const &name1 : all_variables) {
    for (std::string const &name2 : all_variables) {
      if (name1 < name2) {
        z3::expr v1 = c.constant(name1.data(), P);
        z3::expr v2 = c.constant(name2.data(), P);
        s.add(v1 != v2);
        s.add(!subpattern(v1, v2));
        s.add(!subpattern(v2, v1));
      }
    }
  }

  s.add(forall(x, subpattern(x, x)));
  s.add(
      forall(x,
             y,
             z,
             implies(subpattern(x, y) && subpattern(y, z), subpattern(x, z))));

  s.add(forall(x, y, add(x, y) == add(y, x)));
  s.add(forall(x, y, mul(x, y) == mul(y, x)));
  s.add(forall(x, y, z, add(add(x, y), z) == add(x, add(y, z))));
  s.add(forall(x, y, z, mul(mul(x, y), z) == mul(x, mul(y, z))));
  s.add(forall(x, y, z, add(mul(x, z), mul(y, z)) == mul(add(x, y), z)));
  s.add(forall(x, y, z, add(div(x, z), div(y, z)) == div(add(x, y), z)));
  s.add(forall(x, y, z, mul(x, div(y, z)) == div(mul(x, y), z)));

  s.add(forall(x, y, subpattern(x, add(x, y))));
  s.add(forall(x, y, subpattern(x, mul(x, y))));
  s.add(forall(x, y, subpattern(x, div(x, y))));
  s.add(forall(x, y, subpattern(y, div(x, y))));
  s.add(forall(x, subpattern(x, exp(x))));
  s.add(forall(x, subpattern(x, sqrt(x))));
  s.add(forall(x, subpattern(x, silu(x))));
  s.add(forall(x, subpattern(x, gelu(x))));
  s.add(forall(x, subpattern(x, relu(x))));
  s.add(forall(x, f, f1, subpattern(x, clamp(f, f1, x))));
  s.add(forall(x, i, subpattern(x, rms(i, x))));
  s.add(forall(x, i, subpattern(x, sum(i, x))));

  s.add(forall(x, x == sum(c.int_val(0), x)));
  s.add(forall(x, i1, i2, sum(i1, sum(i2, x)) == sum(i1 + i2, x)));
  s.add(forall(to_expr_vector(c, {x, y, i, i1, i2}),
               sum(i, add(sum(i1, x), sum(i2, y))) ==
                   add(sum(i + i1, x), sum(i + i2, y))));
  s.add(forall(x, y, i, sum(i, mul(x, y)) == mul(sum(i, x), y)));
  s.add(forall(x, y, i, sum(i, div(x, y)) == div(sum(i, x), y)));

  // Lemmas
  s.add(
      forall(x, i1, i2, implies(i1 <= i2, subpattern(sum(i1, x), sum(i2, x)))));

  bool has_sub = false;
  std::unordered_map<std::string, size_t> opaque;
  collect_extensions(*this, has_sub, opaque);
  collect_extensions(other, has_sub, opaque);
  if (has_sub) {
    z3::func_decl sub = c.function("sub", P, P, P);
    s.add(forall(x, y, subpattern(x, sub(x, y))));
    s.add(forall(x, y, subpattern(y, sub(x, y))));
  }
  for (auto const &[name, arity] : opaque) {
    z3::sort_vector domain(c);
    z3::expr_vector bound(c);
    for (size_t k = 0; k < arity; k++) {
      domain.push_back(P);
      bound.push_back(c.constant(("a" + std::to_string(k)).c_str(), P));
    }
    z3::func_decl f = c.function(("opaque_" + name).c_str(), domain, P);
    for (size_t k = 0; k < arity; k++) {
      s.add(forall(bound, subpattern(bound[k], f(bound))));
    }
  }

  // Theorem to prove
  s.add(!subpattern(pattern1, pattern2));

  z3::check_result cr = s.check();
  if (cr == z3::unsat) {
    return true;
  }
  if (cr == z3::sat) {
    return false;
  }
  // cr == z3::unknown: Z3 exhausted its resource limit without a verdict. This
  // happens on deeply-nested targets whose rewrite theory (associativity,
  // distributivity, reduction-push) explodes MBQI quantifier instantiation --
  // e.g. attention scores where BOTH matmul operands carry an RMS-norm
  // (reciprocal-sqrt-of-reduction) subtree, so a reduction sits over a product
  // of two reduction-bearing factors. subpattern_to is used by check_pattern as
  // a *necessary-condition* prune, so returning false here would silently drop
  // valid candidates (the numeric ProbabilisticVerifier is the real gate). Fall
  // back to a sound, cheap structural test: if this pattern occurs verbatim as a
  // sub-expression of `other`, it is provably a subpattern (reflexivity plus the
  // one-step subpattern axioms above, closed under transitivity), so keep it.
  // A non-matching unknown stays pruned, which preserves search tractability.
  return other.contains_subexpr(to_string());
}

bool AbstractExpr::contains_subexpr(std::string const &other) const {
  if (to_string() == other) {
    return true;
  }
  for (auto const &child : children(*this)) {
    if (child && child->contains_subexpr(other)) {
      return true;
    }
  }
  return false;
}

Var::Var(std::string const &name) : name(name) {}

z3::expr Var::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  all_variables.insert(name);
  return c.constant(name.data(), P);
}

std::string Var::to_string() const {
  return name;
}

Add::Add(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Add::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl add = c.function("add", P, P, P);
  return add(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

Mul::Mul(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Mul::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl mul = c.function("mul", P, P, P);
  return mul(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

Div::Div(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Div::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl div = c.function("div", P, P, P);
  return div(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

Exp::Exp(std::shared_ptr<AbstractExpr> exponent) : exponent(exponent) {
  assert(exponent);
}

z3::expr Exp::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl exp = c.function("exp", P, P);
  return exp(exponent->to_z3(c, all_variables));
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

Sqrt::Sqrt(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Sqrt::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl sqrt = c.function("sqrt", P, P);
  return sqrt(a->to_z3(c, all_variables));
}

std::string Sqrt::to_string() const {
  return "sqrt(" + a->to_string() + ")";
}

Silu::Silu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Silu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl silu = c.function("silu", P, P);
  return silu(a->to_z3(c, all_variables));
}

std::string Silu::to_string() const {
  return "silu(" + a->to_string() + ")";
}

Gelu::Gelu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Gelu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl gelu = c.function("gelu", P, P);
  return gelu(a->to_z3(c, all_variables));
}

std::string Gelu::to_string() const {
  return "gelu(" + a->to_string() + ")";
}

Relu::Relu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Relu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl relu = c.function("relu", P, P);
  return relu(a->to_z3(c, all_variables));
}

std::string Relu::to_string() const {
  return "relu(" + a->to_string() + ")";
}

Clamp::Clamp(float min_val, float max_val, std::shared_ptr<AbstractExpr> elems)
    : min_val(min_val), max_val(max_val), elems(elems) {
  assert(elems);
}

z3::expr Clamp::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl clamp = c.function("clamp", c.fpa_sort(8, 24), c.fpa_sort(8, 24), P, P);
  return clamp(c.fpa_val(min_val), c.fpa_val(max_val), elems->to_z3(c, all_variables));
}

std::string Clamp::to_string() const {
  return "clamp(" + std::to_string(min_val) + " <= x <= " + std::to_string(max_val) + ", " + elems->to_string() + ")";
}

RMS::RMS(int red_deg, std::shared_ptr<AbstractExpr> elems)
    : red_deg(red_deg), elems(elems) {
  assert(elems);
}

z3::expr RMS::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl rms = c.function("rms", c.int_sort(), P, P);
  return rms(c.int_val(red_deg), elems->to_z3(c, all_variables));
}

std::string RMS::to_string() const {
  return "rms(" + std::to_string(red_deg) + ", " + elems->to_string() + ")";
}

Sum::Sum(int red_deg, std::shared_ptr<AbstractExpr> summand)
    : red_deg_log(std::ceil(std::log2(red_deg))), summand(summand) {
  assert(red_deg > 1);
  assert(summand);
}

z3::expr Sum::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl sum = c.function("sum", c.int_sort(), P, P);
  return sum(c.int_val(red_deg_log), summand->to_z3(c, all_variables));
}

std::string Sum::to_string() const {
  return "sum(" + std::to_string(red_deg_log) + ", " + summand->to_string() + ")";
}

Sub::Sub(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Sub::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl sub = c.function("sub", P, P, P);
  return sub(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Sub::to_string() const {
  return "(" + lhs->to_string() + "-" + rhs->to_string() + ")";
}

Opaque::Opaque(std::string const &name,
               std::vector<std::shared_ptr<AbstractExpr>> args)
    : name(name), args(std::move(args)) {
  for (auto const &a : this->args) {
    assert(a);
  }
}

z3::expr Opaque::to_z3(z3::context &c,
                       std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::sort_vector domain(c);
  z3::expr_vector actual(c);
  for (auto const &a : args) {
    domain.push_back(P);
    actual.push_back(a->to_z3(c, all_variables));
  }
  z3::func_decl f = c.function(("opaque_" + name).c_str(), domain, P);
  return f(actual);
}

std::string Opaque::to_string() const {
  std::string out = name + "(";
  for (size_t k = 0; k < args.size(); k++) {
    out += (k ? "," : "") + args[k]->to_string();
  }
  return out + ")";
}

Split::Split(int split_dim, std::shared_ptr<AbstractExpr> input)
    : split_dim(split_dim), input(input) {
  assert(input);
}

// identity as no algebraic transformation is performed for split
z3::expr Split::to_z3(z3::context &c,
                      std::unordered_set<std::string> &all_variables) const {
  return input->to_z3(c, all_variables);
}

std::string Split::to_string() const {
  return input->to_string();
}

} // namespace search
} // namespace cleave_superoptimizer
