#pragma once

#include "cleave/superoptimizer/utils/hash_utils.h"
#include "cleave/superoptimizer/utils/json_utils.h"
#include "z3++.h"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace cleave_superoptimizer {
namespace search {

class AbstractExpr {
public:
  AbstractExpr() = default;
  virtual ~AbstractExpr() = default;

  virtual z3::expr
      to_z3(z3::context &c,
            std::unordered_set<std::string> &all_variables) const = 0;
  bool subpattern_to(AbstractExpr const &other) const;
  bool operator==(AbstractExpr const &other) const;
  // True if `other` (a to_string()-serialized pattern) equals this expression
  // or any of its transitive sub-expressions. Used as a sound structural
  // fallback in subpattern_to when Z3 cannot decide within its resource limit.
  bool contains_subexpr(std::string const &other) const;
  virtual std::string to_string() const = 0;
};

class Var : public AbstractExpr {
public:
  Var(std::string const &name);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string name;
};

class Add : public AbstractExpr {
public:
  Add(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Mul : public AbstractExpr {
public:
  Mul(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Div : public AbstractExpr {
public:
  Div(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Exp : public AbstractExpr {
public:
  Exp(std::shared_ptr<AbstractExpr> exponent);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> exponent;
};

class Sqrt : public AbstractExpr {
public:
  Sqrt(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Silu : public AbstractExpr {
public:
  Silu(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Gelu : public AbstractExpr {
public:
  Gelu(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Relu : public AbstractExpr {
  public:
    Relu(std::shared_ptr<AbstractExpr> a);
    z3::expr to_z3(z3::context &c,
                   std::unordered_set<std::string> &all_variables) const override;
    std::string to_string() const override;
    std::shared_ptr<AbstractExpr> a;
};

class Clamp : public AbstractExpr {
  public:
    Clamp(float min_val, float max_val, std::shared_ptr<AbstractExpr> elems);
    z3::expr to_z3(z3::context &c,
                   std::unordered_set<std::string> &all_variables) const override;
    std::string to_string() const override;
    float min_val;
    float max_val;
    std::shared_ptr<AbstractExpr> elems;
  };

// Note(@Mengdi): Replace it with Sqr and Sqrt once we have related algebraic
// transformation
class RMS : public AbstractExpr {
public:
  RMS(int red_deg, std::shared_ptr<AbstractExpr> elems);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  int red_deg;
  std::shared_ptr<AbstractExpr> elems;
};

class Red : public AbstractExpr {
public:
  Red(int red_deg, std::shared_ptr<AbstractExpr> summand);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  int red_deg_log;
  std::shared_ptr<AbstractExpr> summand;
};

class Split : public AbstractExpr {
public:
  Split(int split_dim, std::shared_ptr<AbstractExpr> input);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  int split_dim;
  std::shared_ptr<AbstractExpr> input;
};

} // namespace search
} // namespace cleave_superoptimizer
