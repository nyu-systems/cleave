#pragma once

#include "cleave/superoptimizer/kernel/graph.h"
#include "cleave/superoptimizer/search/abstract_expr/abstract_expr.h"

namespace cleave_superoptimizer {
namespace search {

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns);

} // namespace search
} // namespace cleave_superoptimizer
