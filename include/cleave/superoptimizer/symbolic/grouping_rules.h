#pragma once

#include "cleave/superoptimizer/kernel/operator.h"
#include "cleave/superoptimizer/symbolic/dimension_group.h"

namespace cleave_superoptimizer {
namespace symbolic {

/**
 * @brief Applies operator-specific dimension grouping rules
 *
 * Each operator type has specific constraints on which dimensions must be equal.
 * This class encapsulates the grouping logic for all operator types.
 *
 * Grouping Rules by Operator Type:
 *
 * 1. Matmul (A × B → C):
 *    - Reduction dimensions: A.dim[-1] must equal B.dim[-2]
 *    - Batch dimensions: corresponding batch dims must match (unless broadcast)
 *    - Broadcasting: size-1 dimensions marked as broadcast
 *
 * 2. Element-wise Binary (A op B → C):
 *    - Matching dimensions: A.dim[i] == B.dim[i] > 1 are grouped
 *    - Broadcasting: size-1 dimensions marked as broadcast
 *
 * 3. Element-wise Unary (f(A) → B):
 *    - Identity: A.dim[i] == B.dim[i] for all i
 *
 * 4. Reduction (A → B, reducing dim d):
 *    - Non-reduced dimensions: A.dim[i] == B.dim[i] for i != d
 *    - Reduced dimension: A.dim[d] relates to B.dim[d] by reduction factor
 *
 * 5. RMS Norm:
 *    - Similar to element-wise unary with normalization
 */
class GroupingRules {
public:
  static void apply_grouping(kernel::KNOperator const *op,
                             DimensionGrouper &grouper);

private:
  static void group_matmul(kernel::KNOperator const *op,
                          DimensionGrouper &grouper);

  static void group_element_binary(kernel::KNOperator const *op,
                                  DimensionGrouper &grouper);

  static void group_element_unary(kernel::KNOperator const *op,
                                 DimensionGrouper &grouper);

  static void group_reduction(kernel::KNOperator const *op,
                             DimensionGrouper &grouper);

  static void group_rms_norm(kernel::KNOperator const *op,
                            DimensionGrouper &grouper);


  static void group_if_matching_non_broadcast(DimensionGrouper &grouper,
                                              DimensionID dim1,
                                              int size1,
                                              DimensionID dim2,
                                              int size2);

  static void group_if_compatible_batch(DimensionGrouper &grouper,
                                        DimensionID dim1,
                                        int size1,
                                        DimensionID dim2,
                                        int size2);
};

} // namespace symbolic
} // namespace cleave_superoptimizer
