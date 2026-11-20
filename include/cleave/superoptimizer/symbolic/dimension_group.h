#pragma once

#include "cleave/superoptimizer/type.h"

#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cleave_superoptimizer {
namespace symbolic {


struct DimensionID {
  type::GuidType tensor_guid; // Which tensor this dimension belongs to
  int dim_idx;                // Which dimension index (0-4 for MAX_TENSOR_DIMS=5)

  bool operator==(DimensionID const &other) const {
    return tensor_guid == other.tensor_guid && dim_idx == other.dim_idx;
  }

  bool operator!=(DimensionID const &other) const {
    return !(*this == other);
  }

  bool operator<(DimensionID const &other) const {
    if (tensor_guid != other.tensor_guid) {
      return tensor_guid < other.tensor_guid;
    }
    return dim_idx < other.dim_idx;
  }

  // Hash function for use in unordered containers
  struct Hash {
    // Added to the tensor guid before hashing (see DimensionGrouper).
    type::GuidType guid_offset = 0;

    size_t operator()(DimensionID const &id) const {
      // Combine tensor_guid and dim_idx using standard hash combination
      size_t h1 = std::hash<type::GuidType>{}(id.tensor_guid + guid_offset);
      size_t h2 = std::hash<int>{}(id.dim_idx);
      return h1 ^ (h2 << 1);
    }
  };
};

struct DivisibilityRelation {
  DimensionID larger;
  DimensionID smaller;
};

class DimensionGrouper {
public:
  // first_guid: the guid of the first tensor of the graph being grouped.
  explicit DimensionGrouper(type::GuidType first_guid);
  void touch(DimensionID d);
  void merge(DimensionID d1, DimensionID d2);
  DimensionID find(DimensionID d);
  void mark_broadcast(DimensionID d);
  void mark_divisible(DimensionID larger, DimensionID smaller);
  bool is_broadcast(DimensionID d) const;
  std::vector<std::set<DimensionID>> get_groups() const;
  std::set<DimensionID> const &get_broadcast_group() const {
    return broadcast_group;
  }
  std::vector<DivisibilityRelation> const &get_divisibility_relations() const {
    return divisibility_relations;
  }

private:
  // get_groups() returns the groups in the iteration order of `parent` and of
  // a table of representatives, and the symbolizer numbers them in that order.
  // Guids come from a process-wide counter, so hashing a dimension by its guid
  // made that order, and with it the symbols and the checkpoint key of a
  // graph, depend on what the process had built before. The two tables hash a
  // dimension by the place of its tensor in the graph instead: its guid
  // relative to the graph's first tensor, counted from the counter's start.
  // Every graph then gets the symbols it gets as the first graph of a process.
  // (Which group gets which symbol does not change what the search finds, but
  // it prunes by comparing symbols, so it changes how many states it visits.)
  DimensionID::Hash position_hash;
  std::unordered_map<DimensionID, DimensionID, DimensionID::Hash> parent;
  std::unordered_map<DimensionID, int, DimensionID::Hash> rank;
  std::set<DimensionID> broadcast_group;
  std::vector<DivisibilityRelation> divisibility_relations;
};


struct SymbolicGraphConfig {
  int base_symbol = 6;
  bool use_prime_numbers = false;
  int symbol_increment = 4;
};

/**
 * @brief Assigns distinct symbolic values to dimension groups
 *
 * After grouping dimensions, this class assigns small distinct integer values
 * to each group for use in the symbolic graph. These distinct values enable
 * efficient shape matching during graph search.
 *
 * Assignment strategy:
 * 1. All broadcast dimensions (size 1) → symbol 1
 * 2. Each non-broadcast group → distinct symbol (6, 10, ...)
 */
class DimensionSymbolizer {
public:

  DimensionSymbolizer(
      DimensionGrouper const &grouper,
      SymbolicGraphConfig const &config = SymbolicGraphConfig());

  int get_symbol(DimensionID dim) const;

  std::vector<int> get_symbolic_dims(type::GuidType tensor_guid,
                                     int num_dims) const;

private:
  // Map from dimension to its symbolic value
  std::unordered_map<DimensionID, int, DimensionID::Hash> symbol_map;

  DimensionGrouper const &grouper;

  static int next_prime(int n);
};

} // namespace symbolic
} // namespace cleave_superoptimizer
