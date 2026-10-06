#include "cleave/superoptimizer/symbolic/dimension_group.h"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace cleave_superoptimizer {
namespace symbolic {

// ============================================================================
// DimensionGrouper Implementation
// ============================================================================

DimensionGrouper::DimensionGrouper() {}

void DimensionGrouper::touch(DimensionID d) {
  (void)find(d);
}

void DimensionGrouper::merge(DimensionID d1, DimensionID d2) {
  // Find representatives of both dimensions
  DimensionID root1 = find(d1);
  DimensionID root2 = find(d2);

  // Already in same group
  if (root1 == root2) {
    return;
  }

  // Union by rank: attach smaller tree under larger tree
  int rank1 = rank.count(root1) ? rank[root1] : 0;
  int rank2 = rank.count(root2) ? rank[root2] : 0;

  if (rank1 < rank2) {
    parent[root1] = root2;
  } else if (rank1 > rank2) {
    parent[root2] = root1;
  } else {
    // Equal rank: attach root2 under root1 and increment root1's rank
    parent[root2] = root1;
    rank[root1] = rank1 + 1;
  }
}

DimensionID DimensionGrouper::find(DimensionID d) {
  // If dimension has no parent, it's its own representative
  if (parent.count(d) == 0) {
    parent[d] = d;
    return d;
  }

  // Path compression: make all nodes on path point directly to root
  if (parent[d] != d) {
    parent[d] = find(parent[d]);
  }

  return parent[d];
}

void DimensionGrouper::mark_broadcast(DimensionID d) {
  touch(d);
  broadcast_group.insert(d);
}

void DimensionGrouper::mark_divisible(DimensionID larger,
                                      DimensionID smaller) {
  touch(larger);
  touch(smaller);
  divisibility_relations.push_back({larger, smaller});
}

bool DimensionGrouper::is_broadcast(DimensionID d) const {
  return broadcast_group.count(d) > 0;
}

std::vector<std::set<DimensionID>> DimensionGrouper::get_groups() const {
  // Map from representative to all members in that group
  std::unordered_map<DimensionID, std::set<DimensionID>, DimensionID::Hash>
      rep_to_group;

  // Collect all dimensions that have been touched
  for (auto const &[dim, parent_dim] : parent) {
    // Find representative (need const_cast to call non-const find)
    DimensionID rep =
        const_cast<DimensionGrouper *>(this)->find(const_cast<DimensionID &>(dim));
    rep_to_group[rep].insert(dim);
  }

  // Convert to vector
  std::vector<std::set<DimensionID>> groups;
  groups.reserve(rep_to_group.size());
  for (auto const &[rep, group] : rep_to_group) {
    groups.push_back(group);
  }

  return groups;
}

// ============================================================================
// DimensionSymbolizer Implementation
// ============================================================================

int DimensionSymbolizer::next_prime(int n) {
  if (n < 2) {
    return 2;
  }

  // Simple prime checker
  auto is_prime = [](int x) {
    if (x < 2) return false;
    if (x == 2) return true;
    if (x % 2 == 0) return false;
    for (int i = 3; i * i <= x; i += 2) {
      if (x % i == 0) return false;
    }
    return true;
  };

  int candidate = n + 1;
  while (!is_prime(candidate)) {
    candidate++;
  }
  return candidate;
}

DimensionSymbolizer::DimensionSymbolizer(
    DimensionGrouper const &grouper,
    SymbolicGraphConfig const &config)
    : grouper(grouper), max_symbol_used(1), num_groups(0) {

  int next_symbol = config.use_prime_numbers ? 2 * config.symbol_increment : config.base_symbol;

  // Step 1: Assign 1 to all broadcast dimensions
  for (auto const &dim : grouper.get_broadcast_group()) {
    symbol_map[dim] = 1;
  }

  // Step 2: Assign distinct base symbols to each non-broadcast group
  std::unordered_set<DimensionID, DimensionID::Hash> processed_reps;
  std::unordered_map<DimensionID, int, DimensionID::Hash> rep_to_symbol;

  // Get all groups
  auto groups = grouper.get_groups();

  for (auto const &group : groups) {
    if (group.empty()) {
      continue;
    }

    // Get representative of this group
    DimensionID rep =
        const_cast<DimensionGrouper &>(grouper).find(*group.begin());

    assert(processed_reps.count(rep) == 0);
    processed_reps.insert(rep);

    // Skip if this is a broadcast group
    if (grouper.is_broadcast(rep)) {
      continue;
    }

    // Assign next available symbol to all dimensions in this group
    int symbol = next_symbol;
    rep_to_symbol[rep] = symbol;

    // Update next symbol
    if (config.use_prime_numbers) {
      next_symbol = next_prime(next_symbol / config.symbol_increment) * config.symbol_increment;
    } else {
      next_symbol += config.symbol_increment;
    }

    num_groups++;
  }

  // Step 3: Preserve known non-equal divisibility constraints, such as GQA
  // head batches where q_heads is a multiple of kv_heads.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto const &relation : grouper.get_divisibility_relations()) {
      DimensionID larger_rep =
          const_cast<DimensionGrouper &>(grouper).find(relation.larger);
      DimensionID smaller_rep =
          const_cast<DimensionGrouper &>(grouper).find(relation.smaller);
      if (larger_rep == smaller_rep ||
          grouper.is_broadcast(larger_rep) ||
          grouper.is_broadcast(smaller_rep)) {
        continue;
      }
      auto larger_it = rep_to_symbol.find(larger_rep);
      auto smaller_it = rep_to_symbol.find(smaller_rep);
      if (larger_it == rep_to_symbol.end() ||
          smaller_it == rep_to_symbol.end()) {
        continue;
      }

      int required_multiple = smaller_it->second;
      int adjusted_symbol = std::lcm(larger_it->second, required_multiple);
      if (adjusted_symbol != larger_it->second) {
        larger_it->second = adjusted_symbol;
        changed = true;
      }
    }
  }

  // Step 4: Write final group symbols to every dimension in each group
  for (auto const &group : groups) {
    if (group.empty()) {
      continue;
    }

    DimensionID rep = const_cast<DimensionGrouper &>(grouper).find(*group.begin());
    if (grouper.is_broadcast(rep)) {
      continue;
    }

    auto symbol_it = rep_to_symbol.find(rep);
    assert(symbol_it != rep_to_symbol.end());
    int symbol = symbol_it->second;
    for (auto const &dim : group) {
      symbol_map[dim] = symbol;
    }
    max_symbol_used = std::max(max_symbol_used, symbol);
  }
}

int DimensionSymbolizer::get_symbol(DimensionID dim) const {
  // First check if dimension is directly in the symbol map
  auto it = symbol_map.find(dim);
  assert(it != symbol_map.end() && "Dimension not found in symbol map");
  return it->second;
}

std::vector<int> DimensionSymbolizer::get_symbolic_dims(
    type::GuidType tensor_guid,
    int num_dims) const {
  std::vector<int> symbolic_dims;
  symbolic_dims.reserve(num_dims);

  for (int i = 0; i < num_dims; i++) {
    DimensionID dim{tensor_guid, i};
    symbolic_dims.push_back(get_symbol(dim));
  }

  return symbolic_dims;
}

} // namespace symbolic
} // namespace cleave_superoptimizer
