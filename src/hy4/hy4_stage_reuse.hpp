#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hy4 {

// Returns the first slot available for a partial append. On success requested contains
// only the misses. A zero result leaves the requested set on the original refill path.
inline int stage_tail(std::vector<int>& requested, const int* slots, int cached, int capacity) {
    if (cached <= 0 || cached >= capacity || requested.size() > static_cast<std::size_t>(capacity)) return 0;
    std::size_t missing = 0;
    for (int expert : requested) {
        const int slot = slots[expert];
        if (slot >= cached) return 0;
        missing += slot < 0;
    }
    if (missing == 0 || missing == requested.size() || missing > static_cast<std::size_t>(capacity - cached)) return 0;
    requested.erase(std::remove_if(requested.begin(), requested.end(),
                                   [&](int expert) { return slots[expert] >= 0; }), requested.end());
    return cached;
}

inline void stage_slots_for_copy(int* slots, int experts, const int* copied, int count, int base) {
    if (base == 0) std::fill_n(slots, experts, -1);
    for (int i = 0; i < count; ++i) slots[copied[i]] = base + i;
}

}  // namespace hy4
