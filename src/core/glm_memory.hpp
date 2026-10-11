#pragma once

#include <algorithm>
#include <cstddef>

namespace strata::core::glmfast {

// All inputs are bytes. On an APU the GPU pool and pinned host allocations spend
// the same physical memory; the runtime's GTT/carve-out size is not free RAM.
inline size_t expert_pool_budget(bool unified, size_t device_free, size_t ram_available,
                                size_t reserve, size_t ram_headroom, size_t expert_bytes) {
    if (!unified) return device_free > reserve ? device_free - reserve : 0;
    const size_t room = ram_available > ram_headroom ? ram_available - ram_headroom : 0;
    return std::min(expert_bytes, room > reserve ? room - reserve : 0);
}

inline bool minimal_ram_tier(bool unified, bool all_experts_fit, bool explicit_budget) {
    return unified && all_experts_fit && !explicit_budget;
}

inline bool full_unified_pool(bool unified, int slots, int experts) {
    return unified && slots >= experts;
}

// A COMPLETE pool: a layer's main slots (the lendable tail aside) hold every expert AND the spares - the Windows APU's
// carve-out, a big card, a split over many GPUs.  Nothing is ever evicted there (a spare always finds a free slot), the
// prompt path borrows only the tail beyond them, and the decode never misses.
inline bool complete_pool(int main_slots, int experts, int spares) {
    return main_slots >= experts + spares;
}

// The slots a layer gets where the budget (avail bytes; stride_sum: one slot in every MoE layer) holds a complete pool
// plus the lendable tail - every expert, the spares and `tail` slots - else 0 (the pool is sized from the budget)
inline int complete_pool_slots(size_t avail, size_t stride_sum, int experts, int spares, int tail) {
    const size_t per = (size_t) std::max(0, experts + spares + std::max(0, tail));
    return stride_sum > 0 && experts > 0 && avail / stride_sum >= per ? (int) per : 0;
}

// The slots the warm-up fills: every expert where a unified pool holds them all, else all but the spares - and never
// more than there are experts (a complete pool has more slots than experts: the warm-up's order lists each once)
inline int warm_pool_slots(bool unified, int slots, int experts, int spares) {
    return full_unified_pool(unified, slots, experts) ? experts : std::min(experts, std::max(0, slots - spares));
}

}  // namespace strata::core::glmfast
