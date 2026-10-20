#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace strata::core::glmfast {

// The complete pool and the all-resident launch skip (STRATA_GLM_COMPLETE_POOL, STRATA_GLM_RESIDENT_SKIP): on by
// default on AMD, where the Strix / Gorgon Halo APUs hold every expert (measured there); an NVIDIA build keeps the
// capped pool and every route's launches unless the setting is 1.  The setting, when given, wins either way.
inline bool on_by_default_on_amd(const char* env) {
    if (const char* v = std::getenv(env)) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
    return true;
#else
    return false;
#endif
}
// ... and the other way round: on by default on NVIDIA, off on AMD unless the setting is 1
inline bool off_by_default_on_amd(const char* env) {
    if (const char* v = std::getenv(env)) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
    return false;
#else
    return true;
#endif
}
// STRATA_GLM_GROUP_EXPERTS=1: the batched rows' routed experts each run once for all the rows of a group (glm_rows.cu)
inline bool group_experts_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_GROUP_EXPERTS");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
// STRATA_GLM_SKIP_EMPTY=1: the batched rows leave out the launches their routes have no work for (glm_rows.cu)
inline bool skip_empty_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_SKIP_EMPTY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

// All inputs are bytes. On an APU the GPU pool and pinned host allocations spend
// the same physical memory; the runtime's GTT/carve-out size is not free RAM.
inline size_t expert_pool_budget(bool unified, size_t device_free, size_t ram_available,
                                size_t reserve, size_t ram_headroom, size_t expert_bytes) {
    if (!unified) return device_free > reserve ? device_free - reserve : 0;
    const size_t room = ram_available > ram_headroom ? ram_available - ram_headroom : 0;
    return std::min(expert_bytes, room > reserve ? room - reserve : 0);
}

// The VRAM kept free beside a discrete card's expert pool: the reserve, except on a small card where the default
// one would leave the pool short of its floor (a route's experts and the spares in every layer) - there an automatic
// budget lowers it to what leaves exactly the floor, down to `small`.  A reserve set by hand, a fixed budget
// (--expert-cache N) and a card short even at `small` keep it.  {reserve, lowered}
inline std::pair<size_t, bool> pool_reserve(size_t device_free, size_t reserve, bool given, bool fixed, size_t floor,
                                            size_t small) {
    if (given || fixed || device_free >= reserve + floor || device_free < floor + small) return {reserve, false};
    return {device_free - floor, true};
}

// The check of a written budget: the bytes it gives back once `have` of the `asked` bytes were allocated and touched
// and `free_after` was read - what the driver refused, plus what the reserve is short by and 64 MiB.  A read of
// nearly nothing free (under 16 MiB) only says "at least": the first two give back 1 GiB each, later ones a quarter.
inline size_t pool_give_back(size_t asked, size_t have, size_t free_after, size_t reserve, int& zero_reads) {
    constexpr size_t kSlack = size_t{64} << 20, kGiB = size_t{1} << 30;
    size_t give = asked > have ? asked - have : 0;
    if (free_after + kSlack < reserve) {
        size_t g = reserve - free_after + kSlack;
        if (free_after < (size_t{16} << 20)) g = std::max(g, ++zero_reads <= 2 ? kGiB : asked / 4);
        give += g;
    }
    return std::min(give, asked);
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

// Windows' commit on an APU: Windows charges every allocation of an integrated GPU to its commit (RAM + page file),
// though it does not count the carve-out as RAM (DeepSeek's engine measured it on a Radeon 8065S: src/ds4/ds4_model.cu,
// commit_room).  The other programs keep max(8 GiB, 10% of the RAM) of it, as there.
inline size_t commit_margin(size_t ram_total) {
    return std::max(size_t{8} << 30, ram_total / 10);
}
// The expert pool's budget within what Windows can still commit (commit_avail) beside that margin - never below its
// floor (a route's experts and the spares in every layer: the least a decode runs with), nor above `budget`.  `all`:
// the most the pool can ever take (0: no such bound).  {the budget, the bytes of experts that keeps off the GPU}
inline std::pair<size_t, size_t> commit_pool_cap(size_t budget, size_t floor, size_t all, size_t commit_avail,
                                                 size_t ram_total) {
    const size_t margin = commit_margin(ram_total);
    const size_t room = commit_avail > margin ? commit_avail - margin : 0;
    const size_t cap = std::min(budget, std::max(room, floor));
    const size_t top = all > 0 ? all : budget;
    return {cap, std::min(budget, top) - std::min(cap, top)};
}
// The page file (GiB, rounded up) that holds `kept` bytes more: today's - the commit limit less the RAM - and those
inline size_t pagefile_hint_gib(size_t commit_limit, size_t ram_total, size_t kept) {
    const size_t pf = commit_limit > ram_total ? commit_limit - ram_total : 0;
    return (pf + kept + (size_t{1} << 30) - 1) >> 30;
}

// The slots the warm-up fills: every expert where a unified pool holds them all, else all but the spares - and never
// more than there are experts (a complete pool has more slots than experts: the warm-up's order lists each once)
inline int warm_pool_slots(bool unified, int slots, int experts, int spares) {
    return full_unified_pool(unified, slots, experts) ? experts : std::min(experts, std::max(0, slots - spares));
}

}  // namespace strata::core::glmfast
