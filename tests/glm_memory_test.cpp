// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <cassert>
#include <cstdio>

int main() {
    using strata::core::glmfast::expert_pool_budget;
    using strata::core::glmfast::minimal_ram_tier;
    using strata::core::glmfast::full_unified_pool;
    using strata::core::glmfast::warm_pool_slots;
    using strata::core::glmfast::complete_pool;
    using strata::core::glmfast::complete_pool_slots;
    constexpr size_t G = size_t{1} << 30;
    constexpr size_t experts = 80 * G + 174 * (size_t{1} << 20);

    // A 128 GiB APU with 110 GiB available after setup: the 112 GiB runtime
    // aperture (and even a tiny reported free value) cannot change the OS budget.
    assert(expert_pool_budget(true, 112 * G, 110 * G, G, 16 * G, experts) == experts);
    assert(expert_pool_budget(true, G, 110 * G, G, 16 * G, experts) == experts);
    // Memory pressure leaves 53 GiB; missing MemAvailable or less than headroom
    // leaves zero rather than wrapping or trusting HIP's much larger free figure.
    assert(expert_pool_budget(true, 112 * G, 70 * G, G, 16 * G, experts) == 53 * G);
    assert(expert_pool_budget(true, 112 * G, 0, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 16 * G, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 17 * G, G, 16 * G, experts) == 0);
    // User headroom/reserve and the discrete formula retain their meaning.
    assert(expert_pool_budget(true, 112 * G, 70 * G, 2 * G, 8 * G, experts) == 60 * G);
    assert(expert_pool_budget(false, 12 * G, 110 * G, 3 * G, 16 * G, experts) == 9 * G);
    assert(expert_pool_budget(false, G, 110 * G, 3 * G, 16 * G, experts) == 0);
    assert(minimal_ram_tier(true, true, false));
    assert(!minimal_ram_tier(true, true, true));  // STRATA_GLM_RAM_GB / supplied budget wins
    assert(!minimal_ram_tier(true, false, false));
    assert(!minimal_ram_tier(false, true, false));
    // Warm every expert when it fits, and never evict residents to make spares.
    // Smaller APU pools and all discrete pools retain the existing spare quota.
    assert(warm_pool_slots(true, 288, 288, 3) == 288);
    assert(full_unified_pool(true, 288, 288));
    assert(warm_pool_slots(true, 192, 288, 3) == 189);
    assert(!full_unified_pool(true, 192, 288));
    assert(warm_pool_slots(false, 288, 288, 3) == 285);
    assert(!full_unified_pool(false, 288, 288));
    // A complete pool (every expert + the spares in the main slots, the prompt's tail beyond them - the Windows APU's
    // carve-out): the warm-up fills each expert once, never past the 288-entry order
    assert(warm_pool_slots(false, 288 + 3 + 21, 288, 3) == 288);
    assert(warm_pool_slots(false, 290, 288, 3) == 287);
    assert(complete_pool(291, 288, 3));
    assert(!complete_pool(290, 288, 3));
    assert(!complete_pool(288 - 21, 288, 3));
    // 158 GiB of budget, 308 MiB a slot over every MoE layer: 288 + 3 + 18 slots fit; 40 GiB do not; nothing on zeros
    constexpr size_t M = size_t{1} << 20;
    assert(complete_pool_slots(158 * G, 308 * M, 288, 3, 18) == 309);
    assert(complete_pool_slots(309 * 308 * M, 308 * M, 288, 3, 18) == 309);
    assert(complete_pool_slots(309 * 308 * M - 1, 308 * M, 288, 3, 18) == 0);
    assert(complete_pool_slots(40 * G, 308 * M, 288, 3, 18) == 0);
    assert(complete_pool_slots(158 * G, 0, 288, 3, 18) == 0);
    assert(complete_pool_slots(158 * G, 308 * M, 288, 3, -5) == 291);
    // The pool's reserve: 700 MiB, except where it leaves the pool short of its floor (here 2 GiB) - an automatic budget
    // then lowers it to leave exactly the floor, never below 300 MiB; one set by hand or a fixed budget keeps it
    using strata::core::glmfast::pool_reserve;
    const size_t R = 700 * M, S = 300 * M;
    assert(pool_reserve(10 * G, R, false, false, 2 * G, S) == std::make_pair(R, false));            // room for both
    assert(pool_reserve(2 * G + 500 * M, R, false, false, 2 * G, S) == std::make_pair(500 * M, true));
    assert(pool_reserve(2 * G + 300 * M, R, false, false, 2 * G, S) == std::make_pair(S, true));
    assert(pool_reserve(2 * G + 299 * M, R, false, false, 2 * G, S) == std::make_pair(R, false));    // short anyway
    assert(pool_reserve(2 * G + 500 * M, R, true, false, 2 * G, S) == std::make_pair(R, false));     // given
    assert(pool_reserve(2 * G + 500 * M, R, false, true, 2 * G, S) == std::make_pair(R, false));     // --expert-cache N
    // The written budget's check (20 GiB asked, 700 MiB reserve): free as planned gives nothing back; 200 MiB free
    // gives back the 500 short + 64; nearly nothing free 1 GiB twice, then a quarter; a refused piece what it refused
    using strata::core::glmfast::pool_give_back;
    int zr = 0;
    assert(pool_give_back(20 * G, 20 * G, 700 * M, R, zr) == 0);
    assert(pool_give_back(20 * G, 20 * G, 640 * M, R, zr) == 0);
    assert(pool_give_back(20 * G, 20 * G, 200 * M, R, zr) == 564 * M);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == G && zr == 1);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == G && zr == 2);
    assert(pool_give_back(20 * G, 20 * G, 0, R, zr) == 5 * G && zr == 3);
    zr = 0;
    assert(pool_give_back(20 * G, 18 * G, 2 * G, R, zr) == 2 * G);
    assert(pool_give_back(20 * G, 18 * G, 100 * M, R, zr) == 2 * G + 664 * M);
    assert(pool_give_back(G, 0, 0, R, zr) == G);   // never more than was asked
    std::puts("GLM memory policy checks passed (CPU only)");
}
