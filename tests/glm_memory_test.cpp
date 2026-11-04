// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <cassert>
#include <cstdio>

static void check_ram_plan() {
    using namespace strata::core::glmfast;
    // The old weighted split gives class 0 excess slots while starving class 1,
    // even though the byte budget exactly covers both mandatory capacities.
    const uint64_t budget = 600;
    const int64_t old0 = (int64_t) ((double) budget * 768 / 1104 / 12);
    const int64_t old1 = (int64_t) ((double) budget * 336 / 1104 / 4);
    assert(old0 > 20 && old1 < 90);
    std::vector<int64_t> slots;
    assert(ram_class_plan(budget, {12, 4}, {20, 90}, {80, 100}, slots));
    assert((slots == std::vector<int64_t>{20, 90}));
    assert(ram_class_plan(900, {12, 4}, {20, 90}, {80, 100}, slots));
    assert(slots[0] >= 20 && slots[1] >= 90);
    assert((uint64_t) slots[0] * 12 + (uint64_t) slots[1] * 4 <= 900);
    assert(ram_class_plan(2000, {12, 4}, {20, 90}, {80, 100}, slots));
    assert((slots == std::vector<int64_t>{80, 100}));
    const auto previous = slots;
    assert(!ram_class_plan(budget - 1, {12, 4}, {20, 90}, {80, 100}, slots));
    assert(slots == previous);  // no partially successful plan

    // The pool tail may all be lent to prefill; the main pool's three promotion
    // spares are not resident expert capacity either (288 - (24 - 3) = 267).
    const int warm = warm_pool_slots(false, 28, 288, 3);
    const int main = warm_pool_slots(false, 24, 288, 3);
    assert(warm == 25 && main == 21);
    assert(288 - main == (288 - warm) + 4);
    assert(ram_class_plan(267 * 16, {16}, {288 - main}, {300}, slots));
    assert(slots[0] == 267);
    assert(!ram_class_plan(266 * 16, {16}, {288 - main}, {300}, slots));

    // Geometry from the Windows Maya-L experiment; this tests the planner, not
    // a claim that these budgets are available on every Windows system.
    const std::vector<size_t> stride{12976128, 11665408};
    const uint64_t measured_budget = 135674292224ull;
    assert(ram_class_plan(measured_budget, stride, {2168, 9214}, {2200, 9298}, slots));
    assert(slots[0] >= 2168 && slots[1] >= 9214);
    assert((uint64_t) slots[0] * stride[0] + (uint64_t) slots[1] * stride[1] <= measured_budget);
    assert(!ram_class_plan(measured_budget, stride, {2184, 9282}, {2216, 9366}, slots));

    // Reject malformed sizes/counts and multiplication overflow before writing
    // output. Test UINT64_MAX even on MSVC, where long double is only double.
    const auto saved = slots;
    assert(!ram_class_plan(100, {0}, {0}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {-1}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {2}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {1}, {}, slots));
    assert(!ram_class_plan(100, {4}, {1}, {(int64_t) std::numeric_limits<int>::max() + 1}, slots));
    assert(!ram_class_plan(std::numeric_limits<size_t>::max(), {std::numeric_limits<size_t>::max()},
                          {2}, {2}, slots));
    assert(slots == saved);
    assert(ram_class_plan(std::numeric_limits<uint64_t>::max(), {1}, {0}, {1}, slots));
    assert(slots[0] == 1);
    assert(ram_class_plan(0, {}, {}, {}, slots) && slots.empty());
    assert(ram_class_plan(0, {4}, {0}, {0}, slots) && slots[0] == 0);

    // Exhaust small integer budgets, including rounding residue and zero slots.
    for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) {
        for (uint64_t bytes = 0; bytes < 256; ++bytes) {
            const uint64_t mandatory = (uint64_t) a * 4 + (uint64_t) b * 9;
            const bool ok = ram_class_plan(bytes, {4, 9}, {a, b}, {a + 3, b + 5}, slots);
            assert(ok == (bytes >= mandatory));
            if (!ok) continue;
            assert(slots[0] >= a && slots[0] <= a + 3);
            assert(slots[1] >= b && slots[1] <= b + 5);
            assert((uint64_t) slots[0] * 4 + (uint64_t) slots[1] * 9 <= bytes);
        }
    }
    assert(!balanced_resident_ram(false, true, false));  // opt-in only
    assert(!balanced_resident_ram(true, false, false));  // non-resident unchanged
    assert(!balanced_resident_ram(true, true, true));    // staging-only unchanged
    assert(balanced_resident_ram(true, true, false));
}

int main() {
    check_ram_plan();
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
    // #474: the complete pool is asked for on every GPU (the budget above decides); the setting wins either way
    {
        using strata::core::glmfast::complete_pool_on;
        const auto set = [](const char* v) {
#if defined(_WIN32)
            _putenv_s("STRATA_GLM_COMPLETE_POOL", v ? v : "");
#else
            if (v) setenv("STRATA_GLM_COMPLETE_POOL", v, 1); else unsetenv("STRATA_GLM_COMPLETE_POOL");
#endif
        };
        set(nullptr);
        assert(complete_pool_on());
        set("0");
        assert(!complete_pool_on());
        set("1");
        assert(complete_pool_on());
        set(nullptr);
    }
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
    // Windows' commit on an APU (128 GiB RAM: 12.8 GiB kept for the other programs).  A 140 GiB budget, a pool that can
    // take at most 120 GiB, 4 GiB its floor: 100 GiB of commit left caps it at 87.2 GiB and keeps 32.8 off the GPU;
    // with 150 GiB left it is capped to 137.2, still above all the pool takes, so nothing stays off; with 130 it is
    // capped to 117.2 and 2.8 GiB stay off; with no room at all the floor stays; a 16 GiB PC keeps 8 GiB for the others
    using strata::core::glmfast::commit_margin;
    using strata::core::glmfast::commit_pool_cap;
    using strata::core::glmfast::pagefile_hint_gib;
    assert(commit_margin(128 * G) == 128 * G / 10 && commit_margin(16 * G) == 8 * G);
    const size_t mg = 128 * G / 10;
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 100 * G, 128 * G) == std::make_pair(100 * G - mg, 20 * G + mg));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 150 * G, 128 * G) == std::make_pair(150 * G - mg, size_t{0}));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 200 * G, 128 * G) == std::make_pair(140 * G, size_t{0}));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 130 * G, 128 * G) == std::make_pair(130 * G - mg, mg - 10 * G));
    assert(commit_pool_cap(140 * G, 4 * G, 120 * G, 5 * G, 128 * G) == std::make_pair(4 * G, 116 * G));
    assert(commit_pool_cap(140 * G, 4 * G, 0, 100 * G, 128 * G) == std::make_pair(100 * G - mg, 40 * G + mg));
    assert(commit_pool_cap(2 * G, 4 * G, 120 * G, 0, 128 * G) == std::make_pair(2 * G, size_t{0}));   // (<= budget)
    assert(commit_pool_cap(10 * G, G, 0, 12 * G, 16 * G) == std::make_pair(4 * G, 6 * G));
    // the page file that keeps 33 GiB more on the GPU, with 128 GiB of RAM and a 32 GiB page file now: 65 GiB
    assert(pagefile_hint_gib(160 * G, 128 * G, 33 * G) == 65);
    assert(pagefile_hint_gib(160 * G, 128 * G, 33 * G - 1) == 65 && pagefile_hint_gib(128 * G, 128 * G, 1) == 1);
    std::puts("GLM memory policy checks passed (CPU only)");
}
