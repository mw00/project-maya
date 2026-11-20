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
    assert(!minimal_ram_tier(true, true, true));  // MAYA_GLM_RAM_GB / supplied budget wins
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
            _putenv_s("MAYA_GLM_COMPLETE_POOL", v ? v : "");
#else
            if (v) setenv("MAYA_GLM_COMPLETE_POOL", v, 1); else unsetenv("MAYA_GLM_COMPLETE_POOL");
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
    // #899: the "no room" error names a smaller hand-set reserve only where it can make the room; a budget of 0 (the
    // free VRAM read no more than the reserve) or more short than the whole reserve says it cannot; a default: nothing
    using strata::core::glmfast::reserve_hint;
    using strata::core::glmfast::ReserveHint;
    assert(reserve_hint(true, 2 * G, 600 * M, 3072 * M) == ReserveHint::smaller);
    assert(reserve_hint(true, 2 * G, 3072 * M, 3072 * M) == ReserveHint::smaller);
    assert(reserve_hint(true, 0, 600 * M, 3072 * M) == ReserveHint::budget_zero);                   // (#899's case)
    assert(reserve_hint(true, 2 * G, 4 * G, 3072 * M) == ReserveHint::short_beyond);
    assert(reserve_hint(false, 0, 600 * M, R) == ReserveHint::none);
    assert(reserve_hint(false, 2 * G, 600 * M, R) == ReserveHint::none);
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
    // #899 (2x RX 9070 16 GB, Linux HIP, Maya-S24, layers [0,12) on the head card, reserve 512 MiB): 1.93 of 15.92 GB
    // in use before the check, so it wrote 15.92 - 1.93 - 0.5 = 13.49 GB.  Read right after its frees, the free VRAM
    // was still 0.50 GB ("other 13.75 GB"): the pool's budget 0, 0 slots a layer.  Waiting for the frees gives the
    // budget back (13.49 GB, the working run's pool); a card whose frees land at once (CUDA) reads no pause at all;
    // frees that never land stop at the limit with the last read.
    {
        using strata::core::glmfast::await_check_frees;
        using strata::core::glmfast::expert_pool_budget;
        const double GB = 1073741824.0;
        const size_t total = (size_t) (15.92 * GB), before = total - (size_t) (1.93 * GB), rsv = 512 * M;
        const size_t held = total - (size_t) (15.42 * GB);   // 15.42 of 15.92 GB in use: the reserve's 0.50 GB free
        assert(expert_pool_budget(false, held, 0, rsv, 0, 0) == 0);   // the failure: no budget at all
        std::vector<size_t> reads = {held, held, held, before};
        size_t at = 0;
        int polls = -1, pauses = 0;
        const size_t fr = await_check_frees(before, 256 * M, 2000, [&] { return reads[std::min(at++, reads.size() - 1)]; },
                                            [&] { ++pauses; }, polls);
        assert(fr == before && polls == 3 && pauses == 3);
        const size_t budget = expert_pool_budget(false, fr, 0, rsv, 0, 0);
        assert(budget > (size_t) (13.4 * GB) && budget < (size_t) (13.6 * GB));   // the working run's 13.49 GB pool
        // frees already landed (CUDA, a V100): the first read, no pause - the budget exactly as before the fix
        at = 0;
        reads = {before - 4 * M};
        assert(await_check_frees(before, 256 * M, 2000, [&] { return reads[0]; }, [&] { ++pauses; }, polls) ==
                   before - 4 * M && polls == 0);
        // never landing: at most max_polls pauses, then the last read (the start reports it)
        assert(await_check_frees(before, 256 * M, 7, [&] { return held; }, [] {}, polls) == held && polls == 7);
        // max_polls 0 (MAYA_GLM_POOL_CHECK_WAIT_MS=0): one read, no wait
        assert(await_check_frees(before, 256 * M, 0, [&] { return held; }, [] {}, polls) == held && polls == 0);
    }
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
    {
        // #924: the room kept above the RAM tier at a pinning cap.  NVIDIA (or any non-AMD-Windows host): 512 MB
        // whatever the cap; a discrete AMD card under Windows: a tenth of the cap, rounded up to 256 MB - a 25.5 GB
        // cap (the report's 7900 XT) keeps 2.75 GB, so the tier lands ~2.2 GB lower than before, and a larger cap
        // keeps more; never under 512 MB; the setting wins on every host (rounded up, 0 = no room).
        using strata::core::glmfast::pin_cap_room;
        const int64_t M = int64_t{1} << 20, GB = int64_t{1} << 30;
        assert(pin_cap_room(false, 25 * GB, -1) == 512 * M && pin_cap_room(false, 0, -1) == 512 * M);
        assert(pin_cap_room(false, 200 * GB, -1) == 512 * M);
        const int64_t cap = 25 * GB + 512 * M;
        const int64_t room = pin_cap_room(true, cap, -1);
        assert(room == 2816 * M && room >= cap / 10 && room % (256 * M) == 0);
        assert(pin_cap_room(true, 60 * GB, -1) == 6144 * M && pin_cap_room(true, 2 * GB, -1) == 512 * M);
        assert(pin_cap_room(true, -1, -1) == 512 * M);
        assert(pin_cap_room(true, 25 * GB, 4096) == 4 * GB && pin_cap_room(false, 25 * GB, 3000) == 3072 * M);
        assert(pin_cap_room(true, 25 * GB, 0) == 0 && pin_cap_room(false, 25 * GB, 512) == 512 * M);
        // the tier the old and the new room give under the report's cap (17.5 GB more pin over 7.97 pinned)
        const int64_t held = 7970 * M, more = 17920 * M;
        assert(more - pin_cap_room(false, held + more, -1) > more - pin_cap_room(true, held + more, -1) + 2 * GB);
        // a Windows APU's shared-memory margin: an eighth of the limit, 1 to 4 GB, whole 256 MB (15.84 GB -> 2 GB,
        // above the 1.71 GB under the limit its start needed)
        using strata::core::glmfast::apu_shared_margin;
        assert(apu_shared_margin(16220 * M) == 2 * GB && apu_shared_margin(16220 * M) > 1751 * M);
        assert(apu_shared_margin(4 * GB) == GB && apu_shared_margin(0) == GB && apu_shared_margin(-1) == GB);
        assert(apu_shared_margin(25 * GB) == 3 * GB + 256 * M && apu_shared_margin(64 * GB) == 4 * GB);
    }
    {
        // #924 / #1035: the cap measured before the expert pool - Windows + a discrete Radeon, one GPU only; never
        // on NVIDIA, Linux, an APU or a split; the setting wins on one GPU (an empty one counts as unset), never in a
        // split
        using strata::core::glmfast::pin_cap_before_pool;
        assert(pin_cap_before_pool(true, false, 1, nullptr) && pin_cap_before_pool(true, false, 1, ""));
        assert(!pin_cap_before_pool(false, false, 1, nullptr) && !pin_cap_before_pool(true, true, 1, nullptr));
        assert(!pin_cap_before_pool(true, false, 2, nullptr) && !pin_cap_before_pool(false, true, 3, nullptr));
        assert(!pin_cap_before_pool(true, false, 1, "0") && pin_cap_before_pool(false, false, 1, "1"));
        assert(pin_cap_before_pool(false, true, 1, "1") && !pin_cap_before_pool(false, true, 2, "1"));
        using strata::core::glmfast::pin_left_since;
        const int64_t M = int64_t{1} << 20, GB = int64_t{1} << 30;
        assert(pin_left_since(29 * GB, 300 * M) == 29 * GB - 300 * M && pin_left_since(GB, 2 * GB) == 0);
        assert(pin_left_since(GB, -5) == GB && pin_left_since(0, 0) == 0);
        // the classes sized one after another under a measured cap never pass it less the room kept above them,
        // and with room to spare each keeps its planned slots (the caller's min with the plan)
        using strata::core::glmfast::pin_cap_share;
        using strata::core::glmfast::pin_cap_room;
        const int64_t stride[] = {6 * M + 3 * 4096, 9 * M + 512, 2 * M + 7};
        const int64_t plan[] = {3000, 1500, 900};   // ~35 GB planned: more than the cap below
        for (const int64_t left0 : {29 * GB + 768 * M, 12 * GB, 40 * GB, 600 * M}) {
            const int64_t keep = pin_cap_room(true, left0, -1);
            int64_t left = left0, pinned = 0;
            for (int c = 0; c < 3; ++c) {
                int64_t planned = 0;
                for (int k = c; k < 3; ++k) planned += plan[k] * stride[k];
                const int64_t n = std::min(plan[c], pin_cap_share(left, keep, planned, plan[c]));
                assert(n >= 0);
                left -= n * stride[c];
                pinned += n * stride[c];
            }
            assert(pinned + keep <= left0 || (left0 <= keep && pinned == 0));
            if (left0 == 40 * GB) assert(pinned == plan[0] * stride[0] + plan[1] * stride[1] + plan[2] * stride[2]);
            if (left0 == 29 * GB + 768 * M) assert(pinned > left0 - keep - 3 * 9 * M);   // (no more than a slot a class lost)
        }
    }
    std::puts("GLM memory policy checks passed (CPU only)");
}
