// tests/glm_dsa/glm_tier_plan_test.cpp - glm-dsa B8.2 / B8.3 / B8.4 / B8.7 host rules on the CPU (no GPU, no ggml).
//
//   1. B8.2 segments: the cap (16 GiB on glm-dsa, off on glm5-next, STRATA_GLM_RAM_SEG_GB), seg_plan's whole-slot
//      segments, Maya's shrink steps, seg_pin against a simulated pinning cap (only the LAST segment shrinks, the
//      later ones are released, a fall below `least` releases everything), seg_node, the budget reset rule;
//   2. B8.7: STRATA_GLM_THP, /proc/buddyinfo's free 2 MB bytes, the per-segment madvise plan, the AnonHugePages sum of
//      a /proc/self/smaps text over the tier's ranges, the fault-in plans (every byte touched once);
//   3. B8.3: the lane pool mode table (Maya's rules for unset / 0 / 1 unchanged, "socket", node-split);
//   4. B8.4: split_layout at GLM-5.3 (6144 x 2048, IQ2_XXS / IQ2_S gate/up, IQ3_XXS / Q2_K down) and its refusals;
//      split_loc / unsplit_off a bijection; scatter / gather / split_ops / chunked scatter_range agree byte for byte;
//      half_stride; add_partial.
#include "glm_tier_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace gt = strata::core::glmtier;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr size_t GiB = (size_t) 1 << 30;

// ---------------------------------------------------------------- 1. segments

void test_segments() {
    std::printf("B8.2 segments\n");
    check(gt::seg_cap_bytes(true, nullptr) == 16 * GiB, "glm-dsa default cap 16 GiB");
    check(gt::seg_cap_bytes(false, nullptr) == 0, "glm5-next default: off (Maya's single allocation)");
    check(gt::seg_cap_bytes(true, "0") == 0, "STRATA_GLM_RAM_SEG_GB=0: off");
    check(gt::seg_cap_bytes(false, "8") == 8 * GiB, "STRATA_GLM_RAM_SEG_GB=8 on glm5-next: 8 GiB");
    check(gt::seg_cap_bytes(true, "") == 16 * GiB, "empty env: the default");

    // GLM-5.3 UD-IQ2_M: ~9.75 MB a slot (4 KB rounded), 12000 slots ~ 117 GB
    const size_t slot = 9752576;
    const gt::SegPlan p = gt::seg_plan(12000, slot, 16 * GiB);
    bool cap_ok = true, whole = true;
    for (size_t k = 0; k < p.slots.size(); ++k) {
        cap_ok = cap_ok && (size_t) p.slots[k] * slot <= 16 * GiB;
        whole = whole && (k + 1 == p.slots.size() || p.slots[k] == p.seg_slots);
    }
    check(p.total() == 12000, "segments sum to the class");
    check(cap_ok, "every segment <= 16 GiB");
    check(whole, "every segment but the last is full");
    check((int) p.slots.size() == (int) ((12000 + p.seg_slots - 1) / p.seg_slots), "segment count");
    check(p.seg_slots == (int) (16 * GiB / slot), "full segment = floor(cap / slot) slots");
    const gt::SegPlan one = gt::seg_plan(100, slot, 0);
    check(one.slots.size() == 1 && one.slots[0] == 100, "cap 0: one segment");
    const gt::SegPlan big = gt::seg_plan(3, 20 * GiB, 16 * GiB);
    check(big.slots.size() == 3 && big.seg_slots == 1, "a slot over the cap: one slot a segment");
    check(gt::seg_plan(0, slot, 16 * GiB).slots.empty(), "n = 0: no segments");

    check(gt::shrink_step(3200, 0) == 3100 && gt::shrink_step(3200, 3) == 3100, "tries 0..3: n * 31 / 32");
    check(gt::shrink_step(3200, 4) == 2800, "try 4+: n * 7 / 8");

    // seg_pin against a cap of pinnable bytes, segments pinned in order (the parallel round's result is the same set)
    struct Sim {
        int64_t cap_slots;
        int64_t used = 0;
        std::vector<int> live;   // per segment: slots held
        std::vector<int> released;
    };
    const auto run = [&](const gt::SegPlan& plan, int64_t cap_slots, int64_t least, std::vector<int>& pinned, Sim& sim) {
        sim = Sim{cap_slots};
        sim.live.assign(plan.slots.size(), 0);
        const auto first = [&](std::vector<char>& ok) {
            for (size_t k = 0; k < plan.slots.size(); ++k) {
                const bool fits = sim.used + plan.slots[k] <= sim.cap_slots;
                ok[k] = fits;
                if (fits) {
                    sim.used += plan.slots[k];
                    sim.live[k] = plan.slots[k];
                }
            }
        };
        const auto pin = [&](int k, int s) {
            if (sim.used + s > sim.cap_slots) return false;
            sim.used += s;
            sim.live[(size_t) k] = s;
            return true;
        };
        const auto rel = [&](int k, int s) {
            sim.used -= s;
            sim.live[(size_t) k] -= s;
            sim.released.push_back(k);
        };
        return gt::seg_pin(plan, least, first, pin, rel, pinned);
    };
    const gt::SegPlan q = gt::seg_plan(1000, 1, 300);   // 300 / 300 / 300 / 100
    std::vector<int> pinned;
    Sim sim{0};
    int64_t t = run(q, 100000, 16, pinned, sim);
    check(t == 1000 && pinned == std::vector<int>({300, 300, 300, 100}), "all fit: planned sizes");
    t = run(q, 750, 16, pinned, sim);
    // 300 + 300 fit, the third (300) does not, the last (100) fits in the first round (650 <= 750) and is released;
    // the class shrinks 31/32 from 900: 871 -> 843 -> 816 -> 790 -> 691 (7/8) -> 600 + 91 fits
    bool prefix = pinned.size() == 3 && pinned[0] == 300 && pinned[1] == 300 && pinned[2] < 300;
    check(prefix, "refusal: a prefix of whole segments plus one shrunk last");
    check(t == 600 + pinned.back() && t <= 750 && sim.used == t, "the pinned total matches what is held");
    check(sim.live[3] == 0 && std::count(sim.released.begin(), sim.released.end(), 3) == 1,
          "a later segment that pinned in the first round is released");
    int64_t expect = 900;
    for (int tries = 0;; ++tries) {
        expect = gt::shrink_step(expect, tries);
        if (expect <= 750) break;
    }
    check(t == expect, "the last segment shrinks by Maya's steps on the class total (" + std::to_string(expect) + ")");
    t = run(q, 10, 16, pinned, sim);
    check(t == -1 && pinned.empty() && sim.used == 0, "below least: -1, everything released");
    t = run(q, 610, 16, pinned, sim);
    check(t > 0 && t <= 610 && sim.used == t, "a cap just over two segments");

    const std::vector<int> nodes = {0, 1};
    check(gt::seg_node(0, 0, false, nodes) == 0 && gt::seg_node(1, 0, false, nodes) == 1 &&
              gt::seg_node(2, 0, false, nodes) == 0,
          "interleaved class: segments round-robin over the nodes");
    check(gt::seg_node(5, 0, true, nodes) == 0 && gt::seg_node(5, 1, true, nodes) == 1, "split class: half h on node h");
    check(gt::seg_node(0, 1, true, {3}) == 3, "one node: everything there");
    check(gt::budget_reset_due(0) && !gt::budget_reset_due(1) && !gt::budget_reset_due(2),
          "the budget statics start over at part 0 of every load");
}

// ---------------------------------------------------------------- 2. THP

void test_thp() {
    std::printf("B8.7 huge pages\n");
    check(gt::thp_mode(nullptr) == -1 && gt::thp_mode("auto") == -1, "STRATA_GLM_THP unset / auto");
    check(gt::thp_mode("0") == 0 && gt::thp_mode("1") == 1, "STRATA_GLM_THP=0 / 1");
    const std::string buddy =
        "Node 0, zone      DMA      1      1      1      0      2      1      1      0      1      1      3 \n"
        "Node 0, zone    DMA32      5      6      4      5      4      3      3      4      3      3    300 \n"
        "Node 0, zone   Normal   1000    800    600    400    200    100     50     25     12   1000   5000 \n"
        "Node 1, zone   Normal    100    100    100    100    100    100    100    100    100   2000   4000 \n";
    // order 9 = 2 MB, order 10 = 4 MB
    const int64_t want = (1 + 3 * 2 + 3 + 300 * 2 + 1000 + 5000 * 2 + 2000 + 4000 * 2) * ((int64_t) 2 << 20);
    check(gt::buddy_huge_free(buddy) == want, "buddyinfo: orders >= 9 summed in bytes");
    check(gt::buddy_huge_free("garbage") == -1, "buddyinfo unparsable: -1");
    const std::vector<size_t> segs = {16 * GiB, 16 * GiB, 4 * GiB};
    check(gt::thp_plan(segs, 0, 0) == std::vector<char>({0, 0, 0}), "THP off: no madvise");
    check(gt::thp_plan(segs, 0, 1) == std::vector<char>({1, 1, 1}), "THP on: every segment");
    check(gt::thp_plan(segs, -1, -1) == std::vector<char>({1, 1, 1}), "auto, buddyinfo unknown: madvise (Maya)");
    check(gt::thp_plan(segs, (int64_t) (34 * GiB), -1) == std::vector<char>({1, 1, 0}) &&
              gt::thp_plan(segs, (int64_t) (20 * GiB), -1) == std::vector<char>({1, 0, 0}),
          "auto: a segment takes huge pages only while free blocks cover it + 1 GiB");

    const std::string smaps =
        "7f0000000000-7f0400000000 rw-p 00000000 00:00 0 \n"
        "Size:           16777216 kB\n"
        "Rss:            16777216 kB\n"
        "AnonHugePages:  12582912 kB\n"
        "7f0400000000-7f0500000000 rw-p 00000000 00:00 0 \n"
        "Size:            4194304 kB\n"
        "Rss:             4194304 kB\n"
        "AnonHugePages:         0 kB\n"
        "7f1000000000-7f1000200000 rw-p 00000000 00:00 0      [heap]\n"
        "Size:               2048 kB\n"
        "Rss:                2048 kB\n"
        "AnonHugePages:      2048 kB\n";
    const auto s = gt::smaps_sum(smaps, {{(uintptr_t) 0x7f0000000000ull, (size_t) 0x500000000ull}});
    check(s.anon_huge == (int64_t) 12 * (int64_t) GiB && s.rss == (int64_t) 20 * (int64_t) GiB &&
              s.size == (int64_t) 20 * (int64_t) GiB,
          "smaps: AnonHugePages / Rss of the tier's mappings only");

    for (int threads : {1, 3, 4}) {
        for (bool huge : {false, true}) {
            const size_t bytes = (size_t) 37 * (2 << 20) + 12288;
            std::vector<int> hits(bytes / 4096 + 1, 0);
            bool piece_ok = true;
            for (int t = 0; t < threads; ++t)
                for (const auto& r : gt::fault_plan(bytes, huge, threads, t)) {
                    piece_ok = piece_ok && r.first < r.second && r.second <= bytes &&
                               (!huge || threads == 1 || (r.first % (2 << 20) == 0 && r.second - r.first <= (2 << 20)));
                    for (size_t o = r.first; o < r.second; o += 4096) ++hits[o / 4096];
                }
            const bool once = std::all_of(hits.begin(), hits.begin() + (long) ((bytes + 4095) / 4096), [](int h) { return h == 1; });
            check(once && piece_ok, "fault plan: every page once, " + std::to_string(threads) + " thread(s), huge " +
                                        std::to_string((int) huge));
        }
    }
    check(gt::fault_plan(4096 * 5, false, 4, 1).empty(), "no huge pages: one thread faults the segment");
}

// ---------------------------------------------------------------- 3. pool modes

void test_pools() {
    std::printf("B8.3 lane pool modes\n");
    using gt::LanePool;
    check(gt::lane_pool_mode(1, false, nullptr, false, 2) == LanePool::One, "one part: one pool");
    check(gt::lane_pool_mode(2, false, nullptr, false, 2) == LanePool::PerPart, "2 parts, unset: per part (Maya)");
    check(gt::lane_pool_mode(3, false, nullptr, false, 2) == LanePool::Shared, "3 parts, unset: shared (Maya)");
    check(gt::lane_pool_mode(3, false, "0", false, 2) == LanePool::PerPart, "STRATA_GLM_CPU_SHARED=0: per part");
    check(gt::lane_pool_mode(2, false, "1", false, 2) == LanePool::Shared, "STRATA_GLM_CPU_SHARED=1: shared");
    check(gt::lane_pool_mode(3, true, "1", false, 2) == LanePool::PerPart, "STRATA_GLM_CPU_LANE<dev>: its own pool");
    check(gt::lane_pool_mode(3, false, "socket", false, 2) == LanePool::Socket, "socket: a pinned pool per socket");
    check(gt::lane_pool_mode(3, false, "2", false, 2) == LanePool::Socket, "2: a pinned pool per socket");
    check(gt::lane_pool_mode(3, false, "socket", false, 1) == LanePool::Shared, "socket on one node: shared");
    check(gt::lane_pool_mode(3, false, nullptr, true, 2) == LanePool::NodeSplit, "a node-split tier: NodeSplit");
    check(gt::lane_pool_mode(1, false, "0", true, 2) == LanePool::NodeSplit, "node-split on one GPU too");
    check(gt::lane_pool_mode(3, false, nullptr, true, 1) == LanePool::Shared, "node-split needs two nodes");
    // review fix: Flash keeps Maya's reading of STRATA_GLM_CPU_SHARED ("2" any non-zero: shared; "socket": atoi 0)
    check(gt::lane_pool_mode(3, false, "2", false, 2, false) == LanePool::Shared &&
              gt::lane_pool_mode(3, false, "socket", false, 2, false) == LanePool::PerPart &&
              gt::lane_pool_mode(3, false, "1", false, 2, false) == LanePool::Shared,
          "Flash: '2' one shared pool, 'socket' a pool per part (Maya)");
    // review fix: a node's pinned pool is a thread per physical core less 2, not per logical CPU
    check(gt::numa_lane_threads(40, 20) == 18, "test-PC node (20C/40T): 18 lane threads, not 36");
    check(gt::numa_lane_threads(40, 0) == 36 && gt::numa_lane_threads(3, 2) == 2 && gt::numa_lane_threads(256, 128) == 64,
          "topology unknown: logical - 4; within [2, 64]");
    check(gt::lane_pool_maya_shared(LanePool::Shared) && !gt::lane_pool_maya_shared(LanePool::Socket),
          "only Shared sets Maya's shared flag");
    check(std::string(gt::lane_pool_name(LanePool::Socket)).find("socket") != std::string::npos, "mode names");
    check(!gt::numa_split_requested(false, "1"), "STRATA_GLM_NUMA_SPLIT never on glm5-next");
    check(gt::numa_split_requested(true, "1") && !gt::numa_split_requested(true, "0") &&
              !gt::numa_split_requested(true, nullptr),
          "STRATA_GLM_NUMA_SPLIT=1 on glm-dsa only (default off)");
}

// ---------------------------------------------------------------- 4. split layout

bool bijection(const gt::SplitLayout& L) {
    std::vector<char> seen[2] = {std::vector<char>(L.half_bytes, 0), std::vector<char>(L.half_bytes, 0)};
    for (size_t u = 0; u < L.blob; ++u) {
        const gt::SplitLoc at = gt::split_loc(L, u);
        if (at.off >= L.half_bytes || seen[at.half][at.off]) return false;
        seen[at.half][at.off] = 1;
        if (gt::unsplit_off(L, at.half, at.off) != u) return false;
    }
    for (int h = 0; h < 2; ++h)
        for (char c : seen[h])
            if (!c) return false;
    return true;
}

bool layout_paths_agree(const gt::SplitLayout& L, std::mt19937& rng) {
    std::vector<uint8_t> u(L.blob), h0(L.half_bytes), h1(L.half_bytes), r0(L.half_bytes), r1(L.half_bytes);
    for (auto& b : u) b = (uint8_t) rng();
    // the per-byte map
    for (size_t x = 0; x < L.blob; ++x) {
        const gt::SplitLoc at = gt::split_loc(L, x);
        (at.half ? r1 : r0)[at.off] = u[x];
    }
    gt::scatter(L, u.data(), h0.data(), h1.data());
    if (h0 != r0 || h1 != r1) return false;
    // the copy ops cover every byte once
    std::vector<int> cover(L.blob, 0);
    for (const auto& op : gt::split_ops(L))
        for (size_t r = 0; r < op.rows; ++r)
            for (size_t c = 0; c < op.width; ++c) ++cover[op.u_off + r * op.u_pitch + c];
    if (!std::all_of(cover.begin(), cover.end(), [](int c) { return c == 1; })) return false;
    std::vector<uint8_t> back(L.blob, 0);
    gt::gather(L, h0.data(), h1.data(), back.data());
    if (back != u) return false;
    // chunked disk reads (random chunk boundaries, Maya's 4 KB pieces too)
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<uint8_t> c0(L.half_bytes, 0xee), c1(L.half_bytes, 0xee);
        size_t at = 0;
        while (at < L.blob) {
            const size_t len = pass == 0 ? 4096 : 1 + rng() % (pass == 1 ? 97 : 20000);
            const size_t n = std::min(len, L.blob - at);
            gt::scatter_range(L, at, u.data() + at, n, c0.data(), c1.data());
            at += n;
        }
        if (c0 != r0 || c1 != r1) return false;
    }
    return true;
}

void test_split() {
    std::printf("B8.4 node-split layout\n");
    // ggml block sizes: IQ2_XXS 66 B / 256, IQ2_S 82, IQ3_XXS 98, Q2_K 84, IQ4_XS 136, Q4_K 144
    struct Ty {
        const char* name;
        size_t bb;
    };
    const Ty gu[] = {{"IQ2_XXS", 66}, {"IQ2_S", 82}, {"IQ3_XXS", 98}};
    const Ty dn[] = {{"IQ3_XXS", 98}, {"Q2_K", 84}, {"IQ4_XS", 136}, {"Q4_K", 144}};
    for (const auto& a : gu)
        for (const auto& d : dn) {
            gt::SplitLayout L;
            std::string why;
            const size_t gu_row = 6144 / 256 * a.bb, d_row = 2048 / 256 * d.bb;
            const bool ok = gt::split_layout(gu_row, d_row, 2048, 6144, 256, d.bb, L, why);
            const bool sizes = ok && L.H == 1024 && L.hd_row == d_row / 2 && L.blob == 2 * 2048 * gu_row + 6144 * d_row &&
                               2 * L.half_bytes == L.blob && L.h_gu == 1024 * gu_row;
            bool samples = ok;
            std::mt19937_64 rng(7);
            for (int i = 0; i < 20000 && samples; ++i) {
                const size_t u = rng() % L.blob;
                const gt::SplitLoc at = gt::split_loc(L, u);
                samples = at.off < L.half_bytes && gt::unsplit_off(L, at.half, at.off) == u;
            }
            // gate row 1023 -> half 0, row 1024 -> half 1 row 0; down column 1023's block -> half 0, 1024's -> half 1
            const bool edges = ok && gt::split_loc(L, 1023 * gu_row).half == 0 &&
                               gt::split_loc(L, 1024 * gu_row).half == 1 && gt::split_loc(L, 1024 * gu_row).off == 0 &&
                               gt::split_loc(L, L.gu_bytes + 1024 * gu_row).off == L.h_gu &&
                               gt::split_loc(L, L.down_off + 4 * d.bb - 1).half == 0 &&
                               gt::split_loc(L, L.down_off + 4 * d.bb).half == 1 &&
                               gt::split_loc(L, L.down_off + d_row).off == L.h_down_off + L.hd_row;
            check(ok && sizes && samples && edges,
                  std::string("GLM-5.3 ") + a.name + " / " + d.name + ": halves of " +
                      std::to_string(L.half_bytes) + " B, word " + std::to_string(ok ? gt::split_word(L) : 0) +
                      (ok ? "" : " (" + why + ")"));
        }
    gt::SplitLayout L;
    std::string why;
    check(!gt::split_layout(1584, 784, 2047, 6144, 256, 98, L, why), "odd n_ff refused (" + why + ")");
    check(!gt::split_layout(1584, 98 * 3, 768, 6144, 256, 98, L, why), "n_ff / 2 not whole blocks refused (" + why + ")");
    check(!gt::split_layout(1584, 700, 2048, 6144, 256, 98, L, why), "down row not whole blocks refused (" + why + ")");
    check(!gt::split_layout(1582, 784, 2048, 6144, 256, 98, L, why), "gate/up row not 4-byte refused (" + why + ")");
    check(!gt::split_layout(1584, 2 * 98, 512, 6144, 256, 98, L, why) || L.hd_row % 4 == 0,
          "down half row not 4-byte refused");

    std::mt19937 rng(1234);
    // small layouts: every byte checked
    struct Small {
        size_t gu_row;
        int64_t n_ff, n_embd, blk;
        size_t bb;
    };
    const Small sm[] = {{16, 512, 6, 256, 8}, {12, 1024, 5, 256, 98 * 0 + 12}, {24, 2048, 3, 256, 84}, {8, 64, 7, 32, 4}};
    for (const auto& s : sm) {
        gt::SplitLayout l;
        const size_t d_row = (size_t) (s.n_ff / s.blk) * s.bb;
        const bool ok = gt::split_layout(s.gu_row, d_row, s.n_ff, s.n_embd, s.blk, s.bb, l, why);
        check(ok && bijection(l) && layout_paths_agree(l, rng),
              "layout " + std::to_string(s.gu_row) + " B rows, n_ff " + std::to_string(s.n_ff) +
                  ": bijection; scatter = gather^-1 = per-byte map = chunked reads; ops cover once");
    }
    {
        gt::SplitLayout l;
        gt::split_layout(6144 / 256 * 66, 2048 / 256 * 98, 2048, 6144, 256, 98, l, why);
        check(layout_paths_agree(l, rng), "GLM-5.3 IQ2_XXS / IQ3_XXS: scatter, gather, ops, chunked reads agree");
        gt::SplitLayout l2;
        gt::split_layout(6144 / 256 * 82, 2048 / 256 * 84, 2048, 6144, 256, 84, l2, why);
        const size_t hs = gt::half_stride({l, l2});
        check(hs % 4096 == 0 && hs >= std::max(l.half_bytes, l2.half_bytes) && hs < std::max(l.half_bytes, l2.half_bytes) + 4096,
              "half stride: the largest half, 4 KB rounded");
        check(gt::split_word(l) == 8 && gt::split_word(l2) == 16, "device gather word: 8 B (392 B down halves), 16 B (336 B)");
    }
    std::vector<float> y0(6144), y1(6144), y(6144);
    for (int i = 0; i < 6144; ++i) y0[(size_t) i] = 0.5f * (float) i, y1[(size_t) i] = -0.25f * (float) i + 1.0f;
    gt::add_partial(y0.data(), y1.data(), y.data(), 6144);
    bool add_ok = true;
    for (int i = 0; i < 6144; ++i) add_ok = add_ok && y[(size_t) i] == y0[(size_t) i] + y1[(size_t) i];
    check(add_ok, "the partial sums' add: one 6144-float add");
}

}  // namespace

// ---------------------------------------------------------------- 5. W6 runtime fixes

void test_w6() {
    std::printf("W6 runtime fix (B8.2): bound classes capped by each node's free memory\n");
    const std::string mi =
        "Node 0 MemTotal:       98304000 kB\n"
        "Node 0 MemFree:        10485760 kB\n"
        "Node 0 MemUsed:        87818240 kB\n"
        "Node 0 Active(file):    4194304 kB\n"
        "Node 0 Inactive(file):  2097152 kB\n"
        "Node 0 SReclaimable:    1048576 kB\n"
        "Node 0 SUnreclaim:       524288 kB\n";
    check(gt::node_mem_avail(mi) == (int64_t) (10 + 2 + 1) * (int64_t) GiB, "node meminfo: MemFree + Inactive(file) + SReclaimable");
    check(gt::node_mem_avail("garbage") == -1, "node meminfo unparsable: -1 (no limit)");
    check(gt::node_reserve_bytes(nullptr) == (int64_t) (4 * GiB) && gt::node_reserve_bytes("0") == 0 &&
              gt::node_reserve_bytes("2.5") == (int64_t) (5 * GiB / 2),
          "STRATA_GLM_NODE_RESERVE_GB: default 4 GiB, 0, 2.5");
    const std::vector<int> nodes = {0, 1};
    const size_t MiB = (size_t) 1 << 20, st = 10 * MiB;
    // the review's case: node split, every part's half 0 on node 0, node 0 fuller than the machine-wide figure says
    check(gt::node_fit_slots(1000, st, 16 * GiB, true, nodes, {-1, -1}) == 1000, "split, budgets unknown: n unchanged");
    check(gt::node_fit_slots(1000, st, 16 * GiB, true, nodes, {(int64_t) (20000 * MiB), (int64_t) (20000 * MiB)}) == 1000,
          "split, both nodes hold n x stride: n unchanged");
    check(gt::node_fit_slots(1000, st, 16 * GiB, true, nodes, {(int64_t) (5000 * MiB), (int64_t) (20000 * MiB)}) == 500,
          "split, node 0 holds 5000 MiB: 500 slots (each half needs n x stride on its node)");
    check(gt::node_fit_slots(1000, st, 16 * GiB, true, nodes, {(int64_t) (20000 * MiB), 0}) == 0, "split, node 1 full: 0");
    // NUMA_SEG=bind: 1 GiB slots, 16 a segment, segments round-robin 0, 1, 0, 1: node 0 holds 20 GiB
    {
        const int64_t f = gt::node_fit_slots(64, GiB, 16 * GiB, false, nodes, {(int64_t) (20 * GiB), -1});
        // n = 36: segments 16 (n0), 16 (n1), 4 (n0) -> node 0 20 GiB; n = 37 -> 21 GiB
        check(f == 36, "bind_seg, node 0 holds 20 GiB of round-robin 16 GiB segments: 36 of 64 slots (got " +
                           std::to_string(f) + ")");
        check(gt::node_fit_slots(64, GiB, 16 * GiB, false, nodes, {(int64_t) (40 * GiB), (int64_t) (40 * GiB)}) == 64,
              "bind_seg, both nodes hold their 32 GiB: n unchanged");
    }
    check(gt::node_fit_action(100, 100, 16) == 0, "fit >= n: keep");
    check(gt::node_fit_action(100, 80, 16) == 1, "fit 80 of 100 (>= 3/4, >= least): shrink");
    check(gt::node_fit_action(100, 60, 16) == 2, "fit 60 of 100 (< 3/4): interleave instead");
    check(gt::node_fit_action(20, 15, 16) == 2, "fit below least: interleave instead");

    std::printf("W6 runtime fix (B8.7): per-node huge-page budgets for bound units\n");
    const std::string buddy =
        "Node 0, zone      DMA      1      1      1      0      2      1      1      0      1      1      3 \n"
        "Node 0, zone   Normal   1000    800    600    400    200    100     50     25     12   10      20 \n"
        "Node 1, zone   Normal    100    100    100    100    100    100    100    100    100   2000   4000 \n";
    const int64_t M2 = (int64_t) 2 << 20;
    const std::vector<int64_t> per = gt::buddy_huge_free_nodes(buddy, {0, 1, 2});
    check(per.size() == 3 && per[0] == (1 + 3 * 2 + 10 + 20 * 2) * M2 && per[1] == (2000 + 4000 * 2) * M2 && per[2] == -1,
          "buddyinfo per node: node 0, node 1, an absent node -1");
    check(per[0] + per[1] == gt::buddy_huge_free(buddy), "the per-node budgets add up to the machine-wide sum");
    // node 0 has ~114 MB of free 2 MB blocks, node 1 ~20 GB: split units alternate n0, n1
    const std::vector<size_t> units = {8 * GiB, 8 * GiB, 8 * GiB, 8 * GiB};
    const std::vector<int> un = {0, 1, 0, 1};
    const std::vector<int64_t> nf = {(int64_t) (100 * MiB), (int64_t) (20 * GiB)};
    const int64_t all = nf[0] + nf[1];
    check(gt::thp_plan(units, all, -1) == std::vector<char>({1, 1, 0, 0}),
          "(the machine-wide rule would madvise node 0's first unit)");
    check(gt::thp_plan_nodes(units, un, nf, all, -1) == std::vector<char>({0, 1, 0, 1}),
          "per node: node 0's units not madvised, node 1's two take its 20 GiB");
    check(gt::thp_plan_nodes(units, {-1, -1, -1, -1}, nf, all, -1) == gt::thp_plan(units, all, -1),
          "interleaved units (-1): Maya's machine-wide rule unchanged");
    check(gt::thp_plan_nodes(units, un, nf, all, 1) == std::vector<char>({1, 1, 1, 1}) &&
              gt::thp_plan_nodes(units, un, nf, all, 0) == std::vector<char>({0, 0, 0, 0}),
          "STRATA_GLM_THP=1 / 0 unchanged");
    check(gt::thp_plan_nodes(units, un, {-1, -1}, all, -1) == std::vector<char>({1, 1, 1, 1}),
          "a node's buddyinfo unknown: madvise (Maya's rule)");

    std::printf("W6 runtime fix (B8.4): the cpu_threads knob scaled to each node-split pool\n");
    check(gt::lane_threads_scaled(8, 36, 36) == 8, "equal pools: the same count");
    check(gt::lane_threads_scaled(8, 36, 18) == 4, "half the pool: half the count");
    check(gt::lane_threads_scaled(0, 36, 30) == 30, "0: every thread");
    check(gt::lane_threads_scaled(100, 36, 36) == 36 && gt::lane_threads_scaled(1, 36, 10) == 1,
          "clamped to [1, size]");
}

int main() {
    test_segments();
    test_w6();
    test_thp();
    test_pools();
    test_split();
    if (g_fail) {
        std::printf("glm_tier_plan_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_tier_plan_test: all passed\n");
    return 0;
}
