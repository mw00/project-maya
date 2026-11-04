// tests/ds4_place_test.cpp - the trunk's expert placement (src/ds4/ds4_place.hpp): the per-byte queue, the even
// split and each layer's statics / recent slots / spare, today's rule and DS4_RECENT_FIT's (no GPU, no model).
//
//   ds4_place_test [expert-profile.bin] [--table]
//
// 1. Today's rule against a real load: the shipped profile (data/expert-profile-ds4.bin, 40 x 384 counts) on
//    turboderp's DeepSeek-V4.1-Flash EXL3 2.25bpw pack - an expert 11,103,744 bytes in the 2.5-bit layers (0-5,
//    35-39) and 8,891,904 in the 2-bit ones (3 x the trellis k n K / 8 + the suh / svh F16 vectors, each piece
//    256-aligned: the pack's tensors.txt) - and the RTX 3090 box's budget, 15.76 GB a member for experts.  Its
//    --verbose DS4_PLACE_DEBUG=1 load printed the slot vector below, 1697 slots, and "697 of 15360 in VRAM"; the load
//    without a profile (the even split) "336 of 15360", experts 15.76 GB.  Both reproduced exactly.
// 2. The functions at fit 0 against a transcription of the engine's own code at 3589d6f (place_experts,
//    ds4_model.cu:6803-6843 and 7077-7096) on random profiles, units and budgets, ties included: the same slots and
//    the same split, so the test's real-load check covers the engine's rule, not only this header's.
// 3. DS4_RECENT_FIT=1 and 2 on the same load (the audit's simulated vectors), and their invariants over DS4_RECENT
//    0 / 16 / 24 / 32 / 40 / 48, budgets 6-24 GB, the profile and the even split: never over the budget, every layer
//    with 3 or more slots keeps a recent slot, a static and the spare; fit 1 leaves the slots alone; fit 2 keeps the
//    floor or says it does not fit.
#include "../src/ds4/ds4_place.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace place = strata::ds4::place;

static int g_fail = 0;
static long long g_checks = 0;

static void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// ---- the engine's code at 3589d6f, transcribed (the Layer fields it reads; ucount(l) = the layer's unit)
struct ELayer {
    int il = 0, n_exp = 0, grp = 0, n_vram = 0;
    bool mtp = false;
    int n_cls = 1;
    double unit = 0;
};

// ds4_model.cu:6803-6823 (the queue, a profile seen and no DS4_PLACE_UNIFORM)
static void engine_queue(std::vector<ELayer>& L, const std::vector<uint32_t>& prof, int n_exp_g, double budget,
                         size_t gi) {
    auto ucount = [](const ELayer& l) { return l.unit; };
    std::vector<std::vector<double>> sh(L.size());
    using Top = std::pair<double, int>;
    std::priority_queue<Top> pq;
    for (ELayer& l : L) {
        if (l.grp != (int) gi || l.mtp || l.n_cls > 1) continue;
        l.n_vram = 0;
        const uint32_t* p = prof.data() + (size_t) l.il * n_exp_g;
        double sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += p[e];
        std::vector<double>& s = sh[(size_t) l.il];
        for (int e = 0; e < l.n_exp; ++e)
            s.push_back(0.9 * (sum > 0 ? p[e] / sum : 1.0 / l.n_exp) + 0.1 / l.n_exp);
        std::sort(s.begin(), s.end(), std::greater<double>());
        pq.push({s[0] / ucount(l), l.il});
    }
    double used = 0;
    while (!pq.empty()) {
        const int il = pq.top().second;
        pq.pop();
        ELayer& l = L[(size_t) il];
        if (used + ucount(l) > budget) continue;
        used += ucount(l);
        if (++l.n_vram < l.n_exp) pq.push({sh[(size_t) il][(size_t) l.n_vram] / ucount(l), il});
    }
}

// ds4_model.cu:6790-6795 and 6824-6843 (want; the even split)
static void engine_even(std::vector<ELayer>& L, double budget, size_t gi) {
    auto ucount = [](const ELayer& l) { return l.unit; };
    double want = 0;
    for (const ELayer& l : L)
        if (l.grp == (int) gi && !l.mtp && l.n_cls == 1) want += (double) ucount(l) * l.n_exp;
    const double frac = want > 0 ? std::min(1.0, budget / want) : 0.0;
    for (ELayer& l : L) {
        if (l.grp != (int) gi || l.mtp || l.n_cls > 1) continue;
        l.n_vram = std::min(l.n_exp, (int) std::floor(frac * l.n_exp));
        while (l.n_vram > 0 && (double) ucount(l) * l.n_vram > budget) --l.n_vram;
    }
    double used = 0;
    for (ELayer& l : L)
        if (l.grp == (int) gi && !l.mtp && l.n_cls == 1) used += (double) ucount(l) * l.n_vram;
    for (ELayer& l : L) {
        if (l.grp != (int) gi || l.mtp || l.n_cls > 1) continue;
        while (l.n_vram < l.n_exp && used + (double) ucount(l) <= budget) {
            ++l.n_vram;
            used += (double) ucount(l);
        }
    }
}

// ds4_model.cu:7077-7096 (a layer's split; eligible = !l.mtp && (!file_experts || (pk.gguf && ln_land)))
static place::Split engine_split(int n_vram, int n_exp, bool cache, int recent_n, bool eligible) {
    place::Split s;
    s.slots = std::min(n_vram, n_exp);
    if (cache && s.slots >= 2 && s.slots < n_exp) {
        const int r = recent_n > 0 && eligible && s.slots - 1 - recent_n >= 1 ? recent_n : 0;
        s.statics = s.slots - 1 - r;
        s.recent = r;
        s.spare = 1;
    } else {
        s.statics = s.slots;
        s.recent = 0;
        s.spare = 0;
    }
    return s;
}

// ---- the box
constexpr int kNL = 40, kNE = 384;
constexpr double kU25 = 11103744.0, kU2 = 8891904.0, kBudget = 15.76e9;
static bool k25(int il) { return il <= 5 || il >= 35; }
static const int kMeasured[kNL] = {18, 24, 28, 25, 29, 27, 51, 47, 51, 42, 48, 52, 43, 43, 41, 40, 54, 42, 56, 45,
                                   49, 49, 47, 43, 53, 55, 54, 45, 50, 50, 52, 48, 49, 50, 48, 27, 30, 34, 27, 31};
// DS4_RECENT_FIT=2 on the same load, from a separate Python re-implementation of the rule (the audit's simulator with
// the exact bytes above): 1670 slots, 350 statics
static const int kFit2[kNL] = {35, 35, 36, 39, 37, 39, 44, 40, 44, 45, 43, 42, 42, 42, 46, 40, 42, 41, 44, 36,
                               46, 48, 44, 45, 44, 46, 45, 43, 45, 47, 43, 43, 45, 42, 43, 38, 40, 38, 38, 35};

static std::vector<place::QLayer> box_layers(const std::vector<uint32_t>& prof) {
    std::vector<place::QLayer> q;
    for (int il = 0; il < kNL; ++il)
        q.push_back({il, kNE, k25(il) ? kU25 : kU2, place::shares(prof.data() + (size_t) il * kNE, kNE)});
    return q;
}

static double bytes_of(const std::vector<place::QLayer>& q, const std::vector<int>& slots) {
    double b = 0;
    for (size_t i = 0; i < q.size(); ++i) b += q[i].unit * slots[i];
    return b;
}

static std::vector<place::Split> splits(const std::vector<place::QLayer>& q, const std::vector<int>& slots,
                                        int recent_n, int fit, bool cache = true, bool eligible = true) {
    std::vector<place::Split> v;
    for (size_t i = 0; i < q.size(); ++i) v.push_back(place::split(slots[i], q[i].n_exp, cache, recent_n, eligible, fit));
    return v;
}

static int statics_of(const std::vector<place::Split>& v) {
    int n = 0;
    for (const place::Split& s : v) n += s.statics;
    return n;
}

static void print_table(const std::vector<place::Split>& a, const std::vector<place::Split>& b,
                        const std::vector<place::Split>& c) {
    std::printf("  layer  fit0 slots/statics/recent  fit1  fit2\n");
    for (size_t i = 0; i < a.size(); ++i)
        std::printf("  %5zu  %3d %3d %3d   %3d %3d %3d   %3d %3d %3d\n", i, a[i].slots, a[i].statics, a[i].recent,
                    b[i].slots, b[i].statics, b[i].recent, c[i].slots, c[i].statics, c[i].recent);
}

// 1 + 3: the real load
static void test_box(const std::vector<uint32_t>& prof, bool table) {
    const std::vector<place::QLayer> q = box_layers(prof);
    // today: the queue
    std::vector<int> s0;
    expect(place::queue(q, kBudget, 32, 0, s0), "box: fit 0 queue places");
    int sum = 0;
    bool same = s0.size() == (size_t) kNL;
    for (int il = 0; il < kNL && same; ++il) same = s0[(size_t) il] == kMeasured[il];
    for (int v : s0) sum += v;
    expect(same, "box: fit 0 slot vector = the measured DS4_PLACE_DEBUG vector");
    expect(sum == 1697, "box: 1697 slots (measured: 1697 of 15360)");
    const double b0 = bytes_of(q, s0);
    expect(b0 == 15753113088.0, "box: the slots' bytes 15,753,113,088 (measured: experts 15.75 GB)");
    const std::vector<place::Split> p0 = splits(q, s0, 32, 0);
    expect(statics_of(p0) == 697, "box: 697 statics (measured: 697 of 15360 in VRAM, vram_used=697)");
    for (int il = 0; il < kNL; ++il) {
        const bool cliff = il <= 5 || il == 35 || il == 36 || il == 38 || il == 39;
        expect((p0[(size_t) il].recent == 0) == cliff, "box: fit 0 layer " + std::to_string(il) +
                                                           (cliff ? " has no recent slots" : " has 32 recent slots"));
    }
    expect(p0[37].slots == 34 && p0[37].statics == 1 && p0[37].recent == 32, "box: fit 0 layer 37 = 34 slots, 1 static");
    // today: the even split, no profile
    std::vector<int> su;
    place::even(q, kBudget, su);
    bool even_ok = su.size() == (size_t) kNL && su[0] == 57;
    for (int il = 1; il < kNL && even_ok; ++il) even_ok = su[(size_t) il] == 41;
    expect(even_ok, "box: even split = 57, then 41 a layer");
    expect(statics_of(splits(q, su, 32, 0)) == 336, "box: even split 336 statics (measured: 336 of 15360)");
    expect(bytes_of(q, su) == 15757922304.0, "box: even split's bytes 15,757,922,304 (measured: experts 15.76 GB)");

    // fit 1: the slots as today, the cliff layers a partial recent set (the audit's simulated vector)
    std::vector<int> s1;
    expect(place::queue(q, kBudget, 32, 1, s1) && s1 == s0, "box: fit 1 leaves the queue's slots alone");
    const std::vector<place::Split> p1 = splits(q, s1, 32, 1);
    const int want1[10][4] = {{0, 18, 1, 16},  {1, 24, 1, 22},  {2, 28, 1, 26},  {3, 25, 1, 23},  {4, 29, 1, 27},
                              {5, 27, 1, 25},  {35, 27, 1, 25}, {36, 30, 1, 28}, {38, 27, 1, 25}, {39, 31, 1, 29}};
    for (const auto& w : want1) {
        const place::Split& s = p1[(size_t) w[0]];
        expect(s.slots == w[1] && s.statics == w[2] && s.recent == w[3] && s.spare == 1,
               "box: fit 1 layer " + std::to_string(w[0]) + " = (" + std::to_string(w[1]) + ", " +
                   std::to_string(w[2]) + ", " + std::to_string(w[3]) + ")");
    }
    for (int il = 0; il < kNL; ++il)
        if (p0[(size_t) il].recent > 0)
            expect(p1[(size_t) il].statics == p0[(size_t) il].statics && p1[(size_t) il].recent == 32,
                   "box: fit 1 layer " + std::to_string(il) + " as today");
    expect(statics_of(p1) == 451, "box: fit 1 451 statics (697 - 246)");
    int extra1 = 0;
    double extra1_b = 0;
    for (int il = 0; il < kNL; ++il) {
        extra1 += p0[(size_t) il].statics - p1[(size_t) il].statics;
        extra1_b += (p0[(size_t) il].statics - p1[(size_t) il].statics) * q[(size_t) il].unit;
    }
    expect(extra1 == 246 && std::fabs(extra1_b - 2731521024.0) < 1, "box: fit 1 246 more RAM copies, 2.73 GB");

    // fit 2: the floor and the split's own prices
    std::vector<int> s2;
    expect(place::queue(q, kBudget, 32, 2, s2), "box: fit 2 floor fits 15.76 GB");
    const std::vector<place::Split> p2 = splits(q, s2, 32, 2);
    int lo = 1 << 30, hi = 0, sum2 = 0;
    for (int il = 0; il < kNL; ++il) {
        lo = std::min(lo, s2[(size_t) il]);
        hi = std::max(hi, s2[(size_t) il]);
        sum2 += s2[(size_t) il];
        expect(p2[(size_t) il].recent == 32 && p2[(size_t) il].statics >= 1,
               "box: fit 2 layer " + std::to_string(il) + " keeps 32 recent slots and a static");
    }
    expect(lo == 35 && hi == 48, "box: fit 2 35-48 slots a layer (the audit: 35-48)");
    bool same2 = s2.size() == (size_t) kNL;
    for (int il = 0; il < kNL && same2; ++il) same2 = s2[(size_t) il] == kFit2[il];
    expect(same2, "box: fit 2 slot vector = the independent re-implementation's");
    expect(statics_of(p2) == 350, "box: fit 2 350 statics (the audit: 350)");
    expect(bytes_of(q, s2) <= kBudget, "box: fit 2 within the budget");
    expect(kBudget - bytes_of(q, s2) < kU2, "box: fit 2 leaves less than an expert of the budget");
    int extra2 = 0;
    for (int il = 0; il < kNL; ++il) extra2 += p0[(size_t) il].statics - p2[(size_t) il].statics;
    expect(extra2 == 347, "box: fit 2 347 more RAM copies");
    std::printf("box (15.76 GB, DS4_RECENT 32): slots %d / %d / %d, statics %d / %d / %d (fit 0 / 1 / 2); fit 2 slots "
                "%d-%d; even split %d statics\n",
                sum, sum, sum2, statics_of(p0), statics_of(p1), statics_of(p2), lo, hi, statics_of(splits(q, su, 32, 0)));
    if (table) print_table(p0, p1, p2);
}

// 2: the header at fit 0 against the engine's code, random cases
static void test_engine_parity() {
    std::mt19937_64 rng(0x5eed1234ULL);
    int cases = 0, queue_diff = 0, even_diff = 0;
    for (int c = 0; c < 400; ++c) {
        const int nl = 1 + (int) (rng() % 12), ne = 2 + (int) (rng() % 40);
        std::vector<uint32_t> prof((size_t) nl * ne, 0u);
        const int mode = (int) (rng() % 4);   // 0: random, 1: sparse, 2: all zero (ties), 3: equal counts (ties)
        for (uint32_t& v : prof)
            v = mode == 0 ? (uint32_t) (rng() % 1000) : mode == 1 ? (rng() % 5 == 0 ? (uint32_t) (rng() % 50) : 0u)
                : mode == 2 ? 0u : 7u;
        if (mode == 1 && rng() % 2) std::fill(prof.begin(), prof.begin() + ne, 0u);   // (a layer the profile never saw)
        std::vector<ELayer> L((size_t) nl);
        std::vector<place::QLayer> q;
        const double units[3] = {256.0 * 37, 256.0 * 45, 256.0 * 37};
        for (int il = 0; il < nl; ++il) {
            ELayer& l = L[(size_t) il];
            l.il = il;
            l.n_exp = ne;
            l.unit = units[rng() % 3];
            l.mtp = rng() % 10 == 0;   // (not the queue's)
            l.n_cls = rng() % 12 == 0 ? 2 : 1;
            if (!l.mtp && l.n_cls == 1) q.push_back({il, ne, l.unit, place::shares(prof.data() + (size_t) il * ne, ne)});
        }
        double all = 0;
        for (const place::QLayer& x : q) all += x.unit * x.n_exp;
        const double budget = all * (double) (rng() % 1100) / 1000.0 + (double) (rng() % 9000);
        std::vector<ELayer> Lq = L, Le = L;
        engine_queue(Lq, prof, ne, budget, 0);
        engine_even(Le, budget, 0);
        std::vector<int> sq, se;
        place::queue(q, budget, (int) (rng() % 8), 0, sq);
        place::even(q, budget, se);
        bool dq = sq.size() != q.size(), de = se.size() != q.size();
        for (size_t i = 0; i < q.size() && !dq; ++i) dq = sq[i] != Lq[(size_t) q[i].il].n_vram;
        for (size_t i = 0; i < q.size() && !de; ++i) de = se[i] != Le[(size_t) q[i].il].n_vram;
        queue_diff += dq;
        even_diff += de;
        ++cases;
    }
    expect(queue_diff == 0, "parity: the queue at fit 0 = the engine's (" + std::to_string(queue_diff) + " of " +
                                std::to_string(cases) + " differ)");
    expect(even_diff == 0, "parity: the even split = the engine's (" + std::to_string(even_diff) + " of " +
                               std::to_string(cases) + " differ)");
    // the split, every case in range
    int split_diff = 0, split_cases = 0;
    for (int ne = 1; ne <= 70; ++ne)
        for (int nv = 0; nv <= 80; ++nv)
            for (int rn = 0; rn <= 66; rn += (rn < 40 ? 1 : 13))
                for (int cache = 0; cache < 2; ++cache)
                    for (int el = 0; el < 2; ++el) {
                        const place::Split a = place::split(nv, ne, cache, rn, el, 0);
                        const place::Split b = engine_split(nv, ne, cache, rn, el);
                        split_diff += a.slots != b.slots || a.statics != b.statics || a.recent != b.recent ||
                                      a.spare != b.spare;
                        ++split_cases;
                    }
    expect(split_diff == 0, "parity: the split at fit 0 = the engine's (" + std::to_string(split_diff) + " of " +
                                std::to_string(split_cases) + " differ)");
    std::printf("parity with the engine's code at 3589d6f: %d queue and even-split cases, %d split cases, %d differ\n",
                cases, split_cases, queue_diff + even_diff + split_diff);
}

// 3: the invariants over DS4_RECENT and budgets, the profile and the even split
static void test_sweep(const std::vector<uint32_t>& prof) {
    const std::vector<place::QLayer> q = box_layers(prof);
    double floor_min = 0;
    int runs = 0, fit2_fell = 0;
    for (int rn : {0, 16, 24, 32, 40, 48})
        for (double gb = 6.0; gb <= 24.0 + 1e-9; gb += 0.5) {
            const double budget = gb * 1e9;
            for (int src = 0; src < 2; ++src) {   // 0: the queue (a profile), 1: the even split
                std::vector<int> s0, s1, s2;
                bool q2 = false;
                if (src == 0) {
                    place::queue(q, budget, rn, 0, s0);
                    place::queue(q, budget, rn, 1, s1);
                    q2 = place::queue(q, budget, rn, 2, s2);
                } else {
                    place::even(q, budget, s0);
                    s1 = s0;
                }
                floor_min = 0;
                for (const place::QLayer& x : q) floor_min += x.unit * std::min(x.n_exp, rn + 2);
                const std::string at = "rn " + std::to_string(rn) + ", " + std::to_string(gb) + " GB, " +
                                       (src == 0 ? "queue" : "even");
                expect(bytes_of(q, s0) <= budget, at + ": fit 0 within the budget");
                expect(s1 == s0, at + ": fit 1 slots = fit 0's");
                if (src == 0) {
                    expect(q2 == (floor_min <= budget), at + ": fit 2 places iff its floor fits");
                    if (!q2) ++fit2_fell;
                    if (q2) {
                        expect(bytes_of(q, s2) <= budget, at + ": fit 2 within the budget");
                        bool fl = true, full = true;
                        for (size_t i = 0; i < q.size(); ++i) {
                            fl = fl && s2[i] >= std::min(q[i].n_exp, rn + 2);
                            full = full && (s2[i] == q[i].n_exp || bytes_of(q, s2) + q[i].unit > budget);
                        }
                        expect(fl, at + ": fit 2 every layer at its floor or above");
                        expect(full, at + ": fit 2 no layer's next slot still fits");
                    }
                }
                for (int fit = 0; fit <= 2; ++fit) {
                    if (fit == 2 && (src == 1 || !q2)) continue;   // (fit 2 falls back to fit 1 there)
                    const std::vector<int>& s = fit == 0 ? s0 : fit == 1 ? s1 : s2;
                    const std::vector<place::Split> p = splits(q, s, rn, fit);
                    const std::vector<place::Split> p0 = splits(q, s, rn, 0);
                    for (size_t i = 0; i < p.size(); ++i) {
                        const place::Split& x = p[i];
                        const std::string li = at + ", fit " + std::to_string(fit) + ", layer " + std::to_string(i);
                        expect(x.statics + x.recent + x.spare == x.slots && x.slots == std::min(s[i], q[i].n_exp),
                               li + ": statics + recent + spare = slots");
                        expect(x.recent >= 0 && x.recent <= rn, li + ": recent within DS4_RECENT");
                        if (x.spare == 1) expect(x.statics >= 1, li + ": a static beside the spare");
                        if (fit >= 1 && rn > 0 && x.slots >= 3 && x.slots < q[i].n_exp)
                            expect(x.recent >= 1, li + ": a layer of 3 or more slots keeps a recent slot");
                        if (fit >= 1 && p0[i].recent > 0)
                            expect(x.recent == p0[i].recent && x.statics == p0[i].statics,
                                   li + ": the whole set kept where it fits, as today");
                    }
                }
                ++runs;
            }
        }
    expect(fit2_fell > 0, "sweep: some budget too small for fit 2's floor (the fallback is exercised)");
    std::printf("sweep: DS4_RECENT 0/16/24/32/40/48 x budgets 6-24 GB x queue / even: %d runs, fit 2's floor over the "
                "budget in %d\n", runs, fit2_fell);
}

// small layers: recent sets as large as the layer, the cache off, ineligible layers, one expert
static void test_edges() {
    expect(place::recent_slots(34, 32, true, 0) == 32, "edge: 34 slots keep 32 (today)");
    expect(place::recent_slots(33, 32, true, 0) == 0, "edge: 33 slots keep none (today)");
    expect(place::recent_slots(33, 32, true, 1) == 31, "edge: 33 slots keep 31 (fit 1)");
    expect(place::recent_slots(3, 32, true, 1) == 1, "edge: 3 slots keep 1 (fit 1)");
    expect(place::recent_slots(2, 32, true, 1) == 0, "edge: 2 slots keep none (a static and the spare)");
    expect(place::recent_slots(20, 32, false, 2) == 0, "edge: an ineligible layer keeps none");
    expect(place::recent_slots(20, 0, true, 2) == 0, "edge: DS4_RECENT=0 keeps none");
    const place::Split a = place::split(8, 8, true, 32, true, 2);
    expect(a.slots == 8 && a.statics == 8 && a.spare == 0, "edge: every expert fits: no split");
    const place::Split b = place::split(5, 8, false, 32, true, 2);
    expect(b.statics == 5 && b.recent == 0 && b.spare == 0, "edge: the cache off: every slot a static");
    const place::Split c = place::split(1, 8, true, 32, true, 1);
    expect(c.statics == 1 && c.spare == 0, "edge: one slot: a static, no spare");
    // the queue's floor on a layer smaller than the recent set: the whole layer
    std::vector<uint32_t> prof(16, 1u);
    std::vector<place::QLayer> q = {{0, 8, 1000.0, place::shares(prof.data(), 8)},
                                    {1, 8, 1000.0, place::shares(prof.data() + 8, 8)}};
    std::vector<int> s;
    expect(place::queue(q, 16000.0, 32, 2, s) && s == std::vector<int>({8, 8}), "edge: fit 2 floor = the whole layer");
    s = {3, 3};
    expect(!place::queue(q, 15999.0, 32, 2, s), "edge: fit 2 floor over the budget: false");
    // the switch's spelling
    expect(place::recent_fit_of(nullptr) == 0 && place::recent_fit_of("") == 0 && place::recent_fit_of("0") == 0,
           "switch: unset / empty / 0 = off");
    expect(place::recent_fit_of("1") == 1 && place::recent_fit_of("2") == 2, "switch: 1, 2");
    expect(place::recent_fit_of("3") == 2 && place::recent_fit_of("-1") == 0 && place::recent_fit_of("x") == 0,
           "switch: past 2 = 2, below 0 or not a number = off");
}

int main(int argc, char** argv) {
    std::string path = "data/expert-profile-ds4.bin";
    bool table = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--table") == 0) table = true;
        else path = argv[i];
    }
    std::vector<uint32_t> prof((size_t) kNL * kNE, 0u);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    const size_t n = f != nullptr ? std::fread(prof.data(), 4, prof.size(), f) : 0;
    if (f != nullptr) std::fclose(f);
    if (n != prof.size()) {
        std::fprintf(stderr, "ds4_place_test: %s: %zu of %zu counts read (the shipped profile, 61440 bytes)\n",
                     path.c_str(), n, prof.size());
        return 2;
    }
    test_edges();
    test_engine_parity();
    test_box(prof, table);
    test_sweep(prof);
    if (g_fail != 0) {
        std::fprintf(stderr, "ds4_place_test: FAILED %d of %lld checks\n", g_fail, g_checks);
        return 1;
    }
    std::printf("ds4_place_test: ok (%lld checks)\n", g_checks);
    return 0;
}
