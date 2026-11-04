// tests/ds4_cpu_split_test.cpp - DS4_CPU_SPLIT_COLS (src/ds4/ds4_cpu_exl3.hpp): the CPU lane's EXL3 experts with a
// small call's gate / up and down pieces cut into column runs, against the uncut schedule bit for bit (no GPU, no
// model).
//
//   ds4_cpu_split_test [--small] [--threads N,N,...]
//
// DeepSeek-V4.1-Flash's geometry (5120 -> 2304 -> 5120; --small 1280 -> 768 -> 1280 for a slow CPU), random trellis
// words for K 2 (k2 4) and K 2.5 (k2 5), one and two NUMA parts (a pool a part, node 0's caller helping as the engine's
// service thread does), 1, 2, 3 and 6 entries of one token each (decode), of 1-3 tokens and of 4-8 tokens (verify
// windows; the wider one's first entry 8 tokens, the kernels' widest call, the others 4 + i % 5, so a cut call of one or
// two entries reaches the kernels' N >= 4 and N = 8 code too), several thread counts (1, 2, 5, 16, 30 by default).
// Each case's outputs from the four steps uncut (the reference: today's schedule) against the uncut flow, the split
// steps and the split flow, memcmp; the scratch is poisoned with NaN before every run, so a run that left a sum
// unwritten cannot pass on the reference's leftovers.  Also: the runs exl3_split_runs picks (the 3090 box's lane: 29
// workers + the service thread, one part), the batches' job counts with the split, and that the cases cut both steps
// somewhere (else the test proves nothing).
//
// The EXL3 kernels run on the one tier this CPU picks (exl3_cpu_tier(), printed at start-up): 0 scalar, 1 AVX2 + FMA +
// F16C, 2 AVX-512 F/BW, 3 AVX-512 VNNI - the widest the CPU has.  STRATA_FORCE_AVX2=1 keeps an AVX-512 CPU on tier 1
// (ctest ds4_cpu_split_test_avx2; on a CPU without AVX-512 that is the default run again).  Tier 2 cannot be forced on a
// VNNI CPU (it always takes 3), so only a CPU with AVX-512 F/BW and no VNNI tests it.
#include "../src/ds4/ds4_cpu_exl3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;
using strata::ds4::Exl3Lane;
using strata::ds4::LaneEnt;
using strata::ds4::Pool;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// the runs exl3_split_runs picks: (jobs, threads, bands) -> runs
static void test_runs() {
    struct C {
        int jobs, threads, bands, want;
    };
    // the box: 30 threads a call, one part - gate / up 2 GQ = 40 jobs an entry over 18 bands, down DQ = 18 jobs an
    // entry over 40 bands (5120 -> 2304 -> 5120)
    const C cs[] = {
        {40, 30, 18, 3},     // one entry's gate / up: 2 rounds -> 4 of a third (ideal 1.33)
        {80, 30, 18, 3},     // two entries: 3 rounds -> 8 of a third (ideal 2.67)
        {120, 30, 18, 1},    // three: 4 rounds already whole
        {240, 30, 18, 1},    // six: 8 jobs a thread, the cap
        {18, 30, 40, 5},     // one entry's down: 1 round, 12 threads idle -> 3 of a fifth (ideal 0.6)
        {36, 30, 40, 5},     // two: 2 rounds -> 6 of a fifth (ideal 1.2)
        {54, 30, 40, 1},     // three: 2 rounds, no cut does better
        {108, 30, 40, 1},    // six
        {18, 1, 40, 1},      // one thread: nothing to gain
        {18, 0, 40, 1},      // off
        {0, 30, 40, 1},      // no jobs
        {40, 30, 1, 1},      // one band: nothing to cut
        {20, 30, 7, 7},      // a prime band count: 7 runs or none (equal runs only)
        {20, 30, 9, 3},      // two parts' gate / up (9 bands): 1 round of 20 -> 2 of a third (ideal 0.67)
    };
    for (const C& c : cs) {
        const int got = strata::ds4::exl3_split_runs(c.jobs, c.threads, c.bands);
        expect(got == c.want, "exl3_split_runs(" + std::to_string(c.jobs) + ", " + std::to_string(c.threads) + ", " +
                                  std::to_string(c.bands) + ") = " + std::to_string(got) + ", want " +
                                  std::to_string(c.want));
    }
    // whatever the inputs: a divisor of the bands, at most kSplitJobs jobs a thread once cut
    for (int w = 1; w <= 64; ++w)
        for (int j = 1; j <= 300; ++j)
            for (int b : {1, 2, 7, 9, 18, 20, 40}) {
                const int s = strata::ds4::exl3_split_runs(j, w, b);
                if (s < 1 || b % s != 0 || (s > 1 && (long) j * s > (long) strata::ds4::kSplitJobs * w)) {
                    expect(false, "exl3_split_runs(" + std::to_string(j) + ", " + std::to_string(w) + ", " +
                                      std::to_string(b) + ") = " + std::to_string(s) + ": not a divisor or too many");
                    return;
                }
            }
}

int main(int argc, char** argv) {
    bool small = false;
    std::vector<int> thread_counts = {1, 2, 5, 16, 30};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--small") small = true;
        else if (a == "--threads" && i + 1 < argc) {
            thread_counts.clear();
            std::string v = argv[++i];
            for (size_t p = 0; p < v.size();) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                thread_counts.push_back(std::max(1, std::atoi(v.substr(p, q - p).c_str())));
                p = q + 1;
            }
        }
    }
    // the kernels' tier this run tests (src/kernels/cpu/exl3_cpu.cpp detect_tier)
    const int tier = kc::exl3_cpu_tier();
    const char* tier_names[] = {"scalar", "AVX2 + FMA + F16C", "AVX-512 F/BW", "AVX-512 VNNI"};
    const char* force = std::getenv("STRATA_FORCE_AVX2");
    const bool forced = force != nullptr && force[0] == '1';
    std::printf("EXL3 CPU tier %d (%s)%s\n", tier, tier >= 0 && tier <= 3 ? tier_names[tier] : "unknown",
                forced ? ", STRATA_FORCE_AVX2=1" : "");
    expect(tier >= 0 && tier <= 3, "exl3_cpu_tier() = " + std::to_string(tier) + ", not 0-3");
    expect(!forced || tier <= 1, "STRATA_FORCE_AVX2=1 but tier " + std::to_string(tier));
    test_runs();

    const int E = small ? 1280 : 5120, FF = small ? 768 : 2304;
    constexpr int kWin = 16, S = 64, kExp = 6;   // token rows of x, the slots, distinct experts
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rs(11);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> x((size_t) kWin * E);
    for (auto& v : x) v = nd(rs);
    auto scale = [&] { return (uint16_t) (0x3800 + (rs() % 0x0800)); };   // FP16 0.5 .. 1.0

    long cases = 0, cut_gu = 0, cut_dn = 0, differ = 0;
    for (const int k2 : {4, 5}) {
        for (const int NH : {1, 2}) {
            const int FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128, GB = FFh / 128, DB = Eh / 128;
            // [expert][part]: gate | up | down trellis words; the scales as the engine's (suh whole, svh a part's)
            const size_t w_gu = (size_t) (E / 16) * (FFh / 16) * 4 * k2, w_d = (size_t) (FF / 16) * (Eh / 16) * 4 * k2;
            const size_t w_part = 2 * w_gu + w_d;
            std::vector<uint32_t> tr((size_t) kExp * NH * w_part);
            for (auto& w : tr) w = (uint32_t) rs();
            std::vector<uint16_t> suh_gu((size_t) kExp * 2 * E), svh_gu((size_t) kExp * NH * 2 * FFh);
            std::vector<uint16_t> suh_d((size_t) kExp * FF), svh_d((size_t) kExp * NH * Eh);
            for (auto* v : {&suh_gu, &svh_gu, &suh_d, &svh_d})
                for (auto& s : *v) s = scale();
            auto part = [&](int e, int X, int r) {
                kc::Exl3Part p;
                const uint32_t* b = tr.data() + ((size_t) e * NH + X) * w_part;
                p.k2 = k2;
                if (r < 2) {
                    p.trellis = b + (size_t) r * w_gu;
                    p.suh = suh_gu.data() + ((size_t) e * 2 + r) * E;
                    p.svh = svh_gu.data() + (((size_t) e * NH + X) * 2 + r) * FFh;
                    p.k = E;
                    p.n = FFh;
                } else {
                    p.trellis = b + 2 * w_gu;
                    p.suh = suh_d.data() + (size_t) e * FF;
                    p.svh = svh_d.data() + ((size_t) e * NH + X) * Eh;
                    p.k = FF;
                    p.n = Eh;
                }
                return p;
            };
            std::vector<float> xh((size_t) NH * S * 2 * E), xsb((size_t) NH * S * 2 * (E / 128)),
                acc((size_t) NH * S * 2 * GQ * FFh), xhd((size_t) S * FF), xsd((size_t) S * (FF / 128)),
                accd((size_t) NH * S * DQ * Eh), yd((size_t) S * E);
            Exl3Lane c;
            c.E = E;
            c.FF = FF;
            c.NH = NH;
            c.slots = S;
            c.lim = 10.0f;
            c.xh = xh.data();
            c.xsb = xsb.data();
            c.acc = acc.data();
            c.xhd = xhd.data();
            c.xsd = xsd.data();
            c.accd = accd.data();
            c.yd = yd.data();
            c.alloc_counters(NH);
            auto poison = [&] {
                for (auto* v : {&xh, &xsb, &acc, &xhd, &xsd, &accd, &yd}) std::fill(v->begin(), v->end(), nan);
            };
            for (const int W : thread_counts) {
                // the engine's pools: node 0's W - 1 workers and the caller, the other nodes' W workers each
                std::vector<std::unique_ptr<Pool>> pools;
                for (int X = 0; X < NH; ++X) pools.push_back(std::make_unique<Pool>(X == 0 ? W - 1 : W, std::vector<int>{}, 50));
                std::vector<std::pair<int, int>> batches;   // (jobs, step) of each nodes() call
                auto nodes = [&](int n, const std::function<void(int)>* f, int ph) {
                    batches.push_back({n, ph});
                    for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
                    pools[0]->help();
                    for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
                };
                auto nophase = [](int) {};
                for (const int ne : {1, 2, 3, 6})
                    for (const int window : {0, 1, 2}) {   // tokens an entry: 1; 1-3; 4-8 (the first entry 8)
                        std::vector<LaneEnt> ents;
                        int rows = 0;
                        for (int i = 0; i < ne; ++i) {
                            LaneEnt en;
                            en.e = (i + ne + window) % kExp;
                            en.n = window == 2 ? (i == 0 ? kc::kExl3CpuMaxN : 4 + i % 5) : window ? 1 + i % 3 : 1;
                            for (int t = 0; t < en.n; ++t) en.tok[t] = (i * 5 + t * 3 + window) % kWin;
                            en.slot0 = rows;
                            rows += en.n;
                            ents.push_back(en);
                        }
                        expect(rows <= S, "the entries' rows exceed the slots");
                        const std::string what = "K2 " + std::to_string(k2) + ", " + std::to_string(NH) + " part(s), " +
                                                 std::to_string(W) + " threads, " + std::to_string(ne) + " entries" +
                                                 (window == 2 ? " (4-8 tokens)" : window ? " (1-3 tokens)" : "");
                        auto run = [&](int split_w, bool flow) {
                            poison();
                            batches.clear();
                            c.split_w = split_w;
                            strata::ds4::exl3_experts(c, x.data(), (size_t) E, ents.data(), ne, part, nodes, nophase, flow);
                            c.split_w = 0;
                            return std::vector<float>(yd.begin(), yd.begin() + (size_t) rows * E);
                        };
                        const std::vector<float> ref = run(0, false);
                        bool finite = true;
                        for (float v : ref) finite = finite && std::isfinite(v);
                        expect(finite, what + ": the reference has non-finite outputs (the synthetic weights' scale)");
                        const int R0 = strata::ds4::exl3_split_runs(ne * 2 * GQ, W, GB);
                        const int R2 = strata::ds4::exl3_split_runs(ne * DQ, W, DB);
                        cut_gu += R0 > 1;
                        cut_dn += R2 > 1;
                        const int J0 = ne * 2 * GQ * R0, J1 = ne * GB, J2 = ne * DQ * R2, J3 = ne * DB;
                        struct V {
                            const char* name;
                            int split_w;
                            bool flow;
                        };
                        for (const V v : {V{"uncut flow", 0, true}, V{"split steps", W, false}, V{"split flow", W, true}}) {
                            const std::vector<float> got = run(v.split_w, v.flow);
                            if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
                                ++differ;
                                size_t i = 0;
                                while (i < ref.size() && std::memcmp(&ref[i], &got[i], sizeof(float)) == 0) ++i;
                                expect(false, what + ", " + v.name + ": row " + std::to_string(i / (size_t) E) +
                                                  " col " + std::to_string(i % (size_t) E) + " steps " +
                                                  std::to_string(ref[i]) + " got " + std::to_string(got[i]));
                            }
                            // the batches the split asked the pools for
                            if (v.split_w > 0 && !v.flow) {
                                const std::vector<std::pair<int, int>> want = {{J0, 0}, {J1, 1}, {J2, 2}, {J3, 3}};
                                expect(batches == want, what + ", split steps: the batches' job counts (runs " +
                                                            std::to_string(R0) + " / " + std::to_string(R2) + ")");
                            } else if (v.split_w > 0) {
                                const std::vector<std::pair<int, int>> want = {{J0 + J1 + J2 + J3, -1}};
                                expect(batches == want, what + ", split flow: the batch's job count (runs " +
                                                            std::to_string(R0) + " / " + std::to_string(R2) + ")");
                            }
                        }
                        ++cases;
                    }
            }
            std::printf("K2 %d, %d part(s): done\n", k2, NH);
        }
    }
    // a negative control of the matrix itself: some case must have cut each step, else nothing above tested the cut
    expect(cut_gu > 0, "no case cut the gate / up pieces");
    expect(cut_dn > 0, "no case cut the down pieces");
    std::printf("split, tier %d: %ld cases (gate / up cut in %ld, down in %ld), %ld runs differ from the uncut steps, "
                "bit for bit\n",
                tier, cases, cut_gu, cut_dn, differ);
    std::printf("%s\n", g_fail == 0 ? "ds4_cpu_split_test: ok" : "ds4_cpu_split_test: FAIL");
    return g_fail == 0 ? 0 : 1;
}
