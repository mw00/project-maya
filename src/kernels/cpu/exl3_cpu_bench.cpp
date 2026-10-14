// src/kernels/cpu/exl3_cpu_bench.cpp - how fast does this CPU run GLM-5.3-Flash's EXL3 routed experts on the CPU lane?
//
//   exl3_cpu_bench [--K 3,4] [--threads 1,6,12,24] [--tokens 1,2,4,8] [--tiers 0,1,2] [--mb 640] [--per-call 4]
//                  [--pin] [--hot-only | --cold-only]
//
// Random trellis words (any bit pattern is a valid encoding), experts of 4096 -> 2048 -> 4096 (gate, up, down), each
// tier this CPU runs (0 scalar, 1 AVX2 + FMA, 2 AVX-512) through exl3_cpu_band_acc_on:
//  HOT   one band of a gate matrix (every slice: 196 KB at K 3, in L2) again and again on one thread - the kernel's
//        compute speed, weights a second, for 1..N tokens at once;
//  COLD  --mb of experts read from DRAM once a pass, --per-call experts a call (the start-up timing's 4; a decode
//        route gives the lane 1..8) on a pool of T threads in the engine's jobs
//        (Glm5Model::lane_experts: the inputs; per expert, matrix, band and half the byte sums; per expert and band the
//        swiglu and the down input; per expert and band the down rows) - an expert's time on the lane, the number the
//        engine's start-up timing compares with a copy over PCIe, and the trellis GB/s.
// --pin: thread t on CPU t (one a physical core while T <= the core count on Linux's usual numbering).
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace kc = strata::kernels::cpu;
using clk = std::chrono::steady_clock;

namespace {

constexpr int E = 4096, FF = 2048, GB = FF / 128, DB = E / 128, KH = E / 32;   // GLM-5.3-Flash's routed expert

double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void pin_to(int cpu) {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#else
    (void) cpu;
#endif
}

// nt - 1 spinning workers and the caller; run(n, fn) calls fn(0..n-1) (claimed in turn) and returns once every
// worker is done with the batch
class Pool {
public:
    Pool(int nt, bool pin) : nt_(nt) {
        if (pin) pin_to(0);
        for (int t = 1; t < nt; ++t)
            th_.emplace_back([this, t, pin] {
                if (pin) pin_to(t);
                uint64_t seen = 0;
                for (;;) {
                    uint64_t g;
                    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                    }
                    seen = g;
                    if (quit_.load(std::memory_order_relaxed)) return;
                    work();
                    finished_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    ~Pool() {
        quit_.store(true);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : th_) t.join();
    }
    void run(int n, const std::function<void(int)>& fn) {
        fn_ = &fn;
        total_ = n;
        next_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        work();
        while (finished_.load(std::memory_order_acquire) < nt_ - 1) {
        }
    }

private:
    void work() {
        for (;;) {
            const int j = next_.fetch_add(1, std::memory_order_acq_rel);
            if (j >= total_) return;
            (*fn_)(j);
        }
    }
    int nt_;
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_{0}, finished_{0};
    std::atomic<bool> quit_{false};
    const std::function<void(int)>* fn_ = nullptr;
    int total_ = 0;
};

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) (g_rng >> 16);
}
std::vector<uint16_t> scales(int n) {   // FP16 magnitudes in [0.5, 1), random sign
    std::vector<uint16_t> v((size_t) n);
    for (auto& s : v) s = (uint16_t) (0x3800 | (rnd() & 0x3ff) | ((rnd() & 1u) << 15));
    return v;
}

struct Expert {
    std::vector<uint32_t> tr[3];   // gate, up, down
    std::vector<uint16_t> suh[3], svh[3];
    kc::Exl3Part part(int r, int K) const {
        kc::Exl3Part p;
        p.trellis = tr[r].data();
        p.suh = suh[r].data();
        p.svh = svh[r].data();
        p.k = r < 2 ? E : FF;
        p.n = r < 2 ? FF : E;
        p.K = K;
        return p;
    }
};

// an expert-token's rows (the engine's X3Rows)
struct Rows {
    std::vector<float> x[2], acc[2], xd, xds, dn;
    float xs[2] = {0, 0};
    Rows() {
        for (int r = 0; r < 2; ++r) {
            x[r].assign(E, 0.0f);
            acc[r].assign((size_t) GB * 2 * 128, 0.0f);
        }
        xd.assign(FF, 0.0f);
        xds.assign(GB, 0.0f);
        dn.assign(E, 0.0f);
    }
};

std::vector<int> int_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

const char* tier_name(int t) { return t == 2 ? "avx512" : t == 1 ? "avx2" : "scalar"; }

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Ks = {3, 4}, threads = {1, 6, 12, 24}, tokens = {1, 2, 4, 8}, tiers = {0, 1, 2};
    double mb = 640.0;
    int per = 4;
    bool pin = false, hot = true, cold = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (a == "--K" && v) Ks = int_list(v), ++i;
        else if (a == "--threads" && v) threads = int_list(v), ++i;
        else if (a == "--tokens" && v) tokens = int_list(v), ++i;
        else if (a == "--tiers" && v) tiers = int_list(v), ++i;
        else if (a == "--mb" && v) mb = std::atof(v), ++i;
        else if (a == "--per-call" && v) per = std::atoi(v), ++i;
        else if (a == "--pin") pin = true;
        else if (a == "--hot-only") cold = false;
        else if (a == "--cold-only") hot = false;
        else {
            std::fprintf(stderr, "usage: exl3_cpu_bench [--K 3,4] [--threads 1,6,12,24] [--tokens 1,2,4,8] "
                                 "[--tiers 0,1,2] [--mb 640] [--per-call 4] [--pin] [--hot-only | --cold-only]\n");
            return 2;
        }
    }
    for (int K : Ks)
        if (K < 1 || K > 8) {
            std::fprintf(stderr, "ERROR: K must be 1..8\n");
            return 2;
        }
    if (per < 1 || per > 8) {
        std::fprintf(stderr, "ERROR: --per-call must be 1..8\n");
        return 2;
    }
    for (int n : tokens)
        if (n < 1 || n > 64) {
            std::fprintf(stderr, "ERROR: tokens must be 1..64\n");
            return 2;
        }
    std::printf("exl3_cpu_bench: this CPU's lane runs the %s tier; %u hardware threads\n",
                tier_name(kc::exl3_cpu_tier()), std::thread::hardware_concurrency());
    std::vector<float> xin((size_t) 64 * E);
    for (auto& v : xin) v = (float) ((int) (rnd() % 2001) - 1000) / 400.0f;

    for (int K : Ks) {
        const double expert_mb = 3.0 * E * FF * K / 8.0 / 1e6;
        // ---- HOT: band 0 of one gate matrix, all its slices, one thread
        if (hot) {
            Expert ex;
            ex.tr[0].resize((size_t) (E / 16) * (FF / 16) * 8 * K);
            for (auto& w : ex.tr[0]) w = rnd();
            ex.suh[0] = scales(E);
            ex.svh[0] = scales(FF);
            const kc::Exl3Part p = ex.part(0, K);
            std::vector<std::vector<float>> xh(64, std::vector<float>(E)), acc(64, std::vector<float>(128));
            for (int i = 0; i < 64; ++i) kc::exl3_cpu_prepare(p, xin.data() + (size_t) i * E, xh[(size_t) i].data());
            for (int tier : tiers)
                for (int n : tokens) {
                    std::vector<const float*> xp((size_t) n);
                    std::vector<float*> ap((size_t) n);
                    for (int i = 0; i < n; ++i) xp[(size_t) i] = xh[(size_t) i].data(), ap[(size_t) i] = acc[(size_t) i].data();
                    if (!kc::exl3_cpu_band_acc_on(tier, p, xp.data(), n, 0, 0, E / 16, ap.data())) break;
                    const int reps = tier == 0 ? 6 : 200;
                    double best = 1e30;
                    for (int round = 0; round < 3; ++round) {
                        const auto t0 = clk::now();
                        for (int r = 0; r < reps; ++r) kc::exl3_cpu_band_acc_on(tier, p, xp.data(), n, 0, 0, E / 16, ap.data());
                        best = std::min(best, secs(t0, clk::now()) / reps);
                    }
                    const double w = 128.0 * E;   // weights of the band
                    std::printf("HOT  K%d %-6s %2d token%s 1 thread: %8.1f M weights/s (%7.1f M weight-tokens/s), "
                                "%.3f ms an expert-token\n", K, tier_name(tier), n, n > 1 ? "s" : " ", w / best / 1e6,
                                w * n / best / 1e6, best / n * (3.0 * E * FF / w) * 1e3);
                }
        }
        // ---- COLD: experts from DRAM, 4 a call, the engine's jobs on T threads
        if (cold) {
            const int ne = std::max(8, (int) (mb / expert_mb) / 8 * 8);
            std::vector<Expert> xs((size_t) ne);
            for (auto& ex : xs)
                for (int r = 0; r < 3; ++r) {
                    ex.tr[r].resize((size_t) (E / 16) * (FF / 16) * 8 * K);
                    for (auto& w : ex.tr[r]) w = rnd();
                    ex.suh[r] = scales(r < 2 ? E : FF);
                    ex.svh[r] = scales(r < 2 ? FF : E);
                }
            std::printf("COLD K%d: %d experts of %.2f MB (%.0f MB), %d a call\n", K, ne, expert_mb, ne * expert_mb, per);
            const int maxn = *std::max_element(tokens.begin(), tokens.end());
            std::vector<Rows> rows((size_t) 8 * maxn);
            for (int T : threads) {
                Pool pool(T, pin);
                for (int tier : tiers) {
                    std::atomic<bool> runs{true};   // (false: this CPU lacks the tier)
                    for (int n : tokens) {
                        const auto call = [&](int e0) {
                            const auto R = [&](int e, int i) -> Rows& { return rows[(size_t) (e * n + i)]; };
                            pool.run(per * 2 * n, [&](int j) {
                                const int e = j / (2 * n), r = j / n % 2, i = j % n;
                                R(e, i).xs[r] = kc::exl3_cpu_prepare(xs[(size_t) (e0 + e)].part(r, K),
                                                                      xin.data() + (size_t) i * E, R(e, i).x[r].data());
                            });
                            pool.run(per * 2 * GB * 2, [&](int j) {
                                const int h = j & 1, b = (j >> 1) % GB, r = (j >> 1) / GB % 2, e = (j >> 1) / GB / 2;
                                const float* xp[64];
                                float* ap[64];
                                for (int i = 0; i < n; ++i) {
                                    xp[i] = R(e, i).x[r].data();
                                    ap[i] = R(e, i).acc[r].data() + ((size_t) b * 2 + h) * 128;
                                }
                                if (!kc::exl3_cpu_band_acc_on(tier, xs[(size_t) (e0 + e)].part(r, K), xp, n, b * 128,
                                                              h * KH, (h + 1) * KH, ap))
                                    runs = false;
                            });
                            pool.run(per * n * GB, [&](int j) {
                                const int e = j / (n * GB), i = j / GB % n, b = j % GB;
                                const Expert& ex = xs[(size_t) (e0 + e)];
                                float y[2][128], a[128], hb[128];
                                for (int r = 0; r < 2; ++r) {
                                    const float* a0 = R(e, i).acc[r].data() + ((size_t) b * 2) * 128;
                                    for (int c = 0; c < 128; ++c) a[c] = a0[c] + a0[128 + c];
                                    kc::exl3_cpu_band_out(ex.part(r, K), a, R(e, i).xs[r], b * 128, y[r]);
                                }
                                for (int c = 0; c < 128; ++c) hb[c] = y[0][c] / (1.0f + std::exp(-y[0][c])) * y[1][c];
                                R(e, i).xds[(size_t) b] = kc::exl3_cpu_prepare_block(ex.suh[2].data() + b * 128, hb,
                                                                                     R(e, i).xd.data() + (size_t) b * 128);
                            });
                            pool.run(per * DB, [&](int j) {
                                const int e = j / DB, b = j % DB;
                                const kc::Exl3Part pd = xs[(size_t) (e0 + e)].part(2, K);
                                const float* xp[64];
                                float a[64][128];
                                float* ap[64];
                                for (int i = 0; i < n; ++i) xp[i] = R(e, i).xd.data(), ap[i] = a[i];
                                if (!kc::exl3_cpu_band_acc_on(tier, pd, xp, n, b * 128, 0, FF / 16, ap)) runs = false;
                                for (int i = 0; i < n; ++i) {
                                    float s = 0.0f;
                                    for (int q = 0; q < GB; ++q) s += R(e, i).xds[(size_t) q];
                                    kc::exl3_cpu_band_out(pd, a[i], s, b * 128, R(e, i).dn.data() + (size_t) b * 128);
                                }
                            });
                        };
                        call(0);   // (the tier check and a warm-up)
                        if (!runs) break;
                        if (tier == 0 && n > 1) continue;   // the scalar tier does n tokens n times over
                        double best = 1e30;
                        const int passes = tier == 0 ? 1 : 3;
                        for (int pass = 0; pass < passes; ++pass) {
                            const auto t0 = clk::now();
                            for (int e0 = 0; e0 + per <= (tier == 0 ? std::min(ne, 16) : ne); e0 += per) call(e0);
                            best = std::min(best, secs(t0, clk::now()));
                        }
                        const int done = (tier == 0 ? std::min(ne, 16) : ne) / per * per;
                        std::printf("COLD K%d %-6s T%2d %d token%s: %7.1f experts/s, %.3f ms an expert (%.3f an "
                                    "expert-token), trellis %.1f GB/s\n", K, tier_name(tier), T, n, n > 1 ? "s" : " ",
                                    done / best, best / done * 1e3, best / done / n * 1e3,
                                    done * expert_mb / 1e3 / best);
                        std::fflush(stdout);
                    }
                }
            }
        }
    }
    return 0;
}
