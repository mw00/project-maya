// tests/exl3_cpu_shuf_test.cpp - STRATA_EXL3_AVX2_SHUF: the AVX2 tier's byte-aligned windows taken by byte shuffles,
// against the same tier's shifts and against the scalar tier, bit for bit (no GPU, no model).
//
//   exl3_cpu_shuf_test [--env]
//
// Random trellis words (any bit pattern is a valid encoding) for every bitrate the CPU lane takes (K 1..8 and the
// half-integer 1.5 / 2.5 / 3.5), 1..8 activations, the whole slice range and a split-K piece of it:
//  - the bands (exl3_cpu_band_acc_on, tier 1: band_avx2 for one activation, band_avx2_n for 2..8),
//  - the split-K rows (exl3_cpu_rows_acc_cols_n: rows_avx2_n, the DeepSeek lane's kernel) over whole rows and over
//    C1's uneven column runs,
//  - the band GEMM (exl3_cpu_band_gemm: decode_chunk_avx2, the prompt path's),
// each with the shuffles off and on (exl3_cpu_set_avx2_shuf), the outputs NaN-poisoned before every call, memcmp:
//  - integer activations (|x| <= 8: every product and partial sum is an integer below 2^24, exact in F32 whatever the
//    rounding order): every path, off and on, against the scalar tier;
//  - random F32 activations: on against off (the scalar tier rounds otherwise: it has no FMA).
// The rows and the GEMM run on the lane's tier (exl3_cpu_tier()), which must be 1 (AVX2): the ctest entries set
// STRATA_FORCE_AVX2=1 so an AVX-512 CPU runs it too. A CPU without AVX2 (or a build for another architecture) is
// told so and the test exits 77 (ctest: skipped).
// --env: run with STRATA_EXL3_AVX2_SHUF=1 - the switch must be on at the start (the kernels' own start check
// passed), the ctest entry exl3_cpu_shuf_test_env. Without --env and without that variable it must be off (the
// default).
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;

namespace {

constexpr int kK = 512, kN = 384, kNB = kN / 128, kMaxN = kc::kExl3CpuMaxN;
constexpr int kK2s[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};   // exl3_cpu.cpp EXL3_LANE_K2

uint64_t g_rng = 0x2545F4914F6CDD1Dull;
uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) (g_rng >> 32);
}

const char* tier_name(int t) { return t == 3 ? "avx512vnni" : t == 2 ? "avx512" : t == 1 ? "avx2" : "scalar"; }

std::string kname(int k2) { return (k2 & 1) ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2); }

constexpr uint32_t kPoison = 0x7fc0dead;   // a quiet NaN no kernel writes
void poison(std::vector<float>& v) {
    for (float& f : v) std::memcpy(&f, &kPoison, 4);
}
size_t differ(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(a + i, b + i, 4) != 0;
    return d;
}

struct Counts {
    size_t bands = 0, rows = 0, runs = 0, gemm = 0, scalar = 0, compared = 0;
    size_t all() const { return bands + rows + runs + gemm + scalar; }
};

// one bitrate, one set of activations: every path, the shuffles off and on
void run_k2(int k2, const std::vector<std::vector<float>>& xs, bool exact, float* wc, Counts& c) {
    std::vector<uint32_t> tr((size_t) (kK / 16) * (kN / 16) * 4 * k2);
    for (uint32_t& w : tr) w = rnd();
    std::vector<uint16_t> suh(kK, 0x3c00), svh(kN, 0x3c00);
    kc::Exl3Part m;
    m.trellis = tr.data();
    m.suh = suh.data();
    m.svh = svh.data();
    m.k = kK;
    m.n = kN;
    m.k2 = k2;
    const float* xp[kMaxN];
    for (int i = 0; i < kMaxN; ++i) xp[i] = xs[(size_t) i].data();
    const int ranges[2][2] = {{0, kK / 16}, {3, kK / 16 - 1}};   // every slice; a split-K piece
    std::vector<float> ref((size_t) kMaxN * kN), out[2], run((size_t) kMaxN * kN);
    for (auto& o : out) o.assign((size_t) kMaxN * kN, 0.0f);
    float* rp[kMaxN];
    float* op[2][kMaxN];
    float* up[kMaxN];
    for (int i = 0; i < kMaxN; ++i) {
        rp[i] = ref.data() + (size_t) i * kN;
        up[i] = run.data() + (size_t) i * kN;
        for (int s = 0; s < 2; ++s) op[s][i] = out[s].data() + (size_t) i * kN;
    }
    for (const auto& rg : ranges) {
        const int ks0 = rg[0], ks1 = rg[1];
        // the scalar tier's sums of all 8 activations (each one's alone: band_on_n loops band_scalar)
        if (exact) {
            poison(ref);
            for (int b = 0; b < kNB; ++b) {
                float* bp[kMaxN];
                for (int i = 0; i < kMaxN; ++i) bp[i] = rp[i] + b * 128;
                if (!kc::exl3_cpu_band_acc_on(0, m, xp, kMaxN, b * 128, ks0, ks1, bp)) std::abort();
            }
        }
        for (int n = 1; n <= kMaxN; ++n) {
            const size_t len = (size_t) n * kN;
            // the bands on tier 1, one call a band
            for (int s = 0; s < 2; ++s) {
                kc::exl3_cpu_set_avx2_shuf(s == 1);
                poison(out[s]);
                for (int b = 0; b < kNB; ++b) {
                    float* bp[kMaxN];
                    for (int i = 0; i < n; ++i) bp[i] = op[s][i] + b * 128;
                    if (!kc::exl3_cpu_band_acc_on(1, m, xp, n, b * 128, ks0, ks1, bp)) std::abort();
                }
            }
            for (int i = 0; i < n; ++i) {
                c.bands += differ(op[0][i], op[1][i], kN);
                if (exact) c.scalar += differ(op[0][i], rp[i], kN) + differ(op[1][i], rp[i], kN);
            }
            c.compared += len;
            // the split-K rows over every column (the DeepSeek lane's call), then over uneven column runs (C1's)
            for (int s = 0; s < 2; ++s) {
                kc::exl3_cpu_set_avx2_shuf(s == 1);
                poison(out[s]);
                kc::exl3_cpu_rows_acc_cols_n(m, xp, n, ks0, ks1, 0, kN, op[s]);
            }
            kc::exl3_cpu_set_avx2_shuf(true);
            poison(run);
            kc::exl3_cpu_rows_acc_cols_n(m, xp, n, ks0, ks1, 0, 128, up);
            kc::exl3_cpu_rows_acc_cols_n(m, xp, n, ks0, ks1, 128, kN, up);
            for (int i = 0; i < n; ++i) {
                c.rows += differ(op[0][i], op[1][i], kN);
                c.runs += differ(op[0][i], up[i], kN);
                if (exact) c.scalar += differ(op[0][i], rp[i], kN) + differ(op[1][i], rp[i], kN);
            }
            c.compared += len;
            // the band GEMM: every slice, all the bands at once
            if (ks0 == 0 && ks1 == kK / 16) {
                for (int s = 0; s < 2; ++s) {
                    kc::exl3_cpu_set_avx2_shuf(s == 1);
                    poison(out[s]);
                    kc::exl3_cpu_band_gemm(m, xp, n, 0, kNB, op[s], wc);
                }
                for (int i = 0; i < n; ++i) {
                    c.gemm += differ(op[0][i], op[1][i], kN);
                    if (exact) c.scalar += differ(op[0][i], rp[i], kN) + differ(op[1][i], rp[i], kN);
                }
                c.compared += len;
            }
        }
    }
    kc::exl3_cpu_set_avx2_shuf(false);
}

}  // namespace

int main(int argc, char** argv) {
    bool env_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--env") == 0) env_mode = true;
        else {
            std::fprintf(stderr, "usage: exl3_cpu_shuf_test [--env]\n");
            return 2;
        }
    }
    const int tier = kc::exl3_cpu_tier();
    std::printf("exl3_cpu_shuf_test: the lane's EXL3 CPU tier %d (%s)\n", tier, tier_name(tier));
    int fail = 0;
    // the switch: on at the start with STRATA_EXL3_AVX2_SHUF=1 (the start check passed), off by default
    const char* ev = std::getenv("STRATA_EXL3_AVX2_SHUF");
    const bool at_start = kc::exl3_cpu_avx2_shuf();
    if (env_mode && tier == 1 && !at_start) {
        std::printf("  STRATA_EXL3_AVX2_SHUF=%s: off at the start  FAIL\n", ev ? ev : "(unset)");
        ++fail;
    } else if (!env_mode && ev == nullptr && at_start) {
        std::printf("  STRATA_EXL3_AVX2_SHUF unset: on at the start  FAIL (it must be off by default)\n");
        ++fail;
    } else {
        std::printf("  switch at the start: %s (STRATA_EXL3_AVX2_SHUF=%s)  ok\n", at_start ? "on" : "off",
                    ev ? ev : "unset");
    }
    for (int s = 1; s >= 0; --s) {
        kc::exl3_cpu_set_avx2_shuf(s == 1);
        if (kc::exl3_cpu_avx2_shuf() != (s == 1)) {
            std::printf("  exl3_cpu_set_avx2_shuf(%d) not taken  FAIL\n", s);
            ++fail;
        }
    }
    if (tier != 1) {
        std::printf("exl3_cpu_shuf_test: SKIPPED - the AVX2 tier is not the lane's here (tier %d; an AVX-512 CPU needs "
                    "STRATA_FORCE_AVX2=1, a CPU without AVX2 or another architecture cannot run it)\n", tier);
        return fail != 0 ? 1 : 77;
    }
    std::vector<float> wcb(kc::exl3_cpu_gemm_scratch() + 16);
    float* wc = wcb.data();
    while (((uintptr_t) wc & 63) != 0) ++wc;   // 64-byte aligned, as the GEMM wants
    std::vector<std::vector<float>> xi(kMaxN, std::vector<float>(kK)), xf(kMaxN, std::vector<float>(kK));
    for (auto& x : xi)
        for (float& v : x) v = (float) (1 + (int) (rnd() % 8)) * ((rnd() & 1u) ? -1.0f : 1.0f);
    for (auto& x : xf)
        for (float& v : x) v = (float) ((int) (rnd() % 40001) - 20000) / 9973.0f;
    Counts tot;
    for (int k2 : kK2s) {
        Counts ci, cf;
        run_k2(k2, xi, true, wc, ci);
        run_k2(k2, xf, false, wc, cf);
        const bool ok = ci.all() + cf.all() == 0;
        std::printf("  K%-4s 1-%d rows  integer x: bands %zu rows %zu runs %zu gemm %zu vs scalar %zu | F32 x: bands %zu "
                    "rows %zu runs %zu gemm %zu  (of %zu + %zu)  %s\n", kname(k2).c_str(), kMaxN, ci.bands, ci.rows,
                    ci.runs, ci.gemm, ci.scalar, cf.bands, cf.rows, cf.runs, cf.gemm, ci.compared, cf.compared,
                    ok ? "ok" : "FAIL");
        if (!ok) ++fail;
        tot.compared += ci.compared + cf.compared;
        tot.bands += ci.all() + cf.all();
    }
    std::printf("exl3_cpu_shuf_test: tier %d (%s), %zu outputs compared, %zu differ: %s\n", tier, tier_name(tier),
                tot.compared, tot.bands, fail == 0 ? "ok" : "FAIL");
    return fail == 0 ? 0 : 1;
}
