// tests/exl3_cpu_w16_test.cpp - exl3_cpu_set_w16 (the AVX2 tier's EXL3 codebook product and byte sum in 16-bit lanes)
// against the 32-bit AVX2 path and a scalar model, bit for bit.  Synthetic weights, no GPU, no model file.
//
//   exl3_cpu_w16_test
//
// For every bitrate the CPU lane takes (K 1..8 and 1.5, 2.5, 3.5) on random trellis words and random activations:
//  - the band kernels: exl3_cpu_band_acc_on's tier 4 (the AVX2 tier with the switch) against tier 1, 1-8 and 11
//    activations, every band, the whole slice range and partial ones;
//  - the split-K rows (exl3_cpu_rows_acc_cols_n, exl3_cpu_rows_acc_n, exl3_cpu_rows_acc: the DS4 lane's calls) with
//    exl3_cpu_set_w16(true) against false, 1-8 and 11 activations, whole rows and column runs (the columns outside a
//    run must stay untouched);
//  - exl3_cpu_band_acc / exl3_cpu_band_acc_n and the band GEMM (exl3_cpu_band_gemm) with the switch on and off;
//  every output also against a scalar model (each state read bit by bit, its mul1 byte sum, one fma(s, xh, acc) a
//  weight in (slice, row) order - the vector tiers' arithmetic), memcmp, the outputs poisoned with NaN before each run;
//  - all 65536 states: a K 8 matrix whose tiles hold every one of them, each state's byte sum read back through the
//    kernels with one-hot activations (tiers 0, 1 and 4, and the rows with the switch off and on, 1, 3 and 8
//    activations a call), against the scalar byte sum.
// It prints the tier it ran on.  ctest runs it twice, the second time with MAYA_FORCE_AVX2=1, so an AVX-512 host's
// rows and GEMM take the AVX2 tier too (the band entries reach tier 4 on any AVX2 host).  On a CPU without AVX2 the
// switch must report itself off and change nothing, and tier 4 must be refused.
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

int g_fail = 0;

uint64_t g_rng = 0x2545F4914F6CDD1Dull;
uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) (g_rng >> 16);
}
// full-mantissa floats of both signs over a few binades (so a changed rounding shows)
float rnd_x() {
    const float u = (float) (rnd() & 0xFFFFFF) / 16777216.0f - 0.5f;
    return std::ldexp(u, (int) (rnd() % 7) - 3);
}

const char* tier_name(int t) {
    return t == 4 ? "avx2-w16" : t == 3 ? "avx512vnni" : t == 2 ? "avx512" : t == 1 ? "avx2" : "scalar";
}

// step p of a tile sits at row (p/8 % 4) 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1)
struct Steps {
    int p[16][16];
    Steps() {
        for (int q = 0; q < 256; ++q) p[(q / 8 % 4) * 2 + (q & 1) + 8 * (q / 2 & 1)][q / 32 + 8 * (q / 4 & 1)] = q;
    }
};
const Steps kSteps;

// step p's state read bit by bit: the 16-bit window that ends at bit floor((p + 1) k2 / 2) of the tile's MSB-first
// ring of 128 k2 bits
uint32_t model_state(const uint32_t* tile, int k2, int p) {
    const int bits = 128 * k2, end = ((p + 1) * k2) / 2 + bits;
    uint32_t st = 0;
    for (int q = end - 16; q < end; ++q) {
        const int b = q % bits;
        st = (st << 1) | ((tile[b >> 5] >> (31 - (b & 31))) & 1u);
    }
    return st;
}
int byte_sum(uint32_t st) {
    const uint32_t x = st * 0x83DCD12Du;
    return (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
}

struct Mat {
    std::vector<uint32_t> tr;
    std::vector<uint16_t> suh, svh;
    std::vector<int> s;   // byte sums, row-major k x n
    std::vector<uint8_t> seen;   // states present (65536)
    cpu::Exl3Part p;
};

// crafted (K 8 only): tile j's byte ring B[2q] = j % 256, B[2q + 1] = q + 128 (j / 256 % 2), so step 2q + 1's state
// (B[2q], B[2q + 1]) runs through every (a, b) over 512 tiles
Mat make_mat(int k, int n, int k2, bool crafted) {
    Mat m;
    const int ld = n / 16, W = 4 * k2;
    m.tr.resize((size_t) (k / 16) * ld * W);
    if (!crafted) {
        for (auto& w : m.tr) w = rnd();
    } else {
        for (size_t j = 0; j < m.tr.size() / (size_t) W; ++j) {
            uint8_t B[256];
            for (int q = 0; q < 128; ++q) {
                B[2 * q] = (uint8_t) (j % 256);
                B[2 * q + 1] = (uint8_t) (q + 128 * (j / 256 % 2));
            }
            for (int w = 0; w < W; ++w)
                m.tr[j * (size_t) W + (size_t) w] = (uint32_t) B[4 * w] << 24 | (uint32_t) B[4 * w + 1] << 16 |
                                                    (uint32_t) B[4 * w + 2] << 8 | (uint32_t) B[4 * w + 3];
        }
    }
    m.suh.resize((size_t) k);
    m.svh.resize((size_t) n);
    for (auto& v : m.suh) v = (uint16_t) (0x3800 | (rnd() & 0x3ff) | ((rnd() & 1u) << 15));
    for (auto& v : m.svh) v = (uint16_t) (0x3800 | (rnd() & 0x3ff) | ((rnd() & 1u) << 15));
    m.s.resize((size_t) k * n);
    m.seen.assign(65536, 0);
    for (int ks = 0; ks < k / 16; ++ks)
        for (int t = 0; t < ld; ++t) {
            const uint32_t* tile = m.tr.data() + ((size_t) ks * ld + t) * W;
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) {
                    const uint32_t st = model_state(tile, k2, kSteps.p[r][c]);
                    m.seen[st] = 1;
                    m.s[(size_t) (ks * 16 + r) * n + t * 16 + c] = byte_sum(st);
                }
        }
    m.p.trellis = m.tr.data();
    m.p.suh = m.suh.data();
    m.p.svh = m.svh.data();
    m.p.k = k;
    m.p.n = n;
    m.p.k2 = k2;
    return m;
}

// the vector tiers' sum for column col over slices [ks0, ks1)
float model(const Mat& m, const float* xh, int col, int ks0, int ks1) {
    float a = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks)
        for (int r = 0; r < 16; ++r)
            a = std::fma((float) m.s[(size_t) (ks * 16 + r) * m.p.n + col], xh[ks * 16 + r], a);
    return a;
}

std::string kname(int k2) { return (k2 & 1) ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2); }

constexpr float kPoison = std::numeric_limits<float>::quiet_NaN();
size_t differ(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(a + i, b + i, sizeof(float)) != 0;
    return d;
}

struct Tally {
    size_t cases = 0, values = 0, vs_avx2 = 0, vs_model = 0;
    void add(const Tally& o) {
        cases += o.cases;
        values += o.values;
        vs_avx2 += o.vs_avx2;
        vs_model += o.vs_model;
    }
};

const int kNs[] = {1, 2, 3, 4, 5, 6, 7, 8, 11};
constexpr int kMaxNT = 11;

// one bitrate on random words: the four paths
void test_random(int k2, int k, int n, bool exact_model, bool tier4, Tally (&tot)[4]) {
    Mat m = make_mat(k, n, k2, false);
    const int S = k / 16;
    std::vector<std::vector<float>> xh((size_t) kMaxNT, std::vector<float>((size_t) k));
    for (auto& v : xh)
        for (auto& x : v) x = rnd_x();
    const float* xp[kMaxNT];
    for (int i = 0; i < kMaxNT; ++i) xp[i] = xh[(size_t) i].data();
    const int ranges[3][2] = {{0, S}, {3, S - 1}, {S / 2, S / 2 + 1}};
    Tally t[4];   // band, rows, band_acc(_n), gemm

    // the model over every column and range, once
    std::vector<float> want[3];
    for (int g = 0; g < 3; ++g) {
        want[g].resize((size_t) kMaxNT * n);
        for (int i = 0; i < kMaxNT; ++i)
            for (int c = 0; c < n; ++c) want[g][(size_t) i * n + c] = model(m, xp[i], c, ranges[g][0], ranges[g][1]);
    }

    // 1. bands: tier 4 against tier 1 and the model
    {
        std::vector<float> g1((size_t) kMaxNT * 128), g4((size_t) kMaxNT * 128), w((size_t) kMaxNT * 128);
        float *p1[kMaxNT], *p4[kMaxNT];
        for (int i = 0; i < kMaxNT; ++i) p1[i] = g1.data() + (size_t) i * 128, p4[i] = g4.data() + (size_t) i * 128;
        bool refused = false;
        for (int c0 = 0; c0 < n && !refused; c0 += 128)
            for (int g = 0; g < 3 && !refused; ++g)
                for (int nt : kNs) {
                    std::fill(g1.begin(), g1.end(), kPoison);
                    std::fill(g4.begin(), g4.end(), kPoison);
                    const bool r1 = cpu::exl3_cpu_band_acc_on(1, m.p, xp, nt, c0, ranges[g][0], ranges[g][1], p1);
                    const bool r4 = cpu::exl3_cpu_band_acc_on(4, m.p, xp, nt, c0, ranges[g][0], ranges[g][1], p4);
                    if (r1 != tier4 || r4 != tier4) {   // (tier 4 must run exactly where tier 1 does)
                        std::printf("  K%s band: tier 1 ran %d, tier 4 ran %d, expected both %d  FAIL\n",
                                    kname(k2).c_str(), (int) r1, (int) r4, (int) tier4);
                        ++g_fail;
                        refused = true;
                        break;
                    }
                    if (!tier4) continue;
                    for (int i = 0; i < nt; ++i)
                        for (int j = 0; j < 128; ++j) w[(size_t) i * 128 + j] = want[g][(size_t) i * n + c0 + j];
                    const size_t nv = (size_t) nt * 128;
                    ++t[0].cases;
                    t[0].values += nv;
                    t[0].vs_avx2 += differ(g4.data(), g1.data(), nv);
                    t[0].vs_model += differ(g4.data(), w.data(), nv) + differ(g1.data(), w.data(), nv);
                }
    }

    // 2. split-K rows (the DS4 lane's call): the switch on against off, column runs, untouched columns outside
    {
        const int runs[3][2] = {{0, n}, {128, n - 128}, {n / 2, n / 2 + 128}};
        std::vector<float> off((size_t) kMaxNT * n), on((size_t) kMaxNT * n), w((size_t) kMaxNT * n);
        float *po[kMaxNT], *pn[kMaxNT];
        for (int i = 0; i < kMaxNT; ++i) po[i] = off.data() + (size_t) i * n, pn[i] = on.data() + (size_t) i * n;
        for (int g = 0; g < 3; ++g)
            for (const auto& run : runs)
                for (int nt : kNs) {
                    std::fill(off.begin(), off.end(), kPoison);
                    std::fill(on.begin(), on.end(), kPoison);
                    std::fill(w.begin(), w.end(), kPoison);
                    for (int i = 0; i < nt; ++i)
                        for (int c = run[0]; c < run[1]; ++c) w[(size_t) i * n + c] = want[g][(size_t) i * n + c];
                    cpu::exl3_cpu_set_w16(false);
                    cpu::exl3_cpu_rows_acc_cols_n(m.p, xp, nt, ranges[g][0], ranges[g][1], run[0], run[1], po);
                    cpu::exl3_cpu_set_w16(true);
                    cpu::exl3_cpu_rows_acc_cols_n(m.p, xp, nt, ranges[g][0], ranges[g][1], run[0], run[1], pn);
                    cpu::exl3_cpu_set_w16(false);
                    const size_t nv = (size_t) kMaxNT * n;
                    ++t[1].cases;
                    t[1].values += (size_t) nt * (size_t) (run[1] - run[0]);
                    t[1].vs_avx2 += differ(on.data(), off.data(), nv);
                    if (exact_model) t[1].vs_model += differ(on.data(), w.data(), nv) + differ(off.data(), w.data(), nv);
                }
        // the wrappers the lane calls for whole rows: exl3_cpu_rows_acc (one token), exl3_cpu_rows_acc_n
        for (int nt : {1, 3}) {
            std::fill(off.begin(), off.end(), kPoison);
            std::fill(on.begin(), on.end(), kPoison);
            cpu::exl3_cpu_set_w16(true);
            if (nt == 1) cpu::exl3_cpu_rows_acc(m.p, xp[0], 0, S, pn[0]);
            else cpu::exl3_cpu_rows_acc_n(m.p, xp, nt, 0, S, pn);
            cpu::exl3_cpu_set_w16(false);
            if (nt == 1) cpu::exl3_cpu_rows_acc(m.p, xp[0], 0, S, po[0]);
            else cpu::exl3_cpu_rows_acc_n(m.p, xp, nt, 0, S, po);
            ++t[1].cases;
            t[1].values += (size_t) nt * n;
            t[1].vs_avx2 += differ(on.data(), off.data(), (size_t) kMaxNT * n);
            if (exact_model) t[1].vs_model += differ(on.data(), want[0].data(), (size_t) nt * n);
        }
    }

    // 3. exl3_cpu_band_acc / exl3_cpu_band_acc_n (the tier this CPU runs, the switch on and off)
    {
        std::vector<float> off((size_t) kMaxNT * 128), on((size_t) kMaxNT * 128), w((size_t) kMaxNT * 128);
        float *po[kMaxNT], *pn[kMaxNT];
        for (int i = 0; i < kMaxNT; ++i) po[i] = off.data() + (size_t) i * 128, pn[i] = on.data() + (size_t) i * 128;
        for (int c0 = 0; c0 < n; c0 += 128)
            for (int g = 0; g < 3; ++g)
                for (int nt : kNs) {
                    std::fill(off.begin(), off.end(), kPoison);
                    std::fill(on.begin(), on.end(), kPoison);
                    for (int pass = 0; pass < 2; ++pass) {
                        cpu::exl3_cpu_set_w16(pass == 1);
                        float* const* ap = pass == 1 ? pn : po;
                        if (nt == 1) cpu::exl3_cpu_band_acc(m.p, xp[0], c0, ranges[g][0], ranges[g][1], ap[0]);
                        else cpu::exl3_cpu_band_acc_n(m.p, xp, nt, c0, ranges[g][0], ranges[g][1], ap);
                    }
                    cpu::exl3_cpu_set_w16(false);
                    for (int i = 0; i < nt; ++i)
                        for (int j = 0; j < 128; ++j) w[(size_t) i * 128 + j] = want[g][(size_t) i * n + c0 + j];
                    const size_t nv = (size_t) nt * 128;
                    ++t[2].cases;
                    t[2].values += nv;
                    t[2].vs_avx2 += differ(on.data(), off.data(), (size_t) kMaxNT * 128);
                    if (exact_model) t[2].vs_model += differ(on.data(), w.data(), nv) + differ(off.data(), w.data(), nv);
                }
    }

    // 4. the band GEMM (every input; nb bands from c0)
    {
        const size_t ws = cpu::exl3_cpu_gemm_scratch();
        std::vector<float> wbuf(ws + 16);
        float* wc = wbuf.data();
        while (((uintptr_t) wc & 63u) != 0) ++wc;
        std::vector<float> off((size_t) kMaxNT * n), on((size_t) kMaxNT * n), w((size_t) kMaxNT * n);
        float *po[kMaxNT], *pn[kMaxNT];
        const int bands[2][2] = {{0, n / 128}, {128, 2}};
        for (const auto& bb : bands)
            for (int nt : kNs) {
                const int c0 = bb[0], nb = bb[1];
                for (int i = 0; i < kMaxNT; ++i) po[i] = off.data() + (size_t) i * n, pn[i] = on.data() + (size_t) i * n;
                std::fill(off.begin(), off.end(), kPoison);
                std::fill(on.begin(), on.end(), kPoison);
                cpu::exl3_cpu_set_w16(false);
                cpu::exl3_cpu_band_gemm(m.p, xp, nt, c0, nb, po, wc);
                cpu::exl3_cpu_set_w16(true);
                cpu::exl3_cpu_band_gemm(m.p, xp, nt, c0, nb, pn, wc);
                cpu::exl3_cpu_set_w16(false);
                std::fill(w.begin(), w.end(), kPoison);
                for (int i = 0; i < nt; ++i)
                    for (int c = 0; c < 128 * nb; ++c) w[(size_t) i * n + c] = want[0][(size_t) i * n + c0 + c];
                ++t[3].cases;
                t[3].values += (size_t) nt * 128 * nb;
                t[3].vs_avx2 += differ(on.data(), off.data(), (size_t) kMaxNT * n);
                if (exact_model)
                    t[3].vs_model += differ(on.data(), w.data(), (size_t) kMaxNT * n) +
                                     differ(off.data(), w.data(), (size_t) kMaxNT * n);
            }
    }

    const char* what[4] = {"band tier 4 vs 1", "rows on vs off", "band_acc on vs off", "gemm on vs off"};
    for (int i = 0; i < 4; ++i) {
        const bool bad = t[i].vs_avx2 != 0 || t[i].vs_model != 0;
        if (t[i].cases == 0) {
            std::printf("  K%-3s %-18s not run (no tier 4 on this CPU)\n", kname(k2).c_str(), what[i]);
            continue;
        }
        std::printf("  K%-3s %-18s %4zu cases, %9zu values: %zu differ from the 32-bit path, %s  %s\n",
                    kname(k2).c_str(), what[i], t[i].cases, t[i].values, t[i].vs_avx2,
                    exact_model || i == 0 ? (std::to_string(t[i].vs_model) + " from the model").c_str()
                                          : "the model not compared (scalar tier: no FMA)",
                    bad ? "FAIL" : "ok");
        if (bad) ++g_fail;
        tot[i].add(t[i]);
    }
}

// every state's byte sum through the kernels, one-hot activations (a single slice: fma(s, 1, 0) = s, exact on every
// tier, the scalar one too)
void test_states(bool tier4) {
    const int k = 256, n = 512, k2 = 16;   // 16 x 32 = 512 tiles
    Mat m = make_mat(k, n, k2, true);
    size_t present = 0;
    for (uint8_t v : m.seen) present += v;
    std::printf("  states: %zu of 65536 present in the K 8 matrix  %s\n", present, present == 65536 ? "ok" : "FAIL");
    if (present != 65536) ++g_fail;
    std::vector<std::vector<float>> oh(16, std::vector<float>((size_t) k));
    const float* xp[8];
    std::vector<float> got((size_t) 8 * n);
    float* gp[8];
    // 0..2: band tiers 0, 1, 4; 3, 4: rows with the switch off, on
    size_t cmp[5] = {0, 0, 0, 0, 0}, bad[5] = {0, 0, 0, 0, 0};
    bool ran[5] = {false, false, false, false, false};
    for (int ks = 0; ks < k / 16; ++ks)
        for (int nt : {1, 3, 8})
            for (int r0 = 0; r0 < 16; r0 += nt) {
                const int c = std::min(nt, 16 - r0);
                for (int i = 0; i < c; ++i) {
                    std::fill(oh[(size_t) i].begin(), oh[(size_t) i].end(), 0.0f);
                    oh[(size_t) i][(size_t) (ks * 16 + r0 + i)] = 1.0f;
                    xp[i] = oh[(size_t) i].data();
                }
                for (int path = 0; path < 5; ++path) {
                    std::fill(got.begin(), got.end(), kPoison);
                    bool done = true;
                    if (path < 3) {
                        const int tier = path == 0 ? 0 : path == 1 ? 1 : 4;
                        for (int c0 = 0; c0 < n && done; c0 += 128) {
                            for (int i = 0; i < c; ++i) gp[i] = got.data() + (size_t) i * n + c0;
                            done = cpu::exl3_cpu_band_acc_on(tier, m.p, xp, c, c0, ks, ks + 1, gp);
                        }
                    } else {
                        for (int i = 0; i < c; ++i) gp[i] = got.data() + (size_t) i * n;
                        cpu::exl3_cpu_set_w16(path == 4);
                        cpu::exl3_cpu_rows_acc_cols_n(m.p, xp, c, ks, ks + 1, 0, n, gp);
                        cpu::exl3_cpu_set_w16(false);
                    }
                    if (!done) continue;
                    ran[path] = true;
                    for (int i = 0; i < c; ++i)
                        for (int col = 0; col < n; ++col) {
                            const float want = (float) m.s[(size_t) (ks * 16 + r0 + i) * n + col];
                            ++cmp[path];
                            bad[path] += std::memcmp(&got[(size_t) i * n + col], &want, sizeof(float)) != 0;
                        }
                }
            }
    const char* names[5] = {"band tier 0", "band tier 1", "band tier 4", "rows w16 off", "rows w16 on"};
    for (int path = 0; path < 5; ++path) {
        if (!ran[path]) {
            std::printf("  states: %-12s not run (not on this CPU)\n", names[path]);
            if (path == 1 || path == 2) {
                if (tier4) ++g_fail;   // (AVX2 here: tiers 1 and 4 must run)
            }
            continue;
        }
        std::printf("  states: %-12s %8zu byte sums read back, %zu differ from the scalar sum  %s\n", names[path],
                    cmp[path], bad[path], bad[path] == 0 ? "ok" : "FAIL");
        if (bad[path] != 0) ++g_fail;
    }
}

}  // namespace

int main() {
    const int tier = cpu::exl3_cpu_tier();
    std::printf("exl3_cpu_w16_test: EXL3 CPU tier %d (%s)\n", tier, tier_name(tier));
    // the switch: off by default; on, it reports itself on only where the lane runs the AVX2 tier
    const bool w0 = cpu::exl3_cpu_w16();
    cpu::exl3_cpu_set_w16(true);
    const bool w1 = cpu::exl3_cpu_w16();
    cpu::exl3_cpu_set_w16(false);
    const bool w2 = cpu::exl3_cpu_w16();
    const bool sw_ok = !w0 && w1 == (tier == 1) && !w2;
    std::printf("  switch: default %d, set %d, cleared %d (expected 0, %d, 0)  %s\n", (int) w0, (int) w1, (int) w2,
                (int) (tier == 1), sw_ok ? "ok" : "FAIL");
    if (!sw_ok) ++g_fail;
    // tier 4 runs exactly where tier 1 does
    float probe[128];
    float* pp[1] = {probe};
    std::vector<float> z(16, 0.0f);
    std::vector<uint32_t> tz(4 * 4 * 8, 0);
    cpu::Exl3Part pz;
    pz.trellis = tz.data();
    pz.k = 16;
    pz.n = 128;
    pz.k2 = 4;
    const float* zp[1] = {z.data()};
    const bool t1 = cpu::exl3_cpu_band_acc_on(1, pz, zp, 1, 0, 0, 1, pp);
    const bool t4 = cpu::exl3_cpu_band_acc_on(4, pz, zp, 1, 0, 0, 1, pp);
    const bool t5 = cpu::exl3_cpu_band_acc_on(5, pz, zp, 1, 0, 0, 1, pp);
    std::printf("  tier 1 here %d, tier 4 here %d, tier 5 refused %d  %s\n", (int) t1, (int) t4, (int) !t5,
                t1 == t4 && !t5 ? "ok" : "FAIL");
    if (t1 != t4 || t5) ++g_fail;
    // the rows, the GEMM and exl3_cpu_band_acc take this CPU's tier: exact against the model on the vector tiers (the
    // scalar tier adds without FMA)
    const bool exact = tier >= 1;
    if (tier != 1)
        std::printf("  (the lane runs the %s tier here: the switch is a no-op for the rows, the GEMM and band_acc - "
                    "MAYA_FORCE_AVX2=1 makes an AVX-512 host take AVX2)\n", tier_name(tier));
    Tally tot[4];
    // (tier 4 expected wherever tier 1 runs)
    for (int k2 : {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}) test_random(k2, 512, 512, exact, t1, tot);
    test_states(t1);
    std::printf("exl3_cpu_w16_test: band %zu cases / rows %zu / band_acc %zu / gemm %zu, %zu values; tier %s: %s\n",
                tot[0].cases, tot[1].cases, tot[2].cases, tot[3].cases,
                tot[0].values + tot[1].values + tot[2].values + tot[3].values, tier_name(tier),
                g_fail == 0 ? "ok" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
