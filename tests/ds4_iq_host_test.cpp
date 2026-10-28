// tests/ds4_iq_host_test.cpp - maya-ds4's i-quant pieces that run on the host, no GPU:
//   - IQ1_M's code table (the persistent device kernels' decode: src/ds4/ds4_iq1_codes.hpp), built here as the device
//     builds it from ggml's grid: each entry against the codes worked out value by value from ggml's int8 grid, then
//     every value of IQ1_M blocks decoded through it (codes, scales, d / 8: the products the kernels take) equal to
//     ggml's own dequantize_row_iq1_m bit for bit - blocks that take every entry of the table, and random ones;
//   - IQ1_S's codes the same way (the grid and the delta's bias the device's IQ1_S decode takes);
//   - IQ4_NL experts on the CPU lane (iq_cpu_gate_up / iq_cpu_down, as the engine calls them) bit for bit ggml-cpu's
//     own dot, alone and beside other types, DeepSeek-V4.1-Flash's geometry, a tensor-parallel member's (n_ff 1152)
//     and a small one, 1 to 16 tokens a call, whole matrices and row ranges that start mid-way.
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_iq1_codes.hpp"

// (the GPU builds' tables - iq1s_grid_gpu, the device kernels' grid; ggml's C / C++ branch has the CPU's iq1s_grid
// instead, and its SYCL branch declares the GPU ones as plain arrays)
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace iq = strata::ds4::gguf::iq;

namespace {

int g_fail = 0;

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
float f16(uint16_t h) { return ggml_fp16_to_fp32((ggml_fp16_t) h); }
void report(const char* what, size_t bad, size_t n) {
    if (bad) ++g_fail;
    std::printf("%-60s %s  %zu of %zu differ\n", what, bad ? "FAIL" : "ok  ", bad, n);
}

// ---------------------------------------------------------------- IQ1_M's code table
// the table as the persistent kernels build it in shared memory (two words an entry)
std::vector<uint32_t> iq1m_table() {
    std::vector<uint32_t> t(2 * (size_t) iq::kIq1mCodes);
    for (int j = 0; j < iq::kIq1mCodes; ++j)
        iq::iq1m_codes(iq1s_grid_gpu[j & 2047], (uint32_t) j >> 11, t[2 * (size_t) j], t[2 * (size_t) j + 1]);
    return t;
}
// code v (0..7) of entry j as the table holds it
int code_at(const std::vector<uint32_t>& t, uint32_t j, int v) {
    return (int8_t) (uint8_t) (t[2 * (size_t) j + (size_t) (v >> 2)] >> (8 * (v & 3)));
}

// each entry against its codes worked out one value at a time: value v of grid entry g is g's nibble v & 3 (the low one
// for v < 4, the high one from 4 on), n = g + 1 in {0, 1, 2}; the code 8 g + 1, or 8 g - 1 with the delta's sign set
void check_entries(const std::vector<uint32_t>& t) {
    size_t bad = 0;
    for (uint32_t j = 0; j < (uint32_t) iq::kIq1mCodes; ++j) {
        const uint32_t e = iq1s_grid_gpu[j & 2047];
        for (int v = 0; v < 8; ++v) {
            const int n = (int) ((e >> (8 * (v & 3) + 4 * (v >> 2))) & 15u);
            bad += n > 2 || code_at(t, j, v) != 8 * (n - 1) + (j >> 11 ? -1 : 1);
        }
    }
    report("IQ1_M code table: every entry = 8 g +- 1 of the grid's values", bad, (size_t) iq::kIq1mCodes * 8);
}

// nb IQ1_M blocks' values through the table, as the persistent kernels take them: a group's 12 bits (iq1m_entry) the
// entry, codes times 2 s + 1 (s0: values 0..15, s1: 16..31) in integers, then times d / 8 in F32 (values_kernel's
// product)
void iq1m_values(const uint8_t* blocks, int nb, const std::vector<uint32_t>& t, float* out) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = blocks + (size_t) b * sizeof(block_iq1_m);
        const uint32_t s01 = rd32(p + 48), s23 = rd32(p + 52);
        const float d = f16(iq::iq1m_d16(s01, s23)) * 0.125f;
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t q4 = rd32(p + 4 * ib), h = rd16(p + 32 + 2 * ib);
            int s0 = 0, s1 = 0;
            iq::iq1m_scales(s01, s23, ib, s0, s1);
            for (int l = 0; l < 4; ++l)
                for (int v = 0; v < 8; ++v) {
                    const int at = 8 * l + v;
                    out[(size_t) b * 256 + 32 * ib + at] = d * (float) ((at < 16 ? s0 : s1) * code_at(t, iq::iq1m_entry(q4, h, l), v));
                }
        }
    }
}
// nb IQ1_S blocks' values as the device decodes them: the grid entry (8 qs bits, 3 of qh), codes iq1c with the
// sub-block's delta bias, times 2 s + 1, times d / 8
void iq1s_values(const uint8_t* blocks, int nb, float* out) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = blocks + (size_t) b * sizeof(block_iq1_s);
        const float d = f16(rd16(p)) * 0.125f;
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t h = rd16(p + 34 + 2 * ib), bias = iq::iq1_bias(h & 0x8000u);
            const int s = 2 * (int) ((h >> 12) & 7u) + 1;
            for (int l = 0; l < 4; ++l) {
                const uint32_t g = iq1s_grid_gpu[p[2 + 4 * ib + l] | (((h >> (3 * l)) & 7u) << 8)];
                const uint32_t c[2] = {(uint32_t) iq::iq1c(g << 3, bias), (uint32_t) iq::iq1c(g >> 1, bias)};
                for (int v = 0; v < 8; ++v)
                    out[(size_t) b * 256 + 32 * ib + 8 * l + v] = d * (float) (s * (int8_t) (uint8_t) (c[v >> 2] >> (8 * (v & 3))));
            }
        }
    }
}

// an IQ1_M block's F16 scale into the top nibbles of its 4 scale words (bytes 48..55)
void iq1m_put_d(uint8_t* p, float d) {
    const uint16_t u = (uint16_t) ggml_fp32_to_fp16(d);
    for (int i = 0; i < 4; ++i) {
        uint16_t w = rd16(p + 48 + 2 * i);
        w = (uint16_t) ((w & 0x0FFFu) | (((u >> (4 * i)) & 0xFu) << 12));
        std::memcpy(p + 48 + 2 * i, &w, 2);
    }
}
float rand_d(std::mt19937& rng, int i) {   // small normal scales of either sign; a zero now and then
    if (i % 97 == 5) return i % 2 ? 0.0f : -0.0f;
    const float d = std::uniform_real_distribution<float>(1e-4f, 0.05f)(rng);
    return i % 3 == 1 ? -d : d;
}

size_t differ(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0 && bad++ < 4)
            std::printf("  %s value %zu: %.9g, ggml %.9g\n", what, i, a[i], b[i]);
    return bad;
}

void test_iq1m(std::mt19937& rng) {
    const std::vector<uint32_t> t = iq1m_table();
    check_entries(t);
    // (1) blocks that take every entry: block b's group (ib, l) is entry 32 b + 4 ib + l - 128 blocks for the 4096 -
    // (2) random blocks; both with random scales and their d set (random bytes there would be NaNs and infinities)
    for (int pass = 0; pass < 2; ++pass) {
        const int nb = pass == 0 ? iq::kIq1mCodes / 32 : 4000;
        std::vector<uint8_t> w((size_t) nb * sizeof(block_iq1_m));
        for (auto& v : w) v = (uint8_t) rng();
        for (int b = 0; b < nb; ++b) {
            uint8_t* p = w.data() + (size_t) b * sizeof(block_iq1_m);
            if (pass == 0) {
                std::memset(p + 32, 0, 16);
                for (int ib = 0; ib < 8; ++ib)
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t e = (uint32_t) (32 * b + 4 * ib + l);
                        p[4 * ib + l] = (uint8_t) (e & 255u);
                        p[32 + 2 * ib + (l >> 1)] |= (uint8_t) (((e >> 8) & 15u) << (4 * (l & 1)));
                    }
            }
            iq1m_put_d(p, rand_d(rng, b));
        }
        std::vector<float> got((size_t) nb * 256), want((size_t) nb * 256);
        iq1m_values(w.data(), nb, t, got.data());
        ggml_get_type_traits(GGML_TYPE_IQ1_M)->to_float(w.data(), want.data(), (int64_t) nb * 256);
        char what[96];
        std::snprintf(what, sizeof what, "IQ1_M through the code table = ggml's (%s, %d blocks)",
                      pass == 0 ? "every entry" : "random", nb);
        report(what, differ(got, want, "IQ1_M"), got.size());
    }
}

void test_iq1s(std::mt19937& rng) {
    const int nb = 4000;
    std::vector<uint8_t> w((size_t) nb * sizeof(block_iq1_s));
    for (auto& v : w) v = (uint8_t) rng();
    for (int b = 0; b < nb; ++b) {
        const uint16_t d = (uint16_t) ggml_fp32_to_fp16(rand_d(rng, b));
        std::memcpy(w.data() + (size_t) b * sizeof(block_iq1_s), &d, 2);
    }
    std::vector<float> got((size_t) nb * 256), want((size_t) nb * 256);
    iq1s_values(w.data(), nb, got.data());
    ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(w.data(), want.data(), (int64_t) nb * 256);
    report("IQ1_S codes (grid, delta bias) = ggml's (random, 4000 blocks)", differ(got, want, "IQ1_S"), got.size());
}

// ---------------------------------------------------------------- IQ4_NL on the CPU lane
std::vector<uint8_t> rand_rows(int type, int k, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, k), bb = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {   // (the F16 scale: the i-quants' first field)
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[o], &d, sizeof d);
    }
    return w;
}
float dot(int type, int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu((ggml_type) type)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// one layer: gate / up of type tg, down of td, nt tokens, rows [r0, r1) of each - the number of results that differ
int check_lane(int tg, int td, int E, int FF, int nt, int r0g, int r1g, int r0d, int r1d, float limit, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(tg, td, E, FF, f, err)) {
        std::printf("FAIL: iq_cpu_fmt %s/%s %d x %d: %s\n", ggml_type_name((ggml_type) tg), ggml_type_name((ggml_type) td),
                    E, FF, err.c_str());
        return 1;
    }
    const std::vector<uint8_t> gate = rand_rows(tg, E, FF, rng), up = rand_rows(tg, E, FF, rng),
                               down = rand_rows(td, FF, E, rng);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> a(f.act_bytes * (size_t) nt), hq(f.h_bytes * (size_t) nt);
    std::vector<const void*> ap((size_t) nt), hp((size_t) nt);
    std::vector<float> x((size_t) E), h((size_t) FF);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (t % 3 == 2 ? 30.0f : 1.0f);   // (some tokens with large values)
        for (auto& v : h) v = nd(rng) * 0.1f;
        gg::iq_cpu_quant(f.gu_act, x.data(), E, a.data() + f.act_bytes * (size_t) t);
        gg::iq_cpu_quant(f.d_act, h.data(), FF, hq.data() + f.h_bytes * (size_t) t);
        ap[(size_t) t] = a.data() + f.act_bytes * (size_t) t;
        hp[(size_t) t] = hq.data() + f.h_bytes * (size_t) t;
    }
    std::vector<float> ff((size_t) nt * FF, -7.0f), out((size_t) nt * E, -7.0f);
    std::vector<float*> ffp((size_t) nt), op((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ffp[(size_t) t] = ff.data() + (size_t) t * FF;
        op[(size_t) t] = out.data() + (size_t) t * E;
    }
    gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, ffp.data(), r0g, r1g, limit);
    gg::iq_cpu_down(f, down.data(), hp.data(), nt, op.data(), r0d, r1d);
    int bad = 0;
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < FF; ++r) {
            float want = -7.0f;   // (rows outside [r0, r1) untouched)
            if (r >= r0g && r < r1g) {
                float g = dot(tg, E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                float u = dot(tg, E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                if (limit > 0.0f) {
                    g = std::fmin(g, limit);
                    u = std::fmin(std::fmax(u, -limit), limit);
                }
                want = (g / (1.f + std::exp(-g))) * u;
            }
            const float got = ff[(size_t) t * FF + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 4)
                std::printf("  gate/up %dx%d nt=%d token %d row %d: %.9g, ggml %.9g\n", E, FF, nt, t, r, got, want);
        }
        for (int r = 0; r < E; ++r) {
            const float want = r >= r0d && r < r1d ? dot(td, FF, down.data() + (size_t) r * f.d_row, hp[(size_t) t]) : -7.0f;
            const float got = out[(size_t) t * E + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 4)
                std::printf("  down %dx%d nt=%d token %d row %d: %.9g, ggml %.9g\n", E, FF, nt, t, r, got, want);
        }
    }
    return bad;
}

void test_iq4nl_lane(std::mt19937& rng) {
    const int act = gg::iq_cpu_act_type(GGML_TYPE_IQ4_NL);
    if (act != GGML_TYPE_Q8_0) {
        std::printf("FAIL: IQ4_NL's CPU-lane activation type %d (ggml's vec_dot_type is Q8_0)\n", act);
        ++g_fail;
        return;
    }
    struct Case {
        int tg, td, E, FF;
        std::vector<int> nts;
    };
    const Case cases[] = {
        {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, 1024, 512, {1, 2, 3, 5, 8, 9, 16}},
        {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, 5120, 2304, {1, 4}},
        {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, 5120, 1152, {1, 3}},   // (a member of two's share of n_ff)
        {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ2_XXS, 1024, 512, {1, 4}},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 1024, 512, {1, 4}},
    };
    for (const Case& c : cases) {
        int bad = 0, n = 0;
        for (int nt : c.nts) {
            bad += check_lane(c.tg, c.td, c.E, c.FF, nt, 0, c.FF, 0, c.E, nt % 2 ? 10.0f : 0.0f, rng);   // whole matrices
            bad += check_lane(c.tg, c.td, c.E, c.FF, nt, 37, 37 + 64, 5, 5 + 129, 0.0f, rng);           // ranges mid-way
            n += 2;
        }
        if (bad) ++g_fail;
        char what[96];
        std::snprintf(what, sizeof what, "CPU lane %s/%s %dx%d = ggml-cpu's dot (%d calls)", ggml_type_name((ggml_type) c.tg),
                      ggml_type_name((ggml_type) c.td), c.E, c.FF, n);
        std::printf("%-60s %s  %d results differ\n", what, bad ? "FAIL" : "ok  ", bad);
    }
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261028u);
    test_iq1m(rng);
    test_iq1s(rng);
    test_iq4nl_lane(rng);
    std::printf("ds4_iq_host_test: %s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
