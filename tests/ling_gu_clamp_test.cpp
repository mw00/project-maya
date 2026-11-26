// tests/ling_gu_clamp_test.cpp - maya-ling's CPU lane: the gate/up rows with a NEGATIVE swiglu limit (the post-SiLU clamp
// Ling's GGUF asks for: h = min(silu(g), L) * clamp(u, -L, L), L = -limit) against ggml-cpu's own dot, bit for bit.
// src/ds4/ds4_iq_cpu.cpp is built here with MAYA_LING_SWIGLU, as the maya-ling target builds it (every other engine builds
// it without; the same rows with limit > 0 and limit == 0 are checked too, they must keep their own meaning).
// Every ggml type a Ling GGUF's experts use, 1 to 5 tokens a call, whole matrices and ranges that start mid-way.  No GPU.
#include "../src/ds4/ds4_gguf_cpu.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = maya::ds4::gguf;

namespace {

float dot(int type, int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu((ggml_type) type)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// the expected value of one (gate, up) pair for a limit
float want_of(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = std::fmin(g, limit);
        u = std::fmin(std::fmax(u, -limit), limit);
        return (g / (1.f + std::exp(-g))) * u;
    }
    if (limit < 0.0f) {
        const float L = -limit;
        return std::fmin(g / (1.f + std::exp(-g)), L) * std::fmin(std::fmax(u, -L), L);
    }
    return (g / (1.f + std::exp(-g))) * u;
}

std::vector<uint8_t> quant_rows(int type, int rows, int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> src((size_t) rows * n), im((size_t) n, 1.0f);
    for (auto& v : src) v = nd(rng) * 0.05f;
    std::vector<uint8_t> w((size_t) rows * ggml_row_size((ggml_type) type, n));
    ggml_quantize_chunk((ggml_type) type, src.data(), w.data(), 0, rows, n, im.data());
    return w;
}

// one type, one geometry; the number of results that differ
int check(int type, int E, int FF, int nt, int r0, int r1, float limit, const std::vector<uint8_t>& gate,
          const std::vector<uint8_t>& up, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(type, type, E, FF, f, err)) return -1;   // (no ggml-cpu dot for the type: not run)
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> a(f.act_bytes * (size_t) nt);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> x((size_t) E);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (t % 3 == 2 ? 8.0f : 1.0f);   // (some tokens with large values: the clamp bites)
        gg::iq_cpu_quant(f.gu_act, x.data(), E, a.data() + f.act_bytes * (size_t) t);
        ap[(size_t) t] = a.data() + f.act_bytes * (size_t) t;
    }
    std::vector<float> ff((size_t) nt * FF, -7.0f);
    std::vector<float*> ffp((size_t) nt);
    for (int t = 0; t < nt; ++t) ffp[(size_t) t] = ff.data() + (size_t) t * FF;
    gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, ffp.data(), r0, r1, limit);
    int bad = 0, clamped = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < FF; ++r) {
            float want = -7.0f;   // (rows outside [r0, r1) untouched)
            if (r >= r0 && r < r1) {
                const float g = dot(type, E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                const float u = dot(type, E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                want = want_of(g, u, limit);
                if (want != (g / (1.f + std::exp(-g))) * u) ++clamped;
            }
            const float got = ff[(size_t) t * FF + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 3)
                std::printf("  %s %dx%d nt=%d limit=%g token %d row %d: %.9g, ggml %.9g\n", ggml_type_name((ggml_type) type), E,
                            FF, nt, (double) limit, t, r, (double) got, (double) want);
        }
    (void) clamped;
    return bad;
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261126u);
    int bad = 0, cases = 0, skipped = 0;
    const int types[] = {GGML_TYPE_Q8_0,   GGML_TYPE_Q4_K,    GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K,    GGML_TYPE_Q2_K,
                         GGML_TYPE_Q3_K,   GGML_TYPE_IQ4_NL,  GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
                         GGML_TYPE_IQ2_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,  GGML_TYPE_IQ1_S,   GGML_TYPE_IQ1_M};
    for (const int type : types) {
        int ran = 0;
        const std::vector<uint8_t> gate = quant_rows(type, 256, 1024, rng), up = quant_rows(type, 256, 1024, rng);   // (once a type)
        for (int nt : {1, 2, 3, 5})
            for (const float limit : {-6.0f, -0.05f, 6.0f, 0.0f}) {
                int r = check(type, 1024, 256, nt, 0, 256, limit, gate, up, rng);   // whole matrices
                if (r < 0) break;
                bad += r;
                bad += check(type, 1024, 256, nt, 37, 37 + 100, limit, gate, up, rng);   // a range mid-way (the pool's jobs)
                cases += 2;
                ++ran;
            }
        if (ran == 0) {
            ++skipped;
            std::printf("ling_gu_clamp_test: %s: no ggml-cpu dot here, not run\n", ggml_type_name((ggml_type) type));
        }
    }
    std::printf("ling_gu_clamp_test: %d cases (%d types not run), %d results differ from ggml-cpu's dot - %s\n", cases,
                skipped, bad, bad == 0 && cases > 0 ? "OK" : "FAILED");
    return bad == 0 && cases > 0 ? 0 : 1;
}
