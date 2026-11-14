// tests/ds4_iq2_gu_test.cpp - the CPU lane's IQ2_XXS gate / up rows (the fused pair pass, DS4_IQ1_FGU,
// and the two-pass form behind it) against ggml-cpu's dot bit for bit: random IQ2_XXS gate and up rows of
// DeepSeek-V4.1-Flash's geometry, 1..9 tokens, whole matrices and a mid-range row job, both clamped and not.
// No GPU.  DS4_IQ1_FGU=0 runs the two-pass form; =1 the fused one; both must give ggml's rows exactly.
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

namespace gg = strata::ds4::gguf;

namespace {

std::vector<uint8_t> rand_rows(int n, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size(GGML_TYPE_IQ2_XXS, n), bb = ggml_type_size(GGML_TYPE_IQ2_XXS);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {
        float f = sc(rng);
        const uint16_t hf = (uint16_t) ggml_fp32_to_fp16(f);
        uint16_t d = hf;
        std::memcpy(&w[o], &d, sizeof d);
    }
    return w;
}

float dot(int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XXS)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// one geometry: gate / up rows of IQ2_XXS, nt tokens, ranges [r0, r1); the differing float count
int check(int E, int FF, int nt, int r0, int r1, float limit, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, E, FF, f, err)) {
        std::printf("FAIL: iq_cpu_fmt %d x %d: %s\n", E, FF, err.c_str());
        return 1;
    }
    const std::vector<uint8_t> gate = rand_rows(E, FF, rng), up = rand_rows(E, FF, rng);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> a(f.act_bytes * (size_t) nt);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> x((size_t) E);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (t % 3 == 2 ? 30.0f : 1.0f);
        gg::iq_cpu_quant(f.gu_act, x.data(), E, a.data() + f.act_bytes * (size_t) t);
        ap[(size_t) t] = a.data() + f.act_bytes * (size_t) t;
    }
    std::vector<float> ffv((size_t) nt * FF, -7.0f);
    std::vector<float*> ffp((size_t) nt);
    for (int t = 0; t < nt; ++t) ffp[(size_t) t] = ffv.data() + (size_t) t * FF;
    gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, ffp.data(), r0, r1, limit);
    int bad = 0;
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < FF; ++r) {
            float want = -7.0f;
            if (r >= r0 && r < r1) {
                float g = dot(E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                float u = dot(E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                if (limit > 0.0f) {
                    g = std::fmin(g, limit);
                    u = std::fmin(std::fmax(u, -limit), limit);
                }
                want = (g / (1.f + std::exp(-g))) * u;
            }
            const float got = ffv[(size_t) t * FF + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 4)
                std::printf("  gate/up %dx%d nt=%d token %d row %d: %.9g, ggml %.9g\n", E, FF, nt, t, r, got, want);
        }
    }
    return bad;
}

}  // namespace

int main() {
    ggml_cpu_init();
    const char* fenv = std::getenv("DS4_IQ1_FGU");
    std::printf("ds4_iq2_gu_test: DS4_IQ1_FGU=%s, the IQ2_XXS pair pass %s (own IQ2 rows %s)\n",
                fenv != nullptr ? fenv : "(unset)", gg::iq_cpu_own_gu_fused() ? "fused" : "two passes",
                gg::iq_cpu_own_rows(GGML_TYPE_IQ2_XXS) ? "on" : "off");
    std::mt19937 rng(77271110u);
    int bad = 0, cases = 0;
    struct Geo {
        int E, FF;
    };
    for (const Geo g : {Geo{5120, 2304}, Geo{1024, 512}})
        for (int nt : {1, 2, 3, 4, 5, 7, 8, 9}) {
            bad += check(g.E, g.FF, nt, 0, g.FF, nt % 2 ? 10.0f : 0.0f, rng);   // whole matrices
            bad += check(g.E, g.FF, nt, 37, 37 + 64, 0.0f, rng);                // a mid-range job's rows
            cases += 2;
        }
    std::printf("ds4_iq2_gu_test: %d cases, %d results differ from ggml-cpu's dot - %s\n", cases, bad,
                bad == 0 ? "OK" : "FAILED");
    return bad == 0 ? 0 : 1;
}
