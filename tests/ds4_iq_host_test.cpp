// tests/ds4_iq_host_test.cpp - maya-ds4's i-quant pieces that run on the host, no GPU:
//   - IQ4_NL experts on the CPU lane (iq_cpu_gate_up / iq_cpu_down, as the engine calls them) bit for bit ggml-cpu's
//     own dot, alone and beside other types, DeepSeek-V4.1-Flash's geometry, a tensor-parallel member's (n_ff 1152)
//     and a small one, 1 to 16 tokens a call, whole matrices and row ranges that start mid-way.
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

int g_fail = 0;

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
    test_iq4nl_lane(rng);
    std::printf("ds4_iq_host_test: %s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
