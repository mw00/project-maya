// tests/ds4_iq1_rows_test.cpp - the CPU lane's IQ1_S rows (src/ds4/ds4_iq_cpu.cpp: iq_cpu_gate_up / iq_cpu_down, as
// the engine calls them) against ggml-cpu's own dot, bit for bit: every row of random IQ1_S experts (any bit pattern is
// a valid block: 11-bit grid indices, a 3-bit scale and the delta's sign), DeepSeek-V4.1-Flash's geometry and a smaller
// one, 1 to 16 tokens a call, whole matrices and row ranges that start mid-way.  No GPU.
//
//   DS4_IQ1_PDEP=0 ds4_iq1_rows_test   the rows without pdep, on any CPU (the form AMD Zen / Zen+ / Zen 2 take by default)
//   DS4_IQ1_PDEP=1 ds4_iq1_rows_test   the measured choice (ggml-cpu's dot for one token, pdep where it times faster)
#include "../src/ds4/ds4_gguf_cpu.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;

namespace {

std::vector<uint8_t> rand_rows(int n, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size(GGML_TYPE_IQ1_S, n), bb = ggml_type_size(GGML_TYPE_IQ1_S);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {   // (the F16 scale: block_iq1_s's first field)
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[o], &d, sizeof d);
    }
    return w;
}

float dot(int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// one geometry: gate / up / down rows of IQ1_S, nt tokens, rows [r0, r1) of each; the number of results that differ
int check(int E, int FF, int nt, int r0g, int r1g, int r0d, int r1d, float limit, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, E, FF, f, err)) {
        std::printf("FAIL: iq_cpu_fmt %d x %d: %s\n", E, FF, err.c_str());
        return 1;
    }
    const std::vector<uint8_t> gate = rand_rows(E, FF, rng), up = rand_rows(E, FF, rng), down = rand_rows(FF, E, rng);
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
                float g = dot(E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                float u = dot(E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
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
            const float want = r >= r0d && r < r1d ? dot(FF, down.data() + (size_t) r * f.d_row, hp[(size_t) t]) : -7.0f;
            const float got = out[(size_t) t * E + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 4)
                std::printf("  down %dx%d nt=%d token %d row %d: %.9g, ggml %.9g\n", E, FF, nt, t, r, got, want);
        }
    }
    return bad;
}

}  // namespace

int main() {
    ggml_cpu_init();
    const char* e = std::getenv("DS4_IQ1_PDEP");
    std::printf("ds4_iq1_rows_test: DS4_IQ1_PDEP=%s, own IQ1_S rows %s\n", e != nullptr ? e : "(unset)",
                gg::iq_cpu_own_rows(GGML_TYPE_IQ1_S) ? "on" : "off");
    const bool want_own = e != nullptr && std::atoi(e) == 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (want_own && !gg::iq_cpu_own_rows(GGML_TYPE_IQ1_S)) {
        std::printf("FAIL: DS4_IQ1_PDEP=0 but the CPU lane does not take its own IQ1_S rows\n");
        return 1;
    }
#endif
    std::mt19937 rng(20261025u);
    int bad = 0, cases = 0;
    struct Geo {
        int E, FF;
    };
    for (const Geo g : {Geo{4096, 2048}, Geo{1024, 512}})
        for (int nt : {1, 2, 3, 4, 5, 7, 8, 9, 16}) {
            bad += check(g.E, g.FF, nt, 0, g.FF, 0, g.E, nt % 2 ? 10.0f : 0.0f, rng);   // whole matrices
            bad += check(g.E, g.FF, nt, 37, 37 + 64, 5, 5 + 129, 0.0f, rng);           // ranges mid-way (the pool's jobs)
            cases += 2;
        }
    std::printf("ds4_iq1_rows_test: %d cases, %d results differ from ggml-cpu's dot - %s\n", cases, bad,
                bad == 0 ? "OK" : "FAILED");
    return bad == 0 ? 0 : 1;
}
