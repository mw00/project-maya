// tests/ds4_iq512_rows_test.cpp - the CPU lane's exact AVX-512 rows without VBMI (src/ds4/ds4_iq512_kern.inc: IQ2_XS,
// IQ3_XXS, IQ1_M) against ggml-cpu's own dot, bit for bit, through the lane's calls as the engine makes them
// (iq_cpu_rows, iq_cpu_gate_up): random blocks at the experts' widths and others, 1 to 16 tokens a call, whole
// matrices and row ranges that start mid-way (the rows outside untouched), block scales at their edges (zero, -0,
// subnormal, the largest F16) and activations scaled down until the sums are subnormal (IQ1_M's last FMA).  One
// thread, no GPU.
//
//   DS4_IQ512=1 ds4_iq512_rows_test       the rows forced on: on a CPU with AVX-512 F / BW / VL / DQ and BMI2 the
//                                         lane must take them for all three types
//   ds4_iq512_rows_test                   this CPU's default choice (whichever rows serve, ggml-cpu's bits)
//   ds4_iq512_rows_test --bench           single-thread timing of each type's rows on the path this run takes (1, 2
//                                         and 4 tokens, 64-row calls over ~96 MB of weights); run it with
//                                         DS4_IQ512=1, DS4_IQ512=0 and DS4_IQ_OWN=0 to compare the rows; --bench --glm
//                                         also times the GLM lane's AVX-512 multi-token rows against its AVX2 ones
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
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

const int kTypes[] = {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_M};

void put_scale(int type, uint8_t* blk, uint16_t d) {   // (IQ1_M: the F16 in the top nibbles of its 4 scale words)
    if (type != GGML_TYPE_IQ1_M) {
        std::memcpy(blk, &d, 2);
        return;
    }
    for (int j = 0; j < 4; ++j) {
        uint16_t sw;
        std::memcpy(&sw, blk + 48 + 2 * j, 2);
        sw = (uint16_t) ((sw & 0x0fffu) | (((unsigned) d >> (4 * j)) & 15u) << 12);
        std::memcpy(blk + 48 + 2 * j, &sw, 2);
    }
}

// rows of random blocks: every other block's scale one of the edge values, the rest small normal values
std::vector<uint8_t> rand_rows(int type, int n, int rows, std::mt19937& rng, bool edges = true) {
    static const uint16_t edge[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x2400, 0xa400, 0x7bff};
    const size_t rb = ggml_row_size((ggml_type) type, n), bb = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0, k = 0; o < w.size(); o += bb, ++k)
        put_scale(type, &w[o], edges && k % 2 == 0 ? edge[(k / 2) % 8] : ggml_fp32_to_fp16(sc(rng)));
    return w;
}

// nt tokens of n values as Q8_K (ggml's quantizer), some with outliers; shift: their scales times 2^shift after
std::vector<uint8_t> rand_acts(int n, int nt, std::mt19937& rng, int shift = 0) {
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> a(ab * (size_t) nt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (rng() % 64 == 0 ? 20.0f : 1.0f) * (t % 3 == 2 ? 30.0f : 1.0f);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data() + ab * (size_t) t, n);
    }
    if (shift != 0)
        for (size_t o = 0; o < a.size(); o += ggml_type_size(GGML_TYPE_Q8_K)) {   // (block_q8_K: the F32 d first)
            float d;
            std::memcpy(&d, &a[o], 4);
            d = std::ldexp(d, shift);
            std::memcpy(&a[o], &d, 4);
        }
    return a;
}

float dot(int type, int n, const uint8_t* row, const void* act) {
    float s = 0.0f;
    ggml_get_type_traits_cpu((ggml_type) type)->vec_dot(n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// iq_cpu_rows for rows [r0, r1) of `rows` rows, nt tokens: the results that differ from ggml-cpu's dot in any bit
int check_rows(int type, int n, int rows, int nt, int r0, int r1, int shift, std::mt19937& rng) {
    const auto w = rand_rows(type, n, rows, rng);
    const auto a = rand_acts(n, nt, rng, shift);
    const size_t rb = ggml_row_size((ggml_type) type, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> got((size_t) nt * rows, -7.0f);
    std::vector<float*> op((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + ab * (size_t) t;
        op[(size_t) t] = got.data() + (size_t) t * rows;
    }
    gg::iq_cpu_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), r0, r1);
    int bad = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r) {
            const float want = r >= r0 && r < r1 ? dot(type, n, w.data() + (size_t) r * rb, ap[(size_t) t]) : -7.0f;
            const float g = got[(size_t) t * rows + r];
            if (std::memcmp(&g, &want, 4) != 0 && bad++ < 3)
                std::printf("  %s n=%d nt=%d shift=%d token %d row %d: %.9g, ggml %.9g\n", ggml_type_name((ggml_type) type),
                            n, nt, shift, t, r, g, want);
        }
    return bad;
}

// iq_cpu_gate_up (the swiglu epilogue on the rows' values, clamped or not) against ggml's dots
int check_gate_up(int type, int E, int FF, int nt, int r0, int r1, float limit, std::mt19937& rng) {
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(type, type, E, FF, f, err)) {
        std::printf("FAIL: iq_cpu_fmt %s %d x %d: %s\n", ggml_type_name((ggml_type) type), E, FF, err.c_str());
        return 1;
    }
    const auto gate = rand_rows(type, E, FF, rng, false), up = rand_rows(type, E, FF, rng, false);
    const auto a = rand_acts(E, nt, rng);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> ff((size_t) nt * FF, -7.0f);
    std::vector<float*> fp((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + f.act_bytes * (size_t) t;
        fp[(size_t) t] = ff.data() + (size_t) t * FF;
    }
    gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, fp.data(), r0, r1, limit);
    int bad = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < FF; ++r) {
            float want = -7.0f;
            if (r >= r0 && r < r1) {
                float g = dot(type, E, gate.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                float u = dot(type, E, up.data() + (size_t) r * f.gu_row, ap[(size_t) t]);
                if (limit > 0.0f) {
                    g = std::fmin(g, limit);
                    u = std::fmin(std::fmax(u, -limit), limit);
                }
                want = (g / (1.f + std::exp(-g))) * u;
            }
            const float got = ff[(size_t) t * FF + r];
            if (std::memcmp(&got, &want, 4) != 0 && bad++ < 3)
                std::printf("  gate/up %s nt=%d token %d row %d: %.9g, ggml %.9g\n", ggml_type_name((ggml_type) type), nt,
                            t, r, got, want);
        }
    return bad;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --bench: one type's rows on this run's path, one thread, nt tokens, 64-row calls (the lane's jobs) over ~96 MB of
// rows (more than the caches); the best of 7 passes, as GB/s of weights and ns a row and token
using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
void bench(int type, int nt, RowsFn fn = nullptr, const char* name = nullptr) {
    const int n = 4096;
    const size_t rb = ggml_row_size((ggml_type) type, n);
    const int rows = (int) (96e6 / (double) rb) / 64 * 64;
    std::mt19937 rng(7u + (unsigned) type);
    const auto w = rand_rows(type, n, rows, rng, false);
    const auto a = rand_acts(n, nt, rng);
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<const void*> ap((size_t) nt);
    std::vector<float> out((size_t) nt * rows);
    std::vector<float*> op((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        ap[(size_t) t] = a.data() + ab * (size_t) t;
        op[(size_t) t] = out.data() + (size_t) t * rows;
    }
    auto pass = [&] {
        const double t0 = now_us();
        for (int r0 = 0; r0 < rows; r0 += 64) {
            if (fn != nullptr) fn(type, w.data(), rb, n, ap.data(), nt, op.data(), r0, r0 + 64);
            else gg::iq_cpu_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), r0, r0 + 64);
        }
        return now_us() - t0;
    };
    pass();
    double best = 1e30;
    for (int k = 0; k < 7; ++k) best = std::min(best, pass());
    std::printf("%s %-7s %d token%s: %7.2f GB/s of weights, %6.1f ns a row and token (%d rows of %d, own rows %s)\n",
                name != nullptr ? name : "bench", ggml_type_name((ggml_type) type), nt, nt > 1 ? "s" : " ",
                (double) rows * (double) rb / best / 1e3, best * 1e3 / rows / nt, rows, n,
                gg::iq_cpu_own_rows(type) ? "on" : "off");
}

// AVX-512 F / BW / VL / DQ and BMI2 with the OS saving the state (the rows' own check: ds4_iq_cpu.cpp's iq_isa)
bool cpu_x512() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    const char* e = std::getenv("DS4_IQ512");
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        std::printf("ds4_iq512_rows_test --bench: DS4_IQ512=%s DS4_IQ_OWN=%s, one thread\n", e != nullptr ? e : "(unset)",
                    std::getenv("DS4_IQ_OWN") != nullptr ? std::getenv("DS4_IQ_OWN") : "(unset)");
        for (int type : kTypes)
            for (int nt : {1, 2, 4}) bench(type, nt);
        // (the GLM lane's multi-token rows, AVX-512 against AVX2, on the same rows: what one thread says about 512-bit
        // integer code on this CPU)
        if (argc > 2 && std::string(argv[2]) == "--glm" && strata::kernels::cpu::cpu_avx512bw_ok())
            for (int type : {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS})
                for (int nt : {2, 4}) {
                    bench(type, nt, strata::kernels::cpu::iq512_rows, "glm-avx512");
                    bench(type, nt, strata::kernels::cpu::iq256_rows, "glm-avx2");
                }
        return 0;
    }
    const bool forced = e != nullptr && std::atoi(e) != 0;
    std::printf("ds4_iq512_rows_test: DS4_IQ512=%s, AVX-512 F/BW/VL/DQ + BMI2 %s; own rows:", e != nullptr ? e : "(unset)",
                cpu_x512() ? "yes" : "no");
    for (int type : kTypes) std::printf(" %s %s", ggml_type_name((ggml_type) type), gg::iq_cpu_own_rows(type) ? "on" : "off");
    std::printf("\n");
    if (forced && cpu_x512())
        for (int type : kTypes)
            if (!gg::iq_cpu_own_rows(type)) {
                std::printf("FAIL: DS4_IQ512=1 on an AVX-512 CPU but the lane does not take the %s rows\n",
                            ggml_type_name((ggml_type) type));
                return 1;
            }
    std::mt19937 rng(20261028u);
    int bad = 0, cases = 0;
    for (int type : kTypes) {
        int tb = 0, tc = 0;
        for (int n : {256, 2048, 2304, 4096, 5120})
            for (int nt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16}) {
                tb += check_rows(type, n, 17, nt, 0, 17, 0, rng);   // whole matrices
                tb += check_rows(type, n, 23, nt, 5, 19, 0, rng);   // a range mid-way
                tc += 2;
            }
        for (int nt : {1, 2, 5})   // (sums near and in the subnormal range: the float steps' rounding there)
            for (int shift : {-96, -112, -120, -128, -136}) {
                tb += check_rows(type, 2304, 17, nt, 0, 17, shift, rng);
                ++tc;
            }
        for (int nt : {1, 3, 4, 6}) {
            tb += check_gate_up(type, 2048, 512, nt, 0, 512, nt % 2 ? 10.0f : 0.0f, rng);
            tb += check_gate_up(type, 2048, 512, nt, 37, 37 + 64, 0.0f, rng);
            tc += 2;
        }
        std::printf("  %-7s %d cases, %d results differ from ggml-cpu's dot\n", ggml_type_name((ggml_type) type), tc, tb);
        bad += tb;
        cases += tc;
    }
    std::printf("ds4_iq512_rows_test: %d cases, %d results differ - %s\n", cases, bad, bad == 0 ? "OK" : "FAILED");
    return bad == 0 ? 0 : 1;
}
