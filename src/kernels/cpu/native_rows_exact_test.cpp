// src/kernels/cpu/native_rows_exact_test.cpp - the multi-token expert rows (iq_avx512 / iq_avx2) and the one-token
// rows (iq_avx2's rows1, iq_avx512_one) against ggml-cpu's own single-token dot, BIT FOR BIT, for every format they
// take and 1-11 tokens; then GLM's split-base rows with
// several tokens against the same rows one token at a time (what a verify window computes vs the token path).
// Synthetic rows (random codes, small normal block scales), Q8_K activations from ggml's quantizer.  CPU only.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx512_one.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace cpu = strata::kernels::cpu;
using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);

int failures = 0;

std::vector<uint8_t> rand_rows(int type, int n, int rows, std::mt19937& rng) {
    const ggml_type t = (ggml_type) type;
    const size_t rb = ggml_row_size(t, n), bb = ggml_type_size(t);
    std::vector<uint8_t> w(rb * rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> ds(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(ds(rng));
        if (t == GGML_TYPE_IQ1_M) {
            // IQ1_M stores the FP16 scale in the high nibbles of its last four words.
            const uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x2400, 0xa400, 0x7bff};
            const uint16_t scale = scales[(o / bb) % 8];
            for (int j = 0; j < 4; ++j) {
                uint16_t word;
                std::memcpy(&word, &w[o + 48 + 2 * j], sizeof word);
                word = (word & 0x0fff) | (((scale >> (4 * j)) & 15) << 12);
                std::memcpy(&w[o + 48 + 2 * j], &word, sizeof word);
            }
        } else {
            std::memcpy(&w[o], &d, sizeof d);
        }
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(ds(rng));
            std::memcpy(&w[o + sizeof d], &m, sizeof m);
        }
    }
    return w;
}

std::vector<uint8_t> rand_acts(int n, int nt, std::mt19937& rng) {
    const auto* q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> a(ab * nt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n);
    for (int k = 0; k < nt; ++k) {
        for (auto& v : x) v = nd(rng) * (rng() % 64 == 0 ? 20.0f : 1.0f);   // some outliers, like real activations
        q8k->from_float(x.data(), a.data() + ab * k, n);
    }
    return a;
}

// fn's rows for nt tokens vs ggml's vec_dot per (row, token): the count of results that differ in any bit
int compare(RowsFn fn, int type, int n, int rows, int nt, std::mt19937& rng, int act_shift = 0) {
    const auto w = rand_rows(type, n, rows, rng);
    auto a = rand_acts(n, nt, rng);
    // Scale after quantization: preserve valid codes and integer sums without overflowing the quantizer's inverse.
    if (act_shift != 0) {
        for (size_t offset = 0; offset < a.size(); offset += ggml_type_size(GGML_TYPE_Q8_K)) {
            float scale;
            std::memcpy(&scale, a.data() + offset, sizeof scale);
            scale = std::ldexp(scale, act_shift);
            std::memcpy(a.data() + offset, &scale, sizeof scale);
        }
    }
    const size_t rb = ggml_row_size((ggml_type) type, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<const void*> ap(nt);
    std::vector<float> got((size_t) nt * rows), ref((size_t) nt * rows);
    std::vector<float*> op(nt);
    for (int k = 0; k < nt; ++k) {
        ap[k] = a.data() + ab * k;
        op[k] = got.data() + (size_t) k * rows;
    }
    fn(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows);
    const auto dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    int bad = 0;
    for (int k = 0; k < nt; ++k)
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            dot(n, &s, 0, w.data() + (size_t) r * rb, 0, ap[k], 0, 1);
            if (std::memcmp(&s, &got[(size_t) k * rows + r], sizeof s) != 0) {
                if (bad == 0)
                    std::printf("    first mismatch: row %d token %d: ggml %.9g vs %.9g\n", r, k, s, got[(size_t) k * rows + r]);
                ++bad;
            }
        }
    return bad;
}

void check_kernel(const char* name, RowsFn fn, void (*set_q5k)(bool) noexcept, int type, std::mt19937& rng) {
    // Q5_K: the min term's form that matches this build's ggml-cpu (as native_expert.cpp picks it)
    if (type == GGML_TYPE_Q5_K) {   // (306 results, as the engine's probe: a form off in ~7% of them shows)
        set_q5k(true);
        int off = 0;
        for (int nt : {2, 5, 11}) off += compare(fn, type, 2048, 17, nt, rng);
        if (off != 0) {
            set_q5k(false);
            std::printf("  %s q5_K: ggml-cpu's min term is unfused in this build\n", name);
        }
    }
    int bad = 0, total = 0;
    const std::vector<int> widths = type == GGML_TYPE_IQ1_M ? std::vector<int>{256, 512, 2048, 2304, 4096, 5120}
                                                         : std::vector<int>{2048, 4096};
    for (int n : widths)
        for (int nt = 1; nt <= 11; ++nt) {
            bad += compare(fn, type, n, 17, nt, rng);
            total += 17 * nt;
        }
    std::printf("  %-7s %-8s %5d results, %d differ from ggml-cpu's dot\n", name, ggml_type_name((ggml_type) type), total,
                bad);
    if (bad != 0) ++failures;
    if (type == GGML_TYPE_Q5_K) set_q5k(true);
}

// Tiny Q8_K scales expose rounding of IQ1_M's final delta correction at the float subnormal boundary.
void check_iq1m_tiny_activations() {
    std::mt19937 rng(20261024);
    int bad = 0, total = 0;
    for (int n : {256, 2304, 5120})
        for (int nt : {1, 4, 6, 11})
            for (int shift : {-96, -112, -120, -128, -136}) {
                bad += compare(cpu::iq256_rows, GGML_TYPE_IQ1_M, n, 17, nt, rng, shift);
                bad += compare(cpu::iq256_rows1, GGML_TYPE_IQ1_M, n, 17, nt, rng, shift);
                total += 2 * nt * 17;
            }
    std::printf("  IQ1_M tiny activations: %d results, %d differ from ggml\n", total, bad);
    if (bad != 0) ++failures;
}

// Padded source strides and partial/empty output ranges must not touch adjacent rows.
void check_iq1m_row_bounds(std::mt19937& rng) {
    constexpr int rows = 7;
    constexpr float guard = -12345.25f;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_M)->vec_dot;
    for (int n : {256, 2304, 5120}) {
        const size_t rb = ggml_row_size(GGML_TYPE_IQ1_M, n), stride = rb + 16;
        const auto packed = rand_rows(GGML_TYPE_IQ1_M, n, rows, rng);
        std::vector<uint8_t> weights(stride * rows, 0xa5);
        for (int r = 0; r < rows; ++r)
            std::memcpy(weights.data() + stride * r, packed.data() + rb * r, rb);
        for (int nt : {1, 4, 6, 11}) {
            const auto acts = rand_acts(n, nt, rng);
            const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
            std::vector<const void*> ap(nt);
            std::vector<float*> op(nt);
            std::vector<float> got((size_t) nt * rows, guard), expected(got);
            for (int t = 0; t < nt; ++t) {
                ap[t] = acts.data() + ab * t;
                op[t] = got.data() + rows * t;
                for (int r = 2; r < 5; ++r)
                    dot(n, &expected[(size_t) t * rows + r], 0, weights.data() + stride * r, 0, ap[t], 0, 1);
            }
            cpu::iq256_rows(GGML_TYPE_IQ1_M, weights.data(), stride, n, ap.data(), nt, op.data(), 2, 5);
            cpu::iq256_rows(GGML_TYPE_IQ1_M, weights.data(), stride, n, ap.data(), nt, op.data(), 3, 3);
            if (std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) != 0) {
                std::printf("  IQ1_M row bounds differ at width %d tokens %d\n", n, nt);
                ++failures;
            }
        }
    }
}

// GLM's split rows: nt tokens at once vs each token alone (the token path), bit for bit, swiglu clamp 10
void check_split(int type, std::mt19937& rng) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(type, type, 4096, 2048, f, err)) {
        std::printf("  split %s: %s\n", ggml_type_name((ggml_type) type), err.c_str());
        ++failures;
        return;
    }
    const auto gate = rand_rows(type, 4096, 2048, rng), up = rand_rows(type, 4096, 2048, rng),
               down = rand_rows(type, 2048, 4096, rng);
    const int nt = 6;
    const auto a = rand_acts(4096, nt, rng), h = rand_acts(2048, nt, rng);
    std::vector<const void*> ap(nt), hp(nt);
    std::vector<float> ffw((size_t) nt * 2048), ff1((size_t) nt * 2048), dw((size_t) nt * 4096), d1((size_t) nt * 4096);
    std::vector<float*> fw(nt), dwp(nt);
    for (int k = 0; k < nt; ++k) {
        ap[k] = a.data() + f.act_bytes * k;
        hp[k] = h.data() + f.h_bytes * k;
        fw[k] = ffw.data() + (size_t) k * 2048;
        dwp[k] = dw.data() + (size_t) k * 4096;
    }
    // the window's slices as the CPU lane cuts them: a few row ranges
    for (int r0 = 0; r0 < 2048; r0 += 683) cpu::native_gu_rows_split(f, gate.data(), up.data(), ap.data(), nt, fw.data(), r0, std::min(2048, r0 + 683), 10.0f);
    for (int r0 = 0; r0 < 4096; r0 += 1500) cpu::native_down_rows_split(f, down.data(), hp.data(), nt, dwp.data(), r0, std::min(4096, r0 + 1500));
    for (int k = 0; k < nt; ++k) {
        float* o1 = ff1.data() + (size_t) k * 2048;
        float* p1 = d1.data() + (size_t) k * 4096;
        cpu::native_gu_rows_split(f, gate.data(), up.data(), &ap[k], 1, &o1, 0, 2048, 10.0f);
        cpu::native_down_rows_split(f, down.data(), &hp[k], 1, &p1, 0, 4096);
    }
    const bool gu_same = std::memcmp(ffw.data(), ff1.data(), ffw.size() * sizeof(float)) == 0;
    const bool dn_same = std::memcmp(dw.data(), d1.data(), dw.size() * sizeof(float)) == 0;
    std::printf("  split   %-8s window of %d vs one token at a time: gate/up %s, down %s (multi-token rows %s)\n",
                ggml_type_name((ggml_type) type), nt, gu_same ? "identical" : "DIFFER", dn_same ? "identical" : "DIFFER",
                cpu::native_multi_rows_exact(type) ? "on" : "off");
    if (!gu_same || !dn_same) ++failures;
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261013);
    const int types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
                         GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ1_M};
    const bool avx512 = cpu::cpu_avx512bw_ok(), avx2 = cpu::cpu_avx2_ok();
    std::printf("native_rows_exact_test: AVX-512BW %s, AVX2 %s\n", avx512 ? "yes" : "no", avx2 ? "yes" : "no");
    for (int t : types) {
        if (avx512 && cpu::iq512_supported(t)) check_kernel("avx512", cpu::iq512_rows, cpu::iq512_set_q5k_fused, t, rng);
        if (avx2) check_kernel("avx2", cpu::iq256_rows, cpu::iq256_set_q5k_fused, t, rng);
        if (avx2 && cpu::iq256_rows1_supported(t)) check_kernel("avx2-1t", cpu::iq256_rows1, cpu::iq256_set_q5k_fused, t, rng);
        if (cpu::iq512_one_supported(t)) check_kernel("avx512-1t", cpu::iq512_one_rows, cpu::iq512_set_q5k_fused, t, rng);
    }
    if (avx2) {
        check_iq1m_tiny_activations();
        check_iq1m_row_bounds(rng);
    }
    for (int t : types) check_split(t, rng);
    if (failures != 0) {
        std::printf("native_rows_exact_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("native_rows_exact_test: all identical\n");
    return 0;
}
