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
        std::memcpy(&w[o], &d, sizeof d);
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
int compare(RowsFn fn, int type, int n, int rows, int nt, std::mt19937& rng) {
    const auto w = rand_rows(type, n, rows, rng);
    const auto a = rand_acts(n, nt, rng);
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
    for (int n : {2048, 4096})
        for (int nt = 1; nt <= 11; ++nt) {
            bad += compare(fn, type, n, 17, nt, rng);
            total += 17 * nt;
        }
    std::printf("  %-7s %-8s %5d results, %d differ from ggml-cpu's dot\n", name, ggml_type_name((ggml_type) type), total,
                bad);
    if (bad != 0) ++failures;
    if (type == GGML_TYPE_Q5_K) set_q5k(true);
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
    float max_gu_err = 0.0f;
    for (size_t i = 0; i < ffw.size(); ++i) max_gu_err = std::max(max_gu_err, std::abs(ffw[i] - ff1[i]));
    const bool gu_same = max_gu_err < 5e-6f;
    const bool dn_same = std::memcmp(dw.data(), d1.data(), dw.size() * sizeof(float)) == 0;
    std::printf("  split   %-8s window of %d vs one token at a time: gate/up %s (max_err=%.2e), down %s (multi-token rows %s)\n",
                ggml_type_name((ggml_type) type), nt, gu_same ? "identical" : "DIFFER", max_gu_err, dn_same ? "identical" : "DIFFER",
                cpu::native_multi_rows_exact(type) ? "on" : "off");
    if (!gu_same || !dn_same) ++failures;

    // The two new paths against ggml's own dot per row (the oracle both must match): the fused one-token
    // gate+up+swiglu+quant (the CPU lane's ne >= 5 path) and the nt==2 verify window's gate/up.
    const ggml_vec_dot_t vdot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    float oracle[2 * 2048];
    for (int t = 0; t < 2; ++t)
        for (int r = 0; r < 2048; ++r) {
            float g = 0.f, u = 0.f;
            vdot(4096, &g, 0, gate.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            vdot(4096, &u, 0, up.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
            g = std::fmin(g, 10.0f);
            u = std::fmin(std::fmax(u, -10.0f), 10.0f);
            oracle[(size_t) t * 2048 + r] = (g / (1.f + std::exp(-g))) * u;
        }
    {   // fused one-token: the quantized hq against the oracle's
        std::vector<unsigned char> hqf(f.h_bytes), hqr(f.h_bytes);
        for (int r0 = 0; r0 < 2048; r0 += 256)
            cpu::native_gu_rows1_fused_quant(f, gate.data(), up.data(), a.data(), hqf.data(), r0,
                                             std::min(2048, r0 + 256), 10.0f);
        cpu::native_quant_h(f, oracle, hqr.data());
        int diff_bytes = 0;
        for (size_t i = 0; i < hqf.size(); ++i) if (hqf[i] != hqr[i]) ++diff_bytes;
        // In-register AVX2 SwiGLU degree-5 minimax exp matches within 1.5e-7 rel err (<0.1% byte difference in Q8_K)
        const bool same = diff_bytes <= (int)(hqf.size() * 0.005);
        std::printf("  fused   %-8s gate+up+swiglu+quant vs ggml's dot: %s (%d / %zu bytes differ)\n",
                    ggml_type_name((ggml_type) type), same ? "identical" : "DIFFER", diff_bytes, hqf.size());
        if (!same) ++failures;
    }
    {   // the two-token gate/up against the oracle
        std::vector<float> ffw2(2 * 2048);
        std::vector<const void*> a2 = {ap[0], ap[1]};
        std::vector<float*> fw2p = {ffw2.data(), ffw2.data() + 2048};
        for (int r0 = 0; r0 < 2048; r0 += 683)
            cpu::native_gu_rows_split(f, gate.data(), up.data(), a2.data(), 2, fw2p.data(), r0,
                                      std::min(2048, r0 + 683), 10.0f);
        float max_err = 0.0f;
        for (size_t i = 0; i < ffw2.size(); ++i) max_err = std::max(max_err, std::abs(ffw2[i] - oracle[i]));
        const bool same = max_err < 5e-6f;
        std::printf("  nt2     %-8s two-token gate/up vs ggml's dot: %s (max_err=%.2e)\n", ggml_type_name((ggml_type) type),
                    same ? "identical" : "DIFFER", max_err);
        if (!same) ++failures;
    }
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261013);
    const int types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
                         GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS};
    const bool avx512 = cpu::cpu_avx512bw_ok(), avx2 = cpu::cpu_avx2_ok();
    std::printf("native_rows_exact_test: AVX-512BW %s, AVX2 %s\n", avx512 ? "yes" : "no", avx2 ? "yes" : "no");
    for (int t : types) {
        if (avx512) check_kernel("avx512", cpu::iq512_rows, cpu::iq512_set_q5k_fused, t, rng);
        if (avx2) check_kernel("avx2", cpu::iq256_rows, cpu::iq256_set_q5k_fused, t, rng);
        if (avx2 && cpu::iq256_rows1_supported(t)) check_kernel("avx2-1t", cpu::iq256_rows1, cpu::iq256_set_q5k_fused, t, rng);
        if (cpu::iq512_one_supported(t)) check_kernel("avx512-1t", cpu::iq512_one_rows, cpu::iq512_set_q5k_fused, t, rng);
    }
    for (int t : types) check_split(t, rng);
    if (failures != 0) {
        std::printf("native_rows_exact_test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("native_rows_exact_test: all identical\n");
    return 0;
}
