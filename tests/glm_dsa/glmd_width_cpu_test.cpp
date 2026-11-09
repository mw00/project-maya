// tests/glm_dsa/glmd_width_cpu_test.cpp - B2.1 / B5.2 (code part): the routing ring's width limits and the CPU
// lane's native experts at glm-dsa's n_embd 6144.  CPU only (no device): the host-mapped structs' layout and
// limits, native_fmt's activation caps at 6144 for the expert types a UD-IQ2_M pack carries, and one CPU-lane
// call (8 synthetic experts, gate/up then down then the weighted sum, as fast_cpu_experts runs it) against a
// double-precision reference.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace cpu = strata::kernels::cpu;

// the structs the device writes and the host reads: wide enough for 6144, and the residual vectors are float4-aligned
static_assert(gf::kMaxEmbd >= 6144, "glm-dsa's n_embd 6144 must fit the routing ring");
static_assert(gf::kMaxEmbd == 8192, "B2.1 fixes the ring width at 8192");
static_assert(gf::kRingSize == 256, "B2.5: 256 ring entries");
static_assert(sizeof(gf::MoeRequest::x) / sizeof(float) == (size_t) gf::kMaxEmbd, "MoeRequest.x");
static_assert(sizeof(gf::MoeResponse::cpu_part) / sizeof(float) == (size_t) gf::kMaxEmbd, "MoeResponse.cpu_part");
static_assert(sizeof(gf::CpuAnswer::part) / sizeof(float) == (size_t) gf::kMaxEmbd, "CpuAnswer.part");
static_assert(offsetof(gf::CpuAnswer, part) % 16 == 0, "CpuAnswer.part 16-byte aligned");
static_assert(offsetof(gf::MoeRequest, x) % 4 == 0 && sizeof(gf::MoeRequest) % 8 == 0, "MoeRequest layout");

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "glmd_width_cpu_test: FAIL %s\n", what);
        ++g_fail;
    }
}

// a Q8_K activation (ggml's block_q8_K: float d, int8 qs[256], int16 bsums[16]) back to floats
static std::vector<double> dequant_q8k(const uint8_t* p, int n) {
    std::vector<double> out((size_t) n);
    constexpr size_t kBlock = 4 + 256 + 32;
    for (int b = 0; b < n / 256; ++b) {
        float d;
        std::memcpy(&d, p + (size_t) b * kBlock, 4);
        const int8_t* q = (const int8_t*) (p + (size_t) b * kBlock + 4);
        for (int i = 0; i < 256; ++i) out[(size_t) b * 256 + i] = (double) d * q[i];
    }
    return out;
}

static std::vector<float> dequant_rows(int type, const uint8_t* p, int rows, int n_per_row) {
    std::vector<float> out((size_t) rows * n_per_row);
    const size_t rb = ggml_row_size((ggml_type) type, n_per_row);
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    for (int r = 0; r < rows; ++r) tr->to_float(p + (size_t) r * rb, out.data() + (size_t) r * n_per_row, n_per_row);
    return out;
}

static double rel_l2(const std::vector<double>& ref, const float* got, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0.0 ? den : 1.0));
}

static double silu(double g) { return g / (1.0 + std::exp(-g)); }

// One CPU-lane call over ne experts of (gu_type, d_type) at n_embd x n_ff, the arithmetic of fast_cpu_experts
// (split gate/up base, h quantized per expert, down rows, out[r] = sum_i w_i down_i[r]).
static void lane_call(int gu_type, int d_type, int n_embd, int n_ff, int ne, uint32_t seed) {
    char tag[96];
    std::snprintf(tag, sizeof tag, "%s/%s %d x %d", ggml_type_name((ggml_type) gu_type),
                  ggml_type_name((ggml_type) d_type), n_embd, n_ff);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, n_embd, n_ff, f, err)) {
        std::fprintf(stderr, "glmd_width_cpu_test: %s: %s\n", tag, err.c_str());
        check(false, "native_fmt");
        return;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> imat((size_t) std::max(n_embd, n_ff), 1.0f);   // the i-quants need an importance matrix
    const float* im_gu = ggml_quantize_requires_imatrix((ggml_type) gu_type) ? imat.data() : nullptr;
    const float* im_d = ggml_quantize_requires_imatrix((ggml_type) d_type) ? imat.data() : nullptr;
    std::vector<std::vector<uint8_t>> blobs((size_t) ne, std::vector<uint8_t>(f.bytes));
    std::vector<float> wsrc;
    for (int i = 0; i < ne; ++i) {
        wsrc.resize((size_t) 2 * n_ff * n_embd);
        for (auto& v : wsrc) v = nd(rng) * 0.05f;
        ggml_quantize_chunk((ggml_type) gu_type, wsrc.data(), blobs[(size_t) i].data(), 0, 2 * n_ff, n_embd, im_gu);
        wsrc.resize((size_t) n_embd * n_ff);
        for (auto& v : wsrc) v = nd(rng) * 0.05f;
        ggml_quantize_chunk((ggml_type) d_type, wsrc.data(), blobs[(size_t) i].data() + f.down_off, 0, n_embd, n_ff,
                            im_d);
    }
    std::vector<float> x((size_t) n_embd), w((size_t) ne);
    for (auto& v : x) v = nd(rng);
    for (int i = 0; i < ne; ++i) w[(size_t) i] = 0.05f + 0.1f * (float) i;
    // the lane, as fast_cpu_experts calls it
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq((size_t) ne * cpu::kNativeHBytes);
    std::vector<float> ff((size_t) ne * n_ff), dn((size_t) ne * n_embd), out((size_t) n_embd);
    cpu::native_quant_act(f, x.data(), act.data());
    const void* a[1] = {act.data()};
    for (int i = 0; i < ne; ++i) {
        float* o[1] = {ff.data() + (size_t) i * n_ff};
        cpu::native_gu_rows_split(f, blobs[(size_t) i].data(), blobs[(size_t) i].data() + f.up_off, a, 1, o, 0, n_ff,
                                  std::numeric_limits<float>::infinity());   // glm-dsa: no SwiGLU clamp
        cpu::native_quant_h(f, ff.data() + (size_t) i * n_ff, hq.data() + (size_t) i * cpu::kNativeHBytes);
        const void* h[1] = {hq.data() + (size_t) i * cpu::kNativeHBytes};
        float* od[1] = {dn.data() + (size_t) i * n_embd};
        cpu::native_down_rows_split(f, blobs[(size_t) i].data() + f.down_off, h, 1, od, 0, n_embd);
    }
    for (int r = 0; r < n_embd; ++r) {
        float m = 0.0f;
        for (int i = 0; i < ne; ++i) m += w[(size_t) i] * dn[(size_t) i * n_embd + r];
        out[(size_t) r] = m;
    }
    // reference 1 (the dot products themselves): double over the dequantized weights and the lane's own quantized
    // activations - only the summation order differs, so it must agree to float rounding
    const std::vector<double> aq = dequant_q8k(act.data(), n_embd);
    std::vector<double> ff_ref((size_t) ne * n_ff), out_q((size_t) n_embd, 0.0), out_f((size_t) n_embd, 0.0);
    for (int i = 0; i < ne; ++i) {
        const std::vector<float> gu = dequant_rows(gu_type, blobs[(size_t) i].data(), 2 * n_ff, n_embd);
        const std::vector<float> dw = dequant_rows(d_type, blobs[(size_t) i].data() + f.down_off, n_embd, n_ff);
        std::vector<double> hf((size_t) n_ff);
        for (int r = 0; r < n_ff; ++r) {
            double g = 0.0, u = 0.0, gx = 0.0, ux = 0.0;
            for (int e = 0; e < n_embd; ++e) {
                g += (double) gu[(size_t) r * n_embd + e] * aq[(size_t) e];
                u += (double) gu[(size_t) (n_ff + r) * n_embd + e] * aq[(size_t) e];
                gx += (double) gu[(size_t) r * n_embd + e] * x[(size_t) e];
                ux += (double) gu[(size_t) (n_ff + r) * n_embd + e] * x[(size_t) e];
            }
            ff_ref[(size_t) i * n_ff + r] = silu(g) * u;
            hf[(size_t) r] = silu(gx) * ux;
        }
        const std::vector<double> hqd =
            f.d_act == (int) GGML_TYPE_Q8_K ? dequant_q8k(hq.data() + (size_t) i * cpu::kNativeHBytes, n_ff)
                                            : std::vector<double>();
        for (int r = 0; r < n_embd; ++r) {
            double sq = 0.0, sf = 0.0;
            for (int c = 0; c < n_ff; ++c) {
                if (!hqd.empty()) sq += (double) dw[(size_t) r * n_ff + c] * hqd[(size_t) c];
                sf += (double) dw[(size_t) r * n_ff + c] * hf[(size_t) c];
            }
            out_q[(size_t) r] += (double) w[(size_t) i] * sq;
            out_f[(size_t) r] += (double) w[(size_t) i] * sf;
        }
    }
    const double e_ff = rel_l2(ff_ref, ff.data(), ff.size());
    const double e_q = f.d_act == (int) GGML_TYPE_Q8_K ? rel_l2(out_q, out.data(), out.size()) : 0.0;
    // reference 2 (the whole expert): F32 activations end to end; the lane's Q8_K activations round (8 bits a value)
    const double e_f = rel_l2(out_f, out.data(), out.size());
    std::printf("glmd_width_cpu_test: %s, %d experts: gate/up rel L2 %.2e, down (same h quant) %.2e, vs F32-activation "
                "reference %.2e\n", tag, ne, e_ff, e_q, e_f);
    check(e_ff < 1e-5, "gate/up dot vs double (same activation quant)");
    check(e_q < 1e-5, "down + weighted sum vs double (same h quant)");
    check(e_f < 5e-2, "expert vs double with F32 activations");   // ~2e-2 measured (Gaussian x, 8-bit steps)
}

int main() {
    // B2.1: the limits moe_route and fast_setup refuse
    check(gf::moe_limits_error(4096, 8, 256) == nullptr, "glm5-next geometry fits");
    check(gf::moe_limits_error(6144, 8, 256) == nullptr, "glm-dsa geometry fits");
    check(gf::moe_limits_error(8192, 8, 512) == nullptr, "the ring's maxima fit");
    check(gf::moe_limits_error(8193, 8, 256) != nullptr, "n_embd > kMaxEmbd refused");
    check(gf::moe_limits_error(0, 8, 256) != nullptr, "n_embd 0 refused");
    check(gf::moe_limits_error(6144, 9, 256) != nullptr, "k > 8 refused");
    check(gf::moe_limits_error(6144, 0, 256) != nullptr, "k 0 refused");
    check(gf::moe_limits_error(6144, 8, 513) != nullptr, "n_expert > 512 refused");
    // the ring at 8192: 256 entries x ~33 KB host-mapped (was 64 x ~17 KB)
    std::printf("glmd_width_cpu_test: MoeRequest %zu B, ring %zu KB, MoeResponse %zu B, CpuAnswer %zu B\n",
                sizeof(gf::MoeRequest), sizeof(gf::MoeRequest) * gf::kRingSize / 1024, sizeof(gf::MoeResponse),
                sizeof(gf::CpuAnswer));

    // B5.2: native_fmt's caps at 6144 (kNativeActBytes 8192 holds 24 Q8_K blocks = 7008 B; n_ff 2048 h = 2336 B)
    const int types[][2] = {{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS}, {GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K},
                            {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S},     {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q3_K},
                            {GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS},   {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},
                            {GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K},     {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
                            {GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0}};
    for (const auto& t : types) {
        cpu::NativeFmt f;
        std::string err;
        const bool ok = cpu::native_fmt(t[0], t[1], 6144, 2048, f, err);
        if (!ok) std::fprintf(stderr, "glmd_width_cpu_test: %s/%s at 6144: %s\n", ggml_type_name((ggml_type) t[0]),
                              ggml_type_name((ggml_type) t[1]), err.c_str());
        check(ok && f.act_bytes <= cpu::kNativeActBytes && f.h_bytes <= cpu::kNativeHBytes, "6144 x 2048 fits");
    }
    {
        cpu::NativeFmt f;
        std::string err;
        check(cpu::native_fmt(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 6144, 2048, f, err) && f.act_bytes == 24 * 292,
              "Q8_K activation at 6144 is 7008 B");
    }

    // B5.2: one CPU-lane call at 6144: k-quants at the real n_ff, then the i-quants a UD-IQ2_M pack routes
    lane_call(GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 6144, 2048, 8, 1u);
    lane_call(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, 6144, 256, 2, 2u);
    lane_call(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K, 6144, 256, 2, 3u);
    if (g_fail != 0) {
        std::fprintf(stderr, "glmd_width_cpu_test: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::puts("glmd_width_cpu_test: ok");
    return 0;
}
