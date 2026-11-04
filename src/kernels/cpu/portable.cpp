#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx512_one.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

std::atomic<bool> oracle_q8_0{false};

[[noreturn]] void unsupported(const char* what) {
    std::fprintf(stderr,
                 "maya: %s is x86-only (AVX2/AVX-512) and this build has no portable version.\n"
                 "      On this CPU keep every expert in the GPU's expert pool (the CPU expert lane off).\n",
                 what);
    std::abort();
}

}  // namespace

const char* CpuFeatures::reason() const { return usable() ? "ok" : "not an x86 CPU (no AVX-512)"; }

CpuFeatures cpu_features() { return CpuFeatures{}; }

void cpu_require_expert_support() {
    std::fprintf(stderr, "maya: the canonical Q2_0 pack's expert kernel needs AVX512-VNNI/VBMI (x86).\n"
                         "      On this CPU use a native GGUF pack and keep the CPU expert lane off.\n");
    std::exit(1);
}

void expert_set_oracle_q8_0(bool enabled) { oracle_q8_0.store(enabled, std::memory_order_relaxed); }
bool expert_oracle_q8_0_enabled() { return oracle_q8_0.load(std::memory_order_relaxed); }

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { unsupported("the Q2_0 expert kernel"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 expert kernel");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    unsupported("the Q2_0 expert kernel");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 GGUF row kernel");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { unsupported("the Q2_0 scalar oracle"); }

bool iq512_supported(int) noexcept { return false; }
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq512_set_q5k_fused(bool) noexcept {}

bool iq256_supported(int) noexcept { return false; }
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 i-quant kernel");
}
void iq256_set_q5k_fused(bool) noexcept {}
bool iq256_rows1_supported(int) noexcept { return false; }
void iq256_rows1(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 one-token i-quant kernel");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-2 IQ4_NL kernel");
}

bool iq512_one_supported(int) noexcept { return false; }
void iq512_one_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 one-token i-quant kernel");
}

bool kq_avx512_ok() noexcept { return false; }
bool kq_type_ok(int) noexcept { return false; }
size_t kq_act_bytes(int, int) { return 0; }
void kq_pack_act(const void* const*, int, int, void*) { unsupported("the AVX-512 k-quant kernel"); }
void kq_rows_packed(int, const uint8_t*, size_t, int, const void*, int, float* const*, int, int) {
    unsupported("the AVX-512 k-quant kernel");
}
void kq_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 k-quant kernel");
}

bool exl3_cpu_vector() { return false; }
bool exl3_cpu_available() { return false; }
int exl3_cpu_tier() { return 0; }
float exl3_cpu_prepare(const Exl3Part&, const float*, float*) { unsupported("the EXL3 CPU kernel"); }
float exl3_cpu_prepare_block(const uint16_t*, const float*, float*) { unsupported("the EXL3 CPU kernel"); }
void exl3_cpu_band(const Exl3Part&, const float*, float, int, float*) { unsupported("the EXL3 CPU kernel"); }
void exl3_cpu_band_acc(const Exl3Part&, const float*, int, int, int, float*) { unsupported("the EXL3 CPU kernel"); }
void exl3_cpu_band_acc_n(const Exl3Part&, const float* const*, int, int, int, int, float* const*) {
    unsupported("the EXL3 CPU kernel");
}
void exl3_cpu_rows_acc(const Exl3Part&, const float*, int, int, float*) { unsupported("the EXL3 CPU kernel"); }
void exl3_cpu_rows_acc_n(const Exl3Part&, const float* const*, int, int, int, float* const*) {
    unsupported("the EXL3 CPU kernel");
}
void exl3_cpu_rows_acc_cols_n(const Exl3Part&, const float* const*, int, int, int, int, int, float* const*) {
    unsupported("the EXL3 CPU kernel");
}
size_t exl3_cpu_gemm_scratch() { return 0; }
void exl3_cpu_band_gemm(const Exl3Part&, const float* const*, int, int, int, float* const*, float*) {
    unsupported("the EXL3 CPU kernel");
}
bool exl3_cpu_band_acc_on(int, const Exl3Part&, const float* const*, int, int, int, int, float* const*) {
    return false;
}
void exl3_cpu_band_out(const Exl3Part&, const float*, float, int, float*) { unsupported("the EXL3 CPU kernel"); }

}  // namespace strata::kernels::cpu
