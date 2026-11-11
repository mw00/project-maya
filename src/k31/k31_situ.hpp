// src/k31/k31_situ.hpp - Kimi K3.1's SiTU-GLU activation, the one place its formula lives in the engine.
//
//     situ(g, u) = beta * tanh(g / beta) * sigmoid(g)  *  lbeta * tanh(u / lbeta)        (beta 4, lbeta 25)
//
// (configuration_kimi_k3: activation_situ_beta / activation_situ_linear_beta; ref/kimi_k31.py situ; lbeta <= 0
// leaves the up branch linear, llama.cpp's convention.)  It sits between the expert gate/up GEMVs and the down GEMV
// in every routed and shared expert and in the dense layer-0 FFN.  Maya's fused IQ gate/up kernels only know SwiGLU
// and swigluoai (docs/K31-ENGINE.md section 1.2, item 3), so through M3 gate and up are computed separately and this
// activation runs on its own.
//
// Two forms, each available on the device and the host:
//   * precise  : expf / tanhf (CUDA's non-fast-math library, the host's libm).  The engine default.
//   * portable : the same bits on the device and on the CPU lane (MAYA_DS4_LANE_PARITY) for every non-NaN output
//                (NaN payloads differ: NVIDIA canonicalises, x86 propagates): sigmoid and tanh built from
//                src/ds4/ds4_par_exp.h's exp_portable and plain _rn operations; tanh below |x| 0.625 uses Cephes'
//                odd polynomial, so small arguments keep their relative accuracy.  Host callers compile it with
//                -ffp-contract=off (the ds4_par_exp.h rule).  Parity also needs flush-to-zero OFF on both sides
//                (these forms make denormals, unlike exp_portable alone): situ_cpu clears MXCSR FTZ/DAZ around its
//                loop on x86; the device side is the no-fast-math build rule.
// Both are gated by tools/kimi_k31_golden.py's SiTU vectors: within the measured budget (10 float32 ULPs of the
// float64 reference) for g >= -87, an absolute 1e-35 below it, where a float32 sigmoid leaves the normal range.
// The shipped forms' adversarial worst is 6.9 ULPs (analytic bound ~7.8), so the budget's headroom is ~1.3x.
// Never compile the files that use these with --use_fast_math: CUDA 12.4 turns expf into ex2.approx and the
// divisions approximate (tanhf is unchanged), which misses the budget by ~5.6x (55.8 ULPs) and flushes denormals,
// breaking host/device parity.  nvcc defines no macro for it, so this is a build rule (CMake), not a #error.
#pragma once

#include "../ds4/ds4_par_exp.h"

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define K31_HD __host__ __device__ __forceinline__
#else
#define K31_HD inline
#endif

#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
#include <cmath>
#endif

namespace strata::k31 {

inline constexpr float kSituBeta = 4.0f;          // activation_situ_beta (the released config)
inline constexpr float kSituLinearBeta = 25.0f;   // activation_situ_linear_beta

// ---- precise
K31_HD float situ_tanh(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return tanhf(x);
#else
    return std::tanh(x);
#endif
}

K31_HD float situ_sigmoid(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return 1.0f / (1.0f + expf(-x));
#else
    return 1.0f / (1.0f + std::exp(-x));
#endif
}

K31_HD float situ(float g, float u, float beta = kSituBeta, float lbeta = kSituLinearBeta) {
    const float a = beta * situ_tanh(g / beta) * situ_sigmoid(g);
    const float b = lbeta > 0.0f ? lbeta * situ_tanh(u / lbeta) : u;
    return a * b;
}

// ---- portable (identical bits on the device and the host)
K31_HD float tanh_portable(float x) {
    namespace par = strata::ds4::par;
    if (x != x) return x;
    const float ax = par::p_max(x, -x);
    float r;
    if (ax > 0.625f) {
        // 1 - 2 / (e^{2|x|} + 1); e^{2|x|} = inf past |x| ~44.4 gives exactly 1
        r = par::p_sub(1.0f, par::p_div(2.0f, par::p_add(par::exp_portable(par::p_mul(2.0f, ax)), 1.0f)));
        return x < 0.0f ? -r : r;
    }
    if (ax == 0.0f) return x;   // keeps the sign of zero
    // Cephes tanhf: x + x * s * P(s), s = x^2
    const float s = par::p_mul(x, x);
    float p = -5.70498872745e-3f;
    p = par::p_fma(p, s, 2.06390887954e-2f);
    p = par::p_fma(p, s, -5.37397155531e-2f);
    p = par::p_fma(p, s, 1.33314422036e-1f);
    p = par::p_fma(p, s, -3.33332819422e-1f);
    return par::p_fma(par::p_mul(p, s), x, x);
}

K31_HD float sigmoid_portable(float x) {
    namespace par = strata::ds4::par;
    return par::p_div(1.0f, par::p_add(1.0f, par::exp_portable(-x)));
}

K31_HD float situ_portable(float g, float u, float beta = kSituBeta, float lbeta = kSituLinearBeta) {
    namespace par = strata::ds4::par;
    const float a = par::p_mul(par::p_mul(beta, tanh_portable(par::p_div(g, beta))), sigmoid_portable(g));
    const float b = lbeta > 0.0f ? par::p_mul(lbeta, tanh_portable(par::p_div(u, lbeta))) : u;
    return par::p_mul(a, b);
}

// ---- entry points
/// out[i] = situ(gate[i], up[i]) for i < n on the device (out may alias gate or up).  portable selects the
/// lane-parity form.  Returns false (and launches nothing) for n < 0 or null pointers with n > 0.
bool situ_launch(const float* gate, const float* up, float* out, int64_t n, float beta, float lbeta, bool portable,
                 void* stream);

/// The same on the host (the CPU lane).  Compiled with -ffp-contract=off so the portable form matches the device.
void situ_cpu(const float* gate, const float* up, float* out, int64_t n, float beta, float lbeta, bool portable);

}  // namespace strata::k31
