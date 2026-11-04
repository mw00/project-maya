// src/ds4/ds4_par_exp.h - the experts' SwiGLU with an exp of plain IEEE operations, the same bits on the device and on
// the CPU lane (DS4_LANE_PARITY, mw00/project-maya#644).  CUDA's expf goes through ex2.approx (the SFU's own
// approximation), which a CPU cannot reproduce; this exp uses only fma, multiply, add and an exact power-of-two scale,
// each rounded once, so a host built with -ffp-contract=off and the device's _rn intrinsics give every value alike.
// Accuracy: within 1 ulp of e^y for y >= -87 (Cephes' expf polynomial after a Cody-Waite reduction done with FMAs);
// below that (e^-87 = 1.6e-38, just above FLT_MIN) the result is 0, so no result is ever denormal and FTZ on either
// side changes nothing; overflow gives +inf, as expf.  (The SwiGLU takes e^-g: g > 87 is far past DeepSeek's clamp.)
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_PAR_HD __host__ __device__ __forceinline__
#else
#define DS4_PAR_HD inline
#endif

#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
#include <cmath>
#include <cstdint>
#include <cstring>
#endif

namespace strata::ds4::par {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
DS4_PAR_HD float p_fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
DS4_PAR_HD float p_mul(float a, float b) { return __fmul_rn(a, b); }
DS4_PAR_HD float p_add(float a, float b) { return __fadd_rn(a, b); }
DS4_PAR_HD float p_sub(float a, float b) { return __fsub_rn(a, b); }
DS4_PAR_HD float p_div(float a, float b) { return __fdiv_rn(a, b); }
DS4_PAR_HD float p_min(float a, float b) { return fminf(a, b); }
DS4_PAR_HD float p_max(float a, float b) { return fmaxf(a, b); }
DS4_PAR_HD float p_pow2(int n) { return __int_as_float((n + 127) << 23); }   // 2^n, -126 <= n <= 127
DS4_PAR_HD float p_inf() { return __int_as_float(0x7f800000); }
#else
// (host: this header's callers are compiled with -ffp-contract=off, so each line below is one IEEE operation)
DS4_PAR_HD float p_fma(float a, float b, float c) { return std::fma(a, b, c); }
DS4_PAR_HD float p_mul(float a, float b) { return a * b; }
DS4_PAR_HD float p_add(float a, float b) { return a + b; }
DS4_PAR_HD float p_sub(float a, float b) { return a - b; }
DS4_PAR_HD float p_div(float a, float b) { return a / b; }
DS4_PAR_HD float p_min(float a, float b) { return std::fmin(a, b); }   // (IEEE minNum, as the device's fminf)
DS4_PAR_HD float p_max(float a, float b) { return std::fmax(a, b); }
DS4_PAR_HD float p_pow2(int n) {
    const uint32_t b = (uint32_t) (n + 127) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
DS4_PAR_HD float p_inf() { return p_pow2(128); }   // (exponent bits all set, mantissa 0)
#endif

/// e^y in plain IEEE operations (see the top of the file).
DS4_PAR_HD float exp_portable(float y) {
    if (!(y < 89.0f)) return y != y ? y : p_inf();   // (NaN stays NaN; past 89: inf. Below it n <= 128, and n = 128's
                                                     //  scale below overflows to inf exactly where e^y rounds past FLT_MAX)
    if (y < -87.0f) return 0.0f;                                   // (e^-87 > FLT_MIN: every result kept is normal)
    // n = rint(y log2 e) (|n| <= 128: t stays in [2^23, 2^24), so the subtraction is exact); r = y - n ln 2
    const float t = p_fma(y, 1.44269504088896341f, 12582912.0f);
    const float n = p_sub(t, 12582912.0f);
    float r = p_fma(n, -0.693359375f, y);
    r = p_fma(n, 2.12194440e-4f, r);
    // e^r = 1 + r + r^2 P(r), |r| <= ln2 / 2 + (Cephes' expf)
    float p = 1.9875691500e-4f;
    p = p_fma(p, r, 1.3981999507e-3f);
    p = p_fma(p, r, 8.3334519073e-3f);
    p = p_fma(p, r, 4.1665795894e-2f);
    p = p_fma(p, r, 1.6666665459e-1f);
    p = p_fma(p, r, 5.0000001201e-1f);
    const float r2 = p_mul(r, r);
    p = p_fma(p, r2, r);
    p = p_add(p, 1.0f);
    const int ni = (int) n;
    return ni > 127 ? p_mul(p_mul(p, p_pow2(127)), 2.0f) : p_mul(p, p_pow2(ni));
}

/// DeepSeek's clamped SwiGLU (limit 0: no clamp), the device kernels' operations with exp_portable.
DS4_PAR_HD float swiglu_portable(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = p_min(gv, limit);
        uv = p_min(p_max(uv, -limit), limit);
    }
    return p_mul(p_div(gv, p_add(1.0f, exp_portable(-gv))), uv);
}

}  // namespace strata::ds4::par
