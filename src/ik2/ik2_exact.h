// src/ik2/ik2_exact.h - maya-ik2's "exact lane" arithmetic for a routed expert: the same bits on the CPU and on the
// GPU, so an answer does not depend on where an expert ran (the offload path's lane-parity mode, IK2_LANE).
//
//   * a weight decoded to an exact F32 (NVFP4: a code times its sub-block's E4M3 scale; Q8_0: q times the block's
//     FP16 d - both products exact in F32);
//   * a row's dot product as fused multiply-adds in one fixed order: the row in sub-blocks of 16 values, sub-block s
//     summed into chain s % 4 lane by lane (16 lanes, value j of a sub-block in lane j), the four chains added as
//     (c0 + c1) + (c2 + c3), then the 16 lanes as a tree (j + j+8, j + j+4, j + j+2, j + j+1);
//   * the softcapped SwiGLU from IEEE operations only (our exp and tanh: no expf / tanhf, whose last bits differ
//     between CUDA and a C library), its h rounded to FP16 as the GPU stores it;
//   * every operation rounded once: the device's __fmaf_rn / __fmul_rn / __fadd_rn / __fdiv_rn (never contracted),
//     the host's plain operators in a file built without contraction (-ffp-contract=off; MSVC /fp:precise).
// Both sides include this header; tests/ik2_off_test.cu holds the two (and the CPU's SIMD forms) to the same bits.
#pragma once

#include <cstdint>
#include <cstring>
#if !defined(__CUDA_ARCH__)
#include <cmath>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IK2X_HD __host__ __device__ __forceinline__
#else
#define IK2X_HD inline
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)   // (the device's round-to-nearest intrinsics)
#define IK2X_DEVICE 1
#else
#define IK2X_DEVICE 0
#endif

namespace ik2::x {

constexpr int kNVFP4 = 40, kQ8_0 = 8;   // (ik2_kernels.cuh's type ids)

IK2X_HD float add(float a, float b) {
#if IK2X_DEVICE
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
IK2X_HD float sub(float a, float b) {
#if IK2X_DEVICE
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
IK2X_HD float mul(float a, float b) {
#if IK2X_DEVICE
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
IK2X_HD float div(float a, float b) {
#if IK2X_DEVICE
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
IK2X_HD float fma(float a, float b, float c) {
#if IK2X_DEVICE
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
IK2X_HD float rint(float a) {   // (to nearest, ties to even: the default rounding mode on the host)
#if IK2X_DEVICE
    return rintf(a);
#else
    return std::nearbyint(a);
#endif
}
IK2X_HD uint32_t f2u(float f) {
#if IK2X_DEVICE
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}
IK2X_HD float u2f(uint32_t u) {
#if IK2X_DEVICE
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

/// FP16 bits of f, to nearest even (overflow: infinity)
IK2X_HD uint16_t f2h(float f) {
    const uint32_t x = f2u(f), sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // (65520 and up: infinity)
    if (ax < 0x38800000u) {                                      // (under 2^-14: a subnormal half, or zero)
        if (ax < 0x33000000u) return (uint16_t) sign;
        const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
        const uint32_t sh = 126u - e, half = 1u << (sh - 1), rem = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        if (rem > half || (rem == half && (q & 1u))) ++q;
        return (uint16_t) (sign | q);
    }
    uint32_t h = (ax - 0x38000000u) >> 13;
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
IK2X_HD float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) return u2f(sign | f2u(mul((float) m, 5.9604644775390625e-8f)));   // (m 2^-24, exact)
    if (e == 31) return u2f(sign | 0x7f800000u | (m << 13));
    return u2f(sign | ((e + 112u) << 23) | (m << 13));
}
IK2X_HD float round_f16(float f) { return h2f(f2h(f)); }

/// e^x for x in [-87, 88] (clamped), within ~2 ulp: x = n ln 2 + r, a degree-7 polynomial in r, times 2^n
IK2X_HD float exp(float x) {
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    const float n = rint(mul(x, 1.44269502162933349609375f));
    float r = fma(-n, 0.693145751953125f, x);
    r = fma(-n, 1.428606765330187045e-06f, r);
    float p = 1.98412698412698412526e-4f;
    p = fma(p, r, 1.38888888888888894189e-3f);
    p = fma(p, r, 8.33333333333333321769e-3f);
    p = fma(p, r, 4.16666666666666643537e-2f);
    p = fma(p, r, 1.66666666666666657415e-1f);
    p = fma(p, r, 0.5f);
    p = fma(p, r, 1.0f);
    p = fma(p, r, 1.0f);
    return mul(p, u2f((uint32_t) ((int) n + 127) << 23));
}
/// tanh(y): an odd polynomial under 1/16, else 1 - 2 / (e^{2|y|} + 1) with y's sign
IK2X_HD float tanh(float y) {
    const float a = u2f(f2u(y) & 0x7fffffffu);
    if (a < 0.0625f) {
        const float y2 = mul(y, y);
        float p = -5.3968253968253968e-2f;
        p = fma(p, y2, 1.3333333333333333e-1f);
        p = fma(p, y2, -3.3333333333333333e-1f);
        return fma(mul(p, y2), y, y);
    }
    const float t = sub(1.0f, div(2.0f, add(exp(mul(2.0f, a)), 1.0f)));
    return u2f(f2u(t) | (f2u(y) & 0x80000000u));
}
IK2X_HD float sigmoid(float g) { return div(1.0f, add(1.0f, exp(-g))); }
/// h = gcap tanh(g / gcap) sigmoid(g) ucap tanh(u / ucap), left to right, rounded to FP16
IK2X_HD float swiglu(float g, float u, float gcap, float ucap) {
    const float a = mul(gcap, tanh(div(g, gcap)));
    const float b = mul(mul(a, sigmoid(g)), ucap);
    return round_f16(mul(b, tanh(div(u, ucap))));
}

/// an E4M3 scale byte as ggml's NVFP4 reads it (half its value; 0x7F: 0)
IK2X_HD float ue4m3(uint8_t v) {
    const uint32_t e = (v >> 3) & 0xFu, m = v & 7u;
    const float f = e == 0u ? mul((float) m, 0x1.0p-10f) : u2f(((e + 119u) << 23) | (m << 20));
    return v == 0x7Fu ? 0.0f : f;
}
/// the 16 weights of sub-block s of a row (exact F32): NVFP4 (36-byte blocks of 64) or Q8_0 (34-byte blocks of 32)
IK2X_HD void sub_weights(int ty, const uint8_t* row, int s, float* w) {
    if (ty == kNVFP4) {
        const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        const uint8_t* b = row + (size_t) (s >> 2) * 36;
        const int c = s & 3;
        const float d = ue4m3(b[c]);
        const uint8_t* q = b + 4 + c * 8;
        for (int j = 0; j < 8; ++j) {
            w[j] = mul((float) kv[q[j] & 0xF], d);
            w[j + 8] = mul((float) kv[q[j] >> 4], d);
        }
    } else {
        const uint8_t* b = row + (size_t) (s >> 1) * 34;
        const float d = h2f((uint16_t) (b[0] | (b[1] << 8)));
        const int8_t* q = (const int8_t*) (b + 2 + (s & 1) * 16);
        for (int j = 0; j < 16; ++j) w[j] = mul((float) q[j], d);
    }
}
/// bytes of a row of K values
IK2X_HD size_t row_bytes(int ty, int K) { return ty == kNVFP4 ? (size_t) K / 64 * 36 : (size_t) K / 32 * 34; }
/// 16 lanes to one sum: j + j+8, j + j+4, j + j+2, j + j+1 (l is overwritten)
IK2X_HD float tree16(float* l) {
    for (int j = 0; j < 8; ++j) l[j] = add(l[j], l[j + 8]);
    for (int j = 0; j < 4; ++j) l[j] = add(l[j], l[j + 4]);
    for (int j = 0; j < 2; ++j) l[j] = add(l[j], l[j + 2]);
    return add(l[0], l[1]);
}
/// the row's dot product with x [K] in the fixed order (the reference every form matches)
IK2X_HD float row_dot(int ty, const uint8_t* row, int K, const float* x) {
    float acc[4][16];
    for (int c = 0; c < 4; ++c)
        for (int j = 0; j < 16; ++j) acc[c][j] = 0.0f;
    for (int s = 0; s < K / 16; ++s) {
        float w[16];
        sub_weights(ty, row, s, w);
        for (int j = 0; j < 16; ++j) acc[s & 3][j] = fma(x[16 * s + j], w[j], acc[s & 3][j]);
    }
    float l[16];
    for (int j = 0; j < 16; ++j) l[j] = add(add(acc[0][j], acc[1][j]), add(acc[2][j], acc[3][j]));
    return tree16(l);
}

}  // namespace ik2::x
