// src/kernels/cuda/glm_lat8.cuh - the INT8 latent records (STRATA_GLM_KV_INT8; glm_fast.hpp lat8_rec_bytes) read as
// the FP16 an FP16 row would hold, for the batched kernels: per position kv int8 codes, then an FP16 scale per 32.
//
// A value is (half) (q * d), the exact product rounded once.  The prompt attention reads every selected row this way,
// once per head group of every token that selects it, so the conversion is most of what INT8 adds to a prompt (a
// V100 at 8K: its attention +20%).  Four int-to-float conversions and four F32 multiplies a word became two FP16
// pairs: a code goes to FP16 exactly (the bits 0x64, q + 128 are 1024 + q + 128; less 1152 is q), and FP16's own
// multiply rounds the exact product once, as the F32 product (exact: 8 x 11 bits) then rounded to FP16 did; on AMD a
// scale below FP16's normal range takes that F32 product itself - the same bits for every code and every finite scale
// (tests/glm_lat8_test.cu checks all of them).
#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels::glmb {

__device__ __forceinline__ const uint8_t* lat8_rec(const uint16_t* lat, int kv, int cell) {
    return (const uint8_t*) lat + (size_t) (kv + kv / 16) * cell;
}
// four codes (a word, the first in its low byte) times d
__device__ __forceinline__ uint2 lat8_q4(uint32_t w, __half2 d) {
    const uint32_t u = w ^ 0x80808080u;                                  // q + 128 a byte
    const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140), hi = __byte_perm(u, 0x64646464u, 0x7362);
    const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152
    const __half2 a = __hmul2(__hsub2(*(const __half2*) &lo, bias), d);
    const __half2 b = __hmul2(__hsub2(*(const __half2*) &hi, bias), d);
#if defined(STRATA_USE_HIP) && !defined(__gfx942__) && !defined(__gfx950__)
    // a code 0 times a negative scale is -0 in FP16 and +0 through AMD's F32 product: a zero half loses its sign
    // (only a code 0 makes a zero here - a normal scale times any other code is a normal number).  CDNA3 and CDNA4 (gfx942, gfx950)
    // keeps the sign through the F32 product as NVIDIA does, so its FP16 product is already the same bits
    const auto unsigned_zeros = [](unsigned x) {
        return x & (0x7FFF7FFFu | (((x & 0x7FFF7FFFu) + 0x7FFF7FFFu) & 0x80008000u));
    };
    return make_uint2(unsigned_zeros(*(const unsigned*) &a), unsigned_zeros(*(const unsigned*) &b));
#else
    return make_uint2(*(const unsigned*) &a, *(const unsigned*) &b);
#endif
}
// ... and the same through the F32 product, the conversion itself, for a scale below FP16's normal range (exponent
// bits 0: zero or subnormal): there FP16 arithmetic and signed zeros differ by platform (a Radeon 8060S: a code 0
// times a negative subnormal scale gave -0 in FP16, +0 through F32), and such a scale is a block of near-zero latents
__device__ __forceinline__ uint2 lat8_q4_f32(uint32_t w, float d) {
    const __half2 a = __floats2half2_rn((float) (int8_t) w * d, (float) (int8_t) (w >> 8) * d);
    const __half2 b = __floats2half2_rn((float) (int8_t) (w >> 16) * d, (float) (int8_t) (w >> 24) * d);
    return make_uint2(*(const unsigned*) &a, *(const unsigned*) &b);
}
__device__ __forceinline__ bool lat8_normal(__half d) { return (__half_as_ushort(d) & 0x7C00u) != 0; }
// values [4 c4, 4 c4 + 4) of a record (an FP16 row's uint2 there).  NVIDIA's FP16 multiply keeps subnormals and the
// zeros' signs as the F32 product does (every scale checked on a V100), so only AMD takes the F32 product for such a
// scale - the branch alone cost the V100's prompt attention the whole gain (3.42 -> 3.69 s at 8K)
__device__ __forceinline__ uint2 lat8_h4(const uint8_t* rec, int kv, int c4) {
    const uint32_t w = ((const uint32_t*) rec)[c4];
    const __half d = ((const __half*) (rec + kv))[(4 * c4) >> 5];
#if defined(STRATA_USE_HIP)
    if (!lat8_normal(d)) return lat8_q4_f32(w, __half2float(d));
#endif
    return lat8_q4(w, __half2half2(d));
}
// values [8 c8, 8 c8 + 8) of a record (an FP16 row's uint4 there)
__device__ __forceinline__ uint4 lat8_h8(const uint8_t* rec, int kv, int c8) {
    const uint2 w = ((const uint2*) rec)[c8];
    const __half d = ((const __half*) (rec + kv))[c8 >> 2];
#if defined(STRATA_USE_HIP)
    if (!lat8_normal(d)) {
        const uint2 lo = lat8_q4_f32(w.x, __half2float(d)), hi = lat8_q4_f32(w.y, __half2float(d));
        return make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
#endif
    const uint2 lo = lat8_q4(w.x, __half2half2(d)), hi = lat8_q4(w.y, __half2half2(d));
    return make_uint4(lo.x, lo.y, hi.x, hi.y);
}

}  // namespace strata::kernels::glmb
