// Mixed expert GGUF decoding. Layouts/codebooks: pinned ggml-common.h (MIT).
// Values follow ggml's dequantize_row_{iq3_s,iq4_xs,q5_K,q6_K}.
#pragma once
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
namespace hy4::mixed_quant {
__host__ __device__ inline bool supported(int t) {
    return t == 8 || t == 30 || t == 13 || t == 14 || t == 21 || t == 23;
}
__host__ __device__ inline size_t row_bytes(int t, int n) {
    switch (t) {
    case 8:
        return (size_t)(n / 32) * 34;
    case 30:
        return (size_t)n * 2;
    case 13:
        return (size_t)(n / 256) * 176;
    case 14:
        return (size_t)(n / 256) * 210;
    case 21:
        return (size_t)(n / 256) * 110;
    case 23:
        return (size_t)(n / 256) * 136;
    default:
        return 0;
    }
}
__device__ __forceinline__ float value(const uint8_t* p, int t, int k) {
    if (t == 30)
        return __uint_as_float((unsigned)((const uint16_t*)p)[k] << 16);
    if (t == 8) {
        const auto* b = (const block_q8_0*)p + k / 32;
        return __half2float(b->d) * b->qs[k % 32];
    }
    const int j = k % 256, ib = j / 32, l = j % 32;
    if (t == 13) {
        const auto* b = (const block_q5_K*)p + k / 256;
        const uint8_t* s = b->scales;
        const int sc = ib < 4 ? s[ib] & 63 : (s[ib + 4] & 15) | ((s[ib - 4] >> 6) << 4);
        const int mn = ib < 4 ? s[ib + 4] & 63 : (s[ib + 4] >> 4) | ((s[ib] >> 6) << 4);
        const int q = ((b->qs[(ib / 2) * 32 + l] >> (4 * (ib % 2))) & 15) | (((b->qh[l] >> ib) & 1) << 4);
        const float d = __half2float(((const __half*)b)[0]) * sc, m = __half2float(((const __half*)b)[1]) * mn;
        return __fsub_rn(__fmul_rn(d, (float)q), m);
    }
    if (t == 14) {
        const auto* b = (const block_q6_K*)p + k / 256;
        const int hf = j / 128, qt = (j % 128) / 32;
        const int q = ((b->ql[hf * 64 + (qt % 2) * 32 + l] >> (4 * (qt / 2))) & 15) |
                      (((b->qh[hf * 32 + l] >> (2 * qt)) & 3) << 4);
        return (__half2float(b->d) * b->scales[hf * 8 + qt * 2 + l / 16]) * (q - 32);
    }
    if (t == 21) {
        const auto* b = (const block_iq3_s*)p + k / 256;
        const int g = l / 8, sub = (l % 8) / 4;
        const int idx = b->qs[ib * 8 + g * 2 + sub] | (((unsigned)b->qh[ib] << (8 - 2 * g - sub)) & 256);
        const unsigned grid = iq3s_grid[idx];
        const int q = (grid >> (8 * (l % 4))) & 255;
        const float d = __half2float(b->d) * (1 + 2 * ((b->scales[ib / 2] >> (4 * (ib % 2))) & 15));
        return (d * q) * ((b->signs[ib * 4 + g] >> (l % 8)) & 1 ? -1.f : 1.f);
    }
    if (t == 23) {
        const auto* b = (const block_iq4_xs*)p + k / 256;
        const int sc = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 15) | (((b->scales_h >> (2 * ib)) & 3) << 4);
        const int q = (b->qs[ib * 16 + l % 16] >> (4 * (l / 16))) & 15;
        return (__half2float(b->d) * (sc - 32)) * kvalues_iq4nl[q];
    }
    return 0.f; // host dispatcher rejects unsupported types
}
} // namespace hy4::mixed_quant
