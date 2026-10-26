// src/ds4/ds4_q6k.cuh - one value of a ggml Q6_K superblock (256 values, 210 bytes: ql[128] the low 4 bits, qh[64] the
// high 2, scales[16] int8 a 16 values, d F16), as ggml's dequantize_row_q6_K computes it: d * scale * (q - 32), the two
// products in that order in F32 - the same bits (the Engram tables a GGUF may hold in Q6_K).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace strata::ds4 {

constexpr int kQ6kBytes = 210;

__device__ __forceinline__ float q6k_value(const uint8_t* __restrict__ b, int i) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*) (b + 192);
    // (the F16 scale's bits as a half: HIP's headers take no brace-initialized __half_raw from them)
    const float d = __half2float(__ushort_as_half((unsigned short) (b[208] | (b[209] << 8))));
    const int h = i >> 7, w = i & 127, quarter = w >> 5, l = w & 31;   // (ggml's loop: half h, 32-value quarters)
    ql += h * 64;
    qh += h * 32;
    sc += h * 8;
    const int lo = quarter & 1 ? ql[l + 32] : ql[l];
    const int q = ((quarter & 2 ? lo >> 4 : lo & 0xF) | (((qh[l] >> (2 * quarter)) & 3) << 4)) - 32;
    return __fmul_rn(__fmul_rn(d, (float) sc[(l >> 4) + 2 * quarter]), (float) q);
}

// one value of a ggml IQ4_NL block (32 values, 18 bytes: d F16, then value j in byte j's low nibble (j < 16) or byte
// j - 16's high nibble, a code into kvalues_iq4nl), as ggml's dequantize_row_iq4_nl computes it: d * kvalues[code]
constexpr int kIq4nlBytes = 18;

__device__ __forceinline__ float iq4nl_value(const uint8_t* __restrict__ b, int j) {
    const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    const int code = j < 16 ? b[2 + j] & 0xF : b[2 + j - 16] >> 4;
    return __fmul_rn(d, (float) kv[code]);
}

// a ggml Q5_K superblock (256 values, 176 bytes: d, dmin F16, scales[12] 6-bit scale / min pairs, qh[32] the 5th bits,
// qs[128] the low 4) as ggml's dequantize_row_q5_K computes it, on the host (the token embedding a GGUF may hold in Q5_K)
inline float q5k_f16_host(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else {   // (subnormal: normalized)
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) mm <<= 1, ++sh;
            bits = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) bits = sign | 0x7f800000u | (m << 13);
    else bits = sign | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline void q5k_values(const uint8_t* b, float* y) {
    const float d = q5k_f16_host((uint16_t) (b[0] | b[1] << 8)), dmin = q5k_f16_host((uint16_t) (b[2] | b[3] << 8));
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    auto scale_min = [&](int j, int& s, int& m) {
        if (j < 4) {
            s = sc[j] & 63;
            m = sc[j + 4] & 63;
        } else {
            s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
            m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
    };
    unsigned u1 = 1, u2 = 2;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
        int s, m;
        scale_min(is, s, m);
        const float d1 = d * (float) s, m1 = dmin * (float) m;
        scale_min(is + 1, s, m);
        const float d2 = d * (float) s, m2 = dmin * (float) m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    }
}

// value e of an Engram row in format fmt (1 Q6_K, 2 IQ4_NL; the row's blocks from its start)
__device__ __forceinline__ float eg_block_value(const uint8_t* __restrict__ row, int e, int fmt) {
    return fmt == 1 ? q6k_value(row + (size_t) (e / 256) * kQ6kBytes, e % 256)
                    : iq4nl_value(row + (size_t) (e / 32) * kIq4nlBytes, e % 32);
}

}  // namespace strata::ds4
