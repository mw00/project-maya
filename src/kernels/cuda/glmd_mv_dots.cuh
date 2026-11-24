// src/kernels/cuda/glmd_mv_dots.cuh - glm-dsa B10.7: the 32-value block types' q8_1 dots (llama.cpp vecdotq.cuh
// vec_dot_{q4_0,q4_1,q5_0,q5_1,iq4_nl}_q8_1 at VDR 2, the MMVQ form) and the F16 row element, written once for the
// device and the host so the CPU test runs the very arithmetic the GPU GEMV runs (integer dp4a sums are exact; the
// float tail is the same expression).  The float scale products use f32 (llama.cpp's non-FAST_FP16 branch).
//
// NO #include here: glm_fast.cu includes this file INSIDE its anonymous namespace (through glmd_mv_ext.cuh), after
// <cstdint>, <cstring> and <cuda_fp16.h>; a host user includes <cstdint> and <cstring> first (the host half
// conversion uses std::memcpy: MSVC, the host compiler of the Windows CUDA build, has no __builtin_memcpy, and
// tools/test_port_portability.py keeps GNU builtins out of code it compiles).  Block layouts are ggml's
// (static_asserted against ggml-common.h in glmd_mv_ext.cuh).
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMD_MVQ_HD __host__ __device__ __forceinline__
#else
#define GLMD_MVQ_HD inline
#endif
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GLMD_MVQ_DEVICE 1
#else
#define GLMD_MVQ_DEVICE 0
#endif

namespace glmd_mvq {

// ggml type ids
constexpr int kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ5_0 = 6, kQ5_1 = 7, kIQ4_NL = 20;

struct BQ4_0 { uint16_t d; uint8_t qs[16]; };
struct BQ4_1 { uint16_t d, m; uint8_t qs[16]; };
struct BQ5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct BQ5_1 { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
struct BIQ4_NL { uint16_t d; uint8_t qs[16]; };
struct BQ8_1 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(BQ4_0) == 18 && sizeof(BQ4_1) == 20 && sizeof(BQ5_0) == 22 && sizeof(BQ5_1) == 24 &&
                  sizeof(BIQ4_NL) == 18 && sizeof(BQ8_1) == 36,
              "ggml block sizes");

/// Bytes of one row of n values (n % 32 == 0) of a type served here; 0 otherwise.
GLMD_MVQ_HD size_t row_bytes(int type, long long n) {
    if (n <= 0 || n % 32) return 0;
    const long long nb = n / 32;
    switch (type) {
        case kF16: return (size_t) n * 2;
        case kQ4_0: return (size_t) nb * sizeof(BQ4_0);
        case kQ4_1: return (size_t) nb * sizeof(BQ4_1);
        case kQ5_0: return (size_t) nb * sizeof(BQ5_0);
        case kQ5_1: return (size_t) nb * sizeof(BQ5_1);
        case kIQ4_NL: return (size_t) nb * sizeof(BIQ4_NL);
        default: return 0;
    }
}
GLMD_MVQ_HD bool type_ok(int type) {
    return type == kF16 || type == kQ4_0 || type == kQ4_1 || type == kQ5_0 || type == kQ5_1 || type == kIQ4_NL;
}

GLMD_MVQ_HD float h2f(uint16_t h) {
#if GLMD_MVQ_DEVICE
    return __half2float(__ushort_as_half(h));
#else
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        const float v = (float) m * (1.0f / 16777216.0f);
        std::memcpy(&x, &v, 4);
        x |= sign;
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
#endif
}
GLMD_MVQ_HD int dp4a(int a, int b, int c) {
#if GLMD_MVQ_DEVICE
    return __dp4a(a, b, c);
#else
    for (int i = 0; i < 4; ++i) c += (int) (int8_t) (a >> (8 * i)) * (int) (int8_t) (b >> (8 * i));
    return c;
#endif
}
// int i32 of a 2-byte aligned array / a 4-byte aligned one (little endian)
GLMD_MVQ_HD int ld_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int) ((uint32_t) x16[2 * i32] | ((uint32_t) x16[2 * i32 + 1] << 16));
}
GLMD_MVQ_HD int ld_b4(const void* x, int i32) { return ((const int*) x)[i32]; }

// one call: iqs in {0, 2} covers ints iqs, iqs + 1 of the block's 4 code ints (VDR 2); bq8 is the q8_1 block
GLMD_MVQ_HD float dot_q4_0(const void* vbq, const BQ8_1* bq8, int kbx, int iqs) {
    const BQ4_0* b = (const BQ4_0*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = ld_b2(b->qs, iqs + i);
        sumi = dp4a((v >> 0) & 0x0F0F0F0F, ld_b4(bq8->qs, iqs + i), sumi);
        sumi = dp4a((v >> 4) & 0x0F0F0F0F, ld_b4(bq8->qs, iqs + i + 4), sumi);
    }
    const float d8 = h2f(bq8->d), s8 = h2f(bq8->s);
    return h2f(b->d) * (sumi * d8 - 4.0f * s8);   // (8 * vdr / QI4_0) = 4: subtracts 8 from each code
}
GLMD_MVQ_HD float dot_q4_1(const void* vbq, const BQ8_1* bq8, int kbx, int iqs) {
    const BQ4_1* b = (const BQ4_1*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = ld_b4(b->qs, iqs + i);
        sumi = dp4a((v >> 0) & 0x0F0F0F0F, ld_b4(bq8->qs, iqs + i), sumi);
        sumi = dp4a((v >> 4) & 0x0F0F0F0F, ld_b4(bq8->qs, iqs + i + 4), sumi);
    }
    const float d4d8 = h2f(b->d) * h2f(bq8->d), m4s8 = h2f(b->m) * h2f(bq8->s);
    return sumi * d4d8 + m4s8 / 2.0f;   // QI8_1 / (vdr * QR4_1) = 2 calls share the m * s term
}
GLMD_MVQ_HD int q5_lo(int vl, int vh) {
    int v = (vl >> 0) & 0x0F0F0F0F;
    v |= (vh << 4) & 0x00000010;
    v |= (vh << 11) & 0x00001000;
    v |= (vh << 18) & 0x00100000;
    v |= (vh << 25) & 0x10000000;
    return v;
}
GLMD_MVQ_HD int q5_hi(int vl, int vh) {
    int v = (vl >> 4) & 0x0F0F0F0F;
    v |= (vh >> 12) & 0x00000010;
    v |= (vh >> 5) & 0x00001000;
    v |= (vh << 2) & 0x00100000;
    v |= (vh << 9) & 0x10000000;
    return v;
}
GLMD_MVQ_HD float dot_q5_0(const void* vbq, const BQ8_1* bq8, int kbx, int iqs) {
    const BQ5_0* b = (const BQ5_0*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int vl = ld_b2(b->qs, iqs + i);
        const int vh = ld_b2(b->qh, 0) >> (4 * (iqs + i));
        sumi = dp4a(q5_lo(vl, vh), ld_b4(bq8->qs, iqs + i), sumi);
        sumi = dp4a(q5_hi(vl, vh), ld_b4(bq8->qs, iqs + i + 4), sumi);
    }
    const float d8 = h2f(bq8->d), s8 = h2f(bq8->s);
    return h2f(b->d) * (sumi * d8 - 8.0f * s8);   // (16 * vdr / QI5_0) = 8: subtracts 16 from each code
}
GLMD_MVQ_HD float dot_q5_1(const void* vbq, const BQ8_1* bq8, int kbx, int iqs) {
    const BQ5_1* b = (const BQ5_1*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int vl = ld_b4(b->qs, iqs + i);
        const int vh = ld_b4(b->qh, 0) >> (4 * (iqs + i));
        sumi = dp4a(q5_lo(vl, vh), ld_b4(bq8->qs, iqs + i), sumi);
        sumi = dp4a(q5_hi(vl, vh), ld_b4(bq8->qs, iqs + i + 4), sumi);
    }
    const float d5d8 = h2f(b->d) * h2f(bq8->d), m5s8 = h2f(b->m) * h2f(bq8->s);
    return sumi * d5d8 + m5s8 / 2.0f;   // QI5_1 / vdr = 2 calls share the m * s term
}
// IQ4_NL: the 16-entry non-linear codebook (ggml kvalues_iq4nl)
// PRMT (__byte_perm): byte n of the result is byte s[4n + 2 : 4n] of the 8 bytes {x, y}; bit 3 of a nibble is ignored
GLMD_MVQ_HD uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
#if GLMD_MVQ_DEVICE
    return __byte_perm(x, y, s);
#else
    const uint64_t b = (uint64_t) x | ((uint64_t) y << 32);
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) r |= (uint32_t) ((b >> (8 * ((s >> (4 * n)) & 7))) & 0xff) << (8 * n);
    return r;
#endif
}
// 4 codes (the nibbles at `shift` of each byte of v) -> their 4 int8 codebook values: two byte permutes over the
// codebook packed in 4 words (no table in memory), the code's bit 3 picks the upper half (llama.cpp
// get_int_from_table_16's idea with immediate words)
GLMD_MVQ_HD int iq4nl_pack(int v, int shift) {
    const uint32_t q = ((uint32_t) v >> shift) & 0x0F0F0F0Fu;
    const uint32_t sel = (q & 0xFu) | ((q >> 4) & 0xF0u) | ((q >> 8) & 0xF00u) | ((q >> 12) & 0xF000u);
    const uint32_t lo = byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel);   // -127 -104 -83 -65 | -49 -35 -22 -10
    const uint32_t hi = byte_perm(0x26190D01u, 0x71594535u, sel);   // 1 13 25 38 | 53 69 89 113
    const uint32_t msk = ((q >> 3) & 0x01010101u) * 0xFFu;
    return (int) ((lo & ~msk) | (hi & msk));
}
GLMD_MVQ_HD float dot_iq4_nl(const void* vbq, const BQ8_1* bq8, int kbx, int iqs) {
    const BIQ4_NL* b = (const BIQ4_NL*) vbq + kbx;
    int sumi = 0;
    for (int l = 0; l < 2; ++l) {
        const int q = ld_b2(b->qs, iqs + l);
        sumi = dp4a(iq4nl_pack(q, 0), ld_b4(bq8->qs, iqs + l), sumi);
        sumi = dp4a(iq4nl_pack(q, 4), ld_b4(bq8->qs, iqs + l + 4), sumi);
    }
    return h2f(b->d) * h2f(bq8->d) * (float) sumi;
}
/// The call for type T: dot(row, the row's q8_1 activation, block kbx, iqs); 2 calls per 32-value block, iqs 0 / 2.
GLMD_MVQ_HD float dot(int type, const void* row, const BQ8_1* x, int kbx, int iqs) {
    switch (type) {
        case kQ4_0: return dot_q4_0(row, x + kbx, kbx, iqs);
        case kQ4_1: return dot_q4_1(row, x + kbx, kbx, iqs);
        case kQ5_0: return dot_q5_0(row, x + kbx, kbx, iqs);
        case kQ5_1: return dot_q5_1(row, x + kbx, kbx, iqs);
        case kIQ4_NL: return dot_iq4_nl(row, x + kbx, kbx, iqs);
        default: return 0.0f;
    }
}
/// One lane's share of a quantized row dot in the warp-per-row GEMV (lane L takes calls L, L + 32 * nw, ... of the
/// row's 2 * n_in / 32 calls, part p of nw warps starting at call 32 p + L); the caller adds the lanes (warp_sum)
/// and the parts in order.
GLMD_MVQ_HD float lane_part(int type, const void* row, const BQ8_1* x, int n_in, int lane, int part, int nw) {
    float s = 0.0f;
    for (int k = 32 * part + lane; k < (n_in / 32) * 2; k += 32 * nw) s += dot(type, row, x, k / 2, 2 * (k % 2));
    return s;
}
/// F16 row: 8 values per lane step (one uint4), the activation in f32 (xf) or dequantized from q8_1 (xq).
GLMD_MVQ_HD float f16_lane(const uint16_t* row, const float* xf, const BQ8_1* xq, int n_in, int lane) {
    float s = 0.0f;
    for (int k = lane; k < n_in / 8; k += 32) {
        float acc = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const int e = 8 * k + j;
            const float x = xf ? xf[e] : (float) xq[e / 32].qs[e % 32] * h2f(xq[e / 32].d);
            acc += h2f(row[e]) * x;
        }
        s += acc;
    }
    return s;
}

}  // namespace glmd_mvq
