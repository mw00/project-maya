// src/ds4/ds4_lane_parity.cpp - the CPU lane's low-bit experts in the device's arithmetic (see ds4_lane_parity.hpp).
// Built with -ffp-contract=off (CMakeLists.txt): every float operation below is the one written, rounded once.
#include "ds4_lane_parity.hpp"

#include "ds4_par_exp.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define PAR_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif
#if defined(PAR_X86) && (defined(__GNUC__) || defined(__clang__))
#define PAR_AVX2 __attribute__((target("avx2,fma")))
#else
#define PAR_AVX2
#endif

namespace strata::ds4::par {
namespace {

constexpr int kIQ2_XXS = 16, kIQ3_XXS = 18, kIQ3_S = 21, kIQ2_S = 22;
constexpr int kMXFP4 = 39;      // (a GGUF's expert format: 32 values a block, an E8M0 scale then 16 code bytes)
constexpr int kMaxT = 8;        // activations a call (the CPU lane's entries hold at most 8 tokens)
constexpr int kGuLanes = 32;    // ds4_iq.cu: kGuRW 1 -> 32 lanes a gate / up row
constexpr int kDnLanes = 16;    // kDnRW 2 -> 16 lanes a down row

int block_bytes(int type) {
    switch (type) {
        case kIQ2_XXS: return 66;
        case kIQ2_S: return 82;
        case kIQ3_XXS: return 98;
        case kIQ3_S: return 110;
        case kMXFP4: return 17;
        default: return 0;
    }
}

inline float h2f(const uint8_t* p) {   // F16 -> F32, exactly
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;   // m 2^-24 (exact)
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline uint32_t ld2(const uint8_t* p) {   // 4 bytes from a 2-byte aligned address (the device's ld2)
    uint16_t a, b;
    std::memcpy(&a, p, 2);
    std::memcpy(&b, p + 2, 2);
    return (uint32_t) a | ((uint32_t) b << 16);
}
inline uint32_t ksigns8(uint32_t v) {   // 7 sign bits -> 8 (bit 7 their parity): ggml's ksigns_iq2xs
    v &= 127u;
    return v | ((uint32_t) (std::popcount(v) & 1) << 7);
}
// MXFP4: ggml's kvalues_mxfp4 / 2 - a nibble's E2M1 magnitude (ggml's own table holds the values doubled, the block's
// E8M0 scale halved, so the code is the value)
constexpr uint8_t kMxTab[8] = {0, 1, 2, 3, 4, 6, 8, 12};
inline float e8m0(uint8_t e) {   // 2^(e - 128): ggml's GGML_E8M0_TO_FP32_HALF
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t) (e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// A 32-value sub-block as ds4_iq.cu's decode<T> makes it: 4 groups of 8 magnitudes (g[l]: byte j = value 8 l + j's),
// their signs (sg[l] bit j: value 8 l + j negative), the scale d and the halves' multipliers s0 (values 0..15), s1.
struct Sub {
    uint64_t g[4];
    uint32_t sg[4];
    float d;
    int s0, s1;
};
inline void decode(int type, const uint8_t* row, int kb, Sub& s) {
    if (type == kMXFP4) {   // a block is a sub-block: values 0..15 in bytes 0..15's low nibbles, 16..31 in their high ones
        const uint8_t* b = row + (size_t) kb * 17;
        uint64_t g[4] = {0, 0, 0, 0};
        uint32_t sg[4] = {0, 0, 0, 0};
        for (int l = 0; l < 4; ++l)
            for (int i = 0; i < 8; ++i) {
                const int v = 8 * l + i, byi = v & 15;
                const int nib = v < 16 ? (b[1 + byi] & 15) : (b[1 + byi] >> 4);
                g[l] |= (uint64_t) kMxTab[nib & 7] << (8 * i);
                if (nib >> 3) sg[l] |= 1u << i;
            }
        for (int l = 0; l < 4; ++l) s.g[l] = g[l], s.sg[l] = sg[l];
        s.s0 = s.s1 = 1;
        s.d = e8m0(b[0]);
        return;
    }
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t) (kb >> 3) * (size_t) block_bytes(type);
    if (type == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
        for (int l = 0; l < 4; ++l) {
            s.g[l] = iq2xxs_grid[(a0 >> (8 * l)) & 255u];
            s.sg[l] = ksigns8(a1 >> (7 * l));
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = h2f(b) * 0.125f;
    } else if (type == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t i4 = ld2(qs + 4 * ib), sg4 = ld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
        for (int l = 0; l < 4; ++l) {
            s.g[l] = iq2s_grid[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            s.sg[l] = (sg4 >> (8 * l)) & 255u;
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if (type == kIQ3_XXS) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), aux = ld2(qs + 64 + 4 * ib);
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            s.g[l] = (uint64_t) iq3xxs_grid[(w >> sh) & 255u] | ((uint64_t) iq3xxs_grid[(w >> (sh + 8)) & 255u] << 32);
            s.sg[l] = ksigns8(aux >> (7 * l));
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = h2f(b) * 0.25f;
    } else {   // IQ3_S
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), h = b[66 + ib], s4 = ld2(b + 74 + 4 * ib);
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t i1 = ((w >> sh) & 255u) | ((h << (8 - 2 * l)) & 256u);
            const uint32_t i2 = ((w >> (sh + 8)) & 255u) | ((h << (7 - 2 * l)) & 256u);
            s.g[l] = (uint64_t) iq3s_grid[i1] | ((uint64_t) iq3s_grid[i2] << 32);
            s.sg[l] = (s4 >> (8 * l)) & 255u;
        }
        s.s0 = s.s1 = 1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15);
        s.d = h2f(b);
    }
}
inline int code(const Sub& s, int j) {
    const int l = j >> 3, i = j & 7;
    const int m = (int) ((s.g[l] >> (8 * i)) & 255u);
    return (s.sg[l] >> i) & 1u ? -m : m;
}
// the device's sub_dot's integer part: the exact int8 dots of each half, s0 lo + s1 hi in int
inline int sub_dot_int(const Sub& s, const int8_t* xq) {
    int lo = 0, hi = 0;
    for (int j = 0; j < 16; ++j) lo += code(s, j) * (int) xq[j];
    for (int j = 16; j < 32; ++j) hi += code(s, j) * (int) xq[j];
    return s.s0 * lo + s.s1 * hi;
}
// ... then d times it in F32 (the low-bit kernels)
inline float sub_dot(const Sub& s, const int8_t* xq) { return p_mul(s.d, (float) sub_dot_int(s, xq)); }
inline float lanes_tree(float* acc, int lpr) {   // the device's lanes_sum: lane l += lane l + o, o = lpr / 2 .. 1
    for (int o = lpr / 2; o > 0; o >>= 1)
        for (int l = 0; l < o; ++l) acc[l] = p_add(acc[l], acc[l + o]);
    return acc[0];
}
inline const int8_t* act_q(const void* a) { return (const int8_t*) a; }
inline const float* act_d(const void* a, int k) { return (const float*) ((const uint8_t*) a + k); }

// rows [r0, r1) of one matrix against nt activations, scalar: sums[t][r - r0]
// (an MXFP4 block is a sub-block: the value comes from the block, the activation's scale, its own E8M0 scale - in the
// form the device's mx kernels use, fma(scale * activation scale, the block's exact int, sum))
// (rs: the rows' stride in bytes where it is not k's own row - a member's FF band of a whole expert's down, the band's
// blocks first in each row; 0: k's own)
void rows_scalar(int type, const uint8_t* w, int k, const void* const* act, int nt, int lpr, int r0, int r1,
                 float (*sums)[kMaxT], size_t rs) {
    const bool mx = type == kMXFP4;
    const size_t rb = (size_t) (k / (mx ? 32 : 256)) * (size_t) block_bytes(type);
    const size_t stride = rs != 0 ? rs : rb;
    const int nsb = k / 32;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * stride;
        float acc[kMaxT][32] = {};
        Sub s;
        for (int kb = 0; kb < nsb; ++kb) {
            decode(type, row, kb, s);
            const int lane = kb & (lpr - 1);   // (lpr: 16 or 32)
            for (int t = 0; t < nt; ++t)
                if (mx)
                    acc[t][lane] = p_fma(p_mul(s.d, act_d(act[t], k)[kb]), (float) sub_dot_int(s, act_q(act[t]) + 32 * kb),
                                         acc[t][lane]);
                else
                    acc[t][lane] = p_fma(act_d(act[t], k)[kb], sub_dot(s, act_q(act[t]) + 32 * kb), acc[t][lane]);
        }
        for (int t = 0; t < nt; ++t) sums[r - r0][t] = lanes_tree(acc[t], lpr);
    }
}

#if defined(PAR_X86)
// The same rows with the integer dots in AVX2 (exact: the sums are integers either way); every float operation as above
uint64_t g_ksx[256];   // sign byte -> 8 bytes of +1 / -1 (bit j set: byte j -1), for _mm256_sign_epi8
uint64_t g_kx7[128];
void init_ksx() {
    for (int v = 0; v < 256; ++v) {
        uint64_t m = 0;
        for (int j = 0; j < 8; ++j) m |= (uint64_t) (((v >> j) & 1) ? 0xFFu : 0x01u) << (8 * j);
        g_ksx[v] = m;
    }
    for (int v = 0; v < 128; ++v) g_kx7[v] = g_ksx[ksigns8((uint32_t) v)];
}
// ---- the AVX2 rows, a 256-value block (8 sub-blocks) a step, the type and the lane count fixed at compile time:
//  - the weights' signs applied to them once (signed int8, |w| <= 62), the activation as x + 128 in u8 (precomputed a call):
//    maddubs(x + 128, w) pairs stay within int16 (255 62 2 < 32767), so sum (x + 128) w - 128 sum w = sum x w exactly,
//    the weights' half sums (sum w) once a row for every token
//  - the 8 sub-blocks' half sums (lo: values 0..15, hi: 16..31) by an hadd tree - exact integers, any order
//  - then the device's float steps for the 8 sub-blocks at once, lane by lane: v = d (s0 lo + s1 hi), acc = fma(dx, v, acc)
//    in the lanes kb mod LPR (the block's 8 sub-blocks: 8 consecutive lanes), and the lanes' tree at the row's end
// (g_kx7: 7 sign bits, their parity the 8th (ksigns8) -> 8 bytes of +1 / -1)
PAR_AVX2 inline uint32_t ld4u(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
// MXFP4: 8 consecutive blocks (a 256-value step, 136 bytes) -> the 8 sub-blocks' signed codes (a byte a value: ggml's
// kvalues_mxfp4 halved: E2M1's own values, the block's E8M0 scale halved) and the 8 blocks' scales
PAR_AVX2 inline void dec8_mx(const uint8_t* b, __m256i* ws, __m256& dv) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,   //
                                         0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    alignas(32) float sc[8];
    for (int i = 0; i < 8; ++i) {
        const uint8_t* p = b + (size_t) i * 17;
        const __m256i c = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (p + 1)));
        const __m128i lo = _mm256_castsi256_si128(_mm256_shuffle_epi8(lut, _mm256_and_si256(c, m4)));
        const __m128i hi = _mm256_castsi256_si128(_mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(c, 4), m4)));
        ws[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);   // values 0..15 | 16..31
        sc[i] = e8m0(p[0]);
    }
    dv = _mm256_load_ps(sc);
}
template <int TYPE>
PAR_AVX2 inline void dec8(const uint8_t* b, __m256i* ws, __m256i& s0v, __m256i& s1v, float& d) {
    alignas(32) int s0[8], s1[8];
    if constexpr (TYPE == kIQ2_XXS) {
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q = b + 2 + 8 * ib;
            const uint32_t a0 = ld4u(q), a1 = ld4u(q + 4);
            const __m256i g = _mm256_set_epi64x((long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                                                (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
            const __m256i sg = _mm256_set_epi64x((long long) g_kx7[(a1 >> 21) & 127], (long long) g_kx7[(a1 >> 14) & 127],
                                                 (long long) g_kx7[(a1 >> 7) & 127], (long long) g_kx7[a1 & 127]);
            ws[ib] = _mm256_sign_epi8(g, sg);
            s0[ib] = s1[ib] = 2 * (int) (a1 >> 28) + 1;
        }
        d = h2f(b) * 0.125f;
    } else if constexpr (TYPE == kIQ2_S) {
        const uint8_t* qs = b + 2;
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t i4 = ld4u(qs + 4 * ib), sg4 = ld4u(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
            const __m256i g = _mm256_set_epi64x((long long) iq2s_grid[(i4 >> 24) | ((h << 2) & 0x300u)],
                                                (long long) iq2s_grid[((i4 >> 16) & 255u) | ((h << 4) & 0x300u)],
                                                (long long) iq2s_grid[((i4 >> 8) & 255u) | ((h << 6) & 0x300u)],
                                                (long long) iq2s_grid[(i4 & 255u) | ((h << 8) & 0x300u)]);
            const __m256i sg = _mm256_set_epi64x((long long) g_ksx[sg4 >> 24], (long long) g_ksx[(sg4 >> 16) & 255],
                                                 (long long) g_ksx[(sg4 >> 8) & 255], (long long) g_ksx[sg4 & 255]);
            ws[ib] = _mm256_sign_epi8(g, sg);
            s0[ib] = 2 * (int) (sc & 15) + 1;
            s1[ib] = 2 * (int) (sc >> 4) + 1;
        }
        d = h2f(b) * 0.125f;
    } else if constexpr (TYPE == kIQ3_XXS) {
        const uint8_t* qs = b + 2;
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t w0 = ld4u(qs + 8 * ib), w1 = ld4u(qs + 8 * ib + 4), aux = ld4u(qs + 64 + 4 * ib);
            const __m256i g = _mm256_set_epi32((int) iq3xxs_grid[w1 >> 24], (int) iq3xxs_grid[(w1 >> 16) & 255],
                                               (int) iq3xxs_grid[(w1 >> 8) & 255], (int) iq3xxs_grid[w1 & 255],
                                               (int) iq3xxs_grid[w0 >> 24], (int) iq3xxs_grid[(w0 >> 16) & 255],
                                               (int) iq3xxs_grid[(w0 >> 8) & 255], (int) iq3xxs_grid[w0 & 255]);
            const __m256i sg = _mm256_set_epi64x((long long) g_kx7[(aux >> 21) & 127], (long long) g_kx7[(aux >> 14) & 127],
                                                 (long long) g_kx7[(aux >> 7) & 127], (long long) g_kx7[aux & 127]);
            ws[ib] = _mm256_sign_epi8(g, sg);
            s0[ib] = s1[ib] = 2 * (int) (aux >> 28) + 1;
        }
        d = h2f(b) * 0.25f;
    } else {   // IQ3_S
        const uint8_t* qs = b + 2;
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t w0 = ld4u(qs + 8 * ib), w1 = ld4u(qs + 8 * ib + 4), h = b[66 + ib], s4 = ld4u(b + 74 + 4 * ib);
            const __m256i g = _mm256_set_epi32(
                (int) iq3s_grid[(w1 >> 24) | ((h << 1) & 256u)], (int) iq3s_grid[((w1 >> 16) & 255u) | ((h << 2) & 256u)],
                (int) iq3s_grid[((w1 >> 8) & 255u) | ((h << 3) & 256u)], (int) iq3s_grid[(w1 & 255u) | ((h << 4) & 256u)],
                (int) iq3s_grid[(w0 >> 24) | ((h << 5) & 256u)], (int) iq3s_grid[((w0 >> 16) & 255u) | ((h << 6) & 256u)],
                (int) iq3s_grid[((w0 >> 8) & 255u) | ((h << 7) & 256u)], (int) iq3s_grid[(w0 & 255u) | ((h << 8) & 256u)]);
            const __m256i sg = _mm256_set_epi64x((long long) g_ksx[s4 >> 24], (long long) g_ksx[(s4 >> 16) & 255],
                                                 (long long) g_ksx[(s4 >> 8) & 255], (long long) g_ksx[s4 & 255]);
            ws[ib] = _mm256_sign_epi8(g, sg);
            s0[ib] = s1[ib] = 1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15);
        }
        d = h2f(b);
    }
    s0v = _mm256_load_si256((const __m256i*) s0);
    s1v = _mm256_load_si256((const __m256i*) s1);
}
// p[0..7] (8 int32 lanes each: lane j = values 4j..4j+3 of sub-block i) -> lo (8 lanes: sub-block i's values 0..15), hi
PAR_AVX2 inline void halves8(const __m256i* p, __m256i& lo, __m256i& hi) {
    const __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(p[0], p[1]), _mm256_hadd_epi32(p[2], p[3]));
    const __m256i b = _mm256_hadd_epi32(_mm256_hadd_epi32(p[4], p[5]), _mm256_hadd_epi32(p[6], p[7]));
    lo = _mm256_permute2x128_si256(a, b, 0x20);
    hi = _mm256_permute2x128_si256(a, b, 0x31);
}
template <int TYPE, int LPR>
PAR_AVX2 void rows_avx2_t(const uint8_t* w, int k, const void* const* act, int nt, int r0, int r1, float (*sums)[kMaxT],
                          size_t rs) {
    constexpr bool kHalves = TYPE == kIQ2_S;   // (the others: s0 == s1)
    constexpr bool kMx = TYPE == kMXFP4;       // (a block is a sub-block; its own E8M0 scale)
    constexpr int NQ = LPR / 8;                // (the lanes as vectors of 8)
    const size_t rb = (size_t) (k / (kMx ? 32 : 256)) * (size_t) block_bytes(TYPE);
    const size_t stride = rs != 0 ? rs : rb;
    const int nsb = kMx ? k / 32 : (k / 256) * 8;   // (the sub-blocks: a 256-value step is 8 of them)
    const int nb = nsb / 8, tail = nsb - nb * 8;   // (a band of 32-value blocks: the last ones below a step)
    const size_t sbytes = kMx ? (size_t) 8 * 17 : (size_t) block_bytes(TYPE);   // (a step's bytes: 8 blocks, or the one)
    const __m256i ones16 = _mm256_set1_epi16(1), ones8 = _mm256_set1_epi8(1), x80 = _mm256_set1_epi8((char) 0x80);
    thread_local std::vector<uint8_t> xu_buf;
    // (+256: the last step of a band reads a whole 256-value step of x, its blocks past the band's end times zero weights)
    if (xu_buf.size() < (size_t) kMaxT * ((size_t) k + 256)) xu_buf.resize((size_t) kMaxT * ((size_t) k + 256));
    const uint8_t* xu[kMaxT];
    const float* xd[kMaxT];
    for (int t = 0; t < nt; ++t) {   // (x + 128 as u8, a call's once)
        uint8_t* u = xu_buf.data() + (size_t) t * (size_t) k;
        const int8_t* q = act_q(act[t]);
        for (int j = 0; j < k; j += 32)
            _mm256_storeu_si256((__m256i*) (u + j), _mm256_xor_si256(_mm256_loadu_si256((const __m256i*) (q + j)), x80));
        xu[t] = u;
        xd[t] = act_d(act[t], k);
    }
    // one 256-value step (8 sub-blocks) of a row: `wblk` its bytes, `xds` the activations' 8 scale rows, `b` the step
    // (its sub-blocks go to the lanes 8 b .. 8 b + 7 mod LPR)
    __m256 acc[kMaxT][NQ];
    auto step = [&](const uint8_t* wblk, const float* const* xds, int b) PAR_AVX2 {
        __m256i ws[8], s0v, s1v;
        __m256 dv;
        if constexpr (kMx) {
            dec8_mx(wblk, ws, dv);
            s0v = _mm256_set1_epi32(1);
            s1v = s0v;
        } else {
            float d = 0.0f;
            dec8<TYPE>(wblk, ws, s0v, s1v, d);
            dv = _mm256_set1_ps(d);
        }
        __m256i p[8], wlo, whi;
        for (int i = 0; i < 8; ++i) p[i] = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, ws[i]), ones16);
        halves8(p, wlo, whi);
        wlo = _mm256_slli_epi32(wlo, 7);
        whi = _mm256_slli_epi32(whi, 7);
        const int q = b % NQ;   // (sub-blocks 8 b .. 8 b + 7: lanes (8 b) mod LPR ..)
        for (int t = 0; t < nt; ++t) {
            const uint8_t* x = xu[t] + (size_t) b * 256;
            for (int i = 0; i < 8; ++i)
                p[i] = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*) (x + 32 * i)), ws[i]), ones16);
            __m256i lo, hi;
            halves8(p, lo, hi);
            lo = _mm256_sub_epi32(lo, wlo);
            hi = _mm256_sub_epi32(hi, whi);
            const __m256i is = kHalves ? _mm256_add_epi32(_mm256_mullo_epi32(lo, s0v), _mm256_mullo_epi32(hi, s1v))
                                       : _mm256_mullo_epi32(_mm256_add_epi32(lo, hi), s0v);
            const __m256 xs = _mm256_load_ps(xds[t]);
            if constexpr (kMx)   // fma(block scale * activation scale, the exact int, sum) - the mx kernels' form
                acc[t][q] = _mm256_fmadd_ps(_mm256_mul_ps(dv, xs), _mm256_cvtepi32_ps(is), acc[t][q]);
            else {
                const __m256 v = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(is));
                acc[t][q] = _mm256_fmadd_ps(xs, v, acc[t][q]);
            }
        }
    };
    alignas(32) float xds[kMaxT][8];   // (a step's 8 scale rows: the row's own, or the band's last with 0s after it)
    const float* xdp[kMaxT];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * stride;
        for (int t = 0; t < nt; ++t)
            for (int q = 0; q < NQ; ++q) acc[t][q] = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            for (int t = 0; t < nt; ++t) {
                std::memcpy(xds[t], xd[t] + (size_t) b * 8, sizeof xds[t]);
                xdp[t] = xds[t];
            }
            step(row + (size_t) b * sbytes, xdp, b);
        }
        if (kMx && tail > 0) {   // (an IQ type's k is whole 256-value blocks: never a tail here)
            // (an FF band of 32-value blocks has less than a step at its end - 36 blocks: 4 of them.  The rest of the step
            //  is zeros: scale 0, no codes, so the sub-blocks add nothing and the lanes past the band stay empty; the
            //  blocks past the band read zero scales and the x past the band's end meets zero weights.  The scalar form
            //  this replaces cost 5x the row.)
            alignas(32) uint8_t padb[136] = {};
            std::memcpy(padb, row + (size_t) nb * sbytes, (size_t) tail * 17);
            for (int t = 0; t < nt; ++t) {
                for (int j = 0; j < 8; ++j) xds[t][j] = j < tail ? xd[t][(size_t) nb * 8 + (size_t) j] : 0.0f;
                xdp[t] = xds[t];
            }
            step(padb, xdp, nb);
        }
        for (int t = 0; t < nt; ++t) {
            alignas(32) float a[32];
            for (int q = 0; q < NQ; ++q) _mm256_store_ps(a + 8 * q, acc[t][q]);
            sums[r - r0][t] = lanes_tree(a, LPR);
        }
    }
}
template <int TYPE>
PAR_AVX2 void rows_avx2_lpr(const uint8_t* w, int k, const void* const* act, int nt, int lpr, int r0, int r1,
                            float (*sums)[kMaxT], size_t rs) {
    if (lpr == kGuLanes) rows_avx2_t<TYPE, kGuLanes>(w, k, act, nt, r0, r1, sums, rs);
    else rows_avx2_t<TYPE, kDnLanes>(w, k, act, nt, r0, r1, sums, rs);
}
PAR_AVX2 void rows_avx2(int type, const uint8_t* w, int k, const void* const* act, int nt, int lpr, int r0, int r1,
                        float (*sums)[kMaxT], size_t rs) {
    switch (type) {
        case kIQ2_XXS: return rows_avx2_lpr<kIQ2_XXS>(w, k, act, nt, lpr, r0, r1, sums, rs);
        case kIQ2_S: return rows_avx2_lpr<kIQ2_S>(w, k, act, nt, lpr, r0, r1, sums, rs);
        case kIQ3_XXS: return rows_avx2_lpr<kIQ3_XXS>(w, k, act, nt, lpr, r0, r1, sums, rs);
        case kMXFP4: return rows_avx2_lpr<kMXFP4>(w, k, act, nt, lpr, r0, r1, sums, rs);
        default: return rows_avx2_lpr<kIQ3_S>(w, k, act, nt, lpr, r0, r1, sums, rs);
    }
}
bool cpu_avx2() {   // (AVX2 + FMA, the OS saving the YMM state: ds4_iq_cpu.cpp's iq_isa)
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) leaf, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(leaf, 0, r[0], r[1], r[2], r[3]);
#endif
    };
    cpuid(0);
    if (r[0] < 7) return false;
    cpuid(1);
    if (!((r[2] >> 12) & 1u) || !((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return false;   // FMA, OSXSAVE, AVX
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0x6) != 0x6) return false;
    cpuid(7);
    return (r[1] >> 5) & 1u;
}
#endif

// ---- the form: AVX2 where the CPU runs it and it gives the scalar form's bits on random rows of every type (checked
// once); DS4_LANE_PARITY_SCALAR=1 (MAYA_DS4_LANE_PARITY_SCALAR): the scalar form
// (the last argument: the rows' stride in bytes, 0: k's own - a tensor-parallel member's FF band, down_band)
using RowsFn = void (*)(int, const uint8_t*, int, const void* const*, int, int, int, int, float (*)[kMaxT], size_t);
RowsFn g_rows = rows_scalar;
const char* g_form = "scalar";
std::once_flag g_once;

std::vector<uint8_t> rand_rows(int type, int n, int k, std::mt19937& rng) {
    const size_t rbb = (size_t) block_bytes(type);
    const size_t rb = (size_t) (k / (type == kMXFP4 ? 32 : 256)) * rbb;
    std::vector<uint8_t> w(rb * (size_t) n);
    for (auto& v : w) v = (uint8_t) rng();
    if (type == kMXFP4) {   // (the E8M0 scale bytes: finite, a few powers of two around 1)
        for (int r = 0; r < n; ++r)
            for (int b = 0; b < k / 32; ++b) w[(size_t) r * rb + (size_t) b * rbb] = (uint8_t) (126 + (rng() % 5));
        return w;
    }
    for (int r = 0; r < n; ++r)   // (finite F16 scales: exponent below 31, small)
        for (int b = 0; b < k / 256; ++b) {
            uint8_t* p = w.data() + (size_t) r * rb + (size_t) b * rbb;
            const uint16_t h = (uint16_t) (0x1000u + (rng() % 0x1800u)) | (uint16_t) ((rng() & 1u) << 15);
            std::memcpy(p, &h, 2);
        }
    return w;
}
void pick_form() {
#if defined(PAR_X86)
    init_ksx();
    const char* f = std::getenv("STRATA_DS4_LANE_PARITY_SCALAR");
    if (f == nullptr) f = std::getenv("DS4_LANE_PARITY_SCALAR");
    if ((f != nullptr && std::atoi(f) != 0) || !cpu_avx2()) return;
    std::mt19937 rng(0x644u);
    bool same = true;
    for (int type : {kIQ2_XXS, kIQ2_S, kIQ3_XXS, kIQ3_S, kMXFP4})
        // (MXFP4 too in the band sizes a tensor-parallel member's FFN takes: 32-value blocks, not a multiple of 256 -
        //  the AVX2 rows' scalar tail there)
        for (int k : type == kMXFP4 ? std::initializer_list<int>{256, 1056, 1152, 1248, 2304, 5120}
                                    : std::initializer_list<int>{256, 2304, 5120}) {
            const int n = 5, nt = 3;
            const std::vector<uint8_t> w = rand_rows(type, n, k, rng);
            std::vector<std::vector<uint8_t>> a(nt, std::vector<uint8_t>(act_bytes(k)));
            const void* ap[kMaxT];
            for (int t = 0; t < nt; ++t) {
                std::vector<float> x((size_t) k);
                for (auto& v : x) v = std::ldexp((float) ((int) (rng() % 20001u) - 10000), -12);
                quant(type, x.data(), k, a[(size_t) t].data());
                ap[t] = a[(size_t) t].data();
            }
            for (int lpr : {kGuLanes, kDnLanes}) {
                float s1[8][kMaxT], s2[8][kMaxT];
                rows_scalar(type, w.data(), k, ap, nt, lpr, 0, n, s1, 0);
                rows_avx2(type, w.data(), k, ap, nt, lpr, 0, n, s2, 0);
                for (int r = 0; r < n; ++r) same = same && std::memcmp(s1[r], s2[r], (size_t) nt * 4) == 0;
            }
        }
    if (!same) {
        std::fprintf(stderr, "ds4: lane parity: the AVX2 rows differ from the scalar ones; the scalar rows are used\n");
        return;
    }
    g_rows = rows_avx2;
    g_form = "avx2";
#endif
}
RowsFn rows_fn() {
    std::call_once(g_once, pick_form);
    return g_rows;
}

// rows [r0, r1) in chunks of 8 rows (the sums' stack array)
template <class Out>
void run_rows(RowsFn fn, int type, const uint8_t* w, int k, const void* const* act, int nt, int lpr, int r0, int r1,
              Out out, size_t rs) {
    float sums[8][kMaxT];
    for (int a = r0; a < r1; a += 8) {
        const int b = std::min(r1, a + 8);
        fn(type, w, k, act, nt, lpr, a, b, sums, rs);
        out(a, b, sums);
    }
}
// the SwiGLU of a chunk's sums (the same operations either way; the AVX2 + FMA build of it only inlines std::fma)
void swiglu_chunk(int a, int b, int nt, const float (*gs)[kMaxT], const float (*us)[kMaxT], float* const* h,
                  float limit) {
    for (int r = a; r < b; ++r)
        for (int t = 0; t < nt; ++t) h[t][r] = swiglu_portable(gs[r - a][t], us[r - a][t], limit);
}
#if defined(PAR_X86)
PAR_AVX2 void swiglu_chunk_avx2(int a, int b, int nt, const float (*gs)[kMaxT], const float (*us)[kMaxT],
                                float* const* h, float limit) {
    for (int r = a; r < b; ++r)
        for (int t = 0; t < nt; ++t) h[t][r] = swiglu_portable(gs[r - a][t], us[r - a][t], limit);
}
#endif
void gate_up_with(RowsFn fn, int type, const uint8_t* gate, const uint8_t* up, int k, const void* const* act, int nt,
                  float* const* h, int r0, int r1, float limit) {   // (whole rows: no stride)
    float gs[8][kMaxT], us[8][kMaxT];
    const int lpr = lane_lanes(type, true);
    for (int a = r0; a < r1; a += 8) {
        const int b = std::min(r1, a + 8);
        fn(type, gate, k, act, nt, lpr, a, b, gs, 0);
        fn(type, up, k, act, nt, lpr, a, b, us, 0);
#if defined(PAR_X86)
        if (fn == rows_avx2) {
            swiglu_chunk_avx2(a, b, nt, gs, us, h, limit);
            continue;
        }
#endif
        swiglu_chunk(a, b, nt, gs, us, h, limit);
    }
}
void down_with(RowsFn fn, int type, const uint8_t* down, int k, const void* const* act, int nt, float* const* y,
               int r0, int r1, size_t rs) {
    run_rows(fn, type, down, k, act, nt, lane_lanes(type, false), r0, r1, [&](int a, int b, float (*s)[kMaxT]) {
        for (int r = a; r < b; ++r)
            for (int t = 0; t < nt; ++t) y[t][r] = s[r - a][t];
    }, rs);
}

}  // namespace

bool lane_type(int type) { return block_bytes(type) > 0; }
int lane_lanes(int type, bool gate_up) {
    if (type == kMXFP4) return kGuLanes;   // (the mx kernels stripe both roles over 32 lanes)
    return gate_up ? kGuLanes : kDnLanes;
}

void quant(int type, const float* x, int k, void* act) {
    if (type != kMXFP4) return quant(x, k, act);
    int8_t* q = (int8_t*) act;
    float* d = (float*) ((uint8_t*) act + k);
    for (int b = 0; b < k / 32; ++b) {
        const float* xb = x + (size_t) b * 32;
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::fmax(am, std::fabs(xb[j]));
        const float sc = am / 127.0f;   // (the device's q8_quant: the same division and rounding)
        for (int j = 0; j < 32; ++j)
            q[(size_t) b * 32 + (size_t) j] = (int8_t) (am > 0.0f ? std::nearbyint(xb[j] / sc) : 0.0);
        d[b] = sc;
    }
}

void quant(const float* x, int k, void* act) {
    int8_t* q = (int8_t*) act;
    float* d = (float*) ((uint8_t*) act + k);
    for (int b = 0; b < k / 256; ++b) {
        const float* xb = x + (size_t) b * 256;
        float am = 0.0f;
        int ix = 0;
        for (int j = 0; j < 256; ++j) {   // (the largest magnitude's first index, as the device's reduction keeps it)
            const float a = std::fabs(xb[j]);
            if (a > am) am = a, ix = j;
        }
        float dv = 0.0f;
        if (am > 0.0f) {
            const float iscale = -127.0f / xb[ix];
            for (int j = 0; j < 256; ++j) {
                const float val = iscale * xb[j] + 12582912.0f;   // (two roundings: -ffp-contract=off)
                int32_t i;
                std::memcpy(&i, &val, 4);
                q[(size_t) b * 256 + (size_t) j] = (int8_t) std::min(127, (i & 0x007fffff) - 0x00400000);
            }
            dv = 1.0f / iscale;
        } else {
            std::memset(q + (size_t) b * 256, 0, 256);
        }
        for (int c = 0; c < 8; ++c) d[b * 8 + c] = dv;
    }
}

void gate_up(int type, const uint8_t* gate, const uint8_t* up, int k, const void* const* act, int nt, float* const* h,
             int r0, int r1, float limit) {
    gate_up_with(rows_fn(), type, gate, up, k, act, nt, h, r0, r1, limit);
}
void down(int type, const uint8_t* dn, int k, const void* const* act, int nt, float* const* y, int r0, int r1) {
    down_with(rows_fn(), type, dn, k, act, nt, y, r0, r1, 0);
}
void down_band(int type, const uint8_t* dn, size_t rs, int k, const void* const* act, int nt, float* const* y, int r0,
               int r1) {
    down_with(rows_fn(), type, dn, k, act, nt, y, r0, r1, rs);
}
void gate_up_ref(int type, const uint8_t* gate, const uint8_t* up, int k, const void* const* act, int nt,
                 float* const* h, int r0, int r1, float limit) {
    gate_up_with(rows_scalar, type, gate, up, k, act, nt, h, r0, r1, limit);
}
void down_ref(int type, const uint8_t* dn, int k, const void* const* act, int nt, float* const* y, int r0, int r1) {
    down_with(rows_scalar, type, dn, k, act, nt, y, r0, r1, 0);
}
void down_band_ref(int type, const uint8_t* dn, size_t rs, int k, const void* const* act, int nt, float* const* y,
                   int r0, int r1) {
    down_with(rows_scalar, type, dn, k, act, nt, y, r0, r1, rs);
}
const char* form() {
    (void) rows_fn();
    return g_form;
}
float row_ref(int type, const uint8_t* row, int k, const void* act, int lpr) {
    float s[8][kMaxT];
    rows_scalar(type, row, k, &act, 1, lpr, 0, 1, s, 0);
    return s[0][0];
}

}  // namespace strata::ds4::par
