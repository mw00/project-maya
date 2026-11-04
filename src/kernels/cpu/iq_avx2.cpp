// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once.
//
// The AVX-512 kernels' (iq_avx512.cpp) multi-token scheme on the CPUs without AVX-512 (AMD Zen 2/3, Intel
// Core 12th-14th gen and Core Ultra): each 32-value chunk is decoded once - grid lookups, one sign vector,
// one scale vector - and every token of the verify window applies them with a load, a `sign`, a `maddubs`,
// a `madd` and an `add`.  ggml-cpu's own AVX2 dot products for these formats are single-token: every token
// re-does the codebook lookups and the sign expansion.  The sign vector is ggml's bit_selector pattern
// (ggml-cpu/arch/x86/quants.c): pshufb-broadcast of each sign byte, AND with the bit selector, CMPEQ, OR
// with one, `vpsignb`.  The result is BIT-IDENTICAL to ggml's AVX2 dot: the same exact integer lanes per block and
// ggml's float steps in ggml's order (one fmadd per block with d_x * d_y, hsum_float_8, then the format's constant;
// see iq_avx512.cpp).
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23), Q4_K (12), Q5_K (13).
// IQ1_M (29) shares its grid, scale and delta decode across tokens, preserving ggml's two float accumulators.
#include "strata/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

std::atomic<bool> q5k_fused{true};   // see iq_avx512.cpp

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens.
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The two 16-value scales of one 32-value half (IQ2_XS, IQ2_S): int16 lanes 0-7 = a (values 0-15),
// 8-15 = b (values 16-31).  NOT a broadcast dword [a,b] - that would alternate the scales lane by lane.
inline __m256i sc16(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)),
                                   _mm_set1_epi16((short) b), 1);
}
// One scale for the whole 32-value half (IQ2_XXS, IQ3_XXS, IQ3_S).
inline __m256i sc32(int s) { return _mm256_set1_epi16(s); }

// sc16_4[k] = sc16(2 (k & 15) + 1, 2 (k >> 4) + 1) (IQ2_XS's scale byte k), as a table: one load instead of sc16's
// broadcasts and insert (all on port 5, Haswell / Broadwell's only shuffle port).  constexpr, as sc8_3: this TU is
// built with -mavx2, so a run-time constructor would be vectorized and run before main, before cpu_avx2_ok().
struct Sc16Tab {
    alignas(32) int16_t v[256][16]{};
    constexpr Sc16Tab() {
        for (int k = 0; k < 256; ++k)
            for (int l = 0; l < 16; ++l) v[k][l] = (int16_t) (2 * ((l < 8 ? k : k >> 4) & 15) + 1);
    }
    __m256i operator[](unsigned k) const { return _mm256_load_si256((const __m256i*) v[k]); }
};
constexpr Sc16Tab sc16_4;

// qwords t[a] t[b] t[c] t[d] without the shuffle port: broadcast loads put together by dword blends (p015), not
// set_epi64x's inserts (ds4_iq_cpu.cpp's tab4)
inline __m256i bq(const uint64_t* p) { return _mm256_castpd_si256(_mm256_broadcast_sd((const double*) p)); }
inline __m256i tab4(const uint64_t* t, unsigned a, unsigned b, unsigned c, unsigned d) {
    return _mm256_blend_epi32(_mm256_blend_epi32(bq(t + a), bq(t + b), 0x0C),
                              _mm256_blend_epi32(bq(t + c), bq(t + d), 0xC0), 0xF0);
}

// The four 7-bit sign indices of a 32-value half (IQ2_XXS, IQ3_XXS: bits 0-27 of w) -> 32 bytes of -1 (sign bit set)
// / +1, in registers: each index's eighth bit is the parity of its seven (a
// 16-entry pshufb table on the xor of its two nibbles), a byte shuffle spreads each index over its 8 values and the
// bit selector picks each value's bit.  The vector ksigns_iq2xs gives (ggml's keven_signs_q2xs), without the four
// table loads and the set_epi64x of every half.
inline __m256i sgn7_vec(uint32_t w) {
    const __m256i shifts = _mm256_setr_epi32(0, 0, 7, 7, 14, 14, 21, 21);
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12,
                                            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    const __m256i bits = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i parity = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0));
    __m256i s = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) w), shifts), _mm256_set1_epi32(127));
    s = _mm256_or_si256(s, _mm256_shuffle_epi8(parity, _mm256_and_si256(_mm256_xor_si256(s, _mm256_srli_epi32(s, 4)),
                                                                        _mm256_set1_epi8(15))));
    s = _mm256_shuffle_epi8(s, spread);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s, bits), bits), _mm256_set1_epi8(1));
}

// ggml's hsum_float_8: the same additions in the same order (the exact kernels)
inline float hsum_ggml(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct Fmt32;

template <> struct Fmt32<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = sgn7_vec(w1);
        sc = sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Fmt32<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        uint16_t v[8];
        std::memcpy(v, b + 2 + 16 * j, 16);
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[o + 3] & 511], (long long) iq2xs_grid[v[o + 2] & 511],
                              (long long) iq2xs_grid[v[o + 1] & 511], (long long) iq2xs_grid[v[o] & 511]);
        uint32_t s = 0;
        for (int l = 0; l < 4; ++l) s |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
        sgn = sgn_vec(s);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h = b[66 + 2 * j + half];
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2s_grid[qs[o + 3] | ((h << 2) & 0x300)],
                              (long long) iq2s_grid[qs[o + 2] | ((h << 4) & 0x300)],
                              (long long) iq2s_grid[qs[o + 1] | ((h << 6) & 0x300)],
                              (long long) iq2s_grid[qs[o] | ((h << 8) & 0x300)]);
        const uint64_t m = u64(b + 2 + 32 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = sgn7_vec(w);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
#define G3(k) (int) iq3s_grid[q[k] | (((h >> k) & 1u) << 8)]
        g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
        const uint64_t m = u64(b + 74 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

template <> struct Fmt32<23> {   // IQ4_XS: d, scales_h u16, scales_l[4], qs[128] (kvalues_iq4nl nibbles)
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const int ib = 2 * j + half;   // the 32-value chunk: 16 bytes, low nibbles values 0-15, high 16-31
        const __m128i q = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * ib));
        const __m128i lut = _mm_loadu_si128((const __m128i*) kvalues_iq4nl), m4 = _mm_set1_epi8(0x0f);
        const __m256i v = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(lut, _mm_and_si128(q, m4))),
            _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4)), 1);
        g = _mm256_sign_epi8(v, v);   // |w|, and w itself signs the activation: ggml's mul_add_epi8
        sgn = v;
        const uint32_t sh = u16(b + 2);
        const uint8_t sl = b[4 + j];
        sc = sc32((int) ((half ? sl >> 4 : sl & 15) | (((sh >> (2 * ib)) & 3) << 4)) - 32);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = Fmt32<TY>::K * hsum_ggml(accf[t]);
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K):
// the 7-bit sign indices never touch ksigns_iq2xs - the 8th sign bit is reconstructed by parity (two
// shifts, a xor and ggml's bit_helper pshufb) and the per-half sign vectors come out of shuffles of one
// 32-byte load.  The half-scales come from sc16_4 and the grid entries by tab4 (ggml: an unpack + pshufb, and
// set_epi64x - port-5 work that bounds Broadwell).  Every token still only pays a load, a sign,
// a maddubs, a madd and an add per half, into alternating accumulators.
template <int NT>
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80,
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80 };
    static const char shuf_even[32] = {   // sign bits of u16 0,1,2,3 (even bytes of the sign register)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const char shuf_odd[32] = {    // sign bits of u16 4,5,6,7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {  // 0x80 when the 4-bit index has odd popcount (parity bit 7)
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00,
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00 };
    const __m256i bsel = _mm256_loadu_si256((const __m256i*) bit_sel);
    const __m256i shev = _mm256_loadu_si256((const __m256i*) shuf_even);
    const __m256i shod = _mm256_loadu_si256((const __m256i*) shuf_odd);
    const __m256i bhelp = _mm256_loadu_si256((const __m256i*) bit_help);
    const __m256i one8 = _mm256_set1_epi8(1);

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256((const __m256i*) (blk + 2 + 32 * jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);              // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13);             // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7,
                _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));  // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb)); // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            for (int h = 0; h < 4; ++h) {   // the four 32-value halves of this group
                const int H = 4 * jj + h;   // global half index -> scales byte pair
                const uint8_t* gi = blk + 2 + 32 * jj + 8 * h;   // (scalar loads: gcc turns a stored copy into vpextrw)
                const __m256i g = tab4(iq2xs_grid, u16(gi) & 511u, u16(gi + 2) & 511u, u16(gi + 4) & 511u, u16(gi + 6) & 511u);
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = sc16_4[blk[66 + H]];   // (scale byte H: values 0-15 low nibble, 16-31 high)
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = _mm256_add_epi32(acc[t][h & 1],
                        _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = 0.125f * hsum_ggml(accf[t]);
}

// ggml's unpack of a K-quant block's 12 scale bytes: utmp[0..1] = the 8 6-bit scales, utmp[2..3] = the 8 mins
inline void k_scales(const uint8_t* s12, uint32_t utmp[4]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    std::memcpy(utmp, s12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// Q4_K (12) / Q5_K (13): the layout and ggml's min term as in iq_avx512.cpp's row_dot_k; each 32-value chunk's
// unsigned values decoded once, every token a load, a maddubs, a madd and an add per chunk.
template <int TY, int NT>
inline void row_dot_k(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    constexpr int bytes = TY == 12 ? 144 : 176, qs_off = TY == 12 ? 16 : 48;
    __m256 accf[NT];
    __m128 accm[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
        accm[t] = _mm_setzero_ps();
        summs[t] = 0.0f;
    }
    const __m256i m4 = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi8(1);
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * bytes;
        uint32_t utmp[4];
        k_scales(blk + 4, utmp);
        uint8_t s8[8];
        std::memcpy(s8, utmp, 8);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_set_epi32(0, 0, (int) utmp[3], (int) utmp[2]));
        __m256i qh = _mm256_setzero_si256();
        if constexpr (TY == 13) qh = _mm256_loadu_si256((const __m256i*) (blk + 16));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (blk + qs_off + 32 * j));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if constexpr (TY == 13) {
                lo = _mm256_add_epi8(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), one), 4));
                hi = _mm256_add_epi8(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j + 1), one), 4));
            }
            const __m256i sl = sc32(s8[2 * j]), sh = sc32(s8[2 * j + 1]);
            for (int t = 0; t < NT; ++t) {
                const __m256i yl = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j));
                const __m256i yh = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j + 32));
                acci[t] = _mm256_add_epi32(acci[t], _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, yl), sl),
                                                                     _mm256_madd_epi16(_mm256_maddubs_epi16(hi, yh), sh)));
            }
        }
        const float dx = h2f(u16(blk)), mx = h2f(u16(blk + 2));
        for (int t = 0; t < NT; ++t) {
            const float d = y[t][i].d * dx, dmin = -y[t][i].d * mx;
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
            const __m256i bs = _mm256_loadu_si256((const __m256i*) y[t][i].bsums);
            const __m128i prod = _mm_madd_epi16(mins, _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1)));
            if constexpr (TY == 12) {
                accm[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), accm[t]);
            } else {
                const __m128i z = _mm_setzero_si128();
                const float sm = (float) _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(prod, z), z));
                summs[t] = q5k_fused.load(std::memory_order_relaxed) ? std::fma(dmin, sm, summs[t]) : summs[t] + dmin * sm;
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        if constexpr (TY == 12) {
            __m128 am = _mm_add_ps(accm[t], _mm_movehl_ps(accm[t], accm[t]));
            am = _mm_add_ss(am, _mm_movehdup_ps(am));
            res[t] = hsum_ggml(accf[t]) + _mm_cvtss_f32(am);
        } else {
            res[t] = hsum_ggml(accf[t]) + summs[t];
        }
    }
}

// IQ1_M follows the pinned ggml AVX2 dot's float recurrence and its int32 lanes per block; the grid/scales/delta signs
// are decoded once for the entire token group.  Built for Broadwell, where ggml's dot is bound by the number of uops it
// issues: the grid bytes are -1/0/+1, so ggml's |g| x sign(q8, g) is sign(q8, g) and the 2s+1 <= 15 scales go into
// maddubs' unsigned operand (|lane| <= 2 x 15 x 128 = 3840, 8 of them a block <= 30720: int16 sums, one madd a block);
// the delta signs come from a broadcast of the qh bytes (AND / CMPEQ / OR), the grid indices by pdep where it is fast
// (iq1m_pdep()), else in scalar registers into tab4.  The int32 lanes are ggml's exact integers, so the bits are too.
#if defined(_MSC_VER) && !defined(__clang__)
inline uint64_t pdep64(uint64_t a, uint64_t m) { return _pdep_u64(a, m); }
inline void cpuid0(unsigned leaf, unsigned r[4]) {
    int x[4];
    __cpuidex(x, (int) leaf, 0);
    for (int k = 0; k < 4; ++k) r[k] = (unsigned) x[k];
}
#else
inline uint64_t pdep64(uint64_t a, uint64_t m) {   // (asm: this file is built without -mbmi2; run only if iq1m_pdep())
    uint64_t r;
    __asm__("pdep %2, %1, %0" : "=r"(r) : "r"(a), "rm"(m));
    return r;
}
inline void cpuid0(unsigned leaf, unsigned r[4]) { __cpuid_count(leaf, 0, r[0], r[1], r[2], r[3]); }
#endif
// BMI2's pdep where it is fast: Intel since Haswell, AMD since Zen 3 (Zen 1/2 and Hygon run it in microcode).
// DS4_IQ1M_PDEP=0/1 picks the form on a BMI2 CPU (both exact; the start check tests the one in use).  Probed on
// first use (a function-local static), only ever reached behind cpu_avx2_ok(): no code of this TU runs before main.
bool iq1m_pdep() {
    static const bool pd = [] {
        unsigned r[4];
        cpuid0(0, r);
        if (r[0] < 7) return false;
        const bool amd = r[1] == 0x68747541u || r[1] == 0x6f677948u;   // "Auth"enticAMD, "Hygo"nGenuine
        cpuid0(1, r);
        const unsigned family = ((r[0] >> 8) & 15) + ((r[0] >> 20) & 255);
        cpuid0(7, r);
        if (((r[1] >> 8) & 1) == 0) return false;
        if (const char* v = std::getenv("DS4_IQ1M_PDEP")) return std::atoi(v) != 0;
        return !amd || family >= 0x19;
    }();
    return pd;
}

// a 32-value half's four grid entries (qs[0..3], the 3 high index bits in nibbles of qh[0..1])
template <bool PD> inline __m256i iq1m_grid(const uint8_t* qs, const uint8_t* qh) {
    if constexpr (PD) {
        const uint64_t p = pdep64(u32(qs), 0x00ff00ff00ff00ffULL) | pdep64(u16(qh) & 0x7777u, 0x0f000f000f000f00ULL);
        uint16_t ix[4];
        std::memcpy(ix, &p, sizeof ix);
        return _mm256_set_epi64x((long long) iq1s_grid[ix[3]], (long long) iq1s_grid[ix[2]],
                                 (long long) iq1s_grid[ix[1]], (long long) iq1s_grid[ix[0]]);
    } else {
        const unsigned h = u16(qh);
        return tab4(iq1s_grid, qs[0] | ((h << 8) & 0x700), qs[1] | ((h << 4) & 0x700), qs[2] | (h & 0x700),
                    qs[3] | ((h >> 4) & 0x700));
    }
}

// sc8_3[k]: bytes 0-15 2 (k & 7) + 1, bytes 16-31 2 (k >> 3 & 7) + 1 (a 32-value half's two scales, maddubs' u8 side)
struct Sc8Tab {
    alignas(32) int8_t v[64][32]{};
    constexpr Sc8Tab() {
        for (int k = 0; k < 64; ++k)
            for (int l = 0; l < 32; ++l) v[k][l] = (int8_t) (2 * ((l < 16 ? k : k >> 3) & 7) + 1);
    }
    __m256i operator[](unsigned k) const { return _mm256_load_si256((const __m256i*) v[k]); }
};
constexpr Sc8Tab sc8_3;

template <int NT, bool PD>
inline void row_dot_iq1m(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT], deltaf[NT];
    const __m256i ones = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    const __m256i b0 = _mm256_setr_epi32(0x08, 0x08, 0x80, 0x80, 0x800, 0x800, 0x8000, 0x8000);   // qh delta bits
    const __m256i b1 = _mm256_slli_epi32(b0, 16);
    for (int t = 0; t < NT; ++t) accf[t] = deltaf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* block = row + (size_t) i * sizeof(block_iq1_m);
        uint16_t scales[4];
        std::memcpy(scales, block + 48, sizeof scales);
        const uint16_t dh = (scales[0] >> 12) | ((scales[1] >> 8) & 0x00f0) |
                            ((scales[2] >> 4) & 0x0f00) | (scales[3] & 0xf000);
        const float dx = h2f(dh);
        __m256i acc16[NT], del16[NT];
        for (int t = 0; t < NT; ++t) acc16[t] = del16[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const uint8_t* qs = block + 8 * j;
            const uint8_t* qh = block + 32 + 4 * j;
            const __m256i grid0 = iq1m_grid<PD>(qs, qh), grid1 = iq1m_grid<PD>(qs + 4, qh + 2);
            const __m256i h = _mm256_castps_si256(_mm256_broadcast_ss((const float*) qh));   // qh[0..3]
            const __m256i delta0 = _mm256_or_si256(_mm256_cmpeq_epi32(_mm256_and_si256(h, b0), b0), ones);
            const __m256i delta1 = _mm256_or_si256(_mm256_cmpeq_epi32(_mm256_and_si256(h, b1), b1), ones);
            const unsigned sc = scales[j];
            const __m256i scale0 = sc8_3[sc & 63], scale1 = sc8_3[(sc >> 6) & 63];
            for (int t = 0; t < NT; ++t) {
                const __m256i a0 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j));
                const __m256i a1 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j + 32));
                acc16[t] = _mm256_add_epi16(acc16[t], _mm256_add_epi16(
                    _mm256_maddubs_epi16(scale0, _mm256_sign_epi8(a0, grid0)),
                    _mm256_maddubs_epi16(scale1, _mm256_sign_epi8(a1, grid1))));
                del16[t] = _mm256_add_epi16(del16[t], _mm256_add_epi16(
                    _mm256_maddubs_epi16(scale0, _mm256_sign_epi8(a0, delta0)),
                    _mm256_maddubs_epi16(scale1, _mm256_sign_epi8(a1, delta1))));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 d = _mm256_set1_ps(dx * y[t][i].d);
            accf[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_madd_epi16(acc16[t], ones16)), accf[t]);
            deltaf[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_madd_epi16(del16[t], ones16)), deltaf[t]);
        }
    }
    // This TU disables contraction; explicit fusion preserves ggml's rounding for subnormal accumulators too.
    for (int t = 0; t < NT; ++t) {
#if defined(_MSC_VER)
        // ggml-cpu's MSVC build rounds the delta product and the final add separately.
        const float delta = IQ1M_DELTA * hsum_ggml(deltaf[t]);
        res[t] = hsum_ggml(accf[t]) + delta;
#else
        res[t] = std::fma(IQ1M_DELTA, hsum_ggml(deltaf[t]), hsum_ggml(accf[t]));
#endif
    }
}

template <int TY, int NT>
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res, bool pd) {
    if constexpr (TY == 29) {   // (pd: iq1m_pdep(), read once a loop by the caller - review r2 S3)
        if (pd) row_dot_iq1m<NT, true>(row, nblocks, y, res);
        else row_dot_iq1m<NT, false>(row, nblocks, y, res);
    }
    else if constexpr (TY == 17) row_dot_iq2xs<NT>(row, nblocks, y, res);
    else if constexpr (TY == 12 || TY == 13) row_dot_k<TY, NT>(row, nblocks, y, res);
    else                    row_dot<TY, NT>(row, nblocks, y, res);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    const bool pd = TY == 29 && iq1m_pdep();
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(blob + (size_t) r * gu_row, nb, y, g, pd);
        row_dot_any<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u, pd);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    const bool pd = TY == 29 && iq1m_pdep();
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res, pd);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 8: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 8) {   // (more than 8: groups of 8, the rows read again per group)
            const int k = nt - t0 < 8 ? nt - t0 : 8;
            gu_rows_nt<TY>(k, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// ---- one token (a decode step's CPU expert): ggml-cpu's own AVX2 dots for IQ3_XXS / IQ3_S with a cheaper decode -
// IQ3_XXS's sign vectors built in registers (sgn7_vec's steps, as neg masks), IQ3_S's 9-bit grid indices
// built in scalar registers (ggml builds them in vectors, stores and reloads them), the inner loop unrolled.  The
// integer lanes and the float steps are ggml's, so every result is bit-identical to its dot.  (The multi-token rows
// above lose to ggml's dot at one token: Maya-L's IQ3_S gate/up pair on one Xeon 6152 core 4.05 vs 3.05 ms.)
inline float dot1_iq3_xxs(const uint8_t* row, int nb, const block_q8_K* y) {
    const __m256i shifts = _mm256_setr_epi32(0, 0, 7, 7, 14, 14, 21, 21);
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12,
                                            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    const __m256i bits = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i parity = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0));
    const auto neg = [&](uint32_t w) {   // 0xFF where the value's sign bit is set
        __m256i s = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) w), shifts), _mm256_set1_epi32(127));
        s = _mm256_or_si256(s, _mm256_shuffle_epi8(parity, _mm256_and_si256(_mm256_xor_si256(s, _mm256_srli_epi32(s, 4)),
                                                                            _mm256_set1_epi8(15))));
        s = _mm256_shuffle_epi8(s, spread);
        return _mm256_cmpeq_epi8(_mm256_and_si256(s, bits), bits);
    };
    const block_iq3_xxs* x = (const block_iq3_xxs*) row;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* q3 = x[i].qs;
        const uint8_t* gas = x[i].qs + QK_K / 4;
        const int8_t* q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32((int) iq3xxs_grid[q3[7]], (int) iq3xxs_grid[q3[6]], (int) iq3xxs_grid[q3[5]],
                                                  (int) iq3xxs_grid[q3[4]], (int) iq3xxs_grid[q3[3]], (int) iq3xxs_grid[q3[2]],
                                                  (int) iq3xxs_grid[q3[1]], (int) iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32((int) iq3xxs_grid[q3[7]], (int) iq3xxs_grid[q3[6]], (int) iq3xxs_grid[q3[5]],
                                                  (int) iq3xxs_grid[q3[4]], (int) iq3xxs_grid[q3[3]], (int) iq3xxs_grid[q3[2]],
                                                  (int) iq3xxs_grid[q3[1]], (int) iq3xxs_grid[q3[0]]);
            q3 += 8;
            const uint32_t a0 = u32(gas), a1 = u32(gas + 4);
            gas += 8;
            const __m256i n0 = neg(a0), n1 = neg(a1);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(q8_1, n0), n0);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(q8_2, n1), n1);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2 * (a0 >> 28) + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.25f * hsum_ggml(accumf);
}

inline float dot1_iq3_s(const uint8_t* row, int nb, const block_q8_K* y) {
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const block_iq3_s* x = (const block_iq3_s*) row;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        const int8_t* q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const uint32_t h0 = qh[ib32], h1 = qh[ib32 + 1];
#define G3(k, h) (int) iq3s_grid[qs[k] | ((h << (8 - (k & 7))) & 256)]
            const __m256i q2_1 = _mm256_set_epi32(G3(7, h0), G3(6, h0), G3(5, h0), G3(4, h0),
                                                  G3(3, h0), G3(2, h0), G3(1, h0), G3(0, h0));
            const __m256i q2_2 = _mm256_set_epi32(G3(15, h1), G3(14, h1), G3(13, h1), G3(12, h1),
                                                  G3(11, h1), G3(10, h1), G3(9, h1), G3(8, h1));
#undef G3
            qs += 16;
            __m256i aux = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) u32(signs)), mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) u32(signs + 4)), mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 8;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const int sc = x[i].scales[ib32 / 2];
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short) (2 * (sc & 0xf) + 1)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short) (2 * (sc >> 4) + 1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return hsum_ggml(accumf);
}

template <int TY>
void rows1(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t];
            out[t][r] = TY == 18 ? dot1_iq3_xxs(row, nb, y) : dot1_iq3_s(row, nb, y);
        }
    }
}

}  // namespace

bool iq256_rows1_supported(int type) noexcept { return type == 17 || type == 18 || type == 21 || type == 29; }

void iq256_rows1(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    if (type == 18) rows1<18>(w, row_bytes, n, act, nt, out, r0, r1);
    else if (type == 21) rows1<21>(w, row_bytes, n, act, nt, out, r0, r1);
    else if (type == 29) dot_rows_nt<29>(nt, w, row_bytes, n, act, out, r0, r1);
    else if (type == 17) dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1);
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23 || type == 29 || type == 12 || type == 13;
}

void iq256_set_q5k_fused(bool fused) noexcept { q5k_fused.store(fused); }

#define IQ256_TYPES(F) F(12) F(13) F(16) F(17) F(18) F(21) F(22) F(23) F(29)

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
#define CASE(T) case T: gu_rows_nt<T>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        IQ256_TYPES(CASE)
#undef CASE
        default: break;
    }
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
#define CASE(T) case T: dot_rows_nt<T>(nt, w, row_bytes, n, act, out, r0, r1); break;
        IQ256_TYPES(CASE)
#undef CASE
        default: break;
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d + 16 nibble bytes) against Q8_0 activations.
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once
// per block; every token then costs a sign, a maddubs, a madd and an fmadd.  The arithmetic is ggml's - only
// the order of the float additions differs.
template <int NT>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: iq4nl_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl256_down_rows(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
    }
}

}  // namespace strata::kernels::cpu
