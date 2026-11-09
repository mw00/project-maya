// src/kernels/cpu/glm_lane_row4_avx2.cpp - the glm-dsa port B14.9: the lockstep 4-row decode GEMV kernels of the CPU
// expert lane (AVX2 + FMA + F16C).  See include/strata/kernels/cpu/glm_lane_row4.hpp for the contract.
//
// x86-64 only: cmake/glm_dsa.d/70-glm-lane-row4.cmake builds this file with the AVX2 flags and
// STRATA_GLM_LANE_ROW4_KERNELS where STRATA_CPU_X86 holds (and the option of that name is ON).  Every instruction here
// may be AVX2, so nothing in it runs before glm_lane_row4.cpp's CPU check has passed: that file (the switch, the check
// and the two hooks native_expert.cpp calls on every decode step) is compiled for any x86-64 and only then asks
// row4x86::kernel() for a kernel (glm_lane_row4_x86.hpp).
//
// Structure (after ik_llama.cpp's R4 kernels, iqk_gemm_iquants.cpp / iqk_gemm_kquants.cpp, MIT, without the row
// repack, so the GGUF rows are used as they are): per 32-value chunk the Q8_K activation is loaded once and the four
// rows each decode their own grid / signs / scales against it into their own int32 accumulator; per 256-value
// super-block each row does one fmadd; the four rows' 8-lane accumulators are reduced together.
//
// Exactness against ggml-cpu (ggml/src/ggml-cpu/arch/x86/quants.c, the __AVX2__ branches of ggml_vec_dot_*_q8_K):
//  - the int16 products come from the same maddubs operands per 16-bit lane (grid bytes x signed Q8 bytes, the sign
//    applied with vpsignb, which equals ggml's xor/sub form for every int8 including -128), the same madd scale
//    vectors, so every int32 lane receives the same terms (int32 adds are exact mod 2^32: order is free);
//  - per super-block and lane the same float op: acc = fma(d, (float) sumi, acc), d = fp16(x.d) * y.d (and for Q2_K
//    first acc = fma(-y.d * fp16(x.dmin), (float) madd(mins, bsums), acc));
//  - the same horizontal order ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7)) and the same final scale (0.125 / 0.25 / none);
//  - IQ1_S's scalar delta term (accum1 += d * sumi1 per super-block, then hsum + IQ1S_DELTA * accum1) in the form
//    this ggml-cpu was compiled to (one fmaf or a multiply and an add), picked once by a probe (iq1s_form).
// The file is compiled with -ffp-contract=off so the compiler cannot fuse anything ggml does not.
#include "glm_lane_row4_x86.hpp"

#if defined(STRATA_GLM_LANE_ROW4_KERNELS) && (defined(__x86_64__) || defined(_M_X64))

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

using row4x86::kQ8K;   // block_q8_K: float d, int8 qs[256], int16 bsums[16]

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
#if defined(_MSC_VER) && !defined(__clang__)
    // MSVC has no _cvtsh_ss: the same VCVTPH2PS on lane 0 (ggml-cpu's GGML_CPU_FP16_TO_FP32 for MSVC)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    return _cvtsh_ss(h);
#endif
}
inline float f32(const uint8_t* p) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint16_t u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline __m256i ld(const void* p) { return _mm256_loadu_si256((const __m256i*) p); }

// keven_signs_q2xs: one u64 per 7-bit sign index, byte k = 0xFF (-1) when bit k of ksigns_iq2xs[i] is set (the 8th
// bit is the parity), 0x01 otherwise.  Built on first use (the file is compiled for AVX2; no static initializer).
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
inline const uint64_t* even_signs() {
    static const EvenSigns t;
    return t.v;
}


// four 8-lane accumulators -> four row sums, each in ggml's hsum_float_8 order
inline void reduce4(const __m256 acc[4], float scale, bool scaled, float out[4]) {
    __m128 s0 = _mm_add_ps(_mm256_extractf128_ps(acc[0], 1), _mm256_castps256_ps128(acc[0]));
    __m128 s1 = _mm_add_ps(_mm256_extractf128_ps(acc[1], 1), _mm256_castps256_ps128(acc[1]));
    __m128 s2 = _mm_add_ps(_mm256_extractf128_ps(acc[2], 1), _mm256_castps256_ps128(acc[2]));
    __m128 s3 = _mm_add_ps(_mm256_extractf128_ps(acc[3], 1), _mm256_castps256_ps128(acc[3]));
    _MM_TRANSPOSE4_PS(s0, s1, s2, s3);   // s_m = lane m (= x_m + x_{m+4}) of rows 0..3
    __m128 r = _mm_add_ps(_mm_add_ps(s0, s2), _mm_add_ps(s1, s3));
    if (scaled) r = _mm_mul_ps(_mm_set1_ps(scale), r);
    _mm_storeu_ps(out, r);
}

// ---- shared building blocks.  Port 5 (the one shuffle port on Skylake-SP) is what ggml's kernels saturate: its
// set_epi64x / set1_epi16 from scalars are vmovq + vpinsrq + vinserti128 and vmovd + vpbroadcastw.  Here table
// rows are fetched with load-only broadcasts and merged with vpblendd (ports 0/1/5), and per-chunk scales are
// broadcast from a small per-row array in memory, so the lockstep rows mostly leave port 5 to the scale shuffles.

// four u64 table entries -> set_epi64x(t[i3], t[i2], t[i1], t[i0])
inline __m256i tab4(const uint64_t* t, unsigned i0, unsigned i1, unsigned i2, unsigned i3) {
    const __m256i a = _mm256_set1_epi64x((long long) t[i0]), b = _mm256_set1_epi64x((long long) t[i1]);
    const __m256i c = _mm256_set1_epi64x((long long) t[i2]), d = _mm256_set1_epi64x((long long) t[i3]);
    return _mm256_blend_epi32(_mm256_blend_epi32(a, b, 0x0C), _mm256_blend_epi32(c, d, 0xC0), 0xF0);
}
// eight u32 table entries t[q[0..7]] -> set_epi32(t[q7], ..., t[q0])
inline __m256i tab8(const uint32_t* t, const uint8_t* q) {
    const __m256i v0 = _mm256_set1_epi32((int) t[q[0]]), v1 = _mm256_set1_epi32((int) t[q[1]]);
    const __m256i v2 = _mm256_set1_epi32((int) t[q[2]]), v3 = _mm256_set1_epi32((int) t[q[3]]);
    const __m256i v4 = _mm256_set1_epi32((int) t[q[4]]), v5 = _mm256_set1_epi32((int) t[q[5]]);
    const __m256i v6 = _mm256_set1_epi32((int) t[q[6]]), v7 = _mm256_set1_epi32((int) t[q[7]]);
    const __m256i lo = _mm256_blend_epi32(_mm256_blend_epi32(v0, v1, 0x02), _mm256_blend_epi32(v2, v3, 0x08), 0x0C);
    const __m256i hi = _mm256_blend_epi32(_mm256_blend_epi32(v4, v5, 0x20), _mm256_blend_epi32(v6, v7, 0x80), 0xC0);
    return _mm256_blend_epi32(lo, hi, 0xF0);
}

// ggml's get_scale_shuffle_k4(c): within each 128-bit half, int16 lane c (half 0: the low-nibble scale of 32-value
// chunk c, half 1: its high-nibble scale) to every int16 lane
alignas(32) const uint8_t kShufK4[8][32] = {
#define K4(c) {2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, \
               2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, \
               2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1, 2 * c, 2 * c + 1}
    K4(0), K4(1), K4(2), K4(3), K4(4), K4(5), K4(6), K4(7)
#undef K4
};
// ggml's get_scale_shuffle_q3k(k), k = 0..3: int16 scale 2k to lanes 0-7, 2k+1 to lanes 8-15 (per 128-bit half)
alignas(32) const uint8_t kShufQ3k[4][32] = {
#define Q3(k) {4 * k, 4 * k + 1, 4 * k, 4 * k + 1, 4 * k, 4 * k + 1, 4 * k, 4 * k + 1, 4 * k, 4 * k + 1, 4 * k, 4 * k + 1, \
               4 * k, 4 * k + 1, 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3, \
               4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3, 4 * k + 2, 4 * k + 3}
    Q3(0), Q3(1), Q3(2), Q3(3)
#undef Q3
};
inline __m256i ldc(const uint8_t (&m)[32]) { return _mm256_load_si256((const __m256i*) m); }

// 8 nibble-pair scale bytes -> int16 lanes 0-7 = 2*lo+1 of bytes 0-7, lanes 8-15 = 2*hi+1 (ggml's IQ2_S scales16;
// IQ2_XS's unpacklo form, shuffled with get_scale_shuffle, picks the same values per chunk)
inline __m256i scales16(const uint8_t* sc) {
    const __m128i v = _mm_loadl_epi64((const __m128i*) sc);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i nib = _mm_unpacklo_epi64(_mm_and_si128(v, m4), _mm_and_si128(_mm_srli_epi16(v, 4), m4));
    return _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), _mm_set1_epi8(1)));
}

// the int16 scale 2 * (w >> 28) + 1 of an IQ2_XXS / IQ3_XXS chunk word, in both halves of a u32 (for vpbroadcastd)
inline uint32_t xxs_scale(uint32_t w) { return (2u * (w >> 28) + 1u) * 0x10001u; }
inline __m256i bcast32(uint32_t v) { return _mm256_set1_epi32((int) v); }   // by value: callers read through u32()

// The four rows are four separate streams; the hardware prefetchers follow one sequential stream best, so each row
// asks for the lines kPfBlocks super-blocks ahead itself (past the row's end this touches the next row or the next
// tensor's bytes: a prefetch never faults).
constexpr int kPfBlocks = 4;
inline void prefetch_ahead(const uint8_t* xb, size_t B) {
    const char* p = (const char*) xb + kPfBlocks * B;
    _mm_prefetch(p, _MM_HINT_T0);
    _mm_prefetch(p + 64, _MM_HINT_T0);
}

#if defined(_MSC_VER) && !defined(__clang__)   // (cl.exe has no unroll pragma: warning C4068 on each of these loops)
#define ROW4_FOR(r) for (int r = 0; r < 4; ++r)
#else
#define ROW4_FOR(r) _Pragma("GCC unroll 4") for (int r = 0; r < 4; ++r)
#endif
#define ROW4_ACC_INIT __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}

// ---- IQ2_XXS (66 B: d, qs[32] u16; per 32 values u32 of 4 grid bytes, u32 of 4x7 sign bits | scale << 28)
void k_iq2_xxs(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 66;
    const uint64_t* sg = even_signs();
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        __m256i sumi[4];
        alignas(32) uint32_t scl[4][8];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) scl[r][ib] = xxs_scale(u32(xb[r] + 2 + 8 * ib + 4));
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const uint8_t* q = xb[r] + 2 + 8 * ib;
                const uint32_t s = u32(q + 4);
                const __m256i g = tab4(iq2xxs_grid, q[0], q[1], q[2], q[3]);
                const __m256i sgv = tab4(sg, s & 127, (s >> 7) & 127, (s >> 14) & 127, (s >> 21) & 127);
                const __m256i dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(y8, sgv));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, bcast32(scl[r][ib])));
            }
        }
        ROW4_FOR(r) {
            const float d = h2f(xb[r]) * yd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 0.125f, true, out);
}

// ---- IQ2_XS (74 B: d, qs[32] u16 = 9-bit grid index | 7-bit sign index << 9, scales[8])
void k_iq2_xs(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 74;
    const uint64_t* sg = even_signs();
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        alignas(32) __m256i sv[4][8];
        __m256i sumi[4];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            const __m256i s16 = scales16(xb[r] + 66);
            for (int ib = 0; ib < 8; ++ib) sv[r][ib] = _mm256_shuffle_epi8(s16, ldc(kShufK4[ib]));
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const uint8_t* q = xb[r] + 2 + 8 * ib;
                const unsigned v0 = u16(q), v1 = u16(q + 2), v2 = u16(q + 4), v3 = u16(q + 6);
                const __m256i g = tab4(iq2xs_grid, v0 & 511, v1 & 511, v2 & 511, v3 & 511);
                const __m256i sgv = tab4(sg, v0 >> 9, v1 >> 9, v2 >> 9, v3 >> 9);
                const __m256i dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(y8, sgv));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, sv[r][ib]));
            }
        }
        ROW4_FOR(r) {
            const float d = h2f(xb[r]) * yd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 0.125f, true, out);
}

// ---- IQ2_S (82 B: d, qs[64] = 32 grid low bytes + 32 sign bytes, qh[8], scales[8])
void k_iq2_s(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 82;
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i one = _mm256_set1_epi8(1);
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        alignas(32) __m256i sv[4][8];
        __m256i sumi[4];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            const __m256i s16 = scales16(xb[r] + 74);
            for (int ib = 0; ib < 8; ++ib) sv[r][ib] = _mm256_shuffle_epi8(s16, ldc(kShufK4[ib]));
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const uint8_t* qs = xb[r] + 2 + 4 * ib;
                const unsigned qh = xb[r][66 + ib];
                const __m256i g = tab4(iq2s_grid, qs[0] | ((qh << 8) & 0x300), qs[1] | ((qh << 6) & 0x300),
                                       qs[2] | ((qh << 4) & 0x300), qs[3] | ((qh << 2) & 0x300));
                const __m256i b = _mm256_shuffle_epi8(bcast32(u32(xb[r] + 2 + 32 + 4 * ib)), spread);
                const __m256i sgv = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, sel), sel), one);
                const __m256i dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(y8, sgv));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, sv[r][ib]));
            }
        }
        ROW4_FOR(r) {
            const float d = h2f(xb[r]) * yd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 0.125f, true, out);
}

// ---- IQ3_XXS (98 B: d, qs[64] grid bytes, then 8 x u32 = 4x7 sign bits | scale << 28)
void k_iq3_xxs(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 98;
    const uint64_t* sg = even_signs();
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        __m256i sumi[4];
        alignas(32) uint32_t scl[4][8];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) scl[r][ib] = xxs_scale(u32(xb[r] + 66 + 4 * ib));
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const uint32_t s = u32(xb[r] + 66 + 4 * ib);
                const __m256i g = tab8(iq3xxs_grid, xb[r] + 2 + 8 * ib);
                const __m256i sgv = tab4(sg, s & 127, (s >> 7) & 127, (s >> 14) & 127, (s >> 21) & 127);
                const __m256i dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(y8, sgv));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, bcast32(scl[r][ib])));
            }
        }
        ROW4_FOR(r) {
            const float d = h2f(xb[r]) * yd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 0.25f, true, out);
}

// both 128-bit halves = the low (H = 0) or high (H = 1) eight int16 scales of a Q2_K / Q3_K super-block
template <int H> inline __m256i kscales(__m128i s8) {
    const __m128i v = _mm_cvtepi8_epi16(H ? _mm_unpackhi_epi64(s8, s8) : s8);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(v), v, 1);
}

// one 128-value half (J) of a Q2_K super-block for one row: 4 x 32 values, 2-bit quants at shifts 0/2/4/6; sv = the
// row's four shuffled scale vectors of this half (precomputed in memory: they become madd memory operands)
template <int J> inline __m256i q2k_half(const uint8_t* xb, const __m256i* sv, const __m256i y8[4]) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i qb = ld(xb + 16 + 32 * J);
    const __m256i p0 = _mm256_madd_epi16(sv[0], _mm256_maddubs_epi16(_mm256_and_si256(qb, m3), y8[0]));
    const __m256i p1 = _mm256_madd_epi16(sv[1], _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qb, 2), m3), y8[1]));
    const __m256i p2 = _mm256_madd_epi16(sv[2], _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m3), y8[2]));
    const __m256i p3 = _mm256_madd_epi16(sv[3], _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qb, 6), m3), y8[3]));
    return _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
}

// a K-quant super-block's 16 int16 scales (sc8, bytes) -> the eight per-chunk madd vectors ggml builds with
// get_scale_shuffle_q3k: sv[4J + K] for half J, chunk K
inline void kscale_vecs(__m128i sc8, __m256i* sv) {
    const __m256i s0 = kscales<0>(sc8), s1 = kscales<1>(sc8);
    for (int k = 0; k < 4; ++k) {
        sv[k] = _mm256_shuffle_epi8(s0, ldc(kShufQ3k[k]));
        sv[4 + k] = _mm256_shuffle_epi8(s1, ldc(kShufQ3k[k]));
    }
}

// ---- Q2_K (84 B: scales[16] = scale | min << 4, qs[64], d, dmin)
void k_q2_k(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 84;
    const __m128i m4 = _mm_set1_epi8(0xF);
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const __m256i bsums = ld(yb + 260);
        const uint8_t* xb[4];
        alignas(32) __m256i sv[4][8];
        __m256i sumi[4];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            const float dmin = -yd * h2f(xb[r] + 82);
            const __m128i ms = _mm_loadu_si128((const __m128i*) xb[r]);
            kscale_vecs(_mm_and_si128(ms, m4), sv[r]);
            const __m256i prod = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_and_si128(_mm_srli_epi16(ms, 4), m4)), bsums);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc[r]);
        }
        {
            const __m256i y8[4] = {ld(q8), ld(q8 + 32), ld(q8 + 64), ld(q8 + 96)};
            ROW4_FOR(r) sumi[r] = q2k_half<0>(xb[r], sv[r], y8);
        }
        {
            const __m256i y8[4] = {ld(q8 + 128), ld(q8 + 160), ld(q8 + 192), ld(q8 + 224)};
            ROW4_FOR(r) sumi[r] = _mm256_add_epi32(sumi[r], q2k_half<1>(xb[r], sv[r] + 4, y8));
        }
        ROW4_FOR(r) {
            const float d = yd * h2f(xb[r] + 80);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 1.0f, false, out);
}

// one 32-value chunk (K) of a Q3_K half (J): low 2 bits at shift 2K, high bit (inverted, x4) at bit 4J+K
template <int J, int K> inline __m256i q3k_chunk(__m256i qb, __m256i hb, const __m256i* sv, __m256i y8) {
    constexpr int bit = 4 * J + K;
    const __m256i q3l = _mm256_and_si256(_mm256_srli_epi16(qb, 2 * K), _mm256_set1_epi8(3));
    const __m256i q3h = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hb, _mm256_set1_epi8((char) (1 << bit))), bit), 2);
    const __m256i p16 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l, y8), _mm256_maddubs_epi16(q3h, y8));
    return _mm256_madd_epi16(sv[K], p16);
}
template <int J> inline __m256i q3k_half(const uint8_t* xb, const __m256i* sv, const __m256i y8[4]) {
    const __m256i qb = ld(xb + 32 + 32 * J), hb = ld(xb);
    const __m256i p0 = q3k_chunk<J, 0>(qb, hb, sv, y8[0]), p1 = q3k_chunk<J, 1>(qb, hb, sv, y8[1]);
    const __m256i p2 = q3k_chunk<J, 2>(qb, hb, sv, y8[2]), p3 = q3k_chunk<J, 3>(qb, hb, sv, y8[3]);
    return _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
}

// ---- Q3_K (110 B: hmask[32], qs[64], scales[12] (6-bit, packed), d)
void k_q3_k(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 110;
    constexpr uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const __m128i m32 = _mm_set1_epi8(32);
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        alignas(32) __m256i sv[4][8];
        __m256i sumi[4];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            uint32_t aux[3];
            std::memcpy(aux, xb[r] + 96, 12);
            const __m128i s128 = _mm_set_epi32((int) (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
                                               (int) (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
                                               (int) ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
                                               (int) ((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)));
            kscale_vecs(_mm_sub_epi8(s128, m32), sv[r]);
        }
        {
            const __m256i y8[4] = {ld(q8), ld(q8 + 32), ld(q8 + 64), ld(q8 + 96)};
            ROW4_FOR(r) sumi[r] = q3k_half<0>(xb[r], sv[r], y8);
        }
        {
            const __m256i y8[4] = {ld(q8 + 128), ld(q8 + 160), ld(q8 + 192), ld(q8 + 224)};
            ROW4_FOR(r) sumi[r] = _mm256_add_epi32(sumi[r], q3k_half<1>(xb[r], sv[r] + 4, y8));
        }
        ROW4_FOR(r) {
            const float d = yd * h2f(xb[r] + 108);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 1.0f, false, out);
}

inline int hsum4_i32(__m128i v) {   // int32 adds are exact: any order
    v = _mm_add_epi32(v, _mm_unpackhi_epi64(v, v));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 1));
    return _mm_cvtsi128_si32(v);
}

// IQ1_S: qh[ib] into the four int16 lanes of chunk ib (per 128-bit half of a broadcast qh[0..7]): chunks 0-3, 4-7
alignas(32) const uint8_t kShufIq1[2][32] = {
    {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7},
    {8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15}};

// one IQ1_S super-block of one row, before its chunks: the 32 grid indices qs[k] | ((qh[k / 4] >> 3 (k % 4)) & 7) << 8
// (ggml's per-chunk shifts << 8, << 5, << 2, >> 1, then & 0x700) to idx, the chunk scales ls = 2 ((qh >> 12) & 7) + 1
// in both int16 halves of a u32 to scl, and ggml's integer delta sum (returned)
//   sumi1 = sum_ib (bsums[2 ib] + bsums[2 ib + 1]) * (qh[ib] & 0x8000 ? -1 : 1) * ls_ib,
// bp = the eight bsums pairs as int16 (|pair| <= 4096), shared by the four rows
inline int iq1s_prep(const uint8_t* xb, __m128i bp, uint16_t* idx, uint32_t* scl) {
    const __m256i qh2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (xb + 34)));
    const __m128i qh = _mm256_castsi256_si128(qh2);
    // lane k of a chunk: qh * 256 / 32 / 4 / 0 (low 16 bits) | qh * 0 / 0 / 0 / 2^15 (high 16 bits, = qh >> 1)
    const __m256i ml = _mm256_setr_epi16(256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0);
    const __m256i mh = _mm256_setr_epi16(0, 0, 0, (short) 0x8000, 0, 0, 0, (short) 0x8000, 0, 0, 0, (short) 0x8000,
                                         0, 0, 0, (short) 0x8000);
    const __m256i m7 = _mm256_set1_epi16(0x700);
    for (int h = 0; h < 2; ++h) {
        const __m256i hq = _mm256_shuffle_epi8(qh2, ldc(kShufIq1[h]));
        const __m256i hi = _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(hq, ml), _mm256_mulhi_epu16(hq, mh)), m7);
        const __m256i lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (xb + 2 + 16 * h)));
        _mm256_store_si256((__m256i*) (idx + 16 * h), _mm256_or_si256(lo, hi));
    }
    const __m128i ls = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), _mm_set1_epi16(7)), 1),
                                     _mm_set1_epi16(1));
    _mm_store_si128((__m128i*) scl, _mm_unpacklo_epi16(ls, ls));
    _mm_store_si128((__m128i*) (scl + 4), _mm_unpackhi_epi16(ls, ls));
    // -ls where bit 15 of qh is set (| 1: a zero qh must keep +ls under vpsignw)
    const __m128i sls = _mm_sign_epi16(ls, _mm_or_si128(qh, _mm_set1_epi16(1)));
    return hsum4_i32(_mm_madd_epi16(sls, bp));
}

// ---- IQ1_S (50 B: d, qs[32] grid low bytes, qh[8] u16 = 4 x 3 grid high bits | scale << 12 | delta sign << 15).
// ggml's AVX2 branch: per 32 values sumi += madd(mul_add_epi8(grid, q8), ls) (grid bytes -1 / 0 / +1, so its
// sign(x, x) / sign(y, x) form is kept as is), the delta sum sumi1 in integers, per super-block acc = fma(d, sumi, acc)
// and accum1 += d * sumi1 in scalar C, at the end hsum(acc) + IQ1S_DELTA * accum1.  kFused: that C as ggml-cpu's
// gnu11 build with FMA on compiles it (GCC's default -ffp-contract=fast: accum1 = fmaf(d, sumi1, accum1)); otherwise
// a separate multiply and add.  The last step is exact either way (a power-of-two product), so it follows suit.
template <bool kFused> void k_iq1_s(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 50;
    ROW4_ACC_INIT;
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const __m128i bp = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) (yb + 260)),
                                          _mm_loadu_si128((const __m128i*) (yb + 276)));
        const uint8_t* xb[4];
        __m256i sumi[4];
        int sumi1[4];
        alignas(32) uint16_t idx[4][32];
        alignas(32) uint32_t scl[4][8];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            sumi1[r] = iq1s_prep(xb[r], bp, idx[r], scl[r]);
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const uint16_t* ix = idx[r] + 4 * ib;
                const __m256i g = tab4(iq1s_grid, ix[0], ix[1], ix[2], ix[3]);
                const __m256i dot = _mm256_maddubs_epi16(_mm256_sign_epi8(g, g), _mm256_sign_epi8(y8, g));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, bcast32(scl[r][ib])));
            }
        }
        ROW4_FOR(r) {
            const float d = yd * h2f(xb[r]);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
            acc1[r] = kFused ? std::fma(d, (float) sumi1[r], acc1[r]) : acc1[r] + d * (float) sumi1[r];
        }
    }
    reduce4(acc, 1.0f, false, out);
    for (int r = 0; r < 4; ++r)
        out[r] = kFused ? std::fma(IQ1S_DELTA, acc1[r], out[r]) : out[r] + IQ1S_DELTA * acc1[r];
}

// IQ4_XS chunk scales ((scales_l[ib / 2] >> 4 (ib & 1)) & 15 | ((scales_h >> 2 ib) & 3) << 4) - 32, as ggml's ls1 / ls2,
// in both int16 halves of a u32 (for vpbroadcastd)
inline void iq4xs_scales(const uint8_t* xb, uint32_t* scl) {
    const __m128i sl = _mm_cvtsi32_si128((int) u32(xb + 4));
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i lo8 = _mm_unpacklo_epi8(_mm_and_si128(sl, m4), _mm_and_si128(_mm_srli_epi16(sl, 4), m4));   // byte ib
    // u32(xb + 2) = scales_h | scales_l[0..1] << 16: shifts up to 14 and & 3 read scales_h only
    const __m256i sh = _mm256_srlv_epi32(_mm256_set1_epi32((int) u32(xb + 2)),
                                         _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
    const __m256i hi = _mm256_slli_epi32(_mm256_and_si256(sh, _mm256_set1_epi32(3)), 4);
    const __m256i ls = _mm256_sub_epi32(_mm256_or_si256(_mm256_cvtepu8_epi32(lo8), hi), _mm256_set1_epi32(32));
    const __m256i lo16 = _mm256_and_si256(ls, _mm256_set1_epi32(0xFFFF));
    _mm256_store_si256((__m256i*) scl, _mm256_or_si256(lo16, _mm256_slli_epi32(ls, 16)));
}

// ---- IQ4_XS (136 B: d, scales_h u16, scales_l[4], qs[128]: chunk ib's 16 bytes hold values 0-15 in the low nibbles
// and 16-31 in the high ones).  ggml's AVX2 branch: kvalues_iq4nl[nibble] (lo nibbles to the low 128 bits, hi to the
// high), mul_add_epi8 against q8, madd by the chunk scale; per super-block acc = fma(fp16(x.d) * y.d, sumi, acc).
// Here the 16 bytes are broadcast to both halves and one vpshufb looks both nibble sets up.
void k_iq4_xs(int nb, const uint8_t* const w[4], const uint8_t* y, float out[4]) {
    constexpr size_t B = 136;
    const __m256i values = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const __m256i m4 = _mm256_set1_epi8(0xF);
    ROW4_ACC_INIT;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* yb = y + (size_t) i * kQ8K;
        const float yd = f32(yb);
        const int8_t* q8 = (const int8_t*) (yb + 4);
        const uint8_t* xb[4];
        __m256i sumi[4];
        alignas(32) uint32_t scl[4][8];
        ROW4_FOR(r) {
            xb[r] = w[r] + (size_t) i * B;
            prefetch_ahead(xb[r], B);
            sumi[r] = _mm256_setzero_si256();
            iq4xs_scales(xb[r], scl[r]);
        }
        for (int ib = 0; ib < 8; ++ib) {
            const __m256i y8 = ld(q8 + 32 * ib);
            ROW4_FOR(r) {
                const __m256i qb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (xb[r] + 8 + 16 * ib)));
                const __m256i nib = _mm256_and_si256(_mm256_blend_epi32(qb, _mm256_srli_epi16(qb, 4), 0xF0), m4);
                const __m256i q4 = _mm256_shuffle_epi8(values, nib);
                const __m256i dot = _mm256_maddubs_epi16(_mm256_sign_epi8(q4, q4), _mm256_sign_epi8(y8, q4));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, bcast32(scl[r][ib])));
            }
        }
        ROW4_FOR(r) {
            const float d = h2f(xb[r]) * yd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    reduce4(acc, 1.0f, false, out);
}

}  // namespace

namespace row4x86 {

Kernel kernel(int ggml_type, bool iq1s_fused) {
    switch (ggml_type) {
        case 10: return k_q2_k;
        case 11: return k_q3_k;
        case 16: return k_iq2_xxs;
        case 17: return k_iq2_xs;
        case 18: return k_iq3_xxs;
        case 19:
            if (iq1s_fused) return k_iq1_s<true>;
            return k_iq1_s<false>;
        case 22: return k_iq2_s;
        case 23: return k_iq4_xs;
        default: return nullptr;
    }
}

}  // namespace row4x86
}  // namespace strata::kernels::cpu

#endif  // STRATA_GLM_LANE_ROW4_KERNELS on x86-64
