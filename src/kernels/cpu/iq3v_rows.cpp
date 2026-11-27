// src/kernels/cpu/iq3v_rows.cpp - the CPU lane's IQ3_XXS and IQ4_XS rows, second generation, and the rows of its 4-bit
// IQ2_XXS copies (Q4X): ggml-cpu's dot to the bit, faster.  ds4_iq_cpu.cpp checks each form against ggml-cpu's dot and
// takes the fastest this CPU runs (own(): g_iq3v / g_iq4v; iq_cpu_q4x_ok: g_q4xv).  Built for the baseline x86-64
// with each function's instruction set its own (GCC / Clang target attributes; MSVC: /arch:AVX2 on this file, and the
// AVX-512 forms in iq3v_rows_z.cpp under /arch:AVX512 - IQ3V_PART below), so a form runs only where the caller found
// the CPU takes it.
//
// The arithmetic is ggml's: per block the same exact int32 lanes (lane l: every 32-value chunk's values 4 l .. 4 l + 3
// times the chunk's scale), then fma(d_x d_y, lanes, acc) block by block and hsum_float_8 times the format's constant
// (IQ3_XXS 1/4, IQ4_XS 1, Q4X 1/8 as IQ2_XXS).  What differs is how a chunk's operands are made:
//  - IQ3_XXS: a block's 8 sign / scale words taken as one vector (their 7-bit fields spread to bytes, each byte's
//    parity in its bit 7 by a nibble vpshufb - ggml's ksigns_iq2xs in registers -, grouped so that one vpshufb a
//    chunk lays its four sign bytes over its 8-value groups, a bit test + 0x7F making the vpsignb operand), its scales
//    likewise; the grid by vpgatherdd (forms 1 / 2: a row at a time, a pair of chunks a step - Golden Cove's best,
//    whose gather is faster than 8 loads + blends) or by broadcasts + blends (forms 3 / 4: a call's rows one stream
//    of blocks, the signs / scales a block ahead - Zen 5's best AVX2 form); form 5 (AVX-512 VBMI BITALG VNNI): the
//    grid's level codes in 8 zmm (vpermi2w lookups, a block ahead), a pair of chunks a zmm, the signs by
//    vpmultishiftqb + vpermi2b of ksigns_iq2xs and one vpshufbitqmb mask a pair, q8 negated under it, vpdpwssd;
//  - IQ4_XS: a call's rows one stream of blocks, a chunk's values by one broadcast + vpsrlvd + and + vpshufb of
//    kvalues_iq4nl, |w| once for every token (vpabsb), the block's scales at its start; AVX-512: a pair of chunks a
//    zmm, the negatives a vpmovb2m mask;
//  - Q4X: |v| and the signed v by two vpshufb (the activation signed by v, one op a token where the earlier rows took
//    three); AVX-512: a pair of chunks a zmm (its 32 code bytes' low then high nibbles: q8's order).
// Every form gives the same bits (own_check / q4x_check against ggml-cpu's dot, and the lane's own test), so the
// choice is time only.  No Q8_K activation holds -128 (ggml's quantizer: -127 .. 127), so negating by vpsignb or a
// masked subtract never wraps differently from ggml's.

#include "maya/kernels/cpu/iq3v_rows.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <atomic>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define IQX_TARGET(s) __attribute__((target(s)))
#else
#define IQX_TARGET(s)
#endif
#define IQ_AVX2 IQX_TARGET("avx2,fma")
// AVX-VNNI's intrinsics: GCC 11, Clang 12, MSVC 2022 (as iq2v_rows.cpp)
// (MAYA_NO_AVXVNNI: the assembler has no AVX-VNNI - Rocky / RHEL 9's binutils 2.35 - so the tier is left out,
//  vnni_form() says so and the AVX2 form runs in its place; CMake checks it)
#if !defined(MAYA_NO_AVXVNNI) && \
    (defined(_MSC_VER) || (defined(__clang__) && __clang_major__ >= 12) || \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define IQ3V_AVXVNNI 1
#endif
// MSVC's /arch is a file's (#1235), as in iq2v_rows.cpp: on MSVC (CMake: MAYA_IQV_SPLIT) this file holds the AVX2 forms
// and the dispatch (part 1), and iq3v_rows_z.cpp compiles the AVX-512 forms alone under /arch:AVX512 (part 2, pass_z).
// GCC / Clang: one file, each function's own target attribute (part 0) - the same code as before.
#if !defined(IQ3V_PART)
#if defined(MAYA_IQV_SPLIT)
#define IQ3V_PART 1
#else
#define IQ3V_PART 0
#endif
#endif
#if defined(_MSC_VER)
#define IQ3V_FI __forceinline
#else
#define IQ3V_FI inline __attribute__((always_inline))
#endif

namespace maya::kernels::cpu::iq3v {
namespace {
inline float h2f(uint16_t h) {   // F16 -> F32, exactly
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
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
IQ_AVX2 inline float hsum_ggml(__m256 x) {   // ggml's hsum_float_8, the same additions in the same order
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
// IQ4_XS: the block's 8 chunk scales (ggml's ls: low nibble from scales_l, bits 4-5 from scales_h, less 32), each as
// both int16 words of its dword
inline void scales4(const uint8_t* blk, int32_t* scs) {
    uint16_t sh;
    std::memcpy(&sh, blk + 2, 2);
    const uint32_t sl = u32(blk + 4);
    for (int c = 0; c < 8; ++c) {
        const int ls = (int) (((sl >> (4 * c)) & 15u) | (((unsigned) sh >> (2 * c)) & 3u) << 4) - 32;
        const uint32_t w = (uint16_t) (int16_t) ls;
        scs[c] = (int32_t) (w | (w << 16));
    }
}

struct Tabs {
    alignas(64) uint8_t ctrl3[8][32];   // IQ3_XXS chunk c: its four sign bytes over its 8-value groups (vpshufb of T)
    alignas(64) uint8_t bit[64];        // form 5: a pair's byte b <- bit 8 (group) + (value) of its chunk's qword
    alignas(64) uint8_t sc[4][64];      // form 5: pair p's scales (word w <- the low word of chunk 2 p + w / 16)
    alignas(64) uint8_t fld[64];        // form 5: vpmultishiftqb's offsets of a word's four 7-bit fields
    alignas(64) uint16_t pidx[2][32];   // form 5: a 64-value group h: output qword m <- codes 16 h + 2 m, + 1
    alignas(64) uint8_t qsi[4][64];     // Q4X AVX-512, pair p: byte b <- scale 2 p + b / 32
    alignas(64) uint16_t isi[4][32];    // IQ4_XS AVX-512, pair p: word w <- scale 2 p + w / 16
    constexpr Tabs() : ctrl3(), bit(), sc(), fld(), pidx(), qsi(), isi() {
        for (int c = 0; c < 8; ++c)
            for (int b = 0; b < 32; ++b) ctrl3[c][b] = (uint8_t) (2 * c + ((b & 15) < 8 ? 0 : 1));
        for (int b = 0; b < 64; ++b) {
            bit[b] = (uint8_t) (8 * ((b / 8) % 4) + b % 8);
            fld[b] = (uint8_t) ((b % 8) < 4 ? 7 * (b % 8) : 0);
        }
        for (int p = 0; p < 4; ++p)
            for (int b = 0; b < 64; ++b) {
                sc[p][b] = (uint8_t) (8 * (2 * p + b / 32) + (b & 1));
                qsi[p][b] = (uint8_t) (2 * p + b / 32);
            }
        for (int h = 0; h < 2; ++h)
            for (int m = 0; m < 8; ++m) {
                pidx[h][4 * m] = pidx[h][4 * m + 1] = (uint16_t) (16 * h + 2 * m);
                pidx[h][4 * m + 2] = pidx[h][4 * m + 3] = (uint16_t) (16 * h + 2 * m + 1);
            }
        for (int p = 0; p < 4; ++p)
            for (int w = 0; w < 32; ++w) isi[p][w] = (uint16_t) (2 * p + w / 16);
    }
};
constexpr Tabs kT{};

// IQ3_XXS's grid entries as four 3-bit level codes (4, 12, .., 52, 62 -> 0 .. 7); ok false if a byte is no level
struct Codes {
    alignas(64) uint16_t c[256];
    bool ok = true;
    Codes() {
        static const int lx[8] = {4, 12, 20, 28, 36, 44, 52, 62};
        for (int i = 0; i < 256; ++i) {
            int v = 0;
            for (int k = 0; k < 4; ++k) {
                const int b = (int) ((iq3xxs_grid[i] >> (8 * k)) & 255u);
                int l = -1;
                for (int j = 0; j < 8; ++j)
                    if (lx[j] == b) l = j;
                if (l < 0) ok = false;
                v |= (l < 0 ? 0 : l) << (3 * k);
            }
            c[i] = (uint16_t) v;
        }
    }
};
const Codes& codes() {
    static const Codes c;
    return c;
}

#if IQ3V_PART != 2
#define IQ3A_NS a2
#define IQ3A_ATTR IQX_TARGET("avx2,fma")
#define IQ3A_VNNI 0
#include "iq3v_avx2.inc"
#if defined(IQ3V_AVXVNNI)
#define IQ3A_NS a2v
#define IQ3A_ATTR IQX_TARGET("avx2,fma,avxvnni")
#define IQ3A_VNNI 1
#include "iq3v_avx2.inc"
#endif
#endif

// ---- the AVX-512 forms
#if IQ3V_PART != 1
#define IQ3V_Z IQX_TARGET("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512bitalg,avx512vnni,avx2,fma")
IQ3V_Z IQ3V_FI __m512i lookup4(const __m512i* T, __m512i idx) {   // the codes of 32 word indices (0 .. 255)
    __m512i r[4];
    for (int j = 0; j < 4; ++j) r[j] = _mm512_permutex2var_epi16(T[2 * j], idx, T[2 * j + 1]);
    const __mmask32 m6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 m7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    return _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[0], r[1]), _mm512_mask_blend_epi16(m6, r[2], r[3]));
}
struct Z3 {
    __m512i T[8], lut, ctl, seven, fld, ctlb, ks0, ks1, one, zero, pidx[2];
    IQ3V_Z Z3() {
        const Codes& cs = codes();
        for (int k = 0; k < 8; ++k) T[k] = _mm512_load_si512((const void*) (cs.c + 32 * k));
        // code -> grid byte in every 128-bit lane; byte b of an output qword <- the 3 bits at [0, 3, 6, 9] (its first
        // entry's code, word 0) or [32, 35, 38, 41] (the second's, word 2)
        lut = _mm512_set4_epi32(0, 0, 0x3E342C24, 0x1C140C04);
        ctl = _mm512_set1_epi64(0x2926232009060300LL);
        seven = _mm512_set1_epi8(7);
        fld = _mm512_load_si512(kT.fld);
        ctlb = _mm512_load_si512(kT.bit);
        ks0 = _mm512_loadu_si512(ksigns_iq2xs);
        ks1 = _mm512_loadu_si512(ksigns_iq2xs + 64);
        one = _mm512_set1_epi64(1);
        zero = _mm512_setzero_si512();
        pidx[0] = _mm512_load_si512(kT.pidx[0]);
        pidx[1] = _mm512_load_si512(kT.pidx[1]);
    }
};
// a block's codes (c0: chunks 0-3, c1: 4-7), its sign bytes (qword c: chunk c's four) and scales (qword c: 2 s + 1)
IQ3V_Z IQ3V_FI void prepz3(const Z3& Z, const uint8_t* blk, __m512i& c0, __m512i& c1, __m512i& S, __m512i& SC) {
    const __m512i q = _mm512_loadu_si512((const void*) (blk + 2));
    c0 = lookup4(Z.T, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(q)));
    c1 = lookup4(Z.T, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(q, 1)));
    const __m512i R = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*) (blk + 2 + QK_K / 4)));
    S = _mm512_permutex2var_epi8(Z.ks0, _mm512_multishift_epi64_epi8(Z.fld, R), Z.ks1);
    SC = _mm512_or_si512(_mm512_srli_epi64(R, 27), Z.one);
}
IQ3V_Z inline __m256i fold(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }
template <int NT>
IQ3V_Z void rows3z(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const Z3 Z;
    const __mmask64 mw = 0x5555555555555555ull;
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq3_xxs);
    __m512i nC0, nC1, nS, nSC;
    prepz3(Z, blk, nC0, nC1, nS, nSC);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const __m512i C0 = nC0, C1 = nC1, S = nS, SC = nSC;
        prepz3(Z, b + 1 < nbt ? blk + BB : blk, nC0, nC1, nS, nSC);   // (the next block's, ahead)
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
        for (int p = 0; p < 4; ++p) {
            const __m512i rep = _mm512_permutexvar_epi16(Z.pidx[p & 1], p >> 1 ? C1 : C0);
            const __m512i g = _mm512_shuffle_epi8(Z.lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(Z.ctl, rep), Z.seven));
            const __m512i Sp = _mm512_permutexvar_epi64(
                _mm512_set_epi64(2 * p + 1, 2 * p + 1, 2 * p + 1, 2 * p + 1, 2 * p, 2 * p, 2 * p, 2 * p), S);
            const __mmask64 neg = _mm512_bitshuffle_epi64_mask(Sp, Z.ctlb);
            const __m512i cv = _mm512_maskz_permutexvar_epi8(mw, _mm512_load_si512(kT.sc[p]), SC);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8 = _mm512_loadu_si512(y[t][i].qs + 64 * p);
                const __m512i d = _mm512_maddubs_epi16(g, _mm512_mask_sub_epi8(q8, neg, Z.zero, q8));
                acc[t] = _mm512_dpwssd_epi32(acc[t], d, cv);
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.25f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}
// IQ4_XS: a pair of chunks a zmm (its 32 bytes as 128-bit lanes low nibbles of chunk 2 p, high, then 2 p + 1's)
template <int NT>
IQ3V_Z void rows4z(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const __m512i k0f = _mm512_set1_epi8(0x0F), zero = _mm512_setzero_si512();
    const __m512i sh = _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0);
    const __m512i lanes = _mm512_set_epi64(3, 2, 3, 2, 1, 0, 1, 0);
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq4_xs);
    alignas(32) int32_t scs[8];
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        scales4(blk, scs);
        const __m512i S = _mm512_castsi256_si512(_mm256_load_si256((const __m256i*) scs));   // (word 2 c: chunk c's)
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
        for (int p = 0; p < 4; ++p) {
            const __m512i x = _mm512_permutexvar_epi64(lanes, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) (blk + 8 + 32 * p))));
            const __m512i v = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srlv_epi64(x, sh), k0f));
            const __m512i a = _mm512_abs_epi8(v);
            const __mmask64 m = _mm512_movepi8_mask(v);
            const __m512i cv = _mm512_permutexvar_epi16(_mm512_slli_epi16(_mm512_load_si512(kT.isi[p]), 1), S);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8 = _mm512_loadu_si512(y[t][i].qs + 64 * p);
                const __m512i d = _mm512_maddubs_epi16(a, _mm512_mask_sub_epi8(q8, m, zero, q8));
                acc[t] = _mm512_dpwssd_epi32(acc[t], d, cv);
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}
// Q4X: a pair of chunks a zmm (the 32 code bytes' low nibbles, then high: q8's order), |v| by vpshufb, the
// negatives (codes 3-5) a compare mask, the pair's scales one vpermb
template <int NT>
IQ3V_Z void rowsqz(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m512i k0f = _mm512_set1_epi8(0x0F), zero = _mm512_setzero_si512(), two = _mm512_set1_epi8(2);
    const __m512i tab = _mm512_broadcast_i32x4(_mm_setr_epi8(8, 25, 43, 8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    const __mmask64 mw = 0x5555555555555555ull;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        const int8_t* sc = (const int8_t*) (row + (size_t) nb * 128);
        const uint8_t* dd = row + (size_t) nb * (128 + 8);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            __m512i acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
            long long s8q;
            std::memcpy(&s8q, sc + 8 * i, 8);
            const __m512i S8 = _mm512_set1_epi64(s8q);
            for (int p = 0; p < 4; ++p) {
                const __m256i x = _mm256_loadu_si256((const __m256i*) (row + (size_t) i * 128 + 32 * p));
                const __m512i n = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(x), _mm256_srli_epi16(x, 4), 1), k0f);
                const __m512i a = _mm512_shuffle_epi8(tab, n);
                const __mmask64 m = _mm512_cmpgt_epi8_mask(n, two);
                const __m512i cv = _mm512_maskz_permutexvar_epi8(mw, _mm512_load_si512(kT.qsi[p]), S8);
                for (int t = 0; t < NT; ++t) {
                    const __m512i q8 = _mm512_loadu_si512(y[t][i].qs + 64 * p);
                    const __m512i d = _mm512_maddubs_epi16(a, _mm512_mask_sub_epi8(q8, m, zero, q8));
                    acc[t] = _mm512_dpwssd_epi32(acc[t], d, cv);
                }
            }
            uint16_t dh;
            std::memcpy(&dh, dd + 2 * i, 2);
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = 0.125f * hsum_ggml(accf[t]);
    }
}
#undef IQ3V_Z
#endif  // IQ3V_PART != 1

#if IQ3V_PART == 2
}  // namespace
template <int NT>
void pass_zt(int f, const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* o, int r0, int r1) {
    if (f == 0) rows3z<NT>(w, rb, nb, y, o, r0, r1);
    else if (f == 1) rows4z<NT>(w, rb, nb, y, o, r0, r1);
    else rowsqz<NT>(w, rb, nb, y, o, r0, r1);
}
// the AVX-512 forms' rows for part 1's pass (format f: 0 IQ3_XXS form 5, 1 IQ4_XS form 3, 2 Q4X form 3), nt 1-4 tokens
void pass_z(int f, int nt, const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* o, int r0,
            int r1) {
    switch (nt) {
        case 1: pass_zt<1>(f, w, rb, nb, y, o, r0, r1); break;
        case 2: pass_zt<2>(f, w, rb, nb, y, o, r0, r1); break;
        case 3: pass_zt<3>(f, w, rb, nb, y, o, r0, r1); break;
        default: pass_zt<4>(f, w, rb, nb, y, o, r0, r1); break;
    }
}
}  // namespace maya::kernels::cpu::iq3v
#else
#if IQ3V_PART == 1
}  // namespace
void pass_z(int f, int nt, const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* o, int r0,
            int r1);   // (iq3v_rows_z.cpp)
namespace {
#define IQ3V_ZROWS(fn) pass_z(f, NT, w, rb, nb, y, o, r0, r1)
#else
#define IQ3V_ZROWS(fn) fn<NT>(w, rb, nb, y, o, r0, r1)
#endif

// rows [r0, r1) of format f (0 IQ3_XXS, 1 IQ4_XS, 2 Q4X), form v, NT tokens
template <int NT>
void pass(int f, int v, const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* o, int r0, int r1) {
    if (f == 0) {
        switch (v) {
            case 5: return IQ3V_ZROWS(rows3z);
#if defined(IQ3V_AVXVNNI)
            case 2: return a2v::rows3g<NT>(w, rb, nb, y, o, r0, r1);
            case 4: return a2v::rows3b<NT>(w, rb, nb, y, o, r0, r1);
#endif
            case 3: return a2::rows3b<NT>(w, rb, nb, y, o, r0, r1);
            default: return a2::rows3g<NT>(w, rb, nb, y, o, r0, r1);
        }
    }
    if (f == 1) {
        if (v == 3) return IQ3V_ZROWS(rows4z);
#if defined(IQ3V_AVXVNNI)
        if (v == 2) return a2v::rows4<NT>(w, rb, nb, y, o, r0, r1);
#endif
        return a2::rows4<NT>(w, rb, nb, y, o, r0, r1);
    }
    if (v == 3) return IQ3V_ZROWS(rowsqz);
#if defined(IQ3V_AVXVNNI)
    if (v == 2) return a2v::rowsq<NT>(w, rb, nb, y, o, r0, r1);
#endif
    return a2::rowsq<NT>(w, rb, nb, y, o, r0, r1);
}
void rows_f(int f, int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {   // (4 tokens a pass: their accumulators stay in registers)
        const block_q8_K* y[4];
        float* o[4];
        const int c = nt - t0 < 4 ? nt - t0 : 4;
        for (int t = 0; t < c; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            o[t] = out[t0 + t];
        }
        switch (c) {
            case 1: pass<1>(f, v, w, rb, nb, y, o, r0, r1); break;
            case 2: pass<2>(f, v, w, rb, nb, y, o, r0, r1); break;
            case 3: pass<3>(f, v, w, rb, nb, y, o, r0, r1); break;
            default: pass<4>(f, v, w, rb, nb, y, o, r0, r1); break;
        }
    }
}
}  // namespace

void rows_iq3xxs(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
                 int r1) {
    if (v == 5 && !codes().ok) return;   // (the caller's check fails on these rows then)
    rows_f(0, v, w, rb, n, act, nt, out, r0, r1);
}
void rows_iq4xs(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1) {
    rows_f(1, v, w, rb, n, act, nt, out, r0, r1);
}
void rows_q4x(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1) {
    rows_f(2, v, w, rb, n, act, nt, out, r0, r1);
}
bool vnni_form() {
#if defined(IQ3V_AVXVNNI)
    return true;
#else
    return false;
#endif
}

}  // namespace maya::kernels::cpu::iq3v
#undef IQ3V_ZROWS
#endif  // IQ3V_PART == 2

#else   // not x86-64: no forms (the caller never asks)
namespace maya::kernels::cpu::iq3v {
void rows_iq3xxs(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
void rows_iq4xs(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
void rows_q4x(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
bool vnni_form() { return false; }
}  // namespace maya::kernels::cpu::iq3v
#endif
