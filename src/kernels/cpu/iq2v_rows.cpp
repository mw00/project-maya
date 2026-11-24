// src/kernels/cpu/iq2v_rows.cpp - the CPU lane's IQ2_XXS and IQ2_S rows, second generation: ggml-cpu's dot to the
// bit, a third to a half faster than the lane's earlier rows.  ds4_iq_cpu.cpp checks each form against ggml-cpu's dot
// and takes the fastest this CPU runs (own(): g_iq2v / g_iq2sv).  Built for the baseline x86-64 with each function's
// instruction set its own (GCC / Clang target attributes; MSVC: /arch:AVX2 on this file, for its VEX encodings - the
// AVX-512 form's intrinsics compile as they are), so a form runs only where the caller found the CPU takes it.
//
// The arithmetic is ggml's: per block the same exact int32 lanes (lane l: every 32-value chunk's values 4 l .. 4 l + 3,
// |grid| x sign(q8), times the chunk's 2 s + 1), then fma(d_x d_y, lanes, acc) block by block and hsum_float_8 x 1/8.
// What differs is how a chunk's operands are made:
//  - a block's eight sign words are taken as one vector: their 7-bit fields spread to bytes, each byte's parity put in
//    its bit 7 by a nibble vpshufb (ggml's ksigns_iq2xs, in registers), the bytes grouped so that one vpshufb a chunk
//    lays its four sign bytes over its four 8-value groups, a bit test + 0x7F making the vpsignb operand;
//  - the scales likewise, a block at a time;
//  - all of that one block ahead (software-pipelined: the chunks never wait on its chain), and a call's rows run as
//    one stream of blocks (a row's blocks follow the one before's in the file);
//  - the grid by broadcast loads + blends (tab4), its indices by shifts or byte loads.
// Forms (g_iq2v): 1 AVX2, byte-load indices, scales stored a block ahead and broadcast by a load (Golden Cove's best:
// its three vector ports are the limit); 2 AVX2 + AVX-VNNI, shifted indices, scales by vpshufb, vpdpwssd (Zen 5's
// best AVX2 form); 3 AVX-512 (F BW VL DQ VBMI BITALG VNNI): a pair of chunks a zmm, the fields by vpmultishiftqb, the
// sign bytes by vpermi2b from ksigns_iq2xs, a pair's 64 sign bits one vpshufbitqmb, q8 negated under that mask, the
// scales one vpermb a pair (the zmm's two halves - one chunk each - added at the block's end: ggml's lanes).
// Every form gives the same bits (own_check against ggml-cpu's dot, and the lane's own test), so the choice is time only.

#include "maya/kernels/cpu/iq2v_rows.hpp"

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
// AVX-VNNI's intrinsics: GCC 11, Clang 12, MSVC 2022 (as ds4_gguf_cpu.cpp's MXFP4 tier)
// (MAYA_NO_AVXVNNI: the assembler has no AVX-VNNI - Rocky / RHEL 9's binutils 2.35 - so the tier is left out,
//  vnni_form() says so and the AVX2 form runs in its place; CMake checks it)
#if !defined(MAYA_NO_AVXVNNI) && \
    (defined(_MSC_VER) || (defined(__clang__) && __clang_major__ >= 12) || \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define IQ2V_AVXVNNI 1
#endif

namespace maya::kernels::cpu::iq2v {
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
IQ_AVX2 inline __m256i bq(const void* p) { return _mm256_castpd_si256(_mm256_broadcast_sd((const double*) p)); }
IQ_AVX2 inline __m256i tab4(const uint64_t* t, unsigned a, unsigned b, unsigned c, unsigned d) {   // t[a] t[b] t[c] t[d]
    return _mm256_blend_epi32(_mm256_blend_epi32(bq(t + a), bq(t + b), 0x0C), _mm256_blend_epi32(bq(t + c), bq(t + d), 0xC0),
                              0xF0);
}
}  // namespace
}  // namespace maya::kernels::cpu::iq2v

namespace maya::kernels::cpu::iq2v {
namespace {

constexpr int kPos[8] = {0, 1, 4, 5, 2, 3, 6, 7};   // chunk c's place in the block's sign-word vector (shufps order)
struct Tabs {
    alignas(64) uint8_t ctrl[8][32];   // chunk c: its four sign bytes over its 8-value groups (vpshufb of T)
    alignas(64) uint8_t hctl[4][32];   // place k of a lane: the high half of its dword over the lane (scales, form 2)
    alignas(64) uint8_t bit[64];       // form 3: a pair's byte b <- bit 8 (group) + (value) of its chunk's qword
    alignas(64) uint8_t sc[4][64];     // form 3: pair p's scales (word w <- the low word of chunk 2 p + w / 16)
    alignas(64) uint8_t fld[64];       // form 3: vpmultishiftqb's offsets of the four 7-bit fields
    alignas(64) uint8_t wctl[8][32];   // IQ2_S, chunk c: word c of each lane over the lane (its lo / hi scale)
    alignas(64) uint16_t swi[4][32];   // IQ2_S form 3, pair p: word w <- scale 2 (2 p + w / 16) + (w % 16 >= 8)
    alignas(64) uint8_t qhb[64];       // IQ2_S form 3: word b's high byte <- qh[b / 4]
    alignas(64) uint16_t qhs[32];      // IQ2_S form 3: word b's shift (2 (b % 4))
    constexpr Tabs() : ctrl(), hctl(), bit(), sc(), fld(), wctl(), swi(), qhb(), qhs() {
        for (int c = 0; c < 8; ++c)
            for (int b = 0; b < 32; ++b) wctl[c][b] = (uint8_t) (2 * c + (b & 1));
        for (int p = 0; p < 4; ++p)
            for (int w = 0; w < 32; ++w) swi[p][w] = (uint16_t) (2 * (2 * p + w / 16) + ((w % 16) >= 8 ? 1 : 0));
        for (int b = 0; b < 64; ++b) qhb[b] = (uint8_t) (b / 8);   // (odd bytes taken: byte 2 w + 1 <- qh[w / 4])
        for (int w = 0; w < 32; ++w) qhs[w] = (uint16_t) (2 * (w % 4));
        for (int c = 0; c < 8; ++c)
            for (int b = 0; b < 32; ++b) ctrl[c][b] = (uint8_t) (2 * kPos[c] + ((b & 15) < 8 ? 0 : 1));
        for (int k = 0; k < 4; ++k)
            for (int b = 0; b < 32; ++b) hctl[k][b] = (uint8_t) (4 * k + 2 + (b & 1));
        for (int b = 0; b < 64; ++b) {
            bit[b] = (uint8_t) (8 * ((b / 8) % 4) + b % 8);
            fld[b] = (uint8_t) ((b % 8) < 4 ? 32 + 7 * (b % 8) : 0);
        }
        for (int p = 0; p < 4; ++p)
            for (int b = 0; b < 64; ++b) sc[p][b] = (uint8_t) (8 * (2 * p + b / 32) + (b & 1));
    }
};
constexpr Tabs kT{};

#if defined(_MSC_VER)
#define IQ2V_FI __forceinline
#else
#define IQ2V_FI inline __attribute__((always_inline))
#endif
#define IQ2A_NS a2
#define IQ2A_ATTR IQX_TARGET("avx2,fma")
#define IQ2A_VNNI 0
#include "iq2v_avx2.inc"
#if defined(IQ2V_AVXVNNI)
#define IQ2A_NS a2v
#define IQ2A_ATTR IQX_TARGET("avx2,fma,avxvnni")
#define IQ2A_VNNI 1
#include "iq2v_avx2.inc"
#endif

// ---- the AVX-512 form (3)
#define IQ2V_Z IQX_TARGET("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512bitalg,avx512vnni,avx2,fma")
IQ2V_Z IQ2V_FI void prepz(const uint8_t* qs, __m512i fld, __m512i ks0, __m512i ks1, __m512i one, __m512i& S,
                          __m512i& SC) {
    const __m512i R = _mm512_loadu_si512(qs);   // (qword c: chunk c's four grid indices, then its sign word)
    S = _mm512_permutex2var_epi8(ks0, _mm512_multishift_epi64_epi8(fld, R), ks1);   // (bytes 0-3: its sign bytes)
    SC = _mm512_or_si512(_mm512_srli_epi64(R, 59), one);                             // (2 s + 1)
}
template <int NT>
IQ2V_Z void rowsz(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m512i fld = _mm512_load_si512(kT.fld), ctlb = _mm512_load_si512(kT.bit);
    const __m512i ks0 = _mm512_loadu_si512(ksigns_iq2xs), ks1 = _mm512_loadu_si512(ksigns_iq2xs + 64);
    const __m512i one = _mm512_set1_epi64(1), zero = _mm512_setzero_si512();
    const __mmask64 mw = 0x5555555555555555ull;
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq2_xxs);
    __m512i nS, nSC;
    prepz(blk + 2, fld, ks0, ks1, one, nS, nSC);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const __m512i S = nS, SC = nSC;
        const uint8_t* qs = blk + 2;
        prepz(b + 1 < nbt ? qs + BB : qs, fld, ks0, ks1, one, nS, nSC);
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
        for (int p = 0; p < 4; ++p) {
            const uint8_t* q = qs + 16 * p;
            const __m512i g = _mm512_inserti64x4(_mm512_castsi256_si512(tab4(iq2xxs_grid, q[0], q[1], q[2], q[3])),
                                                 tab4(iq2xxs_grid, q[8], q[9], q[10], q[11]), 1);
            const __m512i Sp = _mm512_permutexvar_epi64(
                _mm512_set_epi64(2 * p + 1, 2 * p + 1, 2 * p + 1, 2 * p + 1, 2 * p, 2 * p, 2 * p, 2 * p), S);
            const __mmask64 neg = _mm512_bitshuffle_epi64_mask(Sp, ctlb);
            const __m512i cv = _mm512_maskz_permutexvar_epi8(mw, _mm512_load_si512(kT.sc[p]), SC);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8 = _mm512_loadu_si512(y[t][i].qs + 64 * p);
                const __m512i d = _mm512_maddubs_epi16(g, _mm512_mask_sub_epi8(q8, neg, zero, q8));
                acc[t] = _mm512_dpwssd_epi32(acc[t], d, cv);
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t) {
            const __m256i s8 = _mm256_add_epi32(_mm512_castsi512_si256(acc[t]), _mm512_extracti64x4_epi64(acc[t], 1));
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s8), accf[t]);
        }
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.125f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}
// IQ2_S's form 3: a block ahead, its 32 10-bit indices (natural order) into ix and its 16 scales (2 s + 1, word 2 c:
// chunk c's first 16 values, 2 c + 1: its second); a pair: two tab4 + insert, its 64 sign bits a mask straight from
// the block, q8 negated under it, the scales one vpermw
IQ2V_Z IQ2V_FI __m256i prepzs(const uint8_t* blk, __m512i qhb, __m512i qhs, uint16_t* ix) {
    const __m512i qs16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2)));
    const __m512i qh = _mm512_castsi128_si512(_mm_loadl_epi64((const __m128i*) (blk + 2 + QK_K / 4)));
    const __m512i hw = _mm512_maskz_permutexvar_epi8(0xAAAAAAAAAAAAAAAAull, qhb, qh);   // (word b: qh[b / 4] << 8)
    const __m512i hi = _mm512_and_si512(_mm512_srlv_epi16(hw, qhs), _mm512_set1_epi16(0x300));
    _mm512_store_si512(ix, _mm512_or_si512(qs16, hi));
    const __m128i s8 = _mm_loadl_epi64((const __m128i*) (blk + 2 + QK_K / 4 + QK_K / 32));
    const __m128i k0f = _mm_set1_epi8(0x0F);
    __m128i lh = _mm_unpacklo_epi8(_mm_and_si128(s8, k0f), _mm_and_si128(_mm_srli_epi16(s8, 4), k0f));
    lh = _mm_or_si128(_mm_add_epi8(lh, lh), _mm_set1_epi8(1));
    return _mm256_cvtepu8_epi16(lh);   // (word 2 c: lo, 2 c + 1: hi)
}
template <int NT>
IQ2V_Z void rowszs(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m512i qhb = _mm512_load_si512(kT.qhb), qhs = _mm512_load_si512(kT.qhs), zero = _mm512_setzero_si512();
    alignas(64) uint16_t ix2[2][32];
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq2_s);
    __m256i nZ = prepzs(blk, qhb, qhs, ix2[0]);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const __m512i Z = _mm512_castsi256_si512(nZ);
        const uint16_t* ix = ix2[b & 1];
        nZ = prepzs(b + 1 < nbt ? blk + BB : blk, qhb, qhs, ix2[(b + 1) & 1]);
        std::atomic_signal_fence(std::memory_order_seq_cst);   // (ix read from memory)
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
        for (int p = 0; p < 4; ++p) {
            const uint16_t* xi = ix + 8 * p;
            const __m512i g = _mm512_inserti64x4(_mm512_castsi256_si512(tab4(iq2s_grid, xi[0], xi[1], xi[2], xi[3])),
                                                 tab4(iq2s_grid, xi[4], xi[5], xi[6], xi[7]), 1);
            uint64_t sb;
            std::memcpy(&sb, blk + 2 + QK_K / 8 + 8 * p, 8);
            const __mmask64 neg = _cvtu64_mask64(sb);
            const __m512i cv = _mm512_permutexvar_epi16(_mm512_load_si512(kT.swi[p]), Z);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8 = _mm512_loadu_si512(y[t][i].qs + 64 * p);
                const __m512i d = _mm512_maddubs_epi16(g, _mm512_mask_sub_epi8(q8, neg, zero, q8));
                acc[t] = _mm512_dpwssd_epi32(acc[t], d, cv);
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t) {
            const __m256i s8 = _mm256_add_epi32(_mm512_castsi512_si256(acc[t]), _mm512_extracti64x4_epi64(acc[t], 1));
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s8), accf[t]);
        }
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.125f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}
#undef IQ2V_Z
#undef IQ2V_FI

// rows [r0, r1) of form v for nt tokens, 4 a pass (act: Q8_K rows of n values): v 1-3 IQ2_XXS's forms, 11-13 IQ2_S's
template <int NT>
void pass(int v, const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* o, int r0, int r1) {
    if (v >= 11) {   // (IQ2_S: 10 + its form)
        if (v == 13) return rowszs<NT>(w, rb, nb, y, o, r0, r1);
#if defined(IQ2V_AVXVNNI)
        if (v == 12) return a2v::rows2s<NT>(w, rb, nb, y, o, r0, r1);
#endif
        return a2::rows2s<NT>(w, rb, nb, y, o, r0, r1);
    }
    if (v == 3) return rowsz<NT>(w, rb, nb, y, o, r0, r1);
#if defined(IQ2V_AVXVNNI)
    if (v == 2) return a2v::rows2<NT, 0, 0>(w, rb, nb, y, o, r0, r1);
#endif
    return a2::rows2<NT, 1, 1>(w, rb, nb, y, o, r0, r1);
}
}  // namespace

void rows(int v, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
                 int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const block_q8_K* y[4];
        float* o[4];
        const int c = nt - t0 < 4 ? nt - t0 : 4;
        for (int t = 0; t < c; ++t) {
            y[t] = (const block_q8_K*) act[t0 + t];
            o[t] = out[t0 + t];
        }
        switch (c) {
            case 1: pass<1>(v, w, rb, nb, y, o, r0, r1); break;
            case 2: pass<2>(v, w, rb, nb, y, o, r0, r1); break;
            case 3: pass<3>(v, w, rb, nb, y, o, r0, r1); break;
            default: pass<4>(v, w, rb, nb, y, o, r0, r1); break;
        }
    }
}

bool vnni_form() {
#if defined(IQ2V_AVXVNNI)
    return true;
#else
    return false;
#endif
}

}  // namespace maya::kernels::cpu::iq2v

#else   // not x86-64: no forms (the caller never asks)
namespace maya::kernels::cpu::iq2v {
void rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
bool vnni_form() { return false; }
}  // namespace maya::kernels::cpu::iq2v
#endif
