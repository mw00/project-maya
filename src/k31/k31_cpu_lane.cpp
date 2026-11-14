// src/k31/k31_cpu_lane.cpp - see k31_cpu_lane.hpp.  (-ffp-contract=off, no fast math: CMake.)
#include "k31_cpu_lane.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define K31_CPU_X86 1
#endif

#if defined(__FAST_MATH__)
#error "k31_cpu_lane.cpp must not be built with fast math: its sums are the device's bit for bit"
#endif

namespace k31::cpu {
namespace {

constexpr int kIQ1_S = 19, kIQ2_XXS = 16;   // ggml_type

float f16(uint16_t h) {   // exact
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // a subnormal: m 2^-24
            float f = (float) m * 5.9604644775390625e-8f;
            std::memcpy(&u, &f, 4);
            u |= s;
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IQ1_S's codes 8 g +- 1 for grid entry gi (8 values) with the delta's sign neg: [neg][gi][8]
struct Codes {
    int8_t c[2][NGRID_IQ1S][8];
    Codes() {
        for (int neg = 0; neg < 2; ++neg)
            for (int gi = 0; gi < NGRID_IQ1S; ++gi)
                for (int j = 0; j < 8; ++j) {
                    const int g = (int8_t) (uint8_t) (iq1s_grid[gi] >> (8 * j));
                    c[neg][gi][j] = (int8_t) (8 * g + (neg ? -1 : 1));
                }
    }
};
const Codes& codes() {
    static const Codes t;
    return t;
}

// sub-block kb (32 values) of an IQ1_S row: its 32 codes, d (F16 / 8) and s (2 s + 1)
inline void iq1s_sub(const uint8_t* row, int kb, int8_t* q, float& d, int& s) {
    const uint8_t* b = row + (size_t) (kb >> 3) * 50;
    const int ib = kb & 7;
    uint16_t dh, h;
    std::memcpy(&dh, b, 2);
    std::memcpy(&h, b + 34 + 2 * ib, 2);
    const uint8_t* qs = b + 2 + 4 * ib;
    const Codes& C = codes();
    const int neg = (h & 0x8000u) ? 1 : 0;
    for (int l = 0; l < 4; ++l) std::memcpy(q + 8 * l, C.c[neg][qs[l] | (((h >> (3 * l)) & 7u) << 8)], 8);
    s = 2 * (int) ((h >> 12) & 7u) + 1;
    d = f16(dh) * 0.125f;
}

// sub-block kb of an IQ2_XXS row (66 bytes a block: F16 d, then 8 x (4 grid indices | 4 x 7 sign bits, a 4-bit
// scale)): its 32 codes (grid bytes with ksigns_iq2xs's 8 signs a group), d (F16 / 8) and s (2 s + 1) - the device's
// decode<kIQ2_XXS> (ds4_iq.cu), where s0 = s1 = s
inline void iq2xxs_sub(const uint8_t* row, int kb, int8_t* q, float& d, int& s) {
    const uint8_t* b = row + (size_t) (kb >> 3) * 66;
    const int ib = kb & 7;
    uint16_t dh;
    uint32_t a0, a1;
    std::memcpy(&dh, b, 2);
    std::memcpy(&a0, b + 2 + 8 * ib, 4);
    std::memcpy(&a1, b + 2 + 8 * ib + 4, 4);
    for (int l = 0; l < 4; ++l) {
        const uint64_t g = iq2xxs_grid[(a0 >> (8 * l)) & 255u];
        const uint32_t sg = ksigns_iq2xs[(a1 >> (7 * l)) & 127u];
        for (int j = 0; j < 8; ++j) {
            const int v = (int) ((g >> (8 * j)) & 255u);
            q[8 * l + j] = (int8_t) ((sg >> j) & 1u ? -v : v);
        }
    }
    s = 2 * (int) (a1 >> 28) + 1;
    d = f16(dh) * 0.125f;
}

// a sub-block's exact int8 dot (integers: the same in any order, so every form gives the same bits)
inline int dot32_scalar(const int8_t* a, const int8_t* x) {
    int r = 0;
    for (int j = 0; j < 32; ++j) r += (int) a[j] * (int) x[j];
    return r;
}
#if K31_CPU_X86
__attribute__((target("avx512f,avx512bw"))) inline int dot32_avx512(const int8_t* a, const int8_t* x) {
    const __m512i va = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) a));
    const __m512i vx = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) x));
    return _mm512_reduce_add_epi32(_mm512_madd_epi16(va, vx));
}
// IQ1_S's grid as nibbles g + 1 (iq1s_grid_gpu's layout): entry byte j = value j (low nibble), value 4 + j (high)
struct Nib {
    uint32_t t[NGRID_IQ1S];
    Nib() {
        for (int gi = 0; gi < NGRID_IQ1S; ++gi) {
            uint32_t v = 0;
            for (int j = 0; j < 4; ++j) {
                const int lo = (int8_t) (uint8_t) (iq1s_grid[gi] >> (8 * j)) + 1, hi = (int8_t) (uint8_t) (iq1s_grid[gi] >> (8 * (j + 4))) + 1;
                v |= (uint32_t) (lo | (hi << 4)) << (8 * j);
            }
            t[gi] = v;
        }
    }
};
const Nib& nib() {
    static const Nib t;
    return t;
}

// The rows with AVX-512 VNNI, a 256-value block (8 sub-blocks) at a time: the 32 groups' grid entries gathered as
// nibbles n = g + 1, their exact int8 dots by vpdpbusd against the input permuted to the entries' layout (per column,
// once), each sub-block's code dot 8 dn + (neg ? -9 : -7) sum(x) (the codes 8 g +- 1 = 8 n - 8 +- 1), then the
// device's float steps as 8-wide vectors: v = d * (float) (s dot), acc[lane] = fma(xd, v, acc[lane]) - lane kb & 31,
// so block b's 8 sub-blocks are lanes 8 (b & 3) .. + 7, each lane's sums in the device's order.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma"))) void rows_vnni(
    const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol, float* y, size_t ycs) {
    const int nsb = k / 32, nblk = k / 256;
    thread_local std::vector<int8_t> xp;
    thread_local std::vector<int32_t> sx;
    xp.resize((size_t) ncol * k);
    sx.resize((size_t) ncol * nsb);
    for (int c = 0; c < ncol; ++c)   // per block: [lo 0..15][hi 0..15][lo 16..31][hi 16..31], 4 bytes a group
        for (int b = 0; b < nblk; ++b) {
            const int8_t* x = xq + (size_t) c * k + (size_t) b * 256;
            int8_t* o = xp.data() + (size_t) c * k + (size_t) b * 256;
            for (int g = 0; g < 32; ++g) {
                const int j = g >> 4, i = g & 15;
                std::memcpy(o + 128 * j + 4 * i, x + 8 * g, 4);
                std::memcpy(o + 128 * j + 64 + 4 * i, x + 8 * g + 4, 4);
            }
            for (int sb = 0; sb < 8; ++sb) {
                int t = 0;
                for (int v = 0; v < 32; ++v) t += x[32 * sb + v];
                sx[(size_t) c * nsb + 8 * b + sb] = t;
            }
        }
    const uint32_t* tab = nib().t;
    const __m512i m0f = _mm512_set1_epi32(0x0f0f0f0f), m7 = _mm512_set1_epi32(7);
    const __m512i shv = _mm512_setr_epi32(0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9);
    const __m512i pj0 = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m512i pj1 = _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m512i pick = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0);
    constexpr int kMaxCol = 16;
    alignas(64) float acc[kMaxCol][32];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * ld;
        for (int c = 0; c < ncol && c < kMaxCol; ++c) std::memset(acc[c], 0, sizeof acc[c]);
        for (int c0 = 0; c0 < ncol; c0 += kMaxCol) {
            const int nc = ncol - c0 < kMaxCol ? ncol - c0 : kMaxCol;
            if (c0) for (int c = 0; c < nc; ++c) std::memset(acc[c], 0, sizeof acc[c]);
            for (int b = 0; b < nblk; ++b) {
                const uint8_t* blk = row + (size_t) b * 50;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const __m256 dv = _mm256_set1_ps(f16(dh) * 0.125f);
                const __m512i hall = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (blk + 34))));
                __m512i lo[2], hi[2];
                for (int j = 0; j < 2; ++j) {
                    const __m512i qs = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (blk + 2 + 16 * j)));
                    const __m512i hb = _mm512_and_si512(_mm512_srlv_epi32(_mm512_permutexvar_epi32(j ? pj1 : pj0, hall), shv), m7);
                    const __m512i ix = _mm512_or_si512(qs, _mm512_slli_epi32(hb, 8));
                    const __m512i e = _mm512_i32gather_epi32(ix, (const void*) tab, 4);
                    lo[j] = _mm512_and_si512(e, m0f);
                    hi[j] = _mm512_and_si512(_mm512_srli_epi32(e, 4), m0f);
                }
                const __m256i h8 = _mm512_castsi512_si256(hall);
                const __m256i neg = _mm256_and_si256(_mm256_srli_epi32(h8, 15), _mm256_set1_epi32(1));
                const __m256i corr = _mm256_sub_epi32(_mm256_set1_epi32(-7), _mm256_slli_epi32(neg, 1));
                const __m256i sc = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h8, 12), _mm256_set1_epi32(7)), 1),
                                                    _mm256_set1_epi32(1));
                for (int c = 0; c < nc; ++c) {
                    const int cc = c0 + c;
                    const int8_t* xb = xp.data() + (size_t) cc * k + (size_t) b * 256;
                    __m512i t[2];
                    for (int j = 0; j < 2; ++j) {
                        __m512i dn = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo[j], _mm512_loadu_si512((const void*) (xb + 128 * j)));
                        dn = _mm512_dpbusd_epi32(dn, hi[j], _mm512_loadu_si512((const void*) (xb + 128 * j + 64)));
                        dn = _mm512_add_epi32(dn, _mm512_shuffle_epi32(dn, _MM_PERM_CDAB));
                        t[j] = _mm512_add_epi32(dn, _mm512_shuffle_epi32(dn, _MM_PERM_BADC));
                    }
                    const __m256i dn8 = _mm512_castsi512_si256(_mm512_permutex2var_epi32(t[0], pick, t[1]));
                    const __m256i sx8 = _mm256_loadu_si256((const __m256i*) (sx.data() + (size_t) cc * nsb + 8 * b));
                    const __m256i dot = _mm256_add_epi32(_mm256_slli_epi32(dn8, 3), _mm256_mullo_epi32(corr, sx8));
                    const __m256 v = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_mullo_epi32(sc, dot)));
                    float* a = acc[c] + 8 * (b & 3);
                    const __m256 xd8 = _mm256_loadu_ps(xd + (size_t) cc * nsb + 8 * b);
                    _mm256_store_ps(a, _mm256_fmadd_ps(xd8, v, _mm256_load_ps(a)));
                }
            }
            for (int c = 0; c < nc; ++c) {
                float* a = acc[c];
                for (int o = 16; o > 0; o >>= 1)
                    for (int l = 0; l < o; ++l) a[l] = a[l] + a[l + o];
                y[(size_t) (c0 + c) * ycs + (size_t) (r - r0)] = std::fmaf(1.0f, a[0], 0.0f);
            }
        }
    }
}
// IQ2_XXS with AVX-512 VNNI, a block (8 sub-blocks) at a time: the grid's unsigned bytes (8, 25, 43) are vpdpbusd's
// u8 side as they are, the signs go onto the input instead (one masked negate per 64 bytes; exact: the runner's int8
// input is never -128), each sub-block's int32 dot by an exact reduction, then the same float steps as rows_vnni.
// IQ2_XXS block decode with AVX-512 VBMI: the grid indices and the 7-bit sign fields picked out of the block's 64 code
// bytes, the grid entries by four 64-bit gathers, the 128 sign bytes by one byte permute (vpermi2b).  Out: grid[32]
// (the 32 groups' 8 bytes), sm[4] (the sign masks of sub-blocks 2 j, 2 j + 1), scl[8] (2 s + 1).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"))) void iq2xxs_decode_vbmi(const uint8_t* blk, uint64_t* grid,
                                                                                         uint64_t* sm, int32_t* scl) {
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i ks0 = _mm512_loadu_si512((const void*) ksigns_iq2xs), ks1 = _mm512_loadu_si512((const void*) (ksigns_iq2xs + 64));
    const __m256i m127 = _mm256_set1_epi32(127);
    const __m512i qz = _mm512_loadu_si512((const void*) (blk + 2));   // 16 words: a0 (even), a1 (odd)
    const __m256i a0v = _mm512_castsi512_si256(_mm512_permutexvar_epi32(ev, qz));
    const __m256i a1v = _mm512_castsi512_si256(_mm512_permutexvar_epi32(od, qz));
    // byte 4 ib + l: group l's sign field (a1 >> 7 l) & 127 (the grid indices: a0's bytes, the same order)
    const __m256i si = _mm256_or_si256(
        _mm256_or_si256(_mm256_and_si256(a1v, m127), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(a1v, 7), m127), 8)),
        _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(a1v, 14), m127), 16),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(a1v, 21), m127), 24)));
    _mm256_storeu_si256((__m256i*) sm, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, _mm512_castsi256_si512(si), ks1)));
    alignas(32) uint8_t gix[32];
    _mm256_store_si256((__m256i*) gix, a0v);
    for (int j = 0; j < 4; ++j)
        _mm512_storeu_si512((void*) (grid + 8 * j),
                            _mm512_i32gather_epi64(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (gix + 8 * j))),
                                                   (const void*) iq2xxs_grid, 8));
    _mm256_storeu_si256((__m256i*) scl, _mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(a1v, 28), 1), _mm256_set1_epi32(1)));
}
template <bool VB>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma"))) void rows_vnni_iq2xxs_t(
    const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol, float* y, size_t ycs) {
    const int nsb = k / 32, nblk = k / 256;
    const __m512i pick = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0);
    constexpr int kMaxCol = 16;
    alignas(64) float acc[kMaxCol][32];
    alignas(64) uint64_t grid[32];
    alignas(32) int32_t scl[8];
    __mmask64 sm[4];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * ld;
        for (int c0 = 0; c0 < ncol; c0 += kMaxCol) {
            const int nc = ncol - c0 < kMaxCol ? ncol - c0 : kMaxCol;
            for (int c = 0; c < nc; ++c) std::memset(acc[c], 0, sizeof acc[c]);
            for (int b = 0; b < nblk; ++b) {
                const uint8_t* blk = row + (size_t) b * 66;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const __m256 dv = _mm256_set1_ps(f16(dh) * 0.125f);
                __m512i gz[4];
                __m256i sc;
                if constexpr (VB) {
                    alignas(32) uint64_t smw[4];
                    iq2xxs_decode_vbmi(blk, grid, smw, scl);
                    for (int j = 0; j < 4; ++j) {
                        gz[j] = _mm512_load_si512((const void*) (grid + 8 * j));
                        sm[j] = (__mmask64) smw[j];
                    }
                    sc = _mm256_load_si256((const __m256i*) scl);
                } else {
                    uint64_t sg[8];
                    for (int ib = 0; ib < 8; ++ib) {
                        uint32_t a0, a1;
                        std::memcpy(&a0, blk + 2 + 8 * ib, 4);
                        std::memcpy(&a1, blk + 2 + 8 * ib + 4, 4);
                        uint64_t m = 0;
                        for (int l = 0; l < 4; ++l) {
                            grid[4 * ib + l] = iq2xxs_grid[(a0 >> (8 * l)) & 255u];
                            m |= (uint64_t) ksigns_iq2xs[(a1 >> (7 * l)) & 127u] << (8 * l);
                        }
                        sg[ib] = m;
                        scl[ib] = 2 * (int) (a1 >> 28) + 1;
                    }
                    for (int j = 0; j < 4; ++j) {
                        gz[j] = _mm512_load_si512((const void*) (grid + 8 * j));
                        sm[j] = (__mmask64) (sg[2 * j] | sg[2 * j + 1] << 32);
                    }
                    sc = _mm256_load_si256((const __m256i*) scl);
                }
                for (int c = 0; c < nc; ++c) {
                    const int cc = c0 + c;
                    const int8_t* xb = xq + (size_t) cc * k + (size_t) b * 256;
                    __m512i t[4];
                    for (int j = 0; j < 4; ++j) {
                        const __m512i x = _mm512_loadu_si512((const void*) (xb + 64 * j));
                        const __m512i xs = _mm512_mask_sub_epi8(x, sm[j], _mm512_setzero_si512(), x);
                        __m512i dn = _mm512_dpbusd_epi32(_mm512_setzero_si512(), gz[j], xs);
                        dn = _mm512_add_epi32(dn, _mm512_shuffle_epi32(dn, _MM_PERM_CDAB));
                        t[j] = _mm512_add_epi32(dn, _mm512_shuffle_epi32(dn, _MM_PERM_BADC));   // a 128-bit lane's sum, in each
                    }
                    // quarter sums: sub-block i = quarters 2i, 2i + 1
                    const __m256i q01 = _mm512_castsi512_si256(_mm512_permutex2var_epi32(t[0], pick, t[1]));
                    const __m256i q23 = _mm512_castsi512_si256(_mm512_permutex2var_epi32(t[2], pick, t[3]));
                    const __m256i dot = _mm256_permute4x64_epi64(_mm256_hadd_epi32(q01, q23), 0xD8);   // sub-blocks 0..7
                    const __m256 v = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_mullo_epi32(sc, dot)));
                    float* a = acc[c] + 8 * (b & 3);
                    const __m256 xd8 = _mm256_loadu_ps(xd + (size_t) cc * nsb + 8 * b);
                    _mm256_store_ps(a, _mm256_fmadd_ps(xd8, v, _mm256_load_ps(a)));
                }
            }
            for (int c = 0; c < nc; ++c) {
                float* a = acc[c];
                for (int o = 16; o > 0; o >>= 1)
                    for (int l = 0; l < o; ++l) a[l] = a[l] + a[l + o];
                y[(size_t) (c0 + c) * ycs + (size_t) (r - r0)] = std::fmaf(1.0f, a[0], 0.0f);
            }
        }
    }
}
void rows_vnni_iq2xxs(const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol,
                      float* y, size_t ycs) {
    rows_vnni_iq2xxs_t<false>(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
}
void rows_vbmi_iq2xxs(const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol,
                      float* y, size_t ycs) {
    rows_vnni_iq2xxs_t<true>(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
}
bool have_vbmi() {
    static const bool on = __builtin_cpu_supports("avx512vbmi") && !std::getenv("K31_CPU_NO_VBMI");
    return on;
}
bool have_vnni() {
    static const bool on = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni") &&
                           !std::getenv("K31_CPU_SCALAR") && !std::getenv("K31_CPU_NO_VNNI");
    return on;
}

bool have_avx512() {
    // (K31_CPU_SCALAR=1: the scalar form, for the tests: the same bits)
    static const bool on = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                           !std::getenv("K31_CPU_SCALAR");
    return on;
}
#endif

}  // namespace

bool rows_supported(int type) { return type == kIQ1_S || type == kIQ2_XXS; }

const char* form() {
#if K31_CPU_X86
    if (have_vnni()) return "avx512-vnni";
    if (have_avx512()) return "avx512";
#endif
    return "scalar";
}

namespace {
template <int TY, bool V>
__attribute__((always_inline)) inline void rows_t(const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol, float* y,
            size_t ycs) {
    const int nsb = k / 32;
    constexpr int kMaxCol = 16;
    alignas(64) int8_t q[32];
    for (int c0 = 0; c0 < ncol; c0 += kMaxCol) {
        const int nc = ncol - c0 < kMaxCol ? ncol - c0 : kMaxCol;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* row = w + (size_t) r * ld;
            float acc[kMaxCol][32];
            for (int c = 0; c < nc; ++c)
                for (int l = 0; l < 32; ++l) acc[c][l] = 0.0f;
            for (int kb = 0; kb < nsb; ++kb) {
                float d;
                int s;
                if constexpr (TY == kIQ2_XXS) iq2xxs_sub(row, kb, q, d, s);
                else iq1s_sub(row, kb, q, d, s);
                const int lane = kb & 31;
                for (int c = 0; c < nc; ++c) {
                    const int cc = c0 + c;
                    const int8_t* x = xq + (size_t) cc * k + (size_t) kb * 32;
                    // the device: lo (values 0..15) and hi (16..31) apart, s0 lo + s1 hi with s0 = s1 = s: s (lo + hi)
#if K31_CPU_X86
                    const int dt = V ? dot32_avx512(q, x) : dot32_scalar(q, x);
#else
                    const int dt = dot32_scalar(q, x);
#endif
                    const float v = d * (float) (s * dt);
                    acc[c][lane] = std::fmaf(xd[(size_t) cc * nsb + kb], v, acc[c][lane]);
                }
            }
            for (int c = 0; c < nc; ++c) {
                float* a = acc[c];
                for (int o = 16; o > 0; o >>= 1)
                    for (int l = 0; l < o; ++l) a[l] = a[l] + a[l + o];
                y[(size_t) (c0 + c) * ycs + (size_t) (r - r0)] = std::fmaf(1.0f, a[0], 0.0f);
            }
        }
    }
}
#if K31_CPU_X86
template <int TY>
__attribute__((target("avx512f,avx512bw"))) void rows_avx512(const uint8_t* w, size_t ld, int k, int r0, int r1,
                                                             const int8_t* xq, const float* xd, int ncol, float* y,
                                                             size_t ycs) {
    rows_t<TY, true>(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
}
#endif
template <int TY>
void rows_scalar(const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol,
                 float* y, size_t ycs) {
    rows_t<TY, false>(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
}
}  // namespace

void rows(int type, const uint8_t* w, size_t ld, int k, int r0, int r1, const int8_t* xq, const float* xd, int ncol,
          float* y, size_t ycs) {
    if (!rows_supported(type) || k % 256 != 0) return;   // (callers check rows_supported and k)
    const bool iq2 = type == kIQ2_XXS;
#if K31_CPU_X86
    if (have_vnni()) {
        (iq2 ? (have_vbmi() ? rows_vbmi_iq2xxs : rows_vnni_iq2xxs) : rows_vnni)(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
        return;
    }
    if (have_avx512()) {
        (iq2 ? rows_avx512<kIQ2_XXS> : rows_avx512<kIQ1_S>)(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
        return;
    }
#endif
    (iq2 ? rows_scalar<kIQ2_XXS> : rows_scalar<kIQ1_S>)(w, ld, k, r0, r1, xq, xd, ncol, y, ycs);
}

void q8(const float* x, int k, int rows_, int8_t* q, float* d) {
    for (int r = 0; r < rows_; ++r)
        for (int b = 0; b < k / 32; ++b) {
            const float* v = x + (size_t) r * k + (size_t) b * 32;
            float am = 0.0f;
            for (int j = 0; j < 32; ++j) am = std::fmax(am, std::fabs(v[j]));
            const float sc = am / 127.0f;
            for (int j = 0; j < 32; ++j)
                q[(size_t) r * k + (size_t) b * 32 + j] = (int8_t) (am > 0.0f ? (int) std::nearbyint(v[j] / sc) : 0);
            d[(size_t) r * (k / 32) + b] = sc;
        }
}

Pool::Pool(int threads) {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { loop(); });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> l(mu_);
        stop_ = true;
    }
    go_.notify_all();
    for (auto& t : workers_) t.join();
}
void Pool::drain() {
    for (;;) {
        const int i = next_.fetch_add(1);
        if (i >= n_.load()) return;
        (*fn_)(i);
    }
}
void Pool::loop() {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> l(mu_);
            go_.wait(l, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            ++busy_;
        }
        drain();
        {
            std::lock_guard<std::mutex> l(mu_);
            if (--busy_ == 0) done_.notify_all();
        }
    }
}
void Pool::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    {
        // (a worker still draining the last call - it found nothing left - finishes before the state changes)
        std::unique_lock<std::mutex> l(mu_);
        done_.wait(l, [&] { return busy_ == 0; });
        fn_ = &fn;
        n_ = n;
        next_ = 0;
        ++gen_;
    }
    go_.notify_all();
    drain();
    std::unique_lock<std::mutex> l(mu_);
    done_.wait(l, [&] { return busy_ == 0 && next_.load() >= n_; });
}

}  // namespace k31::cpu
