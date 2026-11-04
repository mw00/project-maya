// src/kernels/cpu/iq_avx512_one.cpp - the decode's one-token CPU expert rows for IQ2_XXS in 512-bit lanes.
//
// One token is what the decode's CPU lane runs: ggml-cpu's AVX2 dot and iq_avx512.cpp's multi-token rows (with one
// token) both spend most of a block on the codebook - 32 scalar grid loads assembled into vectors and 32 sign-table
// loads - at ~4-5 GB/s of weights per core. Here neither table is read:
//   * the grid: every iq2xxs_grid byte is 0x08, 0x19 or 0x2b, so an entry is eight 2-bit codes (a u16) and the 256
//     codes fit in 8 zmm registers. A block's 32 indices are looked up at once (4 two-register word permutes and 3
//     blends), and each 64-value chunk's 8 codes are spread to its 64 bytes by a word permute, a multishift (VBMI),
//     an and and a byte shuffle;
//   * the signs: ksigns_iq2xs[i] is i with its parity in bit 7 - four 7-bit fields are spread to four bytes by one
//     pdep (BMI2) and the parities folded in with three shift-xors.
// The bytes are the grid's own (build_codes checks every entry at first use) and the rest is iq_avx512.cpp's row_dot:
// exact integer sums, folded to ggml's 8 lanes, then ggml's float steps in ggml's order - so every result is
// bit-identical to ggml-cpu's dot (native_expert.cpp checks this against ggml before it uses the rows). On a Ryzen 7
// 9700X (Zen 5), Maya-S24's IQ2_XXS gate/up rows: ~1.4x ggml's dot at 8 threads (a whole expert with its IQ2_S down
// rows: ~1.26x).
#include "strata/kernels/cpu/iq_avx512_one.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstring>

#if defined(_MSC_VER)
#define STRATA_ALWAYS_INLINE __forceinline
#else
#define STRATA_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

namespace strata::kernels::cpu {
namespace {

STRATA_ALWAYS_INLINE float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
STRATA_ALWAYS_INLINE uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
STRATA_ALWAYS_INLINE uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// ggml's hsum_float_8: the same additions in the same order
STRATA_ALWAYS_INLINE float hsum_ggml(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// a block's 16 int32 lanes -> ggml's 8 (lane k + lane k + 8), exact
STRATA_ALWAYS_INLINE __m256i fold(__m512i a) {
    return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
}

// int16 lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3
STRATA_ALWAYS_INLINE __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ksigns_iq2xs for the four 7-bit fields of a word: i | parity(i) << 7 in four bytes
STRATA_ALWAYS_INLINE uint32_t signs4(uint32_t w) {
    const uint64_t s = _pdep_u64(w & 0x0FFFFFFFu, 0x7F7F7F7Full);
    uint64_t p = s ^ ((s >> 4) & 0x0F0F0F0Full);
    p ^= (p >> 2) & 0x03030303ull;
    p ^= (p >> 1) & 0x01010101ull;
    return (uint32_t) (s | ((p & 0x01010101ull) << 7));
}

// the 256 grid entries as 2-bit codes (0x08 -> 0, 0x19 -> 1, 0x2b -> 2), byte k at bits 2k; false if a byte is
// none of the three (a grid this file does not know)
alignas(64) uint16_t g_code16[256];
bool build_codes() {
    for (int i = 0; i < 256; ++i) {
        uint16_t c = 0;
        for (int k = 0; k < 8; ++k) {
            const int b = (int) ((iq2xxs_grid[i] >> (8 * k)) & 255);
            const int cc = b == 0x08 ? 0 : b == 0x19 ? 1 : b == 0x2b ? 2 : -1;
            if (cc < 0) return false;
            c |= (uint16_t) (cc << (2 * k));
        }
        g_code16[i] = c;
    }
    for (int i = 0; i < 128; ++i)   // and the sign bytes the parity fold makes
        if ((signs4((uint32_t) i) & 255u) != ksigns_iq2xs[i]) return false;
    return true;
}

bool cpu_bmi2_ok() {
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, 0, 0);
    if (x[0] < 7) return false;
    __cpuidex(x, 7, 0);
    r[1] = (unsigned) x[1];
#else
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#endif
    return (r[1] >> 8) & 1u;
}

STRATA_ALWAYS_INLINE float row_iq2xxs(const uint8_t* row, int nblocks, const block_q8_K* y, const __m512i* T,
                                      const __m512i* pidx) {
    const __m512i lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);   // code -> grid byte, in every 128-bit lane
    const __m512i ctl = _mm512_set1_epi64(0x0E0C0A0806040200LL);  // byte k of a qword <- bits 2k.. of its low word
    const __m512i three = _mm512_set1_epi8(3), b64 = _mm512_set1_epi16(64), b128 = _mm512_set1_epi16(128);
    const __m512i zero = _mm512_setzero_si512();
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 66;
        // the weight rows stream from DRAM: ask for the bytes a few blocks ahead (as iq_avx512.cpp does)
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        // qs: per chunk j the words a0 (4 grid indices), a1 (4 sign fields, scale), b0, b1 - the even words' bytes
        // are the block's 32 indices, chunk j's at 8j..8j+7 in iq2xxs order
        const __m512i q = _mm512_loadu_si512((const void*) (blk + 2));
        const __m512i idx = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi32(0x5555, q)));
        const __m512i r0 = _mm512_permutex2var_epi16(T[0], idx, T[1]), r1 = _mm512_permutex2var_epi16(T[2], idx, T[3]);
        const __m512i r2 = _mm512_permutex2var_epi16(T[4], idx, T[5]), r3 = _mm512_permutex2var_epi16(T[6], idx, T[7]);
        const __mmask32 m6 = _mm512_test_epi16_mask(idx, b64), m7 = _mm512_test_epi16_mask(idx, b128);
        const __m512i codes =
            _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r0, r1), _mm512_mask_blend_epi16(m6, r2, r3));
        __m512i acc = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i rep = _mm512_permutexvar_epi16(pidx[j], codes);
            const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, rep), three));
            const uint8_t* qq = blk + 2 + 16 * j;
            const uint32_t a1 = u32(qq + 4), b1 = u32(qq + 12);
            const __mmask64 m = _cvtu64_mask64((uint64_t) signs4(a1) | ((uint64_t) signs4(b1) << 32));
            const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
            const __m512i yv = _mm512_loadu_si512((const void*) (y[i].qs + 64 * j));
            const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), scales4(sa, sa, sb, sb)));
        }
        accf = _mm256_fmadd_ps(_mm256_set1_ps(h2f(u16(blk)) * y[i].d), _mm256_cvtepi32_ps(fold(acc)), accf);
    }
    return 0.125f * hsum_ggml(accf);
}

}  // namespace

bool iq512_one_supported(int type) noexcept {
    static const bool ok = cpu_avx512_ok() && cpu_bmi2_ok() && build_codes();
    return ok && type == 16;
}

void iq512_one_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1) {
    if (type != 16) return;
    __m512i T[8], pidx[4];
    for (int k = 0; k < 8; ++k) T[k] = _mm512_load_si512((const void*) (g_code16 + 32 * k));
    for (int j = 0; j < 4; ++j) {
        alignas(64) uint16_t p[32];
        for (int wd = 0; wd < 32; ++wd) p[wd] = (uint16_t) (8 * j + wd / 4);   // qword l <- the chunk's code l
        pidx[j] = _mm512_load_si512((const void*) p);
    }
    const int nb = n / QK_K;
    for (int t = 0; t < nt; ++t) {
        const block_q8_K* y = (const block_q8_K*) act[t];
        for (int r = r0; r < r1; ++r) out[t][r] = row_iq2xxs(w + (size_t) r * row_bytes, nb, y, T, pidx);
    }
}

}  // namespace strata::kernels::cpu
