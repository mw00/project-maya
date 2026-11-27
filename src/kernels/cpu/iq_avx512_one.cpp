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
#include "maya/kernels/cpu/iq_avx512_one.hpp"
#include "maya/kernels/cpu/expert_layout.hpp"
#include "maya/kernels/cpu/iq2v_rows.hpp"
#include "maya/kernels/cpu/iq3v_rows.hpp"

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
#include <cstdlib>

#if defined(_MSC_VER)
#define MAYA_ALWAYS_INLINE __forceinline
#else
#define MAYA_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

namespace maya::kernels::cpu {
namespace {

MAYA_ALWAYS_INLINE float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
MAYA_ALWAYS_INLINE uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
MAYA_ALWAYS_INLINE uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// ggml's hsum_float_8: the same additions in the same order
MAYA_ALWAYS_INLINE float hsum_ggml(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// a block's 16 int32 lanes -> ggml's 8 (lane k + lane k + 8), exact
MAYA_ALWAYS_INLINE __m256i fold(__m512i a) {
    return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
}

// int16 lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3
MAYA_ALWAYS_INLINE __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ksigns_iq2xs for the four 7-bit fields of a word: i | parity(i) << 7 in four bytes
MAYA_ALWAYS_INLINE uint32_t signs4(uint32_t w) {
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

// The additional one-token forms use AVX-512 DQ, VBMI, BITALG and VNNI.
// cpu_avx512_ok() supplies the OS-state and F/BW/VL/VBMI/VNNI checks; keep
// the remaining CPUID bits local so a partial AVX-512 implementation never
// reaches an instruction it cannot execute.
bool cpu_nt1_iq512_ok() {
    if (!cpu_avx512_ok() || !cpu_avx2_ok() || !cpu_bmi2_ok()) return false;
    unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, 1, 0);
    if (((unsigned) x[2] & (1u << 29)) == 0) return false;   // F16C
    __cpuidex(x, 0, 0);
    if (x[0] < 7) return false;
    __cpuidex(x, 7, 0);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
    if ((r[2] & (1u << 29)) == 0) return false;   // F16C
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#endif
    constexpr unsigned kAvx512Dq = 1u << 17;
    constexpr unsigned kAvx512Vnni = 1u << 11;
    constexpr unsigned kAvx512Bitalg = 1u << 12;
    return (r[1] & kAvx512Dq) != 0 && (r[2] & kAvx512Vnni) != 0 && (r[2] & kAvx512Bitalg) != 0;
}

struct Codes3S {
    alignas(64) uint16_t c[512];
    alignas(64) uint16_t pidx[2][32];
    bool ok = true;
    Codes3S() {
        static constexpr int levels[8] = {1, 3, 5, 7, 9, 11, 13, 15};
        for (int i = 0; i < 512; ++i) {
            int v = 0;
            for (int k = 0; k < 4; ++k) {
                const int b = (int) ((iq3s_grid[i] >> (8 * k)) & 255u);
                int level = -1;
                for (int j = 0; j < 8; ++j)
                    if (levels[j] == b) level = j;
                if (level < 0) ok = false;
                v |= (level < 0 ? 0 : level) << (3 * k);
            }
            c[i] = (uint16_t) v;
        }
        for (int half = 0; half < 2; ++half)
            for (int m = 0; m < 8; ++m) {
                pidx[half][4 * m] = pidx[half][4 * m + 1] = (uint16_t) (16 * half + 2 * m);
                pidx[half][4 * m + 2] = pidx[half][4 * m + 3] = (uint16_t) (16 * half + 2 * m + 1);
            }
    }
};

const Codes3S& codes3s() {
    static const Codes3S table;
    return table;
}

#if defined(__GNUC__) || defined(__clang__)
#define MAYA_NT1_IQ512_ATTR __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi,avx512bitalg,avx512vnni,bmi2,avx2,fma,f16c")))
#else
#define MAYA_NT1_IQ512_ATTR
#endif

MAYA_NT1_IQ512_ATTR MAYA_ALWAYS_INLINE uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
MAYA_NT1_IQ512_ATTR MAYA_ALWAYS_INLINE __m512i lookup8(const __m512i* table, __m512i idx) {
    __m512i r[8];
    for (int j = 0; j < 8; ++j) r[j] = _mm512_permutex2var_epi16(table[2 * j], idx, table[2 * j + 1]);
    const __mmask32 m6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 m7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    const __mmask32 m8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
    const __m512i a = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[0], r[1]), _mm512_mask_blend_epi16(m6, r[2], r[3]));
    const __m512i b = _mm512_mask_blend_epi16(m7, _mm512_mask_blend_epi16(m6, r[4], r[5]), _mm512_mask_blend_epi16(m6, r[6], r[7]));
    return _mm512_mask_blend_epi16(m8, a, b);
}

MAYA_NT1_IQ512_ATTR void rows_iq3s_512(const uint8_t* w, size_t rb, int n, const void* const* act,
                                         float* const* out, int r0, int r1) {
    const Codes3S& c = codes3s();
    const int nb = n / QK_K;
    __m512i table[16];
    for (int k = 0; k < 16; ++k) table[k] = _mm512_load_si512((const void*) (c.c + 32 * k));
    const __m512i lut = _mm512_set4_epi32(0, 0, 0x0F0D0B09, 0x07050301);
    const __m512i ctl = _mm512_set1_epi64(0x2926232009060300LL);
    const __m512i seven = _mm512_set1_epi8(7), zero = _mm512_setzero_si512(), hi = _mm512_set1_epi16(256);
    const __m512i pidx0 = _mm512_load_si512(c.pidx[0]), pidx1 = _mm512_load_si512(c.pidx[1]);
    const block_q8_K* y = (const block_q8_K*) act[0];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * sizeof(block_iq3_s);
            const uint8_t* qh = blk + 2 + QK_K / 4;
            const __m512i a = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2)));
            const __m512i b = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 34)));
            const __m512i c0 = lookup8(table, _mm512_or_si512(a, _mm512_maskz_mov_epi16((__mmask32) u32(qh), hi)));
            const __m512i c1 = lookup8(table, _mm512_or_si512(b, _mm512_maskz_mov_epi16((__mmask32) u32(qh + 4), hi)));
            __m512i acc = _mm512_setzero_si512();
            for (int p = 0; p < 4; ++p) {
                const __m512i rep = _mm512_permutexvar_epi16(p & 1 ? pidx1 : pidx0, p >> 1 ? c1 : c0);
                const __m512i g = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, rep), seven));
                const __mmask64 neg = _cvtu64_mask64(u64(blk + 74 + 8 * p));
                const int sc = blk[106 + p];
                const __m512i s = scales4(2 * (sc & 15) + 1, 2 * (sc & 15) + 1, 2 * (sc >> 4) + 1, 2 * (sc >> 4) + 1);
                const __m512i q8 = _mm512_loadu_si512((const void*) (y[i].qs + 64 * p));
                const __m512i d = _mm512_maddubs_epi16(g, _mm512_mask_sub_epi8(q8, neg, zero, q8));
                acc = _mm512_dpwssd_epi32(acc, d, s);
            }
            accf = _mm256_fmadd_ps(_mm256_set1_ps(h2f(u16(blk)) * y[i].d), _mm256_cvtepi32_ps(fold(acc)), accf);
        }
        out[0][r - r0] = hsum_ggml(accf);
    }
}

MAYA_ALWAYS_INLINE float row_iq2xxs(const uint8_t* row, int nblocks, const block_q8_K* y, const __m512i* T,
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
    static const bool extra_on = [] {
        const char* v = std::getenv("MAYA_GLM_CPU512_1T");
        return v != nullptr && std::atoi(v) != 0 && cpu_nt1_iq512_ok() && codes3s().ok;
    }();
    return (ok && type == 16) || (extra_on && (type == 18 || type == 21 || type == 22));
}

void iq512_one_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1) {
    if (r1 <= r0) return;
    if (type == 22 || type == 18 || type == 21) {
        // This entry point permits nt > 1, but these additions are intentionally
        // one-token kernels. iq2v/iq3v stream adjacent blocks and write a range
        // at output offset zero. Keep that fast path for packed matrices; a padded
        // source must be submitted one logical row at a time.
        const size_t packed_row_bytes = (size_t) (n / QK_K) *
                                        (type == 22 ? sizeof(block_iq2_s) : type == 18 ? sizeof(block_iq3_xxs) : 0);
        for (int t = 0; t < nt; ++t) {
            const void* one_act[1] = {act[t]};
            float* one_out[1] = {out[t] + r0};
            if (type == 22) {
                if (row_bytes == packed_row_bytes)
                    iq2v::rows(13, w + (size_t) r0 * row_bytes, row_bytes, n, one_act, 1, one_out, 0, r1 - r0);
                else
                    for (int r = r0; r < r1; ++r) {
                        float* row_out[1] = {out[t] + r};
                        iq2v::rows(13, w + (size_t) r * row_bytes, row_bytes, n, one_act, 1, row_out, 0, 1);
                    }
            } else if (type == 18) {
                if (row_bytes == packed_row_bytes)
                    iq3v::rows_iq3xxs(5, w + (size_t) r0 * row_bytes, row_bytes, n, one_act, 1, one_out, 0, r1 - r0);
                else
                    for (int r = r0; r < r1; ++r) {
                        float* row_out[1] = {out[t] + r};
                        iq3v::rows_iq3xxs(5, w + (size_t) r * row_bytes, row_bytes, n, one_act, 1, row_out, 0, 1);
                    }
            } else {
                rows_iq3s_512(w, row_bytes, n, one_act, one_out, r0, r1);
            }
        }
        return;
    }
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

}  // namespace maya::kernels::cpu
