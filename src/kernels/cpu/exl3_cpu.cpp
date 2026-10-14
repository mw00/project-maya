// src/kernels/cpu/exl3_cpu.cpp - EXL3 routed experts on the CPU (see include/strata/kernels/cpu/exl3_cpu.hpp).
//
// The vector kernel follows exllamav3's CPU expert kernel (MIT, turboderp - exllamav3_ext/cpu/moe_mul1.cpp, its
// AVX-512 BW tier): a tile row's 16 states come out of the tile's words with two cross-lane permutes and a per-lane
// funnel shift from compile-time tables, the codebook product's bytes are added with vpmaddubsw against ones and
// vpmaddwd against ones.  Unlike there, the activations stay F32 (no int8 rows) and the affine offset is the device's
// FP16 constant, so the CPU lane computes what the device kernels compute up to the FP16 rounding of each weight.
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define EXL3_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(EXL3_X86) && (defined(__GNUC__) || defined(__clang__))
#define EXL3_AVX512 __attribute__((target("avx512f,avx512bw,fma")))
#else
#define EXL3_AVX512
#endif

namespace strata::kernels::cpu {
namespace {

constexpr uint32_t kMul1 = 0x83DCD12Du;

float k_inv() { return f32_from_f16(0x1eee); }
float k_zero() { return 1024.0f * f32_from_f16(0x1eee) + f32_from_f16(0xc931); }   // w = s * k_inv + k_zero

// the 128-point Sylvester transform over sqrt(128), in place
void had128(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
    const float s = 0.088388347648f;
    for (int i = 0; i < 128; ++i) v[i] *= s;
}

// step p of a tile: row (p/8 % 4) * 2 + (p & 1) + 8 (p/2 & 1), column p/32 + 8 (p/4 & 1); inv[row * 16 + col] = p
constexpr std::array<uint16_t, 256> make_inv() {
    std::array<uint16_t, 256> inv{};
    for (int p = 0; p < 256; ++p) {
        const int lane = p >> 3, j = p & 7;
        const int r = (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = (lane >> 2) + 8 * (j >> 2);
        inv[(size_t) (r * 16 + c)] = (uint16_t) p;
    }
    return inv;
}

// a tile row's 16 states: column c's 16-bit window ends at bit b1 = (p + 1 + 256) K of the tile's MSB-first stream;
// its words i0 (the window's first bit) and i1 (its last), and the right shift of (w[i0] : w[i1]) that aligns it
template <int K>
struct RowTab {
    int32_t i0[16][16] = {}, i1[16][16] = {}, sr[16][16] = {}, sl[16][16] = {};
    uint16_t hi0[16] = {}, hi1[16] = {};   // columns whose word is in the second register pair (>= 32)
};
template <int K>
constexpr RowTab<K> make_rowtab() {
    RowTab<K> t{};
    constexpr int W = 8 * K;
    const auto inv = make_inv();
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            const int p = inv[(size_t) (r * 16 + c)];
            const int b1 = (p + 1 + 256) * K;
            const int w1 = ((b1 - 1) >> 5) % W, w0 = ((b1 - 16) >> 5) % W;
            const int sh = (((b1 - 1) >> 5) + 1) * 32 - b1;
            t.i0[r][c] = w0;
            t.i1[r][c] = w1;
            t.sr[r][c] = sh;
            t.sl[r][c] = 32 - sh;   // a shift by 32 clears the lane: a window inside one word
            if (w0 >= 32) t.hi0[r] = (uint16_t) (t.hi0[r] | (1u << c));
            if (w1 >= 32) t.hi1[r] = (uint16_t) (t.hi1[r] | (1u << c));
        }
    return t;
}

// ---------------------------------------------------------------- scalar
template <int K>
void band_scalar(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* acc /* 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    const int ld = m.n / 16;
    for (int i = 0; i < 128; ++i) acc[i] = 0.0f;
    for (int ks = ks0; ks < ks1; ++ks)
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = m.trellis + ((size_t) ks * ld + tc0 + t) * W;
            for (int r = 0; r < 16; ++r) {
                const float xv = xh[ks * 16 + r];
                for (int c = 0; c < 16; ++c) {
                    const uint64_t a = w[T.i0[r][c]], b = w[T.i1[r][c]];
                    const uint32_t st = (uint32_t) (((a << 32) | b) >> T.sr[r][c]) & 0xffffu;
                    const uint32_t x = st * kMul1;
                    const int s = (int) ((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
                    acc[t * 16 + c] += (float) s * xv;
                }
            }
        }
}

// ---------------------------------------------------------------- AVX-512
#if defined(EXL3_X86)
template <int K>
EXL3_AVX512 inline __m512i gather_words(const RowTab<K>& T, const int32_t (&idx)[16][16], const uint16_t* hi, int r,
                                        __m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i ix = _mm512_loadu_si512((const void*) idx[r]);
    __m512i v = _mm512_permutex2var_epi32(p0, ix, p1);
    if constexpr (8 * K > 32) {
        if (hi[r] != 0) v = _mm512_mask_blend_epi32((__mmask16) hi[r], v, _mm512_permutex2var_epi32(p2, ix, p3));
    }
    (void) T;
    return v;
}

template <int K>
EXL3_AVX512 void band_avx512(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* out /* 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    __m512 acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = _mm512_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        // the band's 8 tiles of this slice are contiguous; the next slices' are one row of tiles on
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk = xh + ks * 16;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a = acc[t];
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs = _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16);
                a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(bs), _mm512_set1_ps(xk[r]), a);
            }
            acc[t] = a;
        }
    }
    for (int t = 0; t < 8; ++t) _mm512_storeu_ps(out + t * 16, acc[t]);
}

// band_avx512 for N activations at once (xh[i] -> out[i]), as exllamav3's CPU kernel takes up to 4 token rows: each
// tile row's 16 states are pulled out and byte-summed once, then one FMA per activation - activation i's FMAs are
// band_avx512's for it alone, in its order, so out[i] is the same bit for bit.  An extra activation costs one FMA of
// the row's ~12 instructions.  (The tiles' accumulators live in memory, loaded once a tile and slice as there: only
// the N of the tile in hand take registers.)
template <int K, int N>
EXL3_AVX512 void band_avx512_n(const Exl3Part& m, const float* const* xh, int tc0, int ks0, int ks1,
                               float* const* out /* N x 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    __m512 acc[8][N];
    for (int t = 0; t < 8; ++t)
        for (int i = 0; i < N; ++i) acc[t][i] = _mm512_setzero_ps();
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        const float* xk[N];
        for (int i = 0; i < N; ++i) xk[i] = xh[i] + ks * 16;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a[N];
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) a[i] = acc[t][i];
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16));
#pragma GCC unroll 8
                for (int i = 0; i < N; ++i) a[i] = _mm512_fmadd_ps(f, _mm512_set1_ps(xk[i][r]), a[i]);
            }
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) acc[t][i] = a[i];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < 8; ++t) _mm512_storeu_ps(out[i] + t * 16, acc[t][i]);
}

// rows_avx512_n: slices [ks0, ks1) of EVERY column - a split-K piece - for N activations (xh[i] -> acc[i], n floats
// each, overwritten).  The tile rows are read in memory order, where a band's tiles are 1 KB runs a whole tile row
// apart: 40 threads of a 2-socket Xeon 6152 read a K4 expert from DRAM in 0.14 ms in pieces, 0.21 in bands (~90 vs
// ~60 GB/s), the kernel the same.  A tile's 16 sums per activation live in acc between slices; each row's decode and
// FMAs are band_avx512's, so acc[i] is the N = 1 call's for xh[i] alone, bit for bit.
template <int K, int N>
EXL3_AVX512 void rows_avx512_n(const Exl3Part& m, const float* const* xh, int ks0, int ks1, float* const* acc) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < m.n; c += 16) _mm512_storeu_ps(acc[i] + c, _mm512_setzero_ps());
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        float xk[N][16];
        for (int i = 0; i < N; ++i)
            for (int r = 0; r < 16; ++r) xk[i][r] = xh[i][ks * 16 + r];
        for (int t = 0; t < ld; ++t) {
            const uint32_t* w = row + (size_t) t * W;
            for (int l = 0; l < W * 4; l += 64) _mm_prefetch((const char*) (w + 16 * W) + l, _MM_HINT_T0);
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a[N];
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) a[i] = _mm512_loadu_ps(acc[i] + t * 16);
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16));
#pragma GCC unroll 8
                for (int i = 0; i < N; ++i) a[i] = _mm512_fmadd_ps(f, _mm512_set1_ps(xk[i][r]), a[i]);
            }
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) _mm512_storeu_ps(acc[i] + t * 16, a[i]);
        }
    }
}

// band_gemm: a band's byte-sum products for many activations - each chunk of kGemmK inputs decoded ONCE into F32
// (wc, in L2), then multiplied as a GEMM: tiles of TB activations x 64 columns, the sums in registers, two FMAs a
// load.  The prompt path's CPU rows, where an expert serves tens to hundreds of tokens: one decode (~6 instructions a
// 16-weight row) for all of them instead of one per kExl3CpuMaxN.
constexpr int kGemmK = 256;

template <int K>
EXL3_AVX512 void decode_chunk(const Exl3Part& m, int tc0, int ks0, int ks1, float* wc /* (ks1 - ks0) 16 x 128 */) {
    constexpr int W = 8 * K;
    static constexpr RowTab<K> T = make_rowtab<K>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const size_t row_words = (size_t) (m.n / 16) * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words + (size_t) tc0 * W;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        const char* pf = (const char*) (row + 4 * row_words);
        for (int l = 0; l < 8 * W * 4; l += 64) _mm_prefetch(pf + l, _MM_HINT_T0);
        float* wr = wc + (size_t) (ks - ks0) * 16 * 128;
        for (int t = 0; t < 8; ++t) {
            const uint32_t* w = row + t * W;
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs =
                    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_mullo_epi32(st, mult), ones8), ones16);
                _mm512_store_ps(wr + r * 128 + t * 16, _mm512_cvtepi32_ps(bs));
            }
        }
    }
}

// acc[t][0..128) += sum_{k < kr} x[t][k] wc[k][..] for TB activations
template <int TB>
EXL3_AVX512 void gemm_tile(const float* wc, int kr, const float* const* x, float* const* acc) {
    for (int cb = 0; cb < 128; cb += 64) {
        __m512 a[TB][4];
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) a[t][j] = _mm512_loadu_ps(acc[t] + cb + j * 16);
        const float* w = wc + cb;
        for (int k = 0; k < kr; ++k, w += 128) {
            const __m512 w0 = _mm512_load_ps(w), w1 = _mm512_load_ps(w + 16), w2 = _mm512_load_ps(w + 32),
                         w3 = _mm512_load_ps(w + 48);
#pragma GCC unroll 8
            for (int t = 0; t < TB; ++t) {
                const __m512 xv = _mm512_set1_ps(x[t][k]);
                a[t][0] = _mm512_fmadd_ps(xv, w0, a[t][0]);
                a[t][1] = _mm512_fmadd_ps(xv, w1, a[t][1]);
                a[t][2] = _mm512_fmadd_ps(xv, w2, a[t][2]);
                a[t][3] = _mm512_fmadd_ps(xv, w3, a[t][3]);
            }
        }
        for (int t = 0; t < TB; ++t)
            for (int j = 0; j < 4; ++j) _mm512_storeu_ps(acc[t] + cb + j * 16, a[t][j]);
    }
}

// nb bands from tile column tc0 (acc[t]: the activation's 128 nb outputs from there): each chunk of the inputs is
// taken for every band in turn, so the activations' chunk (n x kGemmK floats) comes from L2 for the nb bands - an
// expert's rows are its own tokens, and with a job a band they came from DRAM again for every band (~16x the
// expert's weights in bytes)
template <int K>
EXL3_AVX512 void band_gemm_avx512(const Exl3Part& m, const float* const* xh, int n, int tc0, int nb,
                                  float* const* acc, float* wc) {
    constexpr int KS = kGemmK / 16;
    constexpr int TB = 6;   // 24 sums + 4 weights + the activation: 29 of the 32 registers
    for (int t = 0; t < n; ++t) std::memset(acc[t], 0, (size_t) nb * 128 * sizeof(float));
    std::vector<float*> ab((size_t) n);
    for (int ks0 = 0; ks0 < m.k / 16; ks0 += KS) {
        const int ks1 = std::min(m.k / 16, ks0 + KS), kr = (ks1 - ks0) * 16;
        for (int bb = 0; bb < nb; ++bb) {
            decode_chunk<K>(m, tc0 + 8 * bb, ks0, ks1, wc);
            for (int t = 0; t < n; ++t) ab[(size_t) t] = acc[t] + bb * 128;
            float* const* ap = ab.data();
            int t = 0;
            for (; t + TB <= n; t += TB) {
                const float* xp[TB];
                for (int i = 0; i < TB; ++i) xp[i] = xh[t + i] + ks0 * 16;
                gemm_tile<TB>(wc, kr, xp, ap + t);
            }
            if (t < n) {   // the last 1..TB-1 activations in one pass too
                const float* xp[TB];
                for (int i = 0; i < n - t; ++i) xp[i] = xh[t + i] + ks0 * 16;
                switch (n - t) {
                    case 1: gemm_tile<1>(wc, kr, xp, ap + t); break;
                    case 2: gemm_tile<2>(wc, kr, xp, ap + t); break;
                    case 3: gemm_tile<3>(wc, kr, xp, ap + t); break;
                    case 4: gemm_tile<4>(wc, kr, xp, ap + t); break;
                    default: gemm_tile<5>(wc, kr, xp, ap + t); break;
                }
            }
        }
    }
}

// had128 in 8 registers: every butterfly takes the operands the scalar loop takes (a lane's partner by a permute; the
// lower lane a + b, the upper b's lane t - r = a - b), so the result is had128's bit for bit
EXL3_AVX512 inline __m512 bfly(__m512 r, __m512 t, __mmask16 upper) {
    return _mm512_mask_sub_ps(_mm512_add_ps(r, t), upper, t, r);
}
EXL3_AVX512 void had128_avx512(float* v) {
    __m512 r[8];
    for (int i = 0; i < 8; ++i) r[i] = _mm512_loadu_ps(v + 16 * i);
    for (int i = 0; i < 8; ++i) {
        r[i] = bfly(r[i], _mm512_permute_ps(r[i], 0b10110001), (__mmask16) 0xAAAA);              // h = 1
        r[i] = bfly(r[i], _mm512_permute_ps(r[i], 0b01001110), (__mmask16) 0xCCCC);              // h = 2
        r[i] = bfly(r[i], _mm512_shuffle_f32x4(r[i], r[i], 0b10110001), (__mmask16) 0xF0F0);   // h = 4
        r[i] = bfly(r[i], _mm512_shuffle_f32x4(r[i], r[i], 0b01001110), (__mmask16) 0xFF00);   // h = 8
    }
    for (int h = 1; h < 8; h <<= 1)   // h = 16, 32, 64: whole registers
        for (int i = 0; i < 8; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const __m512 a = r[j], b = r[j + h];
                r[j] = _mm512_add_ps(a, b);
                r[j + h] = _mm512_sub_ps(a, b);
            }
    const __m512 s = _mm512_set1_ps(0.088388347648f);
    for (int i = 0; i < 8; ++i) _mm512_storeu_ps(v + 16 * i, _mm512_mul_ps(r[i], s));
}

// exl3_cpu_prepare_block / exl3_cpu_band_out with the FP16 scales converted 16 at a time (vcvtph2ps: exact, as
// f32_from_f16) and the block's sum in eight double lanes
EXL3_AVX512 float prepare_block_avx512(const uint16_t* suh, const float* x, float* xh) {
    for (int i = 0; i < 128; i += 16) {
        const __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (suh + i)));
        _mm512_storeu_ps(xh + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), sc));
    }
    had128_avx512(xh);
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    for (int i = 0; i < 128; i += 16) {
        s0 = _mm512_add_pd(s0, _mm512_cvtps_pd(_mm256_loadu_ps(xh + i)));
        s1 = _mm512_add_pd(s1, _mm512_cvtps_pd(_mm256_loadu_ps(xh + i + 8)));
    }
    return (float) _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
}
EXL3_AVX512 void band_out_avx512(const float* acc, float ki, float kz, const uint16_t* svh, float* y) {
    const __m512 vk = _mm512_set1_ps(ki), vz = _mm512_set1_ps(kz);
    for (int i = 0; i < 128; i += 16) _mm512_storeu_ps(y + i, _mm512_fmadd_ps(_mm512_loadu_ps(acc + i), vk, vz));
    had128_avx512(y);
    for (int i = 0; i < 128; i += 16) {
        const __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (svh + i)));
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), sc));
    }
}

bool detect_avx512() {
    if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) leaf, (int) sub);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    };
    cpuid(0, 0);
    if (r[0] < 7) return false;
    cpuid(1, 0);
    if (!((r[2] >> 27) & 1u) || !((r[2] >> 12) & 1u)) return false;   // OSXSAVE, FMA
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0xE6) != 0xE6) return false;   // the OS saves the AVX-512 state
    cpuid(7, 0);
    return ((r[1] >> 16) & 1u) && ((r[1] >> 30) & 1u);   // AVX-512F, AVX-512BW
}
#endif

template <int K>
void band_any(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* acc) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) {
        band_avx512<K>(m, xh, tc0, ks0, ks1, acc);
        return;
    }
#endif
    band_scalar<K>(m, xh, tc0, ks0, ks1, acc);
}

// n activations (1..kExl3CpuMaxN) on one band
template <int K>
void band_any_n(const Exl3Part& m, const float* const* xh, int n, int tc0, int ks0, int ks1, float* const* acc) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector() && n > 1) {
        switch (n) {
            case 2: band_avx512_n<K, 2>(m, xh, tc0, ks0, ks1, acc); return;
            case 3: band_avx512_n<K, 3>(m, xh, tc0, ks0, ks1, acc); return;
            case 4: band_avx512_n<K, 4>(m, xh, tc0, ks0, ks1, acc); return;
            case 5: band_avx512_n<K, 5>(m, xh, tc0, ks0, ks1, acc); return;
            case 6: band_avx512_n<K, 6>(m, xh, tc0, ks0, ks1, acc); return;
            case 7: band_avx512_n<K, 7>(m, xh, tc0, ks0, ks1, acc); return;
            default: band_avx512_n<K, 8>(m, xh, tc0, ks0, ks1, acc); return;
        }
    }
#endif
    for (int i = 0; i < n; ++i) band_any<K>(m, xh[i], tc0, ks0, ks1, acc[i]);
}

// a split-K piece for n activations (1..kExl3CpuMaxN); the scalar loop: each column's band_scalar
template <int K>
void rows_any_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, float* const* acc) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) {
        switch (n) {
            case 1: rows_avx512_n<K, 1>(m, xh, ks0, ks1, acc); return;
            case 2: rows_avx512_n<K, 2>(m, xh, ks0, ks1, acc); return;
            case 3: rows_avx512_n<K, 3>(m, xh, ks0, ks1, acc); return;
            case 4: rows_avx512_n<K, 4>(m, xh, ks0, ks1, acc); return;
            case 5: rows_avx512_n<K, 5>(m, xh, ks0, ks1, acc); return;
            case 6: rows_avx512_n<K, 6>(m, xh, ks0, ks1, acc); return;
            case 7: rows_avx512_n<K, 7>(m, xh, ks0, ks1, acc); return;
            default: rows_avx512_n<K, 8>(m, xh, ks0, ks1, acc); return;
        }
    }
#endif
    for (int i = 0; i < n; ++i)
        for (int c0 = 0; c0 < m.n; c0 += 128) band_scalar<K>(m, xh[i], c0 / 16, ks0, ks1, acc[i] + c0);
}

}  // namespace

bool exl3_cpu_vector() {
#if defined(EXL3_X86)
    static const bool ok = detect_avx512();
    return ok;
#else
    return false;
#endif
}

namespace {
void had128_any(float* v) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) {
        had128_avx512(v);
        return;
    }
#endif
    had128(v);
}
}  // namespace

float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) return prepare_block_avx512(suh, x, xh);
#endif
    for (int i = 0; i < 128; ++i) xh[i] = x[i] * f32_from_f16(suh[i]);
    had128(xh);
    double sum = 0.0;
    for (int i = 0; i < 128; ++i) sum += xh[i];
    return (float) sum;
}

float exl3_cpu_prepare(const Exl3Part& m, const float* x, float* xh) {
    double sum = 0.0;
    for (int b = 0; b < m.k; b += 128) sum += exl3_cpu_prepare_block(m.suh + b, x + b, xh + b);
    return (float) sum;
}

void exl3_cpu_band_acc(const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc) {
    const int tc0 = c0 / 16;
    switch (m.K) {
        case 1: band_any<1>(m, xh, tc0, ks0, ks1, acc); break;
        case 2: band_any<2>(m, xh, tc0, ks0, ks1, acc); break;
        case 3: band_any<3>(m, xh, tc0, ks0, ks1, acc); break;
        case 4: band_any<4>(m, xh, tc0, ks0, ks1, acc); break;
        case 5: band_any<5>(m, xh, tc0, ks0, ks1, acc); break;
        case 6: band_any<6>(m, xh, tc0, ks0, ks1, acc); break;
        case 7: band_any<7>(m, xh, tc0, ks0, ks1, acc); break;
        case 8: band_any<8>(m, xh, tc0, ks0, ks1, acc); break;
        default:
            for (int i = 0; i < 128; ++i) acc[i] = 0.0f;
    }
}

void exl3_cpu_band_acc_n(const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                         float* const* acc) {
    const int tc0 = c0 / 16;
    for (int i0 = 0; i0 < n; i0 += kExl3CpuMaxN) {
        const int c = std::min(kExl3CpuMaxN, n - i0);
        switch (m.K) {
            case 1: band_any_n<1>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 2: band_any_n<2>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 3: band_any_n<3>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 4: band_any_n<4>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 5: band_any_n<5>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 6: band_any_n<6>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 7: band_any_n<7>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            case 8: band_any_n<8>(m, xh + i0, c, tc0, ks0, ks1, acc + i0); break;
            default:
                for (int i = 0; i < c; ++i)
                    for (int j = 0; j < 128; ++j) acc[i0 + i][j] = 0.0f;
        }
    }
}

void exl3_cpu_rows_acc_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, float* const* acc) {
    for (int i0 = 0; i0 < n; i0 += kExl3CpuMaxN) {
        const int c = std::min(kExl3CpuMaxN, n - i0);
        switch (m.K) {
            case 1: rows_any_n<1>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 2: rows_any_n<2>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 3: rows_any_n<3>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 4: rows_any_n<4>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 5: rows_any_n<5>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 6: rows_any_n<6>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 7: rows_any_n<7>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            case 8: rows_any_n<8>(m, xh + i0, c, ks0, ks1, acc + i0); break;
            default:
                for (int i = 0; i < c; ++i)
                    for (int j = 0; j < m.n; ++j) acc[i0 + i][j] = 0.0f;
        }
    }
}

size_t exl3_cpu_gemm_scratch() { return (size_t) kGemmK * 128; }

void exl3_cpu_band_gemm(const Exl3Part& m, const float* const* xh, int n, int c0, int nb, float* const* acc,
                        float* wc) {
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) {
        switch (m.K) {
            case 1: band_gemm_avx512<1>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 2: band_gemm_avx512<2>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 3: band_gemm_avx512<3>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 4: band_gemm_avx512<4>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 5: band_gemm_avx512<5>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 6: band_gemm_avx512<6>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 7: band_gemm_avx512<7>(m, xh, n, c0 / 16, nb, acc, wc); return;
            case 8: band_gemm_avx512<8>(m, xh, n, c0 / 16, nb, acc, wc); return;
            default: break;
        }
    }
#endif
    (void) wc;
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < nb; ++b) exl3_cpu_band_acc(m, xh[i], c0 + 128 * b, 0, m.k / 16, acc[i] + 128 * b);
}

void exl3_cpu_rows_acc(const Exl3Part& m, const float* xh, int ks0, int ks1, float* acc) {
    exl3_cpu_rows_acc_n(m, &xh, 1, ks0, ks1, &acc);
}

void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y) {
    const float ki = k_inv(), kz = k_zero() * xsum;
#if defined(EXL3_X86)
    if (exl3_cpu_vector()) {
        band_out_avx512(acc, ki, kz, m.svh + c0, y);
        return;
    }
#endif
    for (int i = 0; i < 128; ++i) y[i] = acc[i] * ki + kz;
    had128(y);
    for (int i = 0; i < 128; ++i) y[i] *= f32_from_f16(m.svh[c0 + i]);
}

void exl3_cpu_band(const Exl3Part& m, const float* xh, float xsum, int c0, float* y) {
    float acc[128];
    exl3_cpu_band_acc(m, xh, c0, 0, m.k / 16, acc);
    exl3_cpu_band_out(m, acc, xsum, c0, y);
}

}  // namespace strata::kernels::cpu
