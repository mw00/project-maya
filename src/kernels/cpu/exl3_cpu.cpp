// src/kernels/cpu/exl3_cpu.cpp - EXL3 routed experts on the CPU (see include/strata/kernels/cpu/exl3_cpu.hpp).
//
// The vector kernel follows exllamav3's CPU expert kernel (MIT, turboderp - exllamav3_ext/cpu/moe_mul1.cpp, its
// AVX-512 BW tier): a tile row's 16 states come out of the tile's words with two cross-lane permutes and a per-lane
// funnel shift from compile-time tables, the codebook product's bytes are added with vpmaddubsw against ones and
// vpmaddwd against ones.  Unlike there, the activations stay F32 (no int8 rows) and the affine offset is the device's
// FP16 constant, so the CPU lane computes what the device kernels compute up to the FP16 rounding of each weight.
#include "strata/kernels/cpu/exl3_cpu.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

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

}  // namespace

bool exl3_cpu_vector() {
#if defined(EXL3_X86)
    static const bool ok = detect_avx512();
    return ok;
#else
    return false;
#endif
}

float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh) {
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

void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y) {
    const float ki = k_inv(), kz = k_zero() * xsum;
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
