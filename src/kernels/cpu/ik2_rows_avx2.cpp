// src/kernels/cpu/ik2_rows_avx2.cpp - the ik2 CPU lane's hand-tuned rows (the interface: ik2_rows.hpp).
//
// NVFP4 first: ggml-cpu's ggml_vec_dot_nvfp4_q8_0 is the generic scalar loop (ggml-quants.c:330) - per sub-block a
// 16-value integer sum - and on this model family it is the type most expert rows carry.  The same sums, 16 values an
// instruction: the 4-bit codes nibble-split, shuffled through kvalues_fp4, widened to 16-bit and _mm256_madd_epi16'd
// against the Q8_0 activation's bytes; the sub-block's UE4M3 scale (x 0.5, its convention) and the Q8_0 block's FP16
// scale applied once a sub-block, as ggml does.
//
// Built with /arch:AVX2 (-mavx2 -mfma): only reached when native_hand_ok() found AVX2 + FMA on the CPU.
#include "ik2_rows.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// ggml's kvalues_fp4 (ggml-common.h:1126): the 16 4-bit codes' values, x 2 the E2M1 floats (the scale halves it).
constexpr int8_t kVal4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ggml's ggml_ue4m3_to_fp32 (ggml-impl.h:514), tabulated once
struct Ue4m3 {
    float v[256];
    Ue4m3() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            const float raw = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
const Ue4m3 g_ue4m3;

inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    uint32_t e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {   // subnormal: normalise
            int sh = 0;
            while (!(m & 0x400)) m <<= 1, ++sh;
            u = sign | ((uint32_t) (127 - 15 - sh) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7F800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// One row of a type-40 (NVFP4) tensor: 36-byte blocks (4 UE4M3 sub-block scales, then 32 bytes of two 4-bit codes a
// byte) against a Q8_0 activation (34-byte blocks: fp16 d, 32 int8).  A sub-block's 16 values and its activation's
// bytes make 8 int32 pair sums; those are scaled by dy * d as floats and starred, so the horizontal reduce is one a
// row - the per-sub-block hadds and the scalar tail were the first form's cost.
void nvfp4_rows(const uint8_t* W, size_t rb, const uint8_t* act, int n, int r0, int r1, float* out) {
    const __m128i tab = _mm_loadu_si128((const __m128i*) kVal4);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = n / 64;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        const uint8_t* a = act;
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int i = 0; i < nb; ++i, b += 36, a += 68) {
            for (int sb = 0; sb < 4; ++sb) {
                const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + sb * 8));
                const __m128i lo = _mm_and_si128(raw, m4);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
                const __m128i codes = _mm_unpacklo_epi64(lo, hi);   // j = 0..7 then 8..15, as ggml reads them
                const __m256i v = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(tab, codes));
                const uint8_t* qp = a + (sb / 2) * 34 + 2 + (sb % 2) * 16;
                const __m256i q = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) qp));
                const __m256 p = _mm256_cvtepi32_ps(_mm256_madd_epi16(v, q));
                uint16_t dh;
                std::memcpy(&dh, a + (sb / 2) * 34, 2);
                const __m256 s = _mm256_set1_ps(h2f(dh) * g_ue4m3.v[b[sb]]);
                acc[sb] = _mm256_fmadd_ps(p, s, acc[sb]);   // (a chain each: 16 fmas into one would be the latency)
            }
        }
        const __m256 a4 = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(a4), _mm256_extractf128_ps(a4, 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
        out[r - r0] = _mm_cvtss_f32(t);
    }
}

}  // namespace

bool hand_rows(int type, size_t row_bytes, const uint8_t* base, const void* act, int n, int r0, int r1, float* out) {
    if (type == 40) {   // GGML_TYPE_NVFP4
        nvfp4_rows(base, row_bytes, (const uint8_t*) act, n, r0, r1, out);
        return true;
    }
    return false;
}

}  // namespace strata::kernels::cpu
