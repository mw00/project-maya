// src/kernels/cpu/kq_avx512.cpp - Q3_K / Q2_K rows against many Q8_K activations, AVX-512BW
// (include/maya/kernels/cpu/kq_avx512.hpp).  Compiled with -mavx512f -mavx512bw; kq_avx512_ok() gates the calls.
#include "maya/kernels/cpu/kq_avx512.hpp"

#include <immintrin.h>

#include <cstring>
#include <vector>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

namespace maya::kernels::cpu {

bool kq_avx512_ok() noexcept {
#if defined(__x86_64__) && !defined(_WIN32)
    static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
    return ok;
#else
    return false;
#endif
}

bool kq_type_ok(int t) noexcept { return t == 10 || t == 11; }

namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }   // (_cvtsh_ss: not in MSVC)

// One weight row, unpacked: the codes as bytes in value order (Q3_K: q + 4h, 0..7; Q2_K: q, 0..3), each 64-value
// slice's scales spread over the 32 int16 lanes vpmaddubsw leaves (8 lanes a 16-value group), the correction
// weights per group (Q3_K: 4 * scale - the codes' -4; Q2_K: the min) and the block scales.
struct Row {
    std::vector<uint8_t> u;       // n bytes
    std::vector<int16_t> sc;      // n / 64 slices x 32 lanes
    std::vector<int16_t> cw;      // n / 16 groups
    std::vector<float> dw, dc;    // per 256-value block: the scale of the main sum and of the correction
};

// the 16 group scales (int16, in a ymm) -> the 4 slices' vpmaddwd operands: slice v's lane l takes group 4v + l / 8
inline void spread_scales(__m256i s16, int16_t* sc) {
    static const __m512i idx[4] = {
        _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi16(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi16(11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8,
                         8, 8, 8, 8, 8),
        _mm512_set_epi16(15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13,
                         12, 12, 12, 12, 12, 12, 12, 12)};
    const __m512i z = _mm512_castsi256_si512(s16);
    for (int v = 0; v < 4; ++v) _mm512_storeu_si512((void*) (sc + 32 * v), _mm512_permutexvar_epi16(idx[v], z));
}

void unpack_q3(const block_q3_K* x, int nb, Row& R) {
    const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1);
    alignas(16) int8_t scales[16];
    for (int i = 0; i < nb; ++i) {
        const __m256i h = _mm256_loadu_si256((const __m256i*) x[i].hmask);
        uint8_t* u = R.u.data() + (size_t) i * 256;
        for (int c = 0; c < 2; ++c) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * c));
            for (int s = 0; s < 4; ++s) {
                const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * s)), m3);
                const __m256i hb = _mm256_and_si256(_mm256_srl_epi16(h, _mm_cvtsi32_si128(4 * c + s)), m1);
                _mm256_storeu_si256((__m256i*) (u + 128 * c + 32 * s), _mm256_add_epi8(lo, _mm256_slli_epi16(hb, 2)));
            }
        }
        // the 6-bit scales (ggml's unpacking), less 32
        uint32_t aux[4];
        std::memcpy(aux, x[i].scales, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        std::memcpy(scales, aux, 16);
        const __m256i s16 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) scales)),
                                             _mm256_set1_epi16(32));
        spread_scales(s16, R.sc.data() + (size_t) i * 4 * 32);
        _mm256_storeu_si256((__m256i*) (R.cw.data() + (size_t) i * 16), _mm256_slli_epi16(s16, 2));
        R.dw[(size_t) i] = h2f(x[i].d);
        R.dc[(size_t) i] = R.dw[(size_t) i];
    }
}

void unpack_q2(const block_q2_K* x, int nb, Row& R) {
    const __m256i m3 = _mm256_set1_epi8(3);
    for (int i = 0; i < nb; ++i) {
        uint8_t* u = R.u.data() + (size_t) i * 256;
        for (int k = 0; k < 2; ++k) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * k));
            for (int s = 0; s < 4; ++s)
                _mm256_storeu_si256((__m256i*) (u + 128 * k + 32 * s),
                                    _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * s)), m3));
        }
        const __m256i b16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x[i].scales));
        spread_scales(_mm256_and_si256(b16, _mm256_set1_epi16(0xF)), R.sc.data() + (size_t) i * 4 * 32);
        _mm256_storeu_si256((__m256i*) (R.cw.data() + (size_t) i * 16), _mm256_srli_epi16(b16, 4));
        R.dw[(size_t) i] = h2f(x[i].d);
        R.dc[(size_t) i] = h2f(x[i].dmin);
    }
}

}  // namespace

size_t kq_act_bytes(int nt, int n) {
    const int nb = n / 256;
    return (size_t) nt * n + (size_t) nt * nb * sizeof(float) + (size_t) nt * nb * 16 * sizeof(int16_t) + 64;
}

void kq_pack_act(const void* const* act, int nt, int n, void* dst) {
    const int nb = n / 256;
    int8_t* qa = (int8_t*) dst;
    float* d = (float*) (qa + (size_t) nt * n);
    int16_t* b = (int16_t*) (d + (size_t) nt * nb);
    for (int t = 0; t < nt; ++t) {
        const block_q8_K* y = (const block_q8_K*) act[t];
        for (int i = 0; i < nb; ++i) {
            std::memcpy(qa + (size_t) t * n + (size_t) i * 256, y[i].qs, 256);
            d[(size_t) t * nb + i] = y[i].d;
            std::memcpy(b + ((size_t) t * nb + i) * 16, y[i].bsums, 32);
        }
    }
}

void kq_rows_packed(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* packed, int nt,
                    float* const* out, int r0, int r1) {
    const int nb = n / 256;
    thread_local Row R;
    if (R.u.size() < (size_t) n) {
        R.u.resize((size_t) n);
        R.sc.resize((size_t) n / 64 * 32);
        R.cw.resize((size_t) n / 16);
    }
    if ((int) R.dw.size() < nb) {
        R.dw.resize((size_t) nb);
        R.dc.resize((size_t) nb);
    }
    const int8_t* qa = (const int8_t*) packed;
    const float* dbuf = (const float*) (qa + (size_t) nt * n);
    const int16_t* bbuf = (const int16_t*) (dbuf + (size_t) nt * nb);
    constexpr int TB = 8;   // tokens a pass over a row (16 zmm accumulators)
    const bool q3 = ggml_type == 11;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        if (q3) unpack_q3((const block_q3_K*) row, nb, R);
        else unpack_q2((const block_q2_K*) row, nb, R);
        for (int t0 = 0; t0 < nt; t0 += TB) {
            const int kk = nt - t0 < TB ? nt - t0 : TB;
            __m512 facc[TB];
            for (int k = 0; k < TB; ++k) facc[k] = _mm512_setzero_ps();
            for (int i = 0; i < nb; ++i) {
                __m512i acc[TB];
                for (int k = 0; k < TB; ++k) acc[k] = _mm512_setzero_si512();
                const uint8_t* u = R.u.data() + (size_t) i * 256;
                const int16_t* sc = R.sc.data() + (size_t) i * 4 * 32;
                const int8_t* qi = qa + (size_t) t0 * n + (size_t) i * 256;
                for (int v = 0; v < 4; ++v) {
                    const __m512i uv = _mm512_loadu_si512((const void*) (u + 64 * v));
                    const __m512i sv = _mm512_loadu_si512((const void*) (sc + 32 * v));
                    for (int k = 0; k < kk; ++k) {
                        const __m512i av = _mm512_load_si512((const void*) (qi + (size_t) k * n + 64 * v));
                        acc[k] = _mm512_add_epi32(acc[k], _mm512_madd_epi16(_mm512_maddubs_epi16(uv, av), sv));
                    }
                }
                const __m256i cw = _mm256_loadu_si256((const __m256i*) (R.cw.data() + (size_t) i * 16));
                for (int k = 0; k < kk; ++k) {
                    const size_t ti = (size_t) (t0 + k) * nb + i;
                    const __m256i bs = _mm256_loadu_si256((const __m256i*) (bbuf + ti * 16));
                    const __m512i corr = _mm512_zextsi256_si512(_mm256_madd_epi16(cw, bs));
                    const float da = dbuf[ti];
                    if (q3) {   // the -4's correction shares the block's scale: one conversion
                        facc[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[k], corr)),
                                                  _mm512_set1_ps(da * R.dw[(size_t) i]), facc[k]);
                    } else {
                        facc[k] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[k]), _mm512_set1_ps(da * R.dw[(size_t) i]),
                                                  facc[k]);
                        facc[k] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(corr), _mm512_set1_ps(da * R.dc[(size_t) i]),
                                                   facc[k]);
                    }
                }
            }
            for (int k = 0; k < kk; ++k) out[t0 + k][r] = _mm512_reduce_add_ps(facc[k]);
        }
    }
}

void kq_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1) {
    thread_local std::vector<uint8_t> buf;
    const size_t need = kq_act_bytes(nt, n);
    if (buf.size() < need) buf.resize(need);
    void* p = (void*) (((uintptr_t) buf.data() + 63) & ~(uintptr_t) 63);
    kq_pack_act(act, nt, n, p);
    kq_rows_packed(ggml_type, w, row_bytes, n, p, nt, out, r0, r1);
}

}  // namespace maya::kernels::cpu
