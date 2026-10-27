// src/mm3/mm3_host.cpp - maya-mm3's x86 host helpers (mm3_host.hpp), moved here from mm3_tp.cu as they were.
#include "mm3_host.hpp"

#include <cmath>
#include <cstddef>

#include <immintrin.h>

namespace mm3::host {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    alignas(32) float l[8];
    _mm256_store_ps(l, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    float s = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void rms_rows(const float* x, const float* w, int rows, int D, float eps, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t) r * D;
        const float k = 1.0f / std::sqrt(dot(xr, xr, D) / (float) D + eps);
        const __m256 kk = _mm256_set1_ps(k);
        float* yr = y + (size_t) r * D;
        for (int i = 0; i < D; i += 8)
            _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), kk), _mm256_loadu_ps(w + i)));
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void sum_into(float* h, const float* const* p, int np, const float* extra, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 s = _mm256_loadu_ps(p[0] + i);
        for (int j = 1; j < np; ++j) s = _mm256_add_ps(s, _mm256_loadu_ps(p[j] + i));
        if (extra) s = _mm256_add_ps(s, _mm256_loadu_ps(extra + i));
        _mm256_storeu_ps(h + i, _mm256_add_ps(_mm256_loadu_ps(h + i), s));
    }
}

void pause() { _mm_pause(); }

}  // namespace mm3::host
