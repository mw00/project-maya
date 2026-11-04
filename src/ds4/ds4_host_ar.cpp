// src/ds4/ds4_host_ar.cpp - the host's part of a chunk's all-reduce (allreduce_host): the members' FP16 partials summed
// on the CPU, ar_rs16_kernel's arithmetic bit for bit - each partial converted to F32 (exact), added in member order
// (IEEE adds, round to nearest), the sum clamped to FP16's range (fmaxf then fminf: a NaN goes where the device's
// goes) and rounded to FP16, nearest even.
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAR_X86 1
#endif

namespace strata::ds4 {

namespace {

float h2f(uint16_t h) {   // (exact; FP16 subnormals are F32 normals)
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            int k = -1;
            uint32_t mm = m;
            while ((mm & 0x400u) == 0) {
                mm <<= 1;
                ++k;
            }
            u = s | ((uint32_t) (127 - 15 - k) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 127 - 15) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f2h(float f) {   // round to nearest even (the value already within FP16's range)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xffu) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xffu) == 0xffu) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0u));
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    if (e <= 0) {   // subnormal (or zero)
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return (uint16_t) (s | r);
}

}  // namespace

#if defined(HAR_X86) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c")))
#endif
void ar_host_sum(const uint16_t* const* rows, int nm, uint16_t* out, size_t i0, size_t i1) {
    size_t i = i0;
#if defined(HAR_X86)
    const __m256 hi = _mm256_set1_ps(65504.0f), lo = _mm256_set1_ps(-65504.0f);
    for (; i + 8 <= i1; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (rows[0] + i)));
        for (int m = 1; m < nm; ++m) v = _mm256_add_ps(v, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (rows[m] + i))));
        v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);   // (fmaxf(v, -m): a NaN gives -m; then fminf)
        _mm_storeu_si128((__m128i*) (out + i), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
#endif
    for (; i < i1; ++i) {
        float v = h2f(rows[0][i]);
        for (int m = 1; m < nm; ++m) v = v + h2f(rows[m][i]);
        v = v != v ? -65504.0f : v < -65504.0f ? -65504.0f : v > 65504.0f ? 65504.0f : v;
        out[i] = f2h(v);
    }
}

}  // namespace strata::ds4
