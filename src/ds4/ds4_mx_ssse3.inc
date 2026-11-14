// Included inside ds4_gguf_cpu.cpp's anonymous namespace, on x86 only.
// SSSE3 is independent of AVX/FMA: integer decode/dot here, the scalar tier's scale/fma/order unchanged.
// LUT entries are w+12 (0..24). A pmaddubsw pair is at most 2*24*128=6144 in magnitude; adding low/high
// halves is at most 12288. Neither 16-bit step saturates, including activation -128. The existing int32
// activation correction removes 12*sum(x), giving the same exact signed integer dot as rows_q_scalar.
template <int N, bool Fast>
MX_TARGET("ssse3") void rows_ssse3_n(const uint8_t* w, size_t rb, int k, int r0, int r1,
                                     const void* const* act, float* const* y, const float* tb, int pf) {
    const __m128i lut = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m128i mask = _mm_set1_epi8(15), ones = _mm_set1_epi16(1);
    const uint8_t* aq[N];
    const float* ad[N];
    const int32_t* ac[N];
    for (int i = 0; i < N; ++i) {
        aq[i] = (const uint8_t*) act[i];
        ad[i] = (const float*) (aq[i] + act_qbytes(k));
        ac[i] = (const int32_t*) (aq[i] + act_qbytes(k) + 32 * act_groups(k));
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* p = w + (size_t) r * rb;
        float acc[N][8] = {};
        for (int b = 0; b < k / kBlock; ++b, p += kBytes) {
            if (pf > 0) _mm_prefetch((const char*) p + pf, _MM_HINT_T0);
            const __m128i q = _mm_loadu_si128((const __m128i*) (p + 1));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, mask));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), mask));
            const int a = act_lane(b), lane = a & 7;
            const size_t o = (size_t) (b >> 2) * 128 + 16 * (b & 3);
            const float dw = tb[p[0]];
            for (int i = 0; i < N; ++i) {
                const __m128i xl = _mm_loadu_si128((const __m128i*) (aq[i] + o));
                const __m128i xh = _mm_loadu_si128((const __m128i*) (aq[i] + o + 64));
                const __m128i pairs = _mm_add_epi16(_mm_maddubs_epi16(lo, xl), _mm_maddubs_epi16(hi, xh));
                __m128i sum = _mm_madd_epi16(pairs, ones);
                sum = _mm_hadd_epi32(sum, sum);
                sum = _mm_hadd_epi32(sum, sum);
                const int s = _mm_cvtsi128_si32(sum) + ac[i][a];
                acc[i][lane] = detail::mx_fma<Fast>((float) s, dw * ad[i][a], acc[i][lane]);
            }
        }
        for (int i = 0; i < N; ++i) {
            const float* A = acc[i];
            y[i][r - r0] = ((A[0] + A[4]) + (A[2] + A[6])) + ((A[1] + A[5]) + (A[3] + A[7]));
        }
    }
}
template <bool Fast>
void rows_ssse3(const uint8_t* w, size_t rb, int k, int r0, int r1, const void* const* act, int n,
                float* const* y, const float* tb, int pf) {
    for (int i = 0; i < n; i += kMxCpuMaxN) {
        const int count = std::min(kMxCpuMaxN, n - i);
        switch (count) {
#define MX_SSSE3_CASE(N) case N: rows_ssse3_n<N, Fast>(w, rb, k, r0, r1, act + i, y + i, tb, pf); break
            MX_SSSE3_CASE(1);
            MX_SSSE3_CASE(2);
            MX_SSSE3_CASE(3);
            MX_SSSE3_CASE(4);
            MX_SSSE3_CASE(5);
            MX_SSSE3_CASE(6);
            MX_SSSE3_CASE(7);
            MX_SSSE3_CASE(8);
#undef MX_SSSE3_CASE
        }
    }
}
