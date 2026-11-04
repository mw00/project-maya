// src/kernels/cpu/exl3_cpu_avx512.inc - the AVX-512 tier's trellis decode, included by exl3_cpu.cpp twice: as the
// AVX-512 F/BW tier (tier 2) and as AVX-512 VNNI (tier 3), whose byte sum is one vpdpbusd where F/BW takes a
// vpmaddubsw and a vpmaddwd. The four bytes of a state's mul1 product are 0..255 times 1, so neither saturates and the
// two give the same integer: the same bits, ~7-12% faster from DRAM on a Xeon Gold 6312U (EXL3-4.05bpw, 22 threads).
// The includer defines EXL3_T (the functions' target), EXL3_BSUM(v) (v's four byte sums a 32-bit lane, as int32) and
// EXL3_BSUM_CONSTS (the constants EXL3_BSUM uses), and opens the namespace.

template <int K2>
EXL3_T inline __m512i gather_words(const RowTab<K2>& T, const int32_t (&idx)[16][16], const uint16_t* hi, int r,
                                   __m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i ix = _mm512_loadu_si512((const void*) idx[r]);
    __m512i v = _mm512_permutex2var_epi32(p0, ix, p1);
    if constexpr (4 * K2 > 32) {
        if (hi[r] != 0) v = _mm512_mask_blend_epi32((__mmask16) hi[r], v, _mm512_permutex2var_epi32(p2, ix, p3));
    }
    (void) T;
    return v;
}

template <int K2>
EXL3_T void band_avx512(const Exl3Part& m, const float* xh, int tc0, int ks0, int ks1, float* out /* 128 */) {
    constexpr int W = 4 * K2;
    static constexpr RowTab<K2> T = make_rowtab<K2>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    EXL3_BSUM_CONSTS
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
                const __m512i wa = gather_words<K2>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K2>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs = EXL3_BSUM(_mm512_mullo_epi32(st, mult));
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
template <int K2, int N>
EXL3_T void band_avx512_n(const Exl3Part& m, const float* const* xh, int tc0, int ks0, int ks1,
                          float* const* out /* N x 128 */) {
    constexpr int W = 4 * K2;
    static constexpr RowTab<K2> T = make_rowtab<K2>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    EXL3_BSUM_CONSTS
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
                const __m512i wa = gather_words<K2>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K2>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(EXL3_BSUM(_mm512_mullo_epi32(st, mult)));
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

// rows_avx512_n: slices [ks0, ks1) of the tile columns [t0, t1) - a split-K piece, or a run of its columns - for N
// activations (xh[i] -> acc[i], indexed by column; columns 16 t0 .. 16 t1 overwritten).  The tile rows are read in
// memory order, where a band's tiles are 1 KB runs a whole tile row apart: 40 threads of a 2-socket Xeon 6152 read a K4
// expert from DRAM in 0.14 ms in pieces, 0.21 in bands (~90 vs ~60 GB/s), the kernel the same.  A tile's 16 sums per
// activation live in acc between slices; each row's decode and FMAs are band_avx512's, so acc[i] is the N = 1 call's
// for xh[i] alone, bit for bit, whatever the columns' run.  The prefetch runs 16 tiles ahead in the job's own order
// (into the next slice past the run's end, not the next job's columns)
template <int K2, int N>
EXL3_T void rows_avx512_n(const Exl3Part& m, const float* const* xh, int ks0, int ks1, int t0, int t1,
                          float* const* acc) {
    constexpr int W = 4 * K2;
    static constexpr RowTab<K2> T = make_rowtab<K2>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const int ld = m.n / 16;
    const size_t row_words = (size_t) ld * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    EXL3_BSUM_CONSTS
    const __m512i low16 = _mm512_set1_epi32(0xffff);
    for (int i = 0; i < N; ++i)
        for (int c = 16 * t0; c < 16 * t1; c += 16) _mm512_storeu_ps(acc[i] + c, _mm512_setzero_ps());
    const int span = t1 - t0;
    const uint32_t* row = m.trellis + (size_t) ks0 * row_words;
    for (int ks = ks0; ks < ks1; ++ks, row += row_words) {
        float xk[N][16];
        for (int i = 0; i < N; ++i)
            for (int r = 0; r < 16; ++r) xk[i][r] = xh[i][ks * 16 + r];
        for (int t = t0; t < t1; ++t) {
            const uint32_t* w = row + (size_t) t * W;
            const uint32_t* pf = t + 16 < t1 || span == ld ? w + 16 * W : row + row_words + (size_t) (t + 16 - span) * W;
            for (int l = 0; l < W * 4; l += 64) _mm_prefetch((const char*) pf + l, _MM_HINT_T0);
            const __m512i p0 = _mm512_maskz_loadu_epi32(m0, w);
            const __m512i p1 = m1 ? _mm512_maskz_loadu_epi32(m1, w + 16) : _mm512_setzero_si512();
            const __m512i p2 = m2 ? _mm512_maskz_loadu_epi32(m2, w + 32) : _mm512_setzero_si512();
            const __m512i p3 = m3 ? _mm512_maskz_loadu_epi32(m3, w + 48) : _mm512_setzero_si512();
            __m512 a[N];
#pragma GCC unroll 8
            for (int i = 0; i < N; ++i) a[i] = _mm512_loadu_ps(acc[i] + t * 16);
#pragma GCC unroll 16
            for (int r = 0; r < 16; ++r) {
                const __m512i wa = gather_words<K2>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K2>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512 f = _mm512_cvtepi32_ps(EXL3_BSUM(_mm512_mullo_epi32(st, mult)));
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
template <int K2>
EXL3_T void decode_chunk(const Exl3Part& m, int tc0, int ks0, int ks1, float* wc /* (ks1 - ks0) 16 x 128 */) {
    constexpr int W = 4 * K2;
    static constexpr RowTab<K2> T = make_rowtab<K2>();
    constexpr auto ld_mask = [](int n) -> __mmask16 {
        return n >= 16 ? (__mmask16) 0xffffu : n <= 0 ? (__mmask16) 0 : (__mmask16) ((1u << n) - 1u);
    };
    constexpr __mmask16 m0 = ld_mask(W), m1 = ld_mask(W - 16), m2 = ld_mask(W - 32), m3 = ld_mask(W - 48);
    const size_t row_words = (size_t) (m.n / 16) * W;
    const __m512i mult = _mm512_set1_epi32((int) kMul1);
    EXL3_BSUM_CONSTS
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
                const __m512i wa = gather_words<K2>(T, T.i0, T.hi0, r, p0, p1, p2, p3);
                const __m512i wb = gather_words<K2>(T, T.i1, T.hi1, r, p0, p1, p2, p3);
                const __m512i st = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(wb, _mm512_loadu_si512((const void*) T.sr[r])),
                                    _mm512_sllv_epi32(wa, _mm512_loadu_si512((const void*) T.sl[r]))),
                    low16);
                const __m512i bs = EXL3_BSUM(_mm512_mullo_epi32(st, mult));
                _mm512_store_ps(wr + r * 128 + t * 16, _mm512_cvtepi32_ps(bs));
            }
        }
    }
}

// acc[t][0..128) += sum_{k < kr} x[t][k] wc[k][..] for TB activations
template <int TB>
EXL3_T void gemm_tile(const float* wc, int kr, const float* const* x, float* const* acc) {
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
template <int K2>
EXL3_T void band_gemm_avx512(const Exl3Part& m, const float* const* xh, int n, int tc0, int nb,
                             float* const* acc, float* wc) {
    constexpr int KS = kGemmK / 16;
    constexpr int TB = 6;   // 24 sums + 4 weights + the activation: 29 of the 32 registers
    for (int t = 0; t < n; ++t) std::memset(acc[t], 0, (size_t) nb * 128 * sizeof(float));
    std::vector<float*> ab((size_t) n);
    for (int ks0 = 0; ks0 < m.k / 16; ks0 += KS) {
        const int ks1 = std::min(m.k / 16, ks0 + KS), kr = (ks1 - ks0) * 16;
        for (int bb = 0; bb < nb; ++bb) {
            decode_chunk<K2>(m, tc0 + 8 * bb, ks0, ks1, wc);
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
