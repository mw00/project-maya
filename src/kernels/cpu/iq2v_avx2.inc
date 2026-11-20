// src/kernels/cpu/iq2v_avx2.inc - the AVX2 forms of iq2v_rows.cpp (see there), included by it once a tier with
//   IQ2A_NS    the tier's namespace (inside iq2v)
//   IQ2A_ATTR  the tier's instruction set (GCC / Clang: target(...); MSVC: nothing)
//   IQ2A_VNNI  1: the AVX-VNNI tier (vpdpwssd); 0: vpmaddwd + vpaddd
// The macros are undefined at the end.

namespace IQ2A_NS {

struct K2 {
    __m256i keep0, m1, keep1, m2, k0f, pt, tsh, bitsel, k7f, one, one16;
    IQ2A_ATTR K2() {
        keep0 = _mm256_set1_epi32(0x00003FFF);   // fields 0, 1 stay, 2, 3 up by 2 - then 0, 2 stay, 1, 3 up by 1
        m1 = _mm256_set1_epi32(0x0FFFC000);
        keep1 = _mm256_set1_epi32(0x007F007F);
        m2 = _mm256_set1_epi32(0x3F803F80);
        k0f = _mm256_set1_epi8(0x0F);
        pt = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0));
        tsh = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,   //
                               0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        bitsel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
        k7f = _mm256_set1_epi8(0x7F);
        one = _mm256_set1_epi32(1);
        one16 = _mm256_set1_epi16(1);
    }
};
struct P2 {
    __m256i T, cA, cB;   // the grouped sign bytes; the scales (form 2: lanes duplicated)
};
// a block's sign bytes and scales (qs: its 64 index / sign bytes; sc: form 1's 2 s + 1 pairs, order kPos)
template <int SB>
IQ2A_ATTR inline P2 prep2(const K2& K, const uint8_t* qs, uint32_t* sc) {
    const __m256i A = _mm256_loadu_si256((const __m256i*) qs), B = _mm256_loadu_si256((const __m256i*) (qs + 32));
    const __m256i W = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(A), _mm256_castsi256_ps(B), 0xDD));
    const __m256i v = _mm256_or_si256(_mm256_and_si256(W, K.keep0), _mm256_slli_epi32(_mm256_and_si256(W, K.m1), 2));
    const __m256i f = _mm256_or_si256(_mm256_and_si256(v, K.keep1), _mm256_slli_epi32(_mm256_and_si256(v, K.m2), 1));
    const __m256i s = _mm256_or_si256(f, _mm256_shuffle_epi8(K.pt, _mm256_and_si256(_mm256_xor_si256(f, _mm256_srli_epi16(f, 4)), K.k0f)));
    P2 p;
    p.T = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(s, K.tsh), 0xD8);
    if constexpr (SB == 1) {
        __m256i c = _mm256_or_si256(_mm256_srli_epi32(W, 27), K.one);
        c = _mm256_or_si256(c, _mm256_slli_epi32(c, 16));
        _mm256_store_si256((__m256i*) sc, c);
        p.cA = p.cB = c;
    } else {
        const __m256i h = _mm256_or_si256(_mm256_srli_epi16(W, 11), K.one16);   // (each dword's high word: 2 s + 1)
        p.cA = _mm256_permute2x128_si256(h, h, 0x00);
        p.cB = _mm256_permute2x128_si256(h, h, 0x11);
    }
    return p;
}
// chunk C of a block for NT tokens (IX 1: byte loads of the indices, else shifts; SB as prep2)
template <int C, int NT, int IX, int SB>
IQ2A_ATTR IQ2V_FI void chunk2(const K2& K, const uint8_t* qs, const P2& pr, const uint32_t* sc,
                                                    const block_q8_K* const* y, int i, __m256i* sumi) {
    const uint8_t* q = qs + 8 * C;
    __m256i g;
    if constexpr (IX == 1) {
        g = tab4(iq2xxs_grid, q[0], q[1], q[2], q[3]);
    } else {
        const uint32_t a0 = u32(q);
        g = tab4(iq2xxs_grid, (uint8_t) a0, (uint8_t) (a0 >> 8), (uint8_t) (a0 >> 16), a0 >> 24);
    }
    const __m256i S = _mm256_shuffle_epi8(pr.T, _mm256_load_si256((const __m256i*) kT.ctrl[C]));
    const __m256i x = _mm256_add_epi8(_mm256_and_si256(S, K.bitsel), K.k7f);   // (negative where the bit is set)
    constexpr int p = kPos[C];
    __m256i cv;
    if constexpr (SB == 1)
        cv = _mm256_castps_si256(_mm256_broadcast_ss((const float*) &sc[p]));
    else
        cv = _mm256_shuffle_epi8(p < 4 ? pr.cA : pr.cB, _mm256_load_si256((const __m256i*) kT.hctl[p & 3]));
    for (int t = 0; t < NT; ++t) {
        const __m256i d = _mm256_maddubs_epi16(g, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * C)), x));
        __m256i& s = sumi[2 * t + (C & 1)];
#if IQ2A_VNNI
        s = _mm256_dpwssd_avx_epi32(s, d, cv);
#else
        s = _mm256_add_epi32(s, _mm256_madd_epi16(d, cv));
#endif
    }
}
// out[t][r - r0] for rows [r0, r1) of nb blocks (rb bytes a row), NT tokens
template <int NT, int IX, int SB>
IQ2A_ATTR void rows2(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y,
                                           float* const* out, int r0, int r1) {
    const K2 K;
    alignas(32) uint32_t sc2[2][8];
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq2_xxs);
    P2 nx = prep2<SB>(K, blk + 2, sc2[0]);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const P2 cu = nx;
        const uint32_t* sc = sc2[b & 1];
        const uint8_t* qs = blk + 2;
        nx = prep2<SB>(K, b + 1 < nbt ? qs + BB : qs, sc2[(b + 1) & 1]);   // (the next block's, ahead)
        if constexpr (SB == 1) std::atomic_signal_fence(std::memory_order_seq_cst);   // (sc read from memory)
        __m256i sumi[2 * NT];
        for (int t = 0; t < 2 * NT; ++t) sumi[t] = _mm256_setzero_si256();
        chunk2<0, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<1, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<2, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<3, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<4, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<5, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<6, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        chunk2<7, NT, IX, SB>(K, qs, cu, sc, y, i, sumi);
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi[2 * t], sumi[2 * t + 1])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.125f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}

// ---- IQ2_S (d | qs[32]: the grid indices' low bytes, 4 a chunk | signs[32]: 4 bytes a chunk | qh[8]: a chunk's 4
// indices' bits 8-9 | scales[8]: a chunk's two nibbles, its first and second 16 values).  A block ahead: its 32 10-bit
// indices into ix (order kPos, as IQ2_XXS's sign words), and its scales Z (lane 0: the chunks' 2 lo + 1 as int16,
// lane 1: 2 hi + 1).  A chunk: 4 index loads + tab4, its explicit sign bits by one vpshufb + bit test + 0x7F, its
// scales one vpshufb of Z
struct K2s {
    __m256i qhx, qhm, mulk, smask, bitsel, k7f;
    __m128i k0f, one8;
    IQ2A_ATTR K2s() {
        qhx = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        qhm = _mm256_set1_epi32((int) 0xC0300C03u);   // (byte 4 c + k: bits 2 k, 2 k + 1 - index k's bits 8-9)
        mulk = _mm256_setr_epi16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
        smask = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        bitsel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
        k7f = _mm256_set1_epi8(0x7F);
        k0f = _mm_set1_epi8(0x0F);
        one8 = _mm_set1_epi8(1);
    }
};
IQ2A_ATTR inline __m256i prep2s(const K2s& K, const uint8_t* blk, uint16_t* ix) {
    const __m256i z = _mm256_setzero_si256();
    const __m256i QS = _mm256_loadu_si256((const __m256i*) (blk + 2));
    const __m256i QH = _mm256_shuffle_epi8(_mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) (blk + 2 + QK_K / 4))), K.qhx);
    const __m256i H = _mm256_and_si256(QH, K.qhm);
    const __m256i ia = _mm256_or_si256(_mm256_unpacklo_epi8(QS, z), _mm256_mullo_epi16(_mm256_unpacklo_epi8(H, z), K.mulk));
    const __m256i ib = _mm256_or_si256(_mm256_unpackhi_epi8(QS, z), _mm256_mullo_epi16(_mm256_unpackhi_epi8(H, z), K.mulk));
    _mm256_store_si256((__m256i*) ix, ia);          // (lane 0: chunks 0, 1, lane 1: 4, 5)
    _mm256_store_si256((__m256i*) (ix + 16), ib);   // (2, 3 | 6, 7)
    const __m128i s8 = _mm_loadl_epi64((const __m128i*) (blk + 2 + QK_K / 4 + QK_K / 32));
    const __m128i lo = _mm_and_si128(s8, K.k0f), hi = _mm_and_si128(_mm_srli_epi16(s8, 4), K.k0f);
    __m128i lh = _mm_unpacklo_epi64(lo, hi);
    lh = _mm_or_si128(_mm_add_epi8(lh, lh), K.one8);
    return _mm256_cvtepu8_epi16(lh);
}
template <int C, int NT>
IQ2A_ATTR IQ2V_FI void chunk2s(const K2s& K, const uint8_t* blk, const uint16_t* ix, __m256i Z, const block_q8_K* const* y,
                                int i, __m256i* sumi) {
    const uint16_t* xi = ix + 4 * kPos[C];
    const __m256i g = tab4(iq2s_grid, xi[0], xi[1], xi[2], xi[3]);
    const __m256i S = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_broadcast_ss((const float*) (blk + 2 + QK_K / 8 + 4 * C))), K.smask);
    const __m256i x = _mm256_add_epi8(_mm256_and_si256(S, K.bitsel), K.k7f);
    const __m256i cv = _mm256_shuffle_epi8(Z, _mm256_load_si256((const __m256i*) kT.wctl[C]));
    for (int t = 0; t < NT; ++t) {
        const __m256i d = _mm256_maddubs_epi16(g, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * C)), x));
        __m256i& s = sumi[2 * t + (C & 1)];
#if IQ2A_VNNI
        s = _mm256_dpwssd_avx_epi32(s, d, cv);
#else
        s = _mm256_add_epi32(s, _mm256_madd_epi16(d, cv));
#endif
    }
}
template <int NT>
IQ2A_ATTR void rows2s(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const K2s K;
    alignas(32) uint16_t ix2[2][32];
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq2_s);
    __m256i nZ = prep2s(K, blk, ix2[0]);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const __m256i Z = nZ;
        const uint16_t* ix = ix2[b & 1];
        nZ = prep2s(K, b + 1 < nbt ? blk + BB : blk, ix2[(b + 1) & 1]);
        std::atomic_signal_fence(std::memory_order_seq_cst);   // (ix read from memory)
        __m256i sumi[2 * NT];
        for (int t = 0; t < 2 * NT; ++t) sumi[t] = _mm256_setzero_si256();
        chunk2s<0, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<1, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<2, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<3, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<4, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<5, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<6, NT>(K, blk, ix, Z, y, i, sumi);
        chunk2s<7, NT>(K, blk, ix, Z, y, i, sumi);
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi[2 * t], sumi[2 * t + 1])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.125f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}

}  // namespace IQ2A_NS

#undef IQ2A_NS
#undef IQ2A_ATTR
#undef IQ2A_VNNI
