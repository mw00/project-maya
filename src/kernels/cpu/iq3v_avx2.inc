// src/kernels/cpu/iq3v_avx2.inc - the AVX2 forms of iq3v_rows.cpp (see there), included by it once a tier with
//   IQ3A_NS    the tier's namespace (inside iq3v)
//   IQ3A_ATTR  the tier's instruction set (GCC / Clang: target(...); MSVC: nothing)
//   IQ3A_VNNI  1: the AVX-VNNI tier (vpdpwssd); 0: vpmaddwd + vpaddd
// The macros are undefined at the end.

namespace IQ3A_NS {

IQ3A_ATTR IQ3V_FI __m256i dacc(__m256i s, __m256i d, __m256i c) {   // s + (d . c by int16 pairs), exact
#if IQ3A_VNNI
    return _mm256_dpwssd_avx_epi32(s, d, c);
#else
    return _mm256_add_epi32(s, _mm256_madd_epi16(d, c));
#endif
}
IQ3A_ATTR IQ3V_FI __m256i bd(const void* p) { return _mm256_castps_si256(_mm256_broadcast_ss((const float*) p)); }

// ---- IQ3_XXS (d | qs[64]: 8 grid indices a chunk | 8 sign / scale words, natural order)
struct K3 {
    __m256i keep0, m1, keep1, m2, k0f, pt, tsh, bitsel, k7f, one;
    IQ3A_ATTR K3() {
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
    }
};
// a block's sign bytes, grouped (T: one vpshufb a chunk lays its four over its 8-value groups), and its scales 2 s + 1
// into sc (both words of dword c: chunk c's)
IQ3A_ATTR IQ3V_FI __m256i prep3(const K3& K, const uint8_t* gas, uint32_t* sc) {
    const __m256i W = _mm256_loadu_si256((const __m256i*) gas);
    const __m256i v = _mm256_or_si256(_mm256_and_si256(W, K.keep0), _mm256_slli_epi32(_mm256_and_si256(W, K.m1), 2));
    const __m256i f = _mm256_or_si256(_mm256_and_si256(v, K.keep1), _mm256_slli_epi32(_mm256_and_si256(v, K.m2), 1));
    const __m256i s = _mm256_or_si256(f, _mm256_shuffle_epi8(K.pt, _mm256_and_si256(_mm256_xor_si256(f, _mm256_srli_epi16(f, 4)), K.k0f)));
    __m256i c = _mm256_or_si256(_mm256_srli_epi32(W, 27), K.one);
    c = _mm256_or_si256(c, _mm256_slli_epi32(c, 16));
    _mm256_store_si256((__m256i*) sc, c);
    return _mm256_permute4x64_epi64(_mm256_shuffle_epi8(s, K.tsh), 0xD8);
}
IQ3A_ATTR IQ3V_FI __m256i sgn3(const K3& K, __m256i T, int c) {   // chunk c's signs: negative bytes where its bit is set
    const __m256i S = _mm256_shuffle_epi8(T, _mm256_load_si256((const __m256i*) kT.ctrl3[c]));
    return _mm256_add_epi8(_mm256_and_si256(S, K.bitsel), K.k7f);
}
IQ3A_ATTR IQ3V_FI __m256i gat8(const uint8_t* q) {
    return _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q)), 4);
}
IQ3A_ATTR IQ3V_FI __m256i tab8s(const uint8_t* q) {   // a chunk's 8 grid entries: a qword of indices, broadcasts, blends
    const uint32_t* t = iq3xxs_grid;
    const uint32_t lo = u32(q), hi = u32(q + 4);
    const __m256i a = _mm256_blend_epi32(bd(t + (lo & 255u)), bd(t + ((lo >> 8) & 255u)), 0x02);
    const __m256i b = _mm256_blend_epi32(bd(t + ((lo >> 16) & 255u)), bd(t + (lo >> 24)), 0x08);
    const __m256i c = _mm256_blend_epi32(bd(t + (hi & 255u)), bd(t + ((hi >> 8) & 255u)), 0x20);
    const __m256i e = _mm256_blend_epi32(bd(t + ((hi >> 16) & 255u)), bd(t + (hi >> 24)), 0x80);
    return _mm256_blend_epi32(_mm256_blend_epi32(a, b, 0x0C), _mm256_blend_epi32(c, e, 0xC0), 0xF0);
}
// form 1 / 2: a row at a time, a pair of chunks a step, the grid by vpgatherdd, the block's signs / scales at its start
template <int NT>
IQ3A_ATTR void rows3g(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    constexpr size_t BB = sizeof(block_iq3_xxs);
    const K3 K;
    alignas(32) uint32_t sc[8];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            const __m256i T = prep3(K, blk + 2 + QK_K / 4, sc);
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const uint8_t* q3 = blk + 2 + 8 * ib;
                const __m256i g1 = gat8(q3), g2 = gat8(q3 + 8);
                const __m256i s1 = sgn3(K, T, ib), s2 = sgn3(K, T, ib + 1);
                const __m256i c1 = bd(&sc[ib]), c2 = bd(&sc[ib + 1]);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + 32 * ib;
                    const __m256i d1 = _mm256_maddubs_epi16(g1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), s1));
                    const __m256i d2 = _mm256_maddubs_epi16(g2, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), s2));
                    sumi[t] = dacc(dacc(sumi[t], d1, c1), d2, c2);
                }
            }
            const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = 0.25f * hsum_ggml(accf[t]);
    }
}
// form 3 / 4: a call's rows one stream of blocks, the signs / scales a block ahead, the grid by loads + blends
template <int C, int NT>
IQ3A_ATTR IQ3V_FI void chunk3b(const K3& K, const uint8_t* qs, __m256i T, const uint32_t* sc, const block_q8_K* const* y,
                                int i, __m256i* sumi) {
    const __m256i g = tab8s(qs + 8 * C);
    const __m256i x = sgn3(K, T, C);
    const __m256i cv = bd(&sc[C]);
    for (int t = 0; t < NT; ++t) {
        const __m256i d = _mm256_maddubs_epi16(g, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * C)), x));
        sumi[2 * t + (C & 1)] = dacc(sumi[2 * t + (C & 1)], d, cv);
    }
}
template <int NT>
IQ3A_ATTR void rows3b(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const K3 K;
    alignas(32) uint32_t sc2[2][8];
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq3_xxs);
    __m256i nT = prep3(K, blk + 2 + QK_K / 4, sc2[0]);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        const __m256i T = nT;
        const uint32_t* sc = sc2[b & 1];
        const uint8_t* qs = blk + 2;
        nT = prep3(K, (b + 1 < nbt ? blk + BB : blk) + 2 + QK_K / 4, sc2[(b + 1) & 1]);   // (the next block's, ahead)
        std::atomic_signal_fence(std::memory_order_seq_cst);                                // (sc read from memory)
        __m256i sumi[2 * NT];
        for (int t = 0; t < 2 * NT; ++t) sumi[t] = _mm256_setzero_si256();
        chunk3b<0, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<1, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<2, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<3, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<4, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<5, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<6, NT>(K, qs, T, sc, y, i, sumi);
        chunk3b<7, NT>(K, qs, T, sc, y, i, sumi);
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi[2 * t], sumi[2 * t + 1])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = 0.25f * hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}

// ---- IQ4_XS (d | scales_h | scales_l[4] | qs[128]: chunk c's 16 bytes, values 0-15 the low nibbles, 16-31 the high):
// a call's rows one stream of blocks; a chunk's 32 values by one 16-byte broadcast, vpsrlvd (lane 1: the high
// nibbles), and, one vpshufb of kvalues_iq4nl; |w| by vpabsb once for every token, the activation signed by w
// (ggml's mul_add_epi8); the block's 8 scales at its start (ggml's ls), stored and broadcast a chunk
template <int NT>
IQ3A_ATTR void rows4(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const __m256i k0f = _mm256_set1_epi8(0x0F), sh4 = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
    const uint8_t* blk = w + (size_t) r0 * rb;
    const long long nbt = (long long) (r1 - r0) * nb;
    if (nbt <= 0) return;
    constexpr size_t BB = sizeof(block_iq4_xs);
    alignas(32) int32_t scs[8];
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    int i = 0, r = 0;
    for (long long b = 0; b < nbt; ++b, blk += BB) {
        scales4(blk, scs);
        std::atomic_signal_fence(std::memory_order_seq_cst);   // (scs read from memory)
        __m256i sumi[2 * NT];
        for (int t = 0; t < 2 * NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int c = 0; c < QK_K / 32; ++c) {
            const __m256i x = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (blk + 8 + 16 * c)));
            const __m256i v = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srlv_epi32(x, sh4), k0f));
            const __m256i a = _mm256_abs_epi8(v);
            const __m256i cv = bd(&scs[c]);
            for (int t = 0; t < NT; ++t) {
                const __m256i d = _mm256_maddubs_epi16(a, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * c)), v));
                sumi[2 * t + (c & 1)] = dacc(sumi[2 * t + (c & 1)], d, cv);
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi[2 * t], sumi[2 * t + 1])), accf[t]);
        if (++i == nb) {
            for (int t = 0; t < NT; ++t) {
                out[t][r] = hsum_ggml(accf[t]);
                accf[t] = _mm256_setzero_ps();
            }
            i = 0;
            ++r;
        }
    }
}

// ---- Q4X (a row: its blocks' 128 code bytes - a pair of chunks 32 bytes, chunk 2 p's weight w the low nibble of byte
// w, chunk 2 p + 1's the high - then the blocks' 8 scales 2 s + 1, then their F16 d): |v| and the signed v by two
// vpshufb, the activation signed by v; ggml's IQ2_XXS integers (no q8 of -128 in a Q8_K row), its float steps
template <int NT>
IQ3A_ATTR void rowsq(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    const __m256i k0f = _mm256_set1_epi8(0x0F);
    const __m256i tbl = _mm256_setr_epi8(8, 25, 43, -8, -25, -43, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   //
                                         8, 25, 43, -8, -25, -43, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m256i tab = _mm256_setr_epi8(8, 25, 43, 8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   //
                                         8, 25, 43, 8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        const int8_t* sc = (const int8_t*) (row + (size_t) nb * 128);
        const uint8_t* dd = row + (size_t) nb * (128 + 8);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int p = 0; p < 4; ++p) {
                const __m256i x = _mm256_loadu_si256((const __m256i*) (row + (size_t) i * 128 + 32 * p));
                const __m256i n1 = _mm256_and_si256(x, k0f), n2 = _mm256_and_si256(_mm256_srli_epi16(x, 4), k0f);
                const __m256i v1 = _mm256_shuffle_epi8(tbl, n1), v2 = _mm256_shuffle_epi8(tbl, n2);
                const __m256i a1 = _mm256_shuffle_epi8(tab, n1), a2 = _mm256_shuffle_epi8(tab, n2);
                const __m256i c1 = _mm256_set1_epi16((short) sc[i * 8 + 2 * p]);
                const __m256i c2 = _mm256_set1_epi16((short) sc[i * 8 + 2 * p + 1]);
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + 64 * p;
                    const __m256i d1 = _mm256_maddubs_epi16(a1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) q8), v1));
                    const __m256i d2 = _mm256_maddubs_epi16(a2, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) (q8 + 32)), v2));
                    sumi[t] = dacc(dacc(sumi[t], d1, c1), d2, c2);
                }
            }
            uint16_t dh;
            std::memcpy(&dh, dd + 2 * i, 2);
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = 0.125f * hsum_ggml(accf[t]);
    }
}

}  // namespace IQ3A_NS

#undef IQ3A_NS
#undef IQ3A_ATTR
#undef IQ3A_VNNI
