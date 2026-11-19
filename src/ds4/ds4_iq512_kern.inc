// src/ds4/ds4_iq512_kern.inc - the CPU lane's exact AVX-512 rows that need no VBMI: IQ2_XS, IQ3_XXS, IQ1_M and IQ1_S.
// ds4_iq_cpu.cpp includes it once a tier, inside its unnamed namespace (after hsum_ggml, tab4, tab8, fold, scales4),
// with
//   IQX_NS    the tier's namespace
//   IQX_ATTR  the tier's instruction set as a function attribute (GCC / Clang: target(...); MSVC: nothing) on every
//             function here, so the file is built for the baseline and each tier runs only where iq_isa() says it can
//   IQX_POP   (the second tier) the sign bytes' parity by VPOPCNTDQ; without it, by a 16-entry table in a register
//             (ggml's own AVX2 trick) - the same bytes, so either tier gives the same bits
//   IQX_VNNI  (the third tier, AVX-512 VNNI) IQ1_S's products by vpdpbusd; without it, vpmaddubsw / vpmaddwd - the
//             same integers, so either tier gives the same bits
// and gets IQX_NS::rows<TY, NT>(...) for TY 17 (IQ2_XS) and 18 (IQ3_XXS), and in the tiers without IQX_POP
// rows_iq1m<NT>(...) (IQ1_M has no sign indices; its grid indices and delta signs come from BMI2's pdep / pext, as
// ggml's) and rows_iq1s<NT>(...) (IQ1_S, NT up to 8).  The macros are undefined at the end.
//
// A 64-value chunk a step, as the GLM lane's AVX-512 rows (src/kernels/cpu/iq_avx512.cpp): its grid bytes (table
// entries put together in two 256-bit halves by tab4 / tab8), one 64-bit sign mask, the scales as int16 lanes - decoded
// once, then every token: a masked subtract, vpmaddubsw, vpmaddwd and an add.  IQ2_XS and IQ3_XXS sign their 8-value
// groups by a 7-bit index whose eighth bit is the parity of the seven (ksigns_iq2xs); the eight indices of a chunk
// get their parities in one vector instead of eight table loads.  The integer lanes of a block are exact and fold to
// ggml's 8 (lane k + lane k + 8), then ggml's float steps in ggml's order: every result is ggml-cpu's AVX2 dot to the
// bit (own_check, before the rows are used).
namespace IQX_NS {

#if defined(IQX_POP)
// ksigns_iq2xs of eight 7-bit indices (a dword lane each) as the 64 bits of a mask: i | parity(i) << 7, the parity
// the count's lowest bit (vpmovdb keeps each lane's low byte)
IQX_ATTR inline uint64_t sign_bytes(__m256i s) {
    const __m256i b = _mm256_or_si256(s, _mm256_slli_epi32(_mm256_popcnt_epi32(s), 7));
    return (uint64_t) _mm_cvtsi128_si64(_mm256_cvtepi32_epi8(b));
}
// IQ2_XS: the indices are the top 7 bits of 8 words
IQX_ATTR inline uint64_t sign_bytes_w(__m128i q) {
    return sign_bytes(_mm256_srli_epi32(_mm256_cvtepu16_epi32(q), 9));
}
#else
// the same bytes without VPOPCNTDQ: the parity of the xor of the index's two nibbles, from a pshufb table
IQX_ATTR inline __m128i odd_nibbles() {
    return _mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
}
IQX_ATTR inline uint64_t sign_bytes(__m256i s) {
    const __m256i nib = _mm256_and_si256(_mm256_xor_si256(s, _mm256_srli_epi32(s, 4)), _mm256_set1_epi32(15));
    const __m256i par = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(odd_nibbles()), nib);
    return (uint64_t) _mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(s, par)));
}
// IQ2_XS in 16-bit lanes (ggml's own steps): index s = q >> 9, the nibbles' xor s ^ (q >> 13) (below 128, so pshufb
// reads its low nibble), the bytes packed
IQX_ATTR inline uint64_t sign_bytes_w(__m128i q) {
    const __m128i s = _mm_srli_epi16(q, 9);
    const __m128i par = _mm_shuffle_epi8(odd_nibbles(), _mm_xor_si128(s, _mm_srli_epi16(q, 13)));
    return (uint64_t) _mm_cvtsi128_si64(_mm_packus_epi16(_mm_or_si128(s, par), _mm_setzero_si128()));
}
#endif

// a step's scale lanes out of a block's 16 scale words (sc), int16 lanes 8 g .. 8 g + 7 <- word base + S g
template <int S>
IQX_ATTR inline __m512i step_scales(__m256i sc, int base) {
    const __m512i idx = _mm512_add_epi16(_mm512_set1_epi16((short) base),
                                         _mm512_set_epi16(3 * S, 3 * S, 3 * S, 3 * S, 3 * S, 3 * S, 3 * S, 3 * S, 2 * S, 2 * S,
                                                          2 * S, 2 * S, 2 * S, 2 * S, 2 * S, 2 * S, S, S, S, S, S, S, S, S, 0,
                                                          0, 0, 0, 0, 0, 0, 0));
    return _mm512_permutexvar_epi16(idx, _mm512_castsi256_si512(sc));
}

// rows [r0, r1) (out[t][r - r0]) of NT tokens, TY 17 = IQ2_XS, 18 = IQ3_XXS
template <int TY, int NT>
IQX_ATTR void rows(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0, int r1) {
    constexpr size_t BB = TY == 17 ? sizeof(block_iq2_xs) : sizeof(block_iq3_xxs);
    constexpr float K = TY == 17 ? 0.125f : 0.25f;
    const __m512i zero = _mm512_setzero_si512();
    const __m256i sh7 = _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * BB;
            // (the rows stream from DRAM: the bytes a few blocks ahead asked for, as the GLM lane's AVX-512 rows do)
            _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
            _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
            __m512i acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
            __m256i scw = _mm256_setzero_si256();   // (IQ2_XS: the block's 16 scales 2 s + 1, word 2 b + h: byte b's nibble h)
            if constexpr (TY == 17) {
                const __m128i b = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (blk + 66)));
                const __m128i lo = _mm_and_si128(b, _mm_set1_epi16(15)), hi = _mm_srli_epi16(b, 4);
                scw = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(lo, hi)), _mm_unpackhi_epi16(lo, hi), 1);
                scw = _mm256_add_epi16(_mm256_add_epi16(scw, scw), _mm256_set1_epi16(1));
            }
            for (int j = 0; j < 4; ++j) {
                __m512i g, sc;
                uint64_t sm;
                if constexpr (TY == 17) {   // 8 words: a 9-bit grid index, a 7-bit sign index; 2 scale bytes
                    const uint8_t* q = blk + 2 + 16 * j;
                    uint16_t v[8];
                    std::memcpy(v, q, 16);
                    g = _mm512_inserti64x4(
                        _mm512_castsi256_si512(tab4(iq2xs_grid, v[0] & 511u, v[1] & 511u, v[2] & 511u, v[3] & 511u)),
                        tab4(iq2xs_grid, v[4] & 511u, v[5] & 511u, v[6] & 511u, v[7] & 511u), 1);
                    sm = sign_bytes_w(_mm_loadu_si128((const __m128i*) q));
                    sc = step_scales<1>(scw, 4 * j);   // (words 4 j .. 4 j + 3)
                } else {   // 16 grid indices; the two chunks' words of four 7-bit sign indices and a scale
                    const uint8_t* q = blk + 2 + 16 * j;
                    unsigned i1[8], i2[8];
                    for (int k = 0; k < 8; ++k) {
                        i1[k] = q[k];
                        i2[k] = q[8 + k];
                    }
                    g = _mm512_inserti64x4(_mm512_castsi256_si512(tab8(iq3xxs_grid, i1)), tab8(iq3xxs_grid, i2), 1);
                    const uint32_t a = u32(blk + 2 + QK_K / 4 + 8 * j), c = u32(blk + 2 + QK_K / 4 + 8 * j + 4);
                    const __m256i ac =
                        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi32((int) a)), _mm_set1_epi32((int) c), 1);
                    sm = sign_bytes(_mm256_and_si256(_mm256_srlv_epi32(ac, sh7), _mm256_set1_epi32(127)));
                    const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
                    sc = scales4(sa, sa, sb, sb);
                }
                const __mmask64 m = _cvtu64_mask64(sm);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                    acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
                }
            }
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t)
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
        }
        for (int t = 0; t < NT; ++t) out[t][r - r0] = K * hsum_ggml(accf[t]);
    }
}

#if !defined(IQX_POP)
// IQ1_M rows [r0, r1) of NT tokens: ggml's AVX2 dot (two sums: the grid's and the deltas') in 512-bit lanes.  The
// grid indices of a 64-value step by two pdep each four (a qs byte and the low 3 bits of a qh nibble), the grid values
// (-1, 0, 1) as |g| and a negative mask; each 8-value group's delta sign (the nibble's bit 3) negates that group's 4
// scale lanes, and the delta sum takes the activations' pair sums - ggml's sign-then-sum (ggml's Q8_K rounds to
// -127 .. 127, where negating a pair's sum is summing its negated values); d is the F16 in the scale words' top nibbles
template <int NT>
IQX_ATTR void rows_iq1m(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0,
                        int r1) {
    const __m512i zero = _mm512_setzero_si512(), one = _mm512_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * rb;
        __m256 accf[NT], accd[NT];
        for (int t = 0; t < NT; ++t) accf[t] = accd[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t) i * sizeof(block_iq1_m);
            _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
            _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + QK_K / 8;
            uint16_t sw[4];
            std::memcpy(sw, blk + QK_K / 8 + QK_K / 16, 8);
            const uint16_t dh = (uint16_t) ((sw[0] >> 12) | ((sw[1] >> 8) & 0x00f0) | ((sw[2] >> 4) & 0x0f00) | (sw[3] & 0xf000));
            // the block's 16 3-bit scales as 2 s + 1, word 4 h + j: scale word j's bits 3 h .. 3 h + 2 (step j's lane
            // groups: words j, 4 + j, 8 + j, 12 + j)
            uint64_t sw64;
            std::memcpy(&sw64, sw, 8);
            __m256i scw = _mm256_srlv_epi64(_mm256_set1_epi64x((long long) sw64), _mm256_setr_epi64x(0, 3, 6, 9));
            scw = _mm256_and_si256(scw, _mm256_set1_epi16(7));
            scw = _mm256_add_epi16(_mm256_add_epi16(scw, scw), _mm256_set1_epi16(1));
            __m512i acc[NT], acd[NT];
            for (int t = 0; t < NT; ++t) acc[t] = acd[t] = _mm512_setzero_si512();
            for (int j = 0; j < 4; ++j) {
                const uint32_t h = u32(qh + 4 * j);
                const uint64_t p1 = _pdep_u64(u32(qs + 8 * j), 0x00ff00ff00ff00ffull) |
                                    _pdep_u64(h & 0x7777u, 0x0f000f000f000f00ull);
                const uint64_t p2 = _pdep_u64(u32(qs + 8 * j + 4), 0x00ff00ff00ff00ffull) |
                                    _pdep_u64((h >> 16) & 0x7777u, 0x0f000f000f000f00ull);
                const __m512i g = _mm512_inserti64x4(
                    _mm512_castsi256_si512(tab4(iq1s_grid, (unsigned) (p1 & 0xffff), (unsigned) ((p1 >> 16) & 0xffff),
                                                (unsigned) ((p1 >> 32) & 0xffff), (unsigned) (p1 >> 48))),
                    tab4(iq1s_grid, (unsigned) (p2 & 0xffff), (unsigned) ((p2 >> 16) & 0xffff),
                         (unsigned) ((p2 >> 32) & 0xffff), (unsigned) (p2 >> 48)),
                    1);
                const __m512i ag = _mm512_abs_epi8(g);
                const __mmask64 gn = _mm512_movepi8_mask(g);
                const __m512i sc = step_scales<4>(scw, j);
                const __mmask32 dn = (__mmask32) (_pdep_u32(_pext_u32(h, 0x88888888u), 0x11111111u) * 15u);
                const __m512i sd = _mm512_mask_sub_epi16(sc, dn, zero, sc);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, gn, zero, yv);
                    acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(_mm512_maddubs_epi16(ag, ys), sc));
                    acd[t] = _mm512_add_epi32(acd[t], _mm512_madd_epi16(_mm512_maddubs_epi16(one, yv), sd));
                }
            }
            const float dx = h2f(dh);
            for (int t = 0; t < NT; ++t) {
                const __m256 d = _mm256_set1_ps(y[t][i].d * dx);
                accf[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold(acc[t])), accf[t]);
                accd[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(fold(acd[t])), accd[t]);
            }
        }
        // (ggml's last line one FMA as gcc builds it, or a multiply then an add: g_iq1m_fma, own()'s check)
        for (int t = 0; t < NT; ++t) {
            const float hf = hsum_ggml(accf[t]), hd = hsum_ggml(accd[t]);
            out[t][r - r0] = g_iq1m_fma ? std::fma(IQ1M_DELTA, hd, hf) : hf + mul_exact(IQ1M_DELTA, hd);
        }
    }
}
#endif

#if !defined(IQX_POP)
// IQ1_S rows [r0, r1) of NT tokens (NT <= 8): ggml_vec_dot_iq1_s_q8_K's AVX2 branch (the grid's sum in 8 int32 lanes,
// the deltas' in one int) in 512-bit lanes, a 64-value step's grid decoded once for all the call's tokens.  The grid
// indices of a step by two pdep each four (ggml's own; on Rocket Lake faster than the indices in a vector stored and
// read back, and far faster than vpgatherqq under the GDS microcode), the grid values (-1, 0, 1) as two masks; the
// chunk's scale 2 s + 1 goes into the unsigned operand (|g| (2 s + 1) <= 15, so no pair sum saturates: the same
// integers as ggml's vpmaddwd by the scale), the grid's sign onto the activations by a masked subtract (ggml's vpsignb:
// -128 stays -128 in both).  With VNNI (IQX_VNNI) vpdpbusd takes the four products of a lane at once; without,
// vpmaddubsw into 16-bit lanes (a block's 4 steps at most 4 x 3840 a lane, exact) and one vpmaddwd by 1 a block.  The
// lanes fold to ggml's 8 (lane k + lane k + 8); the delta sum is ggml's from the activations' bsums (one vpmaddwd by
// the signed scales, summed).  The float steps are ggml's in ggml's order, g_iq1_fma's form (own()'s check): every
// result ggml-cpu's dot to the bit.
template <int NT>
IQX_ATTR void rows_iq1s(const uint8_t* w, size_t rb, int nb, const block_q8_K* const* y, float* const* out, int r0,
                        int r1) {
    const bool fa = (g_iq1_fma & 1) != 0, ff = (g_iq1_fma & 2) != 0;
    const __m512i zero = _mm512_setzero_si512();
#if !defined(IQX_VNNI)
    const __m512i ones16 = _mm512_set1_epi16(1);
#endif
    // step j's scale bytes out of the block's 8 (in every 128-bit lane): bytes 0-31 chunk 2 j's, 32-63 chunk 2 j + 1's
    __m512i ssel[4];
    for (int j = 0; j < 4; ++j)
        ssel[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8((char) (2 * j))), _mm256_set1_epi8((char) (2 * j + 1)), 1);
    for (int r = r0; r < r1; ++r) {
        const block_iq1_s* x = (const block_iq1_s*) (w + (size_t) r * rb);
        __m256 accf[NT];
        float acc1[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            acc1[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            _mm_prefetch((const char*) (x + i) + 2048, _MM_HINT_T0);
            const uint8_t* qs = x[i].qs;
            const uint16_t* qh = x[i].qh;
            // the 8 chunks' scales 2 s + 1 (words), as bytes in every 128-bit lane; signed by the delta bit (bit 15) and
            // each twice (words 2 c, 2 c + 1: chunk c's two bsums) for the delta sum
            const __m128i h = _mm_loadu_si128((const __m128i*) qh);
            __m128i ls = _mm_and_si128(_mm_srli_epi16(h, 12), _mm_set1_epi16(7));
            ls = _mm_add_epi16(_mm_add_epi16(ls, ls), _mm_set1_epi16(1));
            const __m512i lsb = _mm512_broadcast_i32x4(_mm_packus_epi16(ls, ls));
            const __m256i sl = _mm256_cvtepu16_epi32(_mm_mask_sub_epi16(ls, _mm_movepi16_mask(h), _mm_setzero_si128(), ls));
            const __m256i slw = _mm256_or_si256(sl, _mm256_slli_epi32(sl, 16));
            __m512i acc[NT];
            for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
            for (int j = 0; j < 4; ++j) {
                const uint64_t p1 = _pdep_u64(u32(qs + 8 * j), 0x00ff00ff00ff00ffull) |
                                    _pdep_u64(qh[2 * j], 0x0700070007000700ull);
                const uint64_t p2 = _pdep_u64(u32(qs + 8 * j + 4), 0x00ff00ff00ff00ffull) |
                                    _pdep_u64(qh[2 * j + 1], 0x0700070007000700ull);
                const __m512i g = _mm512_inserti64x4(
                    _mm512_castsi256_si512(tab4(iq1s_grid, (unsigned) (p1 & 0xffff), (unsigned) ((p1 >> 16) & 0xffff),
                                                (unsigned) ((p1 >> 32) & 0xffff), (unsigned) (p1 >> 48))),
                    tab4(iq1s_grid, (unsigned) (p2 & 0xffff), (unsigned) ((p2 >> 16) & 0xffff),
                         (unsigned) ((p2 >> 32) & 0xffff), (unsigned) (p2 >> 48)),
                    1);
                const __mmask64 gn = _mm512_movepi8_mask(g);
                const __m512i a = _mm512_maskz_mov_epi8(_mm512_test_epi8_mask(g, g), _mm512_shuffle_epi8(lsb, ssel[j]));
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, gn, zero, yv);
#if defined(IQX_VNNI)
                    acc[t] = _mm512_dpbusd_epi32(acc[t], a, ys);
#else
                    acc[t] = _mm512_add_epi16(acc[t], _mm512_maddubs_epi16(a, ys));
#endif
                }
            }
            const float dx = h2f(x[i].d);
            for (int t = 0; t < NT; ++t) {
#if defined(IQX_VNNI)
                const __m256i s8 = fold(acc[t]);
#else
                const __m256i s8 = fold(_mm512_madd_epi16(acc[t], ones16));
#endif
                const __m256i dv = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[t][i].bsums), slw);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(dv), _mm256_extracti128_si256(dv, 1));
                s4 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 1));
                const float sumi1 = (float) _mm_cvtsi128_si32(s4);
                const float d = y[t][i].d * dx;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s8), accf[t]);
                acc1[t] = fa ? std::fma(d, sumi1, acc1[t]) : acc1[t] + mul_exact(d, sumi1);
            }
        }
        for (int t = 0; t < NT; ++t)
            out[t][r - r0] = ff ? std::fma(IQ1S_DELTA, acc1[t], hsum_ggml(accf[t])) : hsum_ggml(accf[t]) + mul_exact(IQ1S_DELTA, acc1[t]);
    }
}
#endif

}  // namespace IQX_NS

#undef IQX_NS
#undef IQX_ATTR
#undef IQX_POP
#undef IQX_VNNI
