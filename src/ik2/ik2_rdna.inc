// src/ik2/ik2_rdna.inc - maya-ik2's kernels for AMD's matrix cores (RDNA 3 / 3.5 gfx11, RDNA 4 gfx12, wave32), the
// paths where the CUDA kernels' mma.sync has no AMD instruction to map to (ik2_kernels.cu includes this inside
// ik2::k; the interface is in ik2_kernels.cuh).  The products: maya::kernels::wmma_f16 (16 x 16 x 16, F16 in, F32
// sums; include/maya/kernels/rdna_wmma.hpp).  Every sum runs in a fixed order, so a GPU's answers repeat; they are
// not the CUDA kernels' bits (another order of the same sums).
//
// The prompt's routed experts: a layer's experts over a chunk in three launches, straight from their blocks (NVFP4 or
// Q8_0, made FP16 in shared memory - exact: a code times its scale is an FP16; or IQ2_XXS / IQ2_XS / IQ1_S / IQ1_M,
// made the FP16 values decode_f16 gives), instead of a decode to FP16 and two BLAS products an expert.  The rows (a token's pick each) lie grouped by expert, in expert order; a tile is up to 32
// rows of one expert.  gate / up: a block 64 rows of gate and the same 64 of up against a tile's rows, SwiGLU in the
// registers (a lane holds a row's gate and up); down: 128 rows of down.  Then each token's picks summed in expert
// order (the order the CUDA path adds them in).
// (AMD builds only: ik2_kernels.cu includes it under MAYA_USE_HIP, after rdna_wmma.hpp)

namespace {

constexpr int kRmBM = 32, kRmBN = 128, kRmBK = 64, kRmLd = kRmBK + 8;   // (row pad: the lanes' rows on other banks)

// (the tests' reference: rows that are FP16 already - decode_f16's - through the same products)
constexpr int kRmF16 = kRdnaF16Rows;
template <int TY>
constexpr int rm_nw = TY == kRmF16 ? 32 : 17;   // (the words a thread holds for its row's next 64 values)
constexpr bool rm_iq(int ty) { return ty == kIQ2_XXS || ty == kIQ2_XS || ty == kIQ1_S || ty == kIQ1_M; }

// three FP16 in a word: value i at bit 16 i
__device__ __forceinline__ uint64_t rm_lut3(float a, float b, float c) {
    return (uint64_t) f2h(a) | ((uint64_t) f2h(b) << 16) | ((uint64_t) f2h(c) << 32);
}
// IQ2_XXS / IQ2_XS: a grid entry's 8 values (8, 25 or 43 each: byte >> 4 = 0, 1, 2) times the sub-block's scale, as
// FP16 pairs with their signs - lut: the three products rounded as decode_f16 rounds them (a sign flips one bit)
__device__ __forceinline__ void rm_iq2_group(uint64_t g, uint32_t sg, uint64_t lut, uint32_t* o) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t b0 = (uint32_t) (g >> (16 * p)) & 0x30u, b1 = (uint32_t) (g >> (16 * p + 8)) & 0x30u;
        const uint32_t h0 = ((uint32_t) (lut >> b0) & 0xFFFFu) ^ (((sg >> (2 * p)) & 1u) << 15);
        const uint32_t h1 = ((uint32_t) (lut >> b1) & 0xFFFFu) ^ (((sg >> (2 * p + 1)) & 1u) << 15);
        o[p] = h0 | (h1 << 16);
    }
}
// IQ1_S / IQ1_M: a grid entry's 8 values (n: the nibbles g + 1 = 0, 1, 2 - values 0..3 the low nibbles of its bytes,
// 4..7 the high ones), lut: dl (g + delta) for the three g
__device__ __forceinline__ void rm_iq1_group(uint32_t n, uint64_t lut, uint32_t* o) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int s0 = 16 * (p & 1) + 4 * (p >> 1);
        const uint32_t n0 = (n >> s0) & 3u, n1 = (n >> (s0 + 8)) & 3u;
        o[p] = ((uint32_t) (lut >> (16 * n0)) & 0xFFFFu) | (((uint32_t) (lut >> (16 * n1)) & 0xFFFFu) << 16);
    }
}

// a block's W tile row: 64 values of one row (k-step ks) from its blocks into FP16 in shared memory
template <int TY, int N>
__device__ __forceinline__ void rm_load_w(const uint8_t* row, int ks, uint32_t (&w)[N]) {
    if constexpr (TY == kNVFP4) {   // a 64-value block: 4 scales, 32 bytes of codes
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 36);
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(p + i);
    } else if constexpr (TY == kQ8_0) {   // Q8_0: two 32-value blocks (34 bytes each)
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 68);
#pragma unroll
        for (int i = 0; i < 17; ++i) w[i] = __ldg(p + i);
    } else if constexpr (TY == kRmF16) {
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 128);
#pragma unroll
        for (int i = 0; i < 32; ++i) w[i] = __ldg(p + i);
    } else if constexpr (TY == kIQ2_XXS) {   // a quarter of a 66-byte block: d, 2 x (4 grid bytes, signs + scale)
        const uint8_t* b = row + (size_t) (ks >> 2) * 66;
        w[0] = ld2u(b);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[1 + i] = ld4u(b + 2 + 16 * (ks & 3) + 4 * i);
    } else if constexpr (TY == kIQ2_XS) {   // of a 74-byte block: d, 2 x 4 16-bit entries, their 2 scale bytes
        const uint8_t* b = row + (size_t) (ks >> 2) * 74;
        w[0] = ld2u(b) | (ld2u(b + 66 + 2 * (ks & 3)) << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[1 + i] = ld4u(b + 2 + 16 * (ks & 3) + 4 * i);
    } else if constexpr (TY == kIQ1_S) {   // of a 50-byte block: d, 2 x 4 grid bytes, their 2 x 16 qh bits
        const uint8_t* b = row + (size_t) (ks >> 2) * 50;
        w[0] = ld2u(b);
        w[1] = ld4u(b + 2 + 8 * (ks & 3)), w[2] = ld4u(b + 6 + 8 * (ks & 3));
        w[3] = ld4u(b + 34 + 4 * (ks & 3));
    } else {   // IQ1_M, of a 56-byte block: 2 x 4 grid bytes, their 2 x 16 qh bits, the block's 8 scale bytes
        const uint8_t* b = row + (size_t) (ks >> 2) * 56;
        w[0] = ld4u(b + 8 * (ks & 3)), w[1] = ld4u(b + 4 + 8 * (ks & 3));
        w[2] = ld4u(b + 32 + 4 * (ks & 3));
        w[3] = ld4u(b + 48), w[4] = ld4u(b + 52);
    }
}
template <int TY, int N>
__device__ __forceinline__ void rm_store_w(const uint32_t (&w)[N], int ks, uint16_t* dst) {
    uint32_t h[32];
    if constexpr (TY == kNVFP4) {
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {   // sub-block sb: values 0..7 the low nibbles of its 8 bytes, 8..15 the high
            const uint32_t sc = ue4m3_h2((w[0] >> (8 * sb)) & 0xFFu);
            const uint32_t w0 = w[1 + 2 * sb], w1 = w[2 + 2 * sb];
            uint32_t* o = h + 8 * sb;
            o[0] = nv_pair_h2(w0, sc), o[1] = nv_pair_h2(w0 >> 16, sc);
            o[2] = nv_pair_h2(w1, sc), o[3] = nv_pair_h2(w1 >> 16, sc);
            o[4] = nv_pair_h2(w0 >> 4, sc), o[5] = nv_pair_h2(w0 >> 20, sc);
            o[6] = nv_pair_h2(w1 >> 4, sc), o[7] = nv_pair_h2(w1 >> 20, sc);
        }
    } else if constexpr (TY == kRmF16) {
#pragma unroll
        for (int i = 0; i < 32; ++i) h[i] = w[i];
    } else if constexpr (TY == kIQ2_XXS) {   // (the values decode_f16 gives: db * grid value * sign, rounded to FP16)
        const float d = half_bits_to_float((uint16_t) w[0]);
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            const uint32_t qw = w[1 + 2 * sb], aux = w[2 + 2 * sb];
            const float db = d * (0.5f + (float) (aux >> 28)) * 0.25f;
            const uint64_t lut = rm_lut3(db * 8.0f, db * 25.0f, db * 43.0f);
#pragma unroll
            for (int l = 0; l < 4; ++l)
                rm_iq2_group(iq2xxs_grid_at((int) ((qw >> (8 * l)) & 0xFFu)), iq_signs8((aux >> (7 * l)) & 127u), lut,
                             h + 16 * sb + 4 * l);
        }
    } else if constexpr (TY == kIQ2_XS) {
        const float d = half_bits_to_float((uint16_t) w[0]);
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            const uint32_t sc = (w[0] >> (16 + 8 * sb)) & 0xFFu;
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {   // (a 4-bit scale a half of the sub-block)
                const float db = d * (0.5f + (float) ((sc >> (4 * hf)) & 15u)) * 0.25f;
                const uint64_t lut = rm_lut3(db * 8.0f, db * 25.0f, db * 43.0f);
                const uint32_t qq = w[1 + 2 * sb + hf];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const uint32_t q16 = (qq >> (16 * m)) & 0xFFFFu;
                    rm_iq2_group(iq2xs_grid_at((int) (q16 & 511u)), iq_signs8(q16 >> 9), lut,
                                 h + 16 * sb + 8 * hf + 4 * m);
                }
            }
        }
    } else if constexpr (TY == kIQ1_S) {   // (dl (g + delta): three values a sub-block)
        const float d = half_bits_to_float((uint16_t) w[0]);
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            const uint32_t q = w[1 + sb], hh = (w[3] >> (16 * sb)) & 0xFFFFu;
            const float dl = d * (float) (2 * (int) ((hh >> 12) & 7u) + 1);
            const float d8 = (hh & 0x8000u) ? -IQ1S_DELTA : IQ1S_DELTA;
            const uint64_t lut = rm_lut3(dl * (-1.0f + d8), dl * (0.0f + d8), dl * (1.0f + d8));
#pragma unroll
            for (int l = 0; l < 4; ++l)
                rm_iq1_group(__ldg(ik2_iq::dev::iq1s_grid_gpu + (((q >> (8 * l)) & 0xFFu) | (((hh >> (3 * l)) & 7u) << 8))),
                             lut, h + 16 * sb + 4 * l);
        }
    } else if constexpr (TY == kIQ1_M) {   // (a scale a half of the sub-block, the delta's sign a group)
        const float d = half_bits_to_float((uint16_t) (((w[3] >> 12) & 0xFu) | ((w[3] >> 24) & 0xF0u) |
                                                       ((w[4] >> 4) & 0xF00u) | ((w[4] >> 16) & 0xF000u)));
        const uint32_t sw2 = ((ks & 2) ? w[4] : w[3]) >> (16 * (ks & 1));   // (this pair of sub-blocks' 12 scale bits)
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            const uint32_t q = w[sb], hh = (w[2] >> (16 * sb)) & 0xFFFFu, sw = (sw2 & 0xFFFFu) >> (6 * sb);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float dl = d * (float) (2 * (int) ((sw >> (3 * (l >> 1))) & 7u) + 1);
                const uint32_t hn = (hh >> (4 * l)) & 0xFu;
                const float d8 = (hn & 8u) ? -IQ1S_DELTA : IQ1S_DELTA;
                const uint64_t lut = rm_lut3(dl * (-1.0f + d8), dl * (0.0f + d8), dl * (1.0f + d8));
                rm_iq1_group(__ldg(ik2_iq::dev::iq1s_grid_gpu + (((q >> (8 * l)) & 0xFFu) | ((hn & 7u) << 8))), lut,
                             h + 16 * sb + 4 * l);
            }
        }
    } else {
#pragma unroll
        for (int b = 0; b < 2; ++b) {   // block b at byte 34 b: its scale, then 32 int8 from byte 34 b + 2
            const int o0 = 34 * b;
            const uint32_t d = (w[o0 >> 2] >> ((o0 & 2) * 8)) & 0xFFFFu;
            const uint32_t sc = d | (d << 16);
#pragma unroll
            for (int p = 0; p < 16; ++p) {
                const int o = o0 + 2 + 2 * p;
                h[16 * b + p] = q8_h2((w[o >> 2] >> ((o & 2) * 8)) & 0xFFFFu, sc);
            }
        }
    }
    uint4* d4 = (uint4*) dst;
#pragma unroll
    for (int i = 0; i < 8; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}

#if defined(MAYA_RDNA_WMMA)
__device__ __forceinline__ maya::kernels::wmma_h16 rm_frag(const uint16_t* p) {
    union {
        uint4 u[2];
        maya::kernels::wmma_h16 v;
    } f;
    f.u[0] = *(const uint4*) p, f.u[1] = *(const uint4*) (p + 8);
    return f.v;
}
#endif

// GU: h [S][F] = SwiGLU(x . gate^T * sg, x . up^T * su) of a tile's rows; else y [S][C] = h . down^T.  grid (N / 64
// (GU) or C / 128, tiles), 128 threads: warp w the tile's rows 16 (w & 1) .., columns 64 (w >> 1) ..
template <int TY, bool GU>
__global__ void __launch_bounds__(128) experts_rdna_kernel(const uint8_t* const* __restrict__ eptr,
                                                           const int4* __restrict__ tiles, size_t off, size_t rb,
                                                           const uint16_t* __restrict__ x16, int K, int F, int C,
                                                           const float* __restrict__ sg, const float* __restrict__ su,
                                                           float gcap, float ucap, uint16_t* __restrict__ h16,
                                                           float* __restrict__ y32) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Xs[kRmBM * kRmLd];
    __shared__ __align__(16) uint16_t Ws[kRmBN * kRmLd];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mh = warp & 1, nh = warp >> 1;
    const int4 tl = tiles[blockIdx.y];   // (expert, first row, rows)
    const int row0 = tl.y, nrows = tl.z;
    const uint8_t* blob = eptr[tl.x];
    // this thread's W row: GU 64 gate rows then the same 64 of up, else 128 rows of down
    const uint8_t* wrow;
    if constexpr (GU) {
        const int r0 = blockIdx.x * 64;
        wrow = blob + (tid < 64 ? 0 : off) + (size_t) (r0 + (tid & 63)) * rb;
    } else {
        wrow = blob + off + (size_t) (blockIdx.x * 128 + tid) * rb;
    }
    // this thread's part of the X tile: row tid / 4, 16 values from column 16 (tid % 4)
    const int xr = tid >> 2, xc = (tid & 3) * 16;
    const bool xok = xr < nrows;
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (row0 + (xok ? xr : 0)) * K + xc);
    const int nk = K / kRmBK;
    uint32_t w[rm_nw<TY>];
    uint4 xa, xb;
    rm_load_w<TY>(wrow, 0, w);
    xa = xok ? xsrc[0] : make_uint4(0u, 0u, 0u, 0u);
    xb = xok ? xsrc[1] : make_uint4(0u, 0u, 0u, 0u);
    wmma_f8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();   // (the previous step's reads of the tiles are done)
        *(uint4*) (Xs + xr * kRmLd + xc) = xa;
        *(uint4*) (Xs + xr * kRmLd + xc + 8) = xb;
        rm_store_w<TY>(w, ks, Ws + tid * kRmLd);
        __syncthreads();
        if (ks + 1 < nk) {   // (the next step's bytes in flight under this one's products)
            rm_load_w<TY>(wrow, ks + 1, w);
            xa = xok ? xsrc[(ks + 1) * (kRmBK / 8)] : make_uint4(0u, 0u, 0u, 0u);
            xb = xok ? xsrc[(ks + 1) * (kRmBK / 8) + 1] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int kk = 0; kk < kRmBK / 16; ++kk) {
            const wmma_h16 a = rm_frag(Xs + (16 * mh + (lane & 15)) * kRmLd + kk * 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wr = GU ? (j < 2 ? 32 * nh + 16 * j : 64 + 32 * nh + 16 * (j - 2)) : 64 * nh + 16 * j;
                acc[j] = wmma_f16(a, rm_frag(Ws + (wr + (lane & 15)) * kRmLd + kk * 16), acc[j]);
            }
        }
    }
    const int hi = lane >> 4;
    if constexpr (GU) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = blockIdx.x * 64 + 32 * nh + 16 * j + (lane & 15);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * mh + wmma_row(i, hi);
                if (m >= nrows) continue;
                const int r = row0 + m;
                const float g = acc[j][i] * sg[r], u = acc[j + 2][i] * su[r];
                h16[(size_t) r * F + col] =
                    f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = blockIdx.x * 128 + 64 * nh + 16 * j + (lane & 15);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * mh + wmma_row(i, hi);
                if (m < nrows) y32[(size_t) (row0 + m) * C + col] = acc[j][i];
            }
        }
    }
#else
    __trap();   // (never launched: experts_rdna_available() is false without the matrix cores)
#endif
}

// The prompt's fused attention (attn_flash_f16_kernel's job and outputs: a split's unnormalised rows and (max, sum),
// merged with the sink by attn_flash_combine_kernel) on the matrix cores.  A block: 64 queries (4 warps x 16) of one
// head against a split of its keys in tiles of 32; S = Q K^T (Q in the registers, K as it lies), the online softmax
// in the log2 domain on S's layout (a lane: 8 query rows x one key a 16-key column; the relative bias read where it
// lies), P through the warp's own shared memory into the A layout, O += P V from V stored transposed.  24 KB of
// shared memory a block, so that several blocks share a CU (64 keys a tile with the bias in shared memory: 54 KB,
// one block a CU, 1.3x slower).
constexpr int kRfNW = 4, kRfBN = 32, kRfLd = kFaD + 8, kRfVld = kRfBN + 8;
template <class KVT>
__global__ void __launch_bounds__(kRfNW * 32)
attn_flash_rdna_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                       const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                       const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                       float2* __restrict__ ml_part) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Ks[kRfBN * kRfLd];          // [key][dim]
    __shared__ __align__(16) uint16_t Vt[kFaD * kRfVld];          // [dim][key]
    __shared__ __align__(16) uint16_t Pss[kRfNW * 16 * kRfVld];   // each warp's P [16][key]
    constexpr bool F8 = sizeof(KVT) == 1;
    const int tid = threadIdx.x, warp = tid >> 5, hi = (tid & 31) >> 4, ln = tid & 15;
    uint16_t* Ps = Pss + warp * 16 * kRfVld;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    constexpr int BM = 16 * kRfNW;
    // (the tiles are the CUDA kernel's 64-key ones cut in two: the same keys, so attn_flash_splits' counts hold)
    const int i0 = blockIdx.x * BM, i_last = min(Tq, i0 + BM) - 1;
    const int k_hi = qpos0 + i_last + 1, k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kRfBN - 1) / kRfBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);
    const int wi = i0 + warp * 16;
    wmma_h16 qf[kFaD / 16];   // (the lane's query row wi + ln, its 128 dims)
    {
        const int iq = wi + ln;
        const uint16_t* qr = q16 + (size_t) (iq < Tq ? iq : 0) * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) qf[kk] = iq < Tq ? rm_frag(qr + kk * 16) : wmma_h16{};
    }
    wmma_f8 o[kFaD / 16];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f8{};
    float mx[8], ls[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) mx[i] = -INFINITY, ls[i] = 0.0f;
    const float sc = qk_scale * kLog2e;
    // A tile's values: K by (key c / 16, 8 dims from 8 (c % 16)), V by a pair of keys (tid % 16) x 16 dims (tid / 16),
    // so that the transposed V goes in as words, a half-warp a row's 16 words.
    using Raw = typename std::conditional<F8, uint2, uint4>::type;
    Raw zr;
    std::memset(&zr, 0, sizeof zr);
    const int vp = tid & 15, vg = tid >> 4;
    auto h8 = [&](const Raw& r) -> uint4 {
        if constexpr (F8) return e4m3x8_h8(r);
        else return r;
    };
    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * kRfBN;
        __syncthreads();   // (the previous tile's reads done)
#pragma unroll
        for (int it = 0; it < (kRfBN * kFaD / 8) / (kRfNW * 32); ++it) {
            const int c = it * kRfNW * 32 + tid, row = c >> 4, col = (c & 15) * 8, kp = kt + row;
            const bool ok = kp < k_hi;
            const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            *(uint4*) (Ks + row * kRfLd + col) = h8(ok ? *(const Raw*) (kc + (size_t) pr * KD + (size_t) g * kFaD + col) : zr);
        }
        {
            Raw vr[2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kp = kt + 2 * vp + j;
                const bool ok = kp < k_hi;
                const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
                const KVT* src = vc + (size_t) pr * KD + (size_t) g * kFaD + 16 * vg;
                vr[j][0] = ok ? *(const Raw*) src : zr;
                vr[j][1] = ok ? *(const Raw*) (src + 8) : zr;
            }
            uint32_t* Vt32 = (uint32_t*) Vt;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 a = h8(vr[0][i]), b = h8(vr[1][i]);
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int d = 16 * vg + 8 * i + 2 * m;
                    Vt32[d * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x5410);
                    Vt32[(d + 1) * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x7632);
                }
            }
        }
        __syncthreads();
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kRfBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kRfBN - 1) < extent;
        if (!any) continue;
        wmma_f8 s[kRfBN / 16];
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j) s[j] = wmma_f8{};
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk)
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j)
                s[j] = wmma_f16(qf[kk], rm_frag(Ks + (16 * j + ln) * kRfLd + kk * 16), s[j]);
        // logits (log2 units): scale, bias (the table's value at the distance, 0 out of its extent), mask; the rows'
        // new maxima, the output's rescale
        float u[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = wmma_row(i, hi), ia = wi + m, qa = qpos0 + ia;
            const float* brow = bias + ((size_t) (ia < Tq ? ia : 0) * H + h) * extent;
            float mxi = -INFINITY;
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j) {
                const int kp = kt + 16 * j + ln, dist = qa - kp;
                float v = s[j][i] * sc;
                if (near && ia < Tq && dist >= 0 && dist < extent) v += brow[dist] * kLog2e;
                v = (dist < 0 || kp < qa - window + 1 || ia >= Tq) ? -INFINITY : v;
                s[j][i] = v;
                mxi = fmaxf(mxi, v);
            }
#pragma unroll
            for (int sh = 1; sh < 16; sh <<= 1) mxi = fmaxf(mxi, __shfl_xor_sync(0xffffffffu, mxi, sh));
            const float mn = fmaxf(mx[i], mxi);
            const float cf = mn == -INFINITY ? 1.0f : exp2f(mx[i] - mn);
            u[i] = mn == -INFINITY ? 0.0f : mn;
            mx[i] = mn;
            ls[i] *= cf;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][i] *= cf;
        }
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = exp2f(s[j][i] - u[i]);
                ls[i] += p;
                Ps[wmma_row(i, hi) * kRfVld + 16 * j + ln] = f2h(p);
            }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < kRfBN / 16; ++kk) {
            const wmma_h16 a = rm_frag(Ps + ln * kRfVld + kk * 16);
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f16(a, rm_frag(Vt + (16 * j + ln) * kRfVld + kk * 16), o[j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int sh = 1; sh < 16; sh <<= 1) ls[i] += __shfl_xor_sync(0xffffffffu, ls[i], sh);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ia = wi + wmma_row(i, hi);
        if (ia >= Tq) continue;
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + ln;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) op[16 * j] = o[j][i];
        if (ln == 0) ml_part[((size_t) split * Tq + ia) * H + h] = make_float2(mx[i], ls[i]);
    }
#else
    __trap();   // (never launched: attn_flash_available() is false without the matrix cores)
#endif
}

// an RDNA 3 / 3.5 / 4 GPU (the current one), whose matrix-core kernels this build has
bool rdna_wmma_gpu() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    const std::string a = p.gcnArchName;
    return a.rfind("gfx11", 0) == 0 || a.rfind("gfx12", 0) == 0;
}

template <class KVT>
void attn_flash_rdna(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV, int Tq,
                     int qpos0, float qk_scale, const float* bias, int extent, int window, int n_split, float* o_part,
                     float2* ml_part, cudaStream_t s) {
    const dim3 grid((unsigned) ((Tq + 16 * kRfNW - 1) / (16 * kRfNW)), (unsigned) H, (unsigned) n_split);
    attn_flash_rdna_kernel<KVT><<<grid, kRfNW * 32, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale,
                                                            bias, extent, window, n_split, o_part, ml_part);
    check("attn_flash_rdna");
}

// y [T][ldy] = x [T][K] . W [N][K]^T in BF16 for a few tokens (TT >= T): a warp R rows, a lane 8 values a step (16
// bytes of each row and of each token's x), every token's sum in its own register, then the warp's sum.
// (H16: the values are FP16, not BF16 - a head kept as other blocks, in the drafter's FP16 copy)
template <int WARPS, int R, int TT, bool H16 = false>
__global__ void __launch_bounds__(WARPS * 32)
gemm_bf16_cols_kernel(const uint16_t* __restrict__ W, int N, int K, const uint16_t* __restrict__ x, int T,
                      float* __restrict__ y, int ldy) {
    const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const uint4* wr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) wr[i] = (const uint4*) (W + (size_t) min(row0 + i, N - 1) * K);
    float acc[R][TT];
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[i][t] = 0.0f;
    auto lo = [](uint32_t v) {
        return H16 ? __half2float(__ushort_as_half((unsigned short) (v & 0xFFFFu))) : __uint_as_float(v << 16);
    };
    auto hi = [](uint32_t v) {
        return H16 ? __half2float(__ushort_as_half((unsigned short) (v >> 16))) : __uint_as_float(v & 0xFFFF0000u);
    };
    for (int c = lane; c < (K >> 3); c += 32) {
        float wf[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint4 w = __ldg(wr[i] + c);
            wf[i][0] = lo(w.x), wf[i][1] = hi(w.x), wf[i][2] = lo(w.y), wf[i][3] = hi(w.y);
            wf[i][4] = lo(w.z), wf[i][5] = hi(w.z), wf[i][6] = lo(w.w), wf[i][7] = hi(w.w);
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= T) break;
            const uint4 v = __ldg((const uint4*) (x + (size_t) t * K) + c);
            const float xf[8] = {lo(v.x), hi(v.x), lo(v.y), hi(v.y), lo(v.z), hi(v.z), lo(v.w), hi(v.w)};
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][t] = fmaf(wf[i][j], xf[j], acc[i][t]);
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= T) break;
            const float a = wsum(acc[i][t]);
            if (lane == 0 && row0 + i < N) y[(size_t) t * ldy + row0 + i] = a;
        }
}
}  // namespace

bool gemm_bf16_cols(const uint16_t* W, int N, int K, const uint16_t* x, int T, float* y, int ldy, cudaStream_t s) {
    if (N <= 0 || T <= 0 || T > 8 || K % 8 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    constexpr int WARPS = 4, R = 4;
    const unsigned g = (unsigned) ((N + WARPS * R - 1) / (WARPS * R));
    if (T <= 4) gemm_bf16_cols_kernel<WARPS, R, 4><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    else gemm_bf16_cols_kernel<WARPS, R, 8><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    check("gemm_bf16_cols");
    return true;
}
bool gemm_f16_cols(const uint16_t* W, int N, int K, const uint16_t* x, int T, float* y, int ldy, cudaStream_t s) {
    if (N <= 0 || T <= 0 || T > 8 || K % 8 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    constexpr int WARPS = 4, R = 4;
    const unsigned g = (unsigned) ((N + WARPS * R - 1) / (WARPS * R));
    if (T <= 4) gemm_bf16_cols_kernel<WARPS, R, 4, true><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    else gemm_bf16_cols_kernel<WARPS, R, 8, true><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    check("gemm_f16_cols");
    return true;
}


bool experts_rdna_available(int ty_gate, int ty_up, int ty_down, int C, int F) {
    static const int on = [] {   // (the current GPU: an RDNA 3 / 3.5 / 4 one of this build; IK2_RDNA_MOE=0: off)
        if (std::getenv("IK2_RDNA_MOE") && !std::atoi(std::getenv("IK2_RDNA_MOE"))) return 0;
        return rdna_wmma_gpu() ? 1 : 0;
    }();
    // (IK2_RDNA_IQ=0: the IQ block types back on the decode-to-FP16 path)
    static const bool iq_on = !(std::getenv("IK2_RDNA_IQ") && !std::atoi(std::getenv("IK2_RDNA_IQ")));
    auto dense = [](int t) { return t == kNVFP4 || t == kQ8_0; };
    bool fmt = dense(ty_gate) && ty_up == ty_gate && ty_down == ty_gate;
    // gate and up of one type, down of its own, when an IQ type is among them (a 256-value block: 4 steps of 64)
    if (!fmt && iq_on && ty_up == ty_gate && (rm_iq(ty_gate) || rm_iq(ty_down)) &&
        (dense(ty_gate) || rm_iq(ty_gate)) && (dense(ty_down) || rm_iq(ty_down)))
        fmt = C % 256 == 0 && F % 256 == 0;
    return on && fmt && C % kRmBN == 0 && F % kRmBN == 0 && C % kRmBK == 0 && F % kRmBK == 0;
}
namespace {
size_t rm_row_bytes(int ty, int K) { return ty == kRmF16 ? (size_t) K * 2 : mv_row_bytes(ty, K); }
template <bool GU, class... A>
void rm_launch(int ty, dim3 g, cudaStream_t s, A... a) {
    switch (ty) {
    case kNVFP4: experts_rdna_kernel<kNVFP4, GU><<<g, 128, 0, s>>>(a...); break;
    case kQ8_0: experts_rdna_kernel<kQ8_0, GU><<<g, 128, 0, s>>>(a...); break;
    case kIQ2_XXS: experts_rdna_kernel<kIQ2_XXS, GU><<<g, 128, 0, s>>>(a...); break;
    case kIQ2_XS: experts_rdna_kernel<kIQ2_XS, GU><<<g, 128, 0, s>>>(a...); break;
    case kIQ1_S: experts_rdna_kernel<kIQ1_S, GU><<<g, 128, 0, s>>>(a...); break;
    case kIQ1_M: experts_rdna_kernel<kIQ1_M, GU><<<g, 128, 0, s>>>(a...); break;
    case kRmF16: experts_rdna_kernel<kRmF16, GU><<<g, 128, 0, s>>>(a...); break;
    default: std::fprintf(stderr, "ik2: experts_rdna: block type %d\n", ty), std::abort();
    }
}
}  // namespace
void experts_rdna_gu(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_up,
                     const uint16_t* x16, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                     uint16_t* h16, cudaStream_t s) {
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (F / 64), (unsigned) n_tiles);
    rm_launch<true>(ty, g, s, eptr, (const int4*) tiles, off_up, rm_row_bytes(ty, C), x16, C, F, C, sg, su, gcap, ucap,
                    h16, (float*) nullptr);
    check("experts_rdna_gu");
}
void experts_rdna_down(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_down,
                       const uint16_t* h16, int C, int F, float* y32, cudaStream_t s) {
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (C / 128), (unsigned) n_tiles);
    rm_launch<false>(ty, g, s, eptr, (const int4*) tiles, off_down, rm_row_bytes(ty, F), h16, F, F, C,
                     (const float*) nullptr, (const float*) nullptr, 0.0f, 0.0f, (uint16_t*) nullptr, y32);
    check("experts_rdna_down");
}
// ---- One token's attention on RDNA 3 / 3.5 / 4 without the emulated mma16816 (IK2_ATTN_RDNA): attn_decode_mma_kernel's
// job for an FP8 cache - its grid, its warps' steps of 16 keys and its partials, so attn_decode_mma_combine_kernel
// merges them (v_inv times 256: K and V come through e4m3x4_h4_s8 as FP16 times 2^-8).  On AMD the CUDA kernel's
// m16n8k16 is 24 shuffles and 32 FMAs a lane a call, 16 calls a step; here a step is plain v_dot2_f32_f16 and DPP:
//   a warp is two rows of 16 lanes; a row takes 8 keys of the step (row 0 the first 8), a lane 8 of the 128 dims - 8
//   bytes of each key's K and V, the 16 lanes a 128-byte line;
//   Q K^T  v_dot2 over the lane's 8 dims, R heads a key; then the row's 16 sums by a transposing reduction (xor 8,
//          4 and 2 halve the keys a lane keeps, xor 1 adds the pair): a lane ends with one key's logits (its pair
//          lane the same), 8 R moves for 8 keys;
//   softmax  online in log2 units as the CUDA kernel's, the max of a head over the warp's 16 keys;
//   P V    P in FP16 by key pairs (the CUDA kernel's P), from the pair's lane by row_share; v_dot2 over a pair a dim.
// Every sum in a fixed order, so a GPU's answers repeat; they are not attn_decode_mma_kernel's bits (another order).
// A window's tokens (grid z) each take the blocks of their one-token launch: the window is the token path's to the bit.
template <int M>
__device__ __forceinline__ float rd_xor(float v) {
#if defined(MAYA_RDNA_WMMA) && !defined(MAYA_HIP_WAVE64)
    const int x = __builtin_bit_cast(int, v);
    int r;
    if constexpr (M == 1) r = __builtin_amdgcn_update_dpp(0, x, 0xB1, 0xF, 0xF, false);         // quad_perm [1,0,3,2]
    else if constexpr (M == 2) r = __builtin_amdgcn_update_dpp(0, x, 0x4E, 0xF, 0xF, false);    // quad_perm [2,3,0,1]
    else if constexpr (M == 4) r = __builtin_amdgcn_update_dpp(0, x, 0x164, 0xF, 0xF, false);   // row_xmask:4
    else if constexpr (M == 8) r = __builtin_amdgcn_update_dpp(0, x, 0x168, 0xF, 0xF, false);   // row_xmask:8
    else {
        static_assert(M == 16, "rd_xor: 1, 2, 4, 8 or 16");
        r = (int) __builtin_amdgcn_permlanex16((unsigned) x, (unsigned) x, 0x76543210u, 0xFEDCBA98u, false, false);
    }
    return __builtin_bit_cast(float, r);
#else
    return __shfl_xor_sync(0xffffffffu, v, M);
#endif
}
// lane N of this lane's row of 16 (row_share)
template <int N>
__device__ __forceinline__ uint32_t rd_row_lane(uint32_t x) {
#if defined(MAYA_RDNA_WMMA) && !defined(MAYA_HIP_WAVE64)
    return (uint32_t) __builtin_amdgcn_update_dpp(0, (int) x, 0x150 + N, 0xF, 0xF, false);
#else
    return __shfl_sync(0xffffffffu, x, (int) (threadIdx.x & 16u) + N);
#endif
}
__device__ __forceinline__ float rd_dot2(uint32_t a, uint32_t b, float c) {
#if defined(MAYA_RDNA_WMMA)
    typedef _Float16 rd_h2 __attribute__((ext_vector_type(2)));
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(rd_h2, a), __builtin_bit_cast(rd_h2, b), c, false);
#else
    const float2 x = u32_to_f2(a), y = u32_to_f2(b);
    return fmaf(x.y, y.y, fmaf(x.x, y.x, c));
#endif
}

template <int R>
__global__ void __launch_bounds__(kWaNW * 32)
attn_decode_rdna_kernel(const uint16_t* __restrict__ q16_0, const uint8_t* __restrict__ kc,
                        const uint8_t* __restrict__ vc, int kv_cap, int H, int Hkv, const int* pos_d, int qpos,
                        float qk_scale, const float* __restrict__ r_0, const float* __restrict__ proj, int d_rel,
                        bool rel4, int extent, int window, int n_split, float* __restrict__ o_part_0,
                        float2* __restrict__ ml_part_0, int blk0, int tok_q, int tok_r, size_t tok_part) {
#if defined(MAYA_RDNA_WMMA) && !defined(MAYA_HIP_WAVE64)
    const int tz = blockIdx.z;
    const uint16_t* __restrict__ q16 = q16_0 + (size_t) tz * tok_q;
    const float* __restrict__ r = r_0 + (size_t) tz * tok_r;
    float* __restrict__ o_part = o_part_0 + (size_t) tz * tok_part;
    float2* __restrict__ ml_part = (float2*) ((float*) ml_part_0 + (size_t) tz * tok_part);
    if (pos_d) qpos = pos_d[tz];
    const int lane = threadIdx.x & 31, row = lane >> 4, li = lane & 15;
    const int g = blockIdx.x, KD = Hkv * kFaD;
    const int w = (blockIdx.y + blk0) * kWaNW + (threadIdx.x >> 5);   // (attn_decode_mma_kernel's warps and steps)
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    uint32_t qf[R][4];   // the lane's 8 dims of each head's q, FP16 pairs
#pragma unroll
    for (int h = 0; h < R; ++h) {
        const uint4 v = __ldg((const uint4*) (q16 + (size_t) (g * R + h) * kFaD + 8 * li));
        qf[h][0] = v.x, qf[h][1] = v.y, qf[h][2] = v.z, qf[h][3] = v.w;
    }
    float o[R][8], m[R], l[R];
#pragma unroll
    for (int h = 0; h < R; ++h) {
        m[h] = -INFINITY, l[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) o[h][d] = 0.0f;
    }
    const float sc = qk_scale * kLog2e * 256.0f;   // (K as FP16 times 2^-8)
    const int kj = ((li >> 3) & 1) * 4 + ((li >> 2) & 1) * 2 + ((li >> 1) & 1);   // the key whose logits the lane ends with
    const bool u8 = li & 8, u4 = li & 4, u2 = li & 2;
    const uint8_t* kg = kc + (size_t) g * kFaD + 8 * li;
    const uint8_t* vg = vc + (size_t) g * kFaD + 8 * li;
    auto load = [&](int st, uint2* kb, uint2* vb) {
        const int kt = k_lo + st * 16, kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
#pragma unroll
        for (int j = 0; j < 8; ++j) {   // (a key past the token's: the token's row, its logits masked)
            int rw = row0 + min(8 * row + j, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            kb[j] = __ldg((const uint2*) (kg + (size_t) rw * KD));
            vb[j] = __ldg((const uint2*) (vg + (size_t) rw * KD));
        }
    };
    uint2 kb[8], vb[8];
    if (st0 < st1) load(st0, kb, vb);
    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16;
        // Q K^T: the lane's part of its row's 8 keys, R heads each
        float s[8][R];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t k01, k23, k45, k67;
            e4m3x4_h4_s8(kb[j].x, k01, k23);
            e4m3x4_h4_s8(kb[j].y, k45, k67);
#pragma unroll
            for (int h = 0; h < R; ++h)
                s[j][h] = rd_dot2(k67, qf[h][3], rd_dot2(k45, qf[h][2], rd_dot2(k23, qf[h][1], rd_dot2(k01, qf[h][0], 0.0f))));
        }
        // the row's sums: 8 keys to 4, 2, 1 a lane (the halves kept by the lane's bits 3, 2, 1), then the pair's
        float c[R];
        {
            float t4[4][R], t2[2][R];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int h = 0; h < R; ++h) {
                    const float keep = u8 ? s[j + 4][h] : s[j][h], give = u8 ? s[j][h] : s[j + 4][h];
                    t4[j][h] = keep + rd_xor<8>(give);
                }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int h = 0; h < R; ++h) {
                    const float keep = u4 ? t4[j + 2][h] : t4[j][h], give = u4 ? t4[j][h] : t4[j + 2][h];
                    t2[j][h] = keep + rd_xor<4>(give);
                }
#pragma unroll
            for (int h = 0; h < R; ++h) {
                const float keep = u2 ? t2[1][h] : t2[0][h], give = u2 ? t2[0][h] : t2[1][h];
                const float t1 = keep + rd_xor<2>(give);
                c[h] = (t1 + rd_xor<1>(t1)) * sc;
            }
        }
        // the logits (log2 units): bias within the extent, mask past the token
        const int key = kt + 8 * row + kj;
        if (qpos - (kt + 15) < extent) {
            const int dd = qpos - key;
            if (dd >= 0 && dd < extent) {
                const float* pj = proj + (size_t) dd * d_rel;
#pragma unroll
                for (int h = 0; h < R; ++h) {
                    const float* rh = r + (size_t) (g * R + h) * d_rel;
                    float b = 0.0f;
                    if (rel4) {
                        for (int e = 0; e < d_rel; e += 4) {
                            const float4 x = __ldg((const float4*) (pj + e)), y = __ldg((const float4*) (rh + e));
                            b = fmaf(x.w, y.w, fmaf(x.z, y.z, fmaf(x.y, y.y, fmaf(x.x, y.x, b))));
                        }
                    } else {
                        for (int e = 0; e < d_rel; ++e) b = fmaf(__ldg(pj + e), __ldg(rh + e), b);
                    }
                    c[h] = fmaf(b, kLog2e, c[h]);
                }
            }
        }
        float p[R];
#pragma unroll
        for (int h = 0; h < R; ++h) {
            if (key > qpos) c[h] = -INFINITY;
            float mx = fmaxf(c[h], rd_xor<2>(c[h]));   // (over the warp's 16 keys: lanes 2 apart hold the next key)
            mx = fmaxf(mx, rd_xor<4>(mx));
            mx = fmaxf(mx, rd_xor<8>(mx));
            mx = fmaxf(mx, rd_xor<16>(mx));
            if (mx > m[h]) {   // (the warp's: every lane has the same mx and m)
                const float f = exp2f(m[h] - mx);   // (m = -inf: 0, with o and l still 0)
                m[h] = mx;
                l[h] *= f;
#pragma unroll
                for (int d = 0; d < 8; ++d) o[h][d] *= f;
            }
            p[h] = exp2f(c[h] - (m[h] == -INFINITY ? 0.0f : m[h]));
            if (!(li & 1)) l[h] += p[h];   // (a key's two lanes: the even one counts it)
        }
        // P by key pairs (2 i, 2 i + 1): the pair's lanes are 2 apart; lane 4 i of the row holds pair i
        uint32_t pk[R];
#pragma unroll
        for (int h = 0; h < R; ++h) {
            const float other = rd_xor<2>(p[h]);
            pk[h] = u2 ? f2_to_u32(other, p[h]) : f2_to_u32(p[h], other);
        }
        // P V: o[h][d] += P (keys 2 i, 2 i + 1) . V (the same keys, dim d), pair after pair
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t pp[R];
#pragma unroll
            for (int h = 0; h < R; ++h) {
                if (i == 0) pp[h] = rd_row_lane<0>(pk[h]);
                else if (i == 1) pp[h] = rd_row_lane<4>(pk[h]);
                else if (i == 2) pp[h] = rd_row_lane<8>(pk[h]);
                else pp[h] = rd_row_lane<12>(pk[h]);
            }
            const uint2 a = vb[2 * i], b = vb[2 * i + 1];
            uint32_t v2[8];   // dim d of the two keys, FP16 times 2^-8
            e4m3x4_h4_s8(__byte_perm(a.x, b.x, 0x5140u), v2[0], v2[1]);
            e4m3x4_h4_s8(__byte_perm(a.x, b.x, 0x7362u), v2[2], v2[3]);
            e4m3x4_h4_s8(__byte_perm(a.y, b.y, 0x5140u), v2[4], v2[5]);
            e4m3x4_h4_s8(__byte_perm(a.y, b.y, 0x7362u), v2[6], v2[7]);
#pragma unroll
            for (int h = 0; h < R; ++h)
#pragma unroll
                for (int d = 0; d < 8; ++d) o[h][d] = rd_dot2(v2[d], pp[h], o[h][d]);
        }
        if (st + 1 < st1) load(st + 1, kb, vb);
    }
    // the two rows' sums (the same max), the lanes' l; row 0 writes the warp's partials
#pragma unroll
    for (int h = 0; h < R; ++h) {
#pragma unroll
        for (int d = 0; d < 8; ++d) o[h][d] += rd_xor<16>(o[h][d]);
        float lt = l[h];
        lt += rd_xor<1>(lt);
        lt += rd_xor<2>(lt);
        lt += rd_xor<4>(lt);
        lt += rd_xor<8>(lt);
        lt += rd_xor<16>(lt);
        l[h] = lt;
    }
    if (row == 0) {
#pragma unroll
        for (int h = 0; h < R; ++h) {
            float* op = o_part + ((size_t) w * H + g * R + h) * kFaD + 8 * li;
            *(float4*) op = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
            *(float4*) (op + 4) = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
            if (li == 0) ml_part[(size_t) w * H + g * R + h] = make_float2(m[h], l[h]);
        }
    }
#else
    __trap();   // (never launched: attn_decode_rdna_on() is false off RDNA 3 / 4 wave32)
#endif
}

// the RDNA kernel takes this layer's one-token FP8 attention (a K / V head's R up to 8)
static bool attn_decode_rdna_on(int H, int Hkv) {
    static const bool gpu = rdna_wmma_gpu();
    return g_attn_rdna > 0 && gpu && Hkv > 0 && H % Hkv == 0 && H / Hkv >= 1 && H / Hkv <= 8;
}
static void attn_decode_rdna_launch(dim3 grid, cudaStream_t s, const uint16_t* q16, const uint8_t* kc,
                                    const uint8_t* vc, int kv_cap, int H, int Hkv, const int* pos, int qpos,
                                    float qk_scale, const float* r, const float* proj, int d_rel, bool rel4,
                                    int extent, int window, int n_split, float* o_part, float2* ml_part, int blk0, int tok_q,
                                    int tok_r, size_t tok_part) {
#define IK2_RDNA_LAUNCH(RR)                                                                                            \
    attn_decode_rdna_kernel<RR><<<grid, kWaNW * 32, 0, s>>>(q16, kc, vc, kv_cap, H, Hkv, pos, qpos, qk_scale, r, proj,  \
                                                            d_rel, rel4, extent, window, n_split, o_part, ml_part, blk0,  \
                                                            tok_q, tok_r, tok_part)
    switch (H / Hkv) {
        case 1: IK2_RDNA_LAUNCH(1); break;
        case 2: IK2_RDNA_LAUNCH(2); break;
        case 3: IK2_RDNA_LAUNCH(3); break;
        case 4: IK2_RDNA_LAUNCH(4); break;
        case 5: IK2_RDNA_LAUNCH(5); break;
        case 6: IK2_RDNA_LAUNCH(6); break;
        case 7: IK2_RDNA_LAUNCH(7); break;
        default: IK2_RDNA_LAUNCH(8); break;
    }
#undef IK2_RDNA_LAUNCH
    check("attn_decode_rdna");
}
