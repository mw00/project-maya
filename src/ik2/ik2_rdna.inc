// src/ik2/ik2_rdna.inc - maya-ik2's kernels for AMD's matrix cores (RDNA 3 / 3.5 gfx11, RDNA 4 gfx12, wave32), the
// paths where the CUDA kernels' mma.sync has no AMD instruction to map to (ik2_kernels.cu includes this inside
// ik2::k; the interface is in ik2_kernels.cuh).  The products: maya::kernels::wmma_f16 (16 x 16 x 16, F16 in, F32
// sums; include/strata/kernels/rdna_wmma.hpp).  Every sum runs in a fixed order, so a GPU's answers repeat; they are
// not the CUDA kernels' bits (another order of the same sums).
//
// The prompt's routed experts: a layer's experts over a chunk in three launches, straight from their blocks (NVFP4 or
// Q8_0, made FP16 in shared memory - exact: a code times its scale is an FP16), instead of a decode to FP16 and two
// BLAS products an expert.  The rows (a token's pick each) lie grouped by expert, in expert order; a tile is up to 32
// rows of one expert.  gate / up: a block 64 rows of gate and the same 64 of up against a tile's rows, SwiGLU in the
// registers (a lane holds a row's gate and up); down: 128 rows of down.  Then each token's picks summed in expert
// order (the order the CUDA path adds them in).
// (AMD builds only: ik2_kernels.cu includes it under STRATA_USE_HIP, after rdna_wmma.hpp)

namespace {

constexpr int kRmBM = 32, kRmBN = 128, kRmBK = 64, kRmLd = kRmBK + 8;   // (row pad: the lanes' rows on other banks)

// a block's W tile row: 64 values of one row (k-step ks) from its blocks into FP16 in shared memory
template <int TY>
__device__ __forceinline__ void rm_load_w(const uint8_t* row, int ks, uint32_t (&w)[17]) {
    if constexpr (TY == kNVFP4) {   // a 64-value block: 4 scales, 32 bytes of codes
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 36);
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(p + i);
    } else {   // Q8_0: two 32-value blocks (34 bytes each)
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 68);
#pragma unroll
        for (int i = 0; i < 17; ++i) w[i] = __ldg(p + i);
    }
}
template <int TY>
__device__ __forceinline__ void rm_store_w(const uint32_t (&w)[17], uint16_t* dst) {
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
    uint32_t w[17];
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
        rm_store_w<TY>(w, Ws + tid * kRmLd);
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

// out [T][C] = sum of a token's picks' wtr[row] y[row], in row order (= expert order); a block a token
__global__ void moe_rows_combine_kernel(const float* __restrict__ y, const int* __restrict__ inv,
                                        const float* __restrict__ wtr, int K, int C, float* __restrict__ out) {
    const int t = blockIdx.x;
    int r[16];
    const int kk = K < 16 ? K : 16;
    for (int k = 0; k < kk; ++k) {   // (insertion sort of a token's few rows)
        int v = inv[(size_t) t * K + k], j = k;
        while (j > 0 && r[j - 1] > v) r[j] = r[j - 1], --j;
        r[j] = v;
    }
    for (int c4 = threadIdx.x; c4 < C / 4; c4 += blockDim.x) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int k = 0; k < kk; ++k) {
            const float w = wtr[r[k]];
            const float4 v = ((const float4*) (y + (size_t) r[k] * C))[c4];
            a.x = fmaf(w, v.x, a.x), a.y = fmaf(w, v.y, a.y), a.z = fmaf(w, v.z, a.z), a.w = fmaf(w, v.w, a.w);
        }
        ((float4*) (out + (size_t) t * C))[c4] = a;
    }
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
template <int WARPS, int R, int TT>
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
    auto lo = [](uint32_t v) { return __uint_as_float(v << 16); };
    auto hi = [](uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); };
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


bool experts_rdna_available(int ty_gate, int ty_up, int ty_down, int C, int F) {
    static const int on = [] {   // (the current GPU: an RDNA 3 / 3.5 / 4 one of this build; IK2_RDNA_MOE=0: off)
        if (std::getenv("IK2_RDNA_MOE") && !std::atoi(std::getenv("IK2_RDNA_MOE"))) return 0;
        return rdna_wmma_gpu() ? 1 : 0;
    }();
    const bool fmt = (ty_gate == kNVFP4 || ty_gate == kQ8_0) && ty_up == ty_gate && ty_down == ty_gate;
    return on && fmt && C % kRmBN == 0 && F % kRmBN == 0 && C % kRmBK == 0 && F % kRmBK == 0;
}
void experts_rdna_gu(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_up,
                     const uint16_t* x16, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                     uint16_t* h16, cudaStream_t s) {
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (F / 64), (unsigned) n_tiles);
    const size_t rb = mv_row_bytes(ty, C);
    if (ty == kNVFP4)
        experts_rdna_kernel<kNVFP4, true><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                            su, gcap, ucap, h16, nullptr);
    else
        experts_rdna_kernel<kQ8_0, true><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                           su, gcap, ucap, h16, nullptr);
    check("experts_rdna_gu");
}
void experts_rdna_down(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_down,
                       const uint16_t* h16, int C, int F, float* y32, cudaStream_t s) {
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (C / 128), (unsigned) n_tiles);
    const size_t rb = mv_row_bytes(ty, F);
    if (ty == kNVFP4)
        experts_rdna_kernel<kNVFP4, false><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                             nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
    else
        experts_rdna_kernel<kQ8_0, false><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                            nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
    check("experts_rdna_down");
}
void moe_rows_combine(const float* y32, const int* inv, const float* wtr, int T, int K, int C, float* out,
                      cudaStream_t s) {
    if (T <= 0) return;
    moe_rows_combine_kernel<<<T, 256, 0, s>>>(y32, inv, wtr, K, C, out);
    check("moe_rows_combine");
}
