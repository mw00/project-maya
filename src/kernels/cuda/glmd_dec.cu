// src/kernels/cuda/glmd_dec.cu - glm-dsa L1-4: the decode attention of mla2 shaped for the GPU (see
// include/maya/kernels/glmd_dec.hpp).  glm-dsa only: mla2 (glmd_fast.cu) calls it.  CUDA and HIP builds; on HIP the
// warps are hip_compat's 32-lane ones (wave64 cards: the half-wavefront mapping), and dev_cfg's device policy decides
// where it runs (S7b).
#include "maya/kernels/glmd_dec.hpp"

#include "maya/core/glm_env.hpp"
#include "maya/kernels/glmd_f16_guard.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace glmd {
namespace {

constexpr int kLora = 512;                   // kv_lora
constexpr int kRope = 64;                    // qk_rope
constexpr int kK = kLora + kRope;            // a cell's staged values
constexpr int kTile = 32;                    // cells a tile: one a lane in the scores
constexpr int kWarps = 8;                    // a block's warps
constexpr int kThreads = kWarps * 32;
constexpr int kStride = kK + 4;              // halves a staged cell: 1,160 B, 8-byte rows, the lanes' reads conflict-free
constexpr int kRec8 = kLora + kLora / 16;    // Maya's INT8 latent record: 512 codes + 16 FP16 scales (544 B)
constexpr int kPart = kLora + 2;             // a partial: O, m, l
constexpr int kMaxSplits = 64;

bool launch_ok(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "glmd_dec %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int n_live(int n_sel, const int* n_sel_dev) {
    return n_sel_dev != nullptr ? min(n_sel, *n_sel_dev) : n_sel;
}
// the splits for n live cells: whole tiles, at least min_tiles a split, at most s_max (the combine computes the same;
// min_tiles 1 for mla_dec_kernel, the RDNA kernel's own for mla_dec_rdna_kernel)
__device__ __forceinline__ int n_splits(int n, int s_max, int min_tiles = 1) {
    const int n_tiles = (n + kTile - 1) / kTile;
    return min((n_tiles + min_tiles - 1) / min_tiles, s_max);
}

// 4 INT8 codes (one word) times an FP16 scale, as 2 half2: each value is __float2half((float) code * scale) - the
// FP16 product of two exact halves is the correctly rounded exact product, which that f32 product (19 bits) is too.
// A byte b: half 0x6400 | (b ^ 0x80) is 1152 + b exactly.
__device__ __forceinline__ void dec4(uint32_t w, __half2 sc2, uint32_t& lo, uint32_t& hi) {
    const uint32_t x = w ^ 0x80808080u;
    const uint32_t p0 = __byte_perm(x, 0x64646464u, 0x5140), p1 = __byte_perm(x, 0x64646464u, 0x7362);
    const __half2 off = __floats2half2_rn(1152.0f, 1152.0f);
    const __half2 h0 = __hmul2(__hsub2(*(const __half2*) &p0, off), sc2);
    const __half2 h1 = __hmul2(__hsub2(*(const __half2*) &p1, off), sc2);
    lo = *(const uint32_t*) &h0;
    hi = *(const uint32_t*) &h1;
}

// grid (s_max, n_head / (8 HPW)): block (b, g) walks tiles [b T / S, (b + 1) T / S) for heads 8 HPW g ..; warp w
// holds heads 8 HPW g + HPW w ..  Dynamic shared memory: the tile (kTile x kStride halves), q (HPW x 8 heads x kK f32),
// the tile's cells.
template <int LQ8, int HPW>
__global__ void __launch_bounds__(kThreads, 1)
    mla_dec_kernel(const float* __restrict__ q_abs, const float* __restrict__ q, int q_stride, int q_pe_off,
                   const void* __restrict__ lat, const uint16_t* __restrict__ k_pe, const int* __restrict__ cells,
                   int n_sel, const int* __restrict__ n_sel_dev, int n_head, float scale, int s_max,
                   float* __restrict__ part) {
    constexpr int HB = kWarps * HPW;
    extern __shared__ __align__(16) unsigned char smem[];
    uint16_t* sL = (uint16_t*) smem;
    float* sQ = (float*) (smem + kTile * kStride * 2);
    int* s_cell = (int*) (sQ + HB * kK);
    const int n = n_live(n_sel, n_sel_dev);
    const int S = n_splits(n, s_max);
    const int b = blockIdx.x;
    if (b >= S) return;
    const int n_tiles = (n + kTile - 1) / kTile;
    const int t0 = (int) ((long long) b * n_tiles / S), t1 = (int) ((long long) (b + 1) * n_tiles / S);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int h0 = blockIdx.y * HB + warp * HPW;
    // the warp's heads' [q_abs | q_pe] (f32) into shared memory
#pragma unroll
    for (int e = 0; e < HPW; ++e) {
        float4* qs = (float4*) (sQ + (warp * HPW + e) * kK);
        const float4* a4 = (const float4*) (q_abs + (size_t) (h0 + e) * kLora);
        const float4* p4 = (const float4*) (q + (size_t) (h0 + e) * q_stride + q_pe_off);
        for (int i = lane; i < kK / 4; i += 32) qs[i] = i < kLora / 4 ? a4[i] : p4[i - kLora / 4];
    }
    const float* qs = sQ + warp * HPW * kK;
    float m[HPW], l[HPW], o[HPW][16];
#pragma unroll
    for (int e = 0; e < HPW; ++e) {
        m[e] = -INFINITY;
        l[e] = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) o[e][j] = 0.0f;
    }
    for (int t = t0; t < t1; ++t) {
        __syncthreads();   // (the previous tile's reads of sL / s_cell are done; q is staged)
        if (tid < kTile) {
            const int i = t * kTile + tid;
            s_cell[tid] = i < n ? cells[i] : -1;
        }
        __syncthreads();
        // stage the tile: latents as FP16 (an INT8 record decoded as mla2 stages it), then k_pe; -1 = zeros
        if constexpr (LQ8 == 0) {
            for (int i = tid; i < kTile * (kLora / 8); i += kThreads) {
                const int s = i >> 6, j = i & 63;
                const int c = s_cell[s];
                uint4 v = make_uint4(0, 0, 0, 0);
                if (c >= 0) v = ((const uint4*) lat)[(size_t) c * (kLora / 8) + j];
                uint2* d = (uint2*) (sL + s * kStride + 8 * j);
                d[0] = make_uint2(v.x, v.y);
                d[1] = make_uint2(v.z, v.w);
            }
        } else {
            for (int i = tid; i < kTile * (kLora / 16); i += kThreads) {
                const int s = i >> 5, j = i & 31;   // codes [16 j, 16 j + 16) of cell s
                const int c = s_cell[s];
                uint32_t pk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (c >= 0) {
                    const uint8_t* rec = (const uint8_t*) lat + (size_t) c * kRec8;
                    const uint4 v = *(const uint4*) (rec + 16 * j);
                    const __half sh = ((const __half*) (rec + kLora))[j >> 1];
                    const __half2 sc2 = __halves2half2(sh, sh);
                    dec4(v.x, sc2, pk[0], pk[1]);
                    dec4(v.y, sc2, pk[2], pk[3]);
                    dec4(v.z, sc2, pk[4], pk[5]);
                    dec4(v.w, sc2, pk[6], pk[7]);
                }
                uint2* d = (uint2*) (sL + s * kStride + 16 * j);
#pragma unroll
                for (int e = 0; e < 4; ++e) d[e] = make_uint2(pk[2 * e], pk[2 * e + 1]);
            }
        }
        for (int i = tid; i < kTile * kRope / 8; i += kThreads) {
            const int s = i >> 3, j = i & 7;
            const int c = s_cell[s];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c >= 0) v = ((const uint4*) k_pe)[(size_t) c * (kRope / 8) + j];
            uint2* d = (uint2*) (sL + s * kStride + kLora + 8 * j);
            d[0] = make_uint2(v.x, v.y);
            d[1] = make_uint2(v.z, v.w);
        }
        __syncthreads();
        // scores: lane = cell; four partial sums a head (each staged value read once for the warp's heads)
        const uint16_t* Lc = sL + lane * kStride;
        float acc[HPW][4];
#pragma unroll
        for (int e = 0; e < HPW; ++e) acc[e][0] = acc[e][1] = acc[e][2] = acc[e][3] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kK; d += 4) {
            const uint2 lv = *(const uint2*) (Lc + d);
            const float2 f0 = __half22float2(*(const __half2*) &lv.x), f1 = __half22float2(*(const __half2*) &lv.y);
#pragma unroll
            for (int e = 0; e < HPW; ++e) {
                const float4 x = *(const float4*) (qs + e * kK + d);
                acc[e][0] = fmaf(f0.x, x.x, acc[e][0]);
                acc[e][1] = fmaf(f0.y, x.y, acc[e][1]);
                acc[e][2] = fmaf(f1.x, x.z, acc[e][2]);
                acc[e][3] = fmaf(f1.y, x.w, acc[e][3]);
            }
        }
        // the online softmax (f32 m, l; a tile of padding changes nothing)
        const bool valid = s_cell[lane] >= 0;
        float ew[HPW];
#pragma unroll
        for (int e = 0; e < HPW; ++e) {
            const float sc = valid ? ((acc[e][0] + acc[e][1]) + (acc[e][2] + acc[e][3])) * scale : -INFINITY;
            const float nm = fmaxf(m[e], warp_max(sc));
            const float al = m[e] == -INFINITY ? 0.0f : expf(m[e] - nm);
            ew[e] = sc == -INFINITY ? 0.0f : expf(sc - nm);
            l[e] = l[e] * al + warp_sum(ew[e]);
            m[e] = nm;
#pragma unroll
            for (int j = 0; j < 16; ++j) o[e][j] *= al;
        }
        // O += P . L: lane holds values {2 lane + 64 j, + 1} of its heads; each staged pair read once for them
#pragma unroll 4
        for (int c = 0; c < kTile; ++c) {
            float w[HPW];
#pragma unroll
            for (int e = 0; e < HPW; ++e) w[e] = __shfl_sync(0xffffffffu, ew[e], c);
            const __half2* Lr = (const __half2*) (sL + c * kStride) + lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 f = __half22float2(Lr[32 * j]);
#pragma unroll
                for (int e = 0; e < HPW; ++e) {
                    o[e][2 * j] = fmaf(w[e], f.x, o[e][2 * j]);
                    o[e][2 * j + 1] = fmaf(w[e], f.y, o[e][2 * j + 1]);
                }
            }
        }
    }
#pragma unroll
    for (int e = 0; e < HPW; ++e) {
        float* p = part + ((size_t) b * n_head + h0 + e) * kPart;
#pragma unroll
        for (int j = 0; j < 8; ++j) *(float2*) (p + 2 * lane + 64 * j) = make_float2(o[e][2 * j], o[e][2 * j + 1]);
        if (lane == 0) {
            p[kLora] = m[e];
            p[kLora + 1] = l[e];
        }
    }
}

// ---- RDNA 3 / 3.5 (gfx11, wave32 WMMA): the same partials from the matrix cores (MAYA_GLMD_MLA_DEC_RDNA=1)
// mla_dec_kernel on AMD holds 8 heads a block (64 KB of LDS), so the 32 heads of a layer read every cell four times,
// with f32 FMAs fed from LDS.  This kernel is glmd_batch.cu's mla_prompt_rdna_kernel (the prompt attention on the same
// archs) for one query row, split over the cells as mla_dec_kernel is: a block per (16 heads, split), the two head
// blocks of a split next to each other in the grid (the L2 serves the second read), Q in registers as WMMA A operands
// (FP16 with the scale folded in and the B15.1 guard), the cells 16 at a time in FP16 (an INT8 record decoded as
// mla_dec_kernel stages it), S = Q K^T and O += P V on the matrix cores (F32 sums, P in FP16), the online softmax in
// F32.  Each block writes its partial (O, m, l) in mla_dec_kernel's layout and mla_dec_combine_kernel merges them.
// Arithmetic: the FP16 products of the CUDA tensor-core decode (ling_kernels.cu mla_dec_wmma) rather than f32 - the
// logits differ from mla_dec_kernel's by rounding.
// gfx11 (RDNA 3 / 3.5, wave32): D = A B + C for 16 x 16 x 16 FP16 tiles, F32 sums.  Lane l holds row (l & 15) of A
// and column (l & 15) of B, all 16 k values (lanes 16-31 the same as 0-15); element i of D's 8 a lane is row
// 2 i + (l >> 4), column l & 15 (glmd_batch.cu's rw::wmma for gfx11).  Other devices never launch the kernel.
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ &&                                                       \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) ||                  \
     defined(__gfx1151__))
#define GLMD_DEC_RDNA 1
#else
#define GLMD_DEC_RDNA 0
#endif
#if defined(MAYA_USE_HIP)
namespace rd {
using h16 = _Float16 __attribute__((ext_vector_type(16)));
using f8 = float __attribute__((ext_vector_type(8)));
[[maybe_unused]] __device__ __forceinline__ f8 wmma(h16 a, h16 b, f8 c) {
#if GLMD_DEC_RDNA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a;
    (void) b;
    __builtin_trap();
    return c;
#endif
}
[[maybe_unused]] __device__ __forceinline__ int row_of(int i, int hi) { return 2 * i + hi; }
}  // namespace rd

[[maybe_unused]] __device__ __forceinline__ uint4 rdna_lat_h8(const void* lat, int lat_q8, int cell, int c8) {
    if (!lat_q8) return ((const uint4*) lat)[(size_t) cell * (kLora / 8) + c8];
    const uint8_t* rec = (const uint8_t*) lat + (size_t) cell * kRec8;
    const float d = __half2float(((const __half*) (rec + kLora))[c8 >> 2]);   // a scale per 32 codes
    const int8_t* q8 = (const int8_t*) rec + c8 * 8;
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half lo = __float2half((float) q8[2 * i] * d), hi = __float2half((float) q8[2 * i + 1] * d);
        w[i] = (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__global__ void __launch_bounds__(kThreads)
    mla_dec_rdna_kernel(const float* __restrict__ q_abs, const float* __restrict__ q, int q_stride, int q_pe_off,
                        const void* __restrict__ lat, int lat_q8, const uint16_t* __restrict__ k_pe,
                        const int* __restrict__ cells,
                        int n_sel, const int* __restrict__ n_sel_dev, int n_head, float scale, int s_max,
                        int min_tiles, float* __restrict__ part) {
#if GLMD_DEC_RDNA
    using wmma_h16 = rd::h16;
    using wmma_f8 = rd::f8;
    constexpr int K = kLora, R = kRope, CH = 16, HB = 16;
    __shared__ __align__(32) _Float16 sK[CH][K];   // the 16 cells' latents (16 KiB)
    __shared__ __align__(32) _Float16 sR[CH][R];   // ... and k_pe (2 KiB)
    __shared__ float sPart[kWarps][HB][CH];        // each wave's partial scores (8 KiB)
    __shared__ __align__(32) _Float16 sP[HB][CH];  // P = exp(S - m), FP16
    __shared__ float sM[HB], sL[HB], sA[HB];
    __shared__ int s_cell[kTile];
    __shared__ float s_qmax[kWarps], s_qs;
    const int n = n_live(n_sel, n_sel_dev);
    const int S = n_splits(n, s_max, min_tiles);
    const int b = blockIdx.y;
    if (b >= S) return;
    const int n_tiles = (n + kTile - 1) / kTile;
    const int t0 = (int) ((long long) b * n_tiles / S), t1 = (int) ((long long) (b + 1) * n_tiles / S);
    const int h0 = blockIdx.x * HB, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int l16 = lane & 15, hi = lane >> 4;
    const bool rope_kb = wave < R / 16;   // (wave-uniform) waves 0-3 also hold a rope k-block
    // this lane's A rows: head h0 + l16; wave w holds latent dims [64 w, 64 w + 64) and rope dims [16 w, 16 w + 16)
    const float* qa = q_abs + (size_t) (h0 + l16) * K + wave * 64;
    const float* qp = q + (size_t) (h0 + l16) * q_stride + q_pe_off + (rope_kb ? wave * 16 : 0);
    // B15.1: the FP16 range guard of the block's Q tile (its 16 heads), as mla_prompt_rdna_kernel
    float qmax = 0.0f;
#pragma unroll
    for (int k = 0; k < 64; ++k) qmax = fmaxf(qmax, fabsf(qa[k] * scale));
    if (rope_kb)
        for (int k = 0; k < 16; ++k) qmax = fmaxf(qmax, fabsf(qp[k] * scale));
    qmax = warp_max(qmax);
    if (lane == 0) s_qmax[wave] = qmax;
    if (tid < HB) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < kWarps; ++w) m = fmaxf(m, s_qmax[w]);
        s_qs = glmd::f16_guard_scale(m);
    }
    __syncthreads();
    const float qs = s_qs, qinv = 1.0f / qs;
    wmma_h16 qf[5];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float v = qa[kb * 16 + j] * scale;
            qf[kb][j] = (_Float16) (qs != 1.0f ? v * qinv : v);
        }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float v = rope_kb ? qp[j] * scale : 0.0f;
        qf[4][j] = (_Float16) (qs != 1.0f ? v * qinv : v);
    }
    wmma_f8 acc[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    for (int t = t0; t < t1; ++t) {
        __syncthreads();   // the previous tile's readers of s_cell are done
        if (tid < kTile) {
            const int i = t * kTile + tid;
            s_cell[tid] = i < n ? cells[i] : -1;
        }
        __syncthreads();
        for (int half = 0; half < 2; ++half) {
            const int* cl = s_cell + half * CH;
            bool any = false;
#pragma unroll
            for (int i = 0; i < CH; ++i) any = any || cl[i] >= 0;
            if (!any) continue;   // (block-uniform: every thread read the same cells)
            __syncthreads();      // the previous tile's readers of sK / sR / sP are done
            for (int i = tid; i < CH * K / 8; i += kThreads) {
                const int s = i / (K / 8), c8 = i - s * (K / 8);
                const int cell = cl[s];
                ((uint4*) &sK[s][0])[c8] = cell < 0 ? make_uint4(0u, 0u, 0u, 0u) : rdna_lat_h8(lat, lat_q8, cell, c8);
            }
            for (int i = tid; i < CH * R / 8; i += kThreads) {
                const int s = i / (R / 8), c8 = i - s * (R / 8);
                const int cell = cl[s];
                ((uint4*) &sR[s][0])[c8] =
                    cell < 0 ? make_uint4(0u, 0u, 0u, 0u) : ((const uint4*) (k_pe + (size_t) R * cell))[c8];
            }
            __syncthreads();
            // this wave's share of S (16 heads x 16 cells) = Q K^T
            wmma_f8 dot;
#pragma unroll
            for (int i = 0; i < 8; ++i) dot[i] = 0.0f;
#pragma unroll
            for (int kb = 0; kb < 4; ++kb)
                dot = rd::wmma(qf[kb], *(const wmma_h16*) &sK[l16][wave * 64 + kb * 16], dot);
            if (rope_kb) dot = rd::wmma(qf[4], *(const wmma_h16*) &sR[l16][wave * 16], dot);
#pragma unroll
            for (int i = 0; i < 8; ++i) sPart[wave][rd::row_of(i, hi)][l16] = dot[i];
            __syncthreads();
            // the online softmax: 16 threads a head row (a row's threads are 16 lanes of one wave)
            {
                const int r = tid >> 4, col = tid & 15;
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < kWarps; ++w) v += sPart[w][r][col];
                v = cl[col] >= 0 ? v * qs : -INFINITY;   // (B15.1: qs 1 is exact)
                float cm = v;
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, o));
                const float mo = sM[r];
                const float mn = fmaxf(mo, cm);
                float p = 0.0f, alpha = 1.0f;
                if (mn != -INFINITY) {
                    alpha = mo == -INFINITY ? 0.0f : expf(mo - mn);
                    p = v == -INFINITY ? 0.0f : expf(v - mn);
                }
                float ps = p;
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
                sP[r][col] = (_Float16) p;
                if (col == 0) {
                    sL[r] = sL[r] * alpha + ps;
                    sM[r] = mn;
                    sA[r] = alpha;
                }
            }
            __syncthreads();
            // O (16 x 512) = alpha O + P (16 x 16 cells) V (16 cells x 512): this wave's four column tiles
            const wmma_h16 pa = *(const wmma_h16*) &sP[l16][0];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[nt][i] *= sA[rd::row_of(i, hi)];
                wmma_h16 bv;
#pragma unroll
                for (int k = 0; k < 16; ++k) bv[k] = sK[k][wave * 64 + nt * 16 + l16];
                acc[nt] = rd::wmma(pa, bv, acc[nt]);
            }
        }
    }
    __syncthreads();
    // the partial: O unnormalised, then m and l (mla_dec_kernel's layout)
    float* pb = part + ((size_t) b * n_head + h0) * kPart;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            asm volatile("" ::"v"(acc[nt][i]));   // (every matrix lane live before the stores, as the prompt kernel)
            pb[(size_t) rd::row_of(i, hi) * kPart + wave * 64 + nt * 16 + l16] = acc[nt][i];
        }
    if (tid < HB) {
        pb[(size_t) tid * kPart + kLora] = sM[tid];
        pb[(size_t) tid * kPart + kLora + 1] = sL[tid];
    }
#else
    (void) q_abs; (void) q; (void) q_stride; (void) q_pe_off; (void) lat; (void) lat_q8; (void) k_pe; (void) cells;
    (void) n_sel; (void) n_sel_dev; (void) n_head; (void) scale; (void) s_max; (void) min_tiles; (void) part;
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    __builtin_trap();   // (no WMMA on this pass: never launched, rdna_dec_cfg() refuses the device)
#endif
#endif
}
#endif  // MAYA_USE_HIP

// one block a head: the S partials merged in f32 (mla2_combine_kernel's rule), ctx = O / l
__global__ void __launch_bounds__(kThreads) mla_dec_combine_kernel(const float* __restrict__ part, int n_sel,
                                                                   const int* __restrict__ n_sel_dev, int n_head,
                                                                   int s_max, float* __restrict__ ctx,
                                                                   int min_tiles = 1) {
    __shared__ float s_w[kMaxSplits];
    __shared__ float s_inv;
    const int h = blockIdx.x, tid = threadIdx.x;
    const int S = n_splits(n_live(n_sel, n_sel_dev), s_max, min_tiles);
    if (tid < 32) {
        float mb[2], lb[2], M = -INFINITY;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int b = tid + 32 * k;
            const float* pb = part + ((size_t) b * n_head + h) * kPart;
            mb[k] = b < S ? pb[kLora] : -INFINITY;
            lb[k] = b < S ? pb[kLora + 1] : 0.0f;
            M = fmaxf(M, mb[k]);
        }
        M = warp_max(M);
        float L = 0.0f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const float w = mb[k] == -INFINITY ? 0.0f : expf(mb[k] - M);
            s_w[tid + 32 * k] = w;
            L += w * lb[k];
        }
        L = warp_sum(L);
        if (tid == 0) s_inv = L > 0.0f ? 1.0f / L : 0.0f;
    }
    __syncthreads();
    for (int c = tid; c < kLora; c += kThreads) {
        float acc = 0.0f;
        for (int b = 0; b < S; ++b) acc += s_w[b] * part[((size_t) b * n_head + h) * kPart + c];
        ctx[(size_t) kLora * h + c] = acc * s_inv;
    }
}

size_t smem_bytes(int hpw) { return (size_t) kTile * kStride * 2 + (size_t) kWarps * hpw * kK * 4 + kTile * 4; }

// the device's choice (cached): heads a warp (2: 16 heads a block, 74 KB of shared memory; 1: 8 heads, 55 KB, the
// devices with 64 KB a block - the AMD GPUs; 0: neither fits - mla2 keeps its own kernels), the blocks a multiprocessor
// holds, and whether mla2 takes it on this device (on: it fits and the policy below says yes)
struct DevCfg {
    int hpw = -1, per_sm = 0, sms = 0;
    bool on = false;
};

// The device's default (MAYA_GLMD_MLA_DEC unset or auto).  A CUDA build: wherever it fits (L1-4's default,
// unchanged).  A HIP build: only on the archs where it was measured faster than mla2's own kernels, in the bench and in
// GLM-5.3's decode - gfx1151 (Radeon 8060S / 8065S, Strix / Gorgon Halo; SUN-NOTES S7b) - and mla2's kernels on every
// other AMD arch until someone measures it there (it builds and fits on all of them: 55 KB of LDS a block).
// MAYA_GLMD_MLA_DEC=1 takes it on any device it fits; =0 keeps mla2's kernels everywhere (decode_mla's legacy_attn).
bool auto_device(int dev) {
#if defined(MAYA_USE_HIP)
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
#else
    (void) dev;
    return true;
#endif
}
const DevCfg& dev_cfg() {
    static DevCfg cfg[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    DevCfg& c = cfg[dev];
    if (c.hpw >= 0) return c;
    c.hpw = 0;
    int optin = 0;
    cudaDeviceGetAttribute(&c.sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    for (const int hpw : {2, 1}) {
        const size_t sm = smem_bytes(hpw);
        if ((size_t) optin < sm) continue;
        bool ok = true;
        int per = 0;
        if (hpw == 2) {
            ok = cudaFuncSetAttribute(mla_dec_kernel<0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) ==
                     cudaSuccess &&
                 cudaFuncSetAttribute(mla_dec_kernel<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) ==
                     cudaSuccess &&
                 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, mla_dec_kernel<1, 2>, kThreads, sm) == cudaSuccess;
        } else {
            ok = cudaFuncSetAttribute(mla_dec_kernel<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) ==
                     cudaSuccess &&
                 cudaFuncSetAttribute(mla_dec_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) ==
                     cudaSuccess &&
                 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, mla_dec_kernel<1, 1>, kThreads, sm) == cudaSuccess;
        }
        if (ok && per > 0) {
            c.hpw = hpw;
            c.per_sm = per;
            break;
        }
        cudaGetLastError();
    }
    const int sw = maya::glmenv::parse(std::getenv("MAYA_GLMD_MLA_DEC"));
    c.on = c.hpw > 0 && (sw == maya::glmenv::kOn || auto_device(dev));
    return c;
}

// MAYA_GLMD_MLA_DEC_RDNA=1 (off by default): the decode attention from the matrix cores (mla_dec_rdna_kernel) on
// gfx11 (RDNA 3 / 3.5) where mla_dec runs; MAYA_GLMD_MLA_DEC_RDNA_SPLITS (1-64, default 64) caps a head block's
// splits, MAYA_GLMD_MLA_DEC_RDNA_MIN_TILES (1-64, default 2) is the fewest 32-cell tiles a split takes.
#if defined(MAYA_USE_HIP)
struct RdnaDecCfg {
    int state = 0;   // 0 not read, 1 on, 2 off
    int s_max = kMaxSplits;
    int min_tiles = 2;
};
const RdnaDecCfg& rdna_dec_cfg() {
    static RdnaDecCfg cfg[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    RdnaDecCfg& c = cfg[dev];
    if (c.state != 0) return c;
    c.state = 2;
    if (maya::glmenv::parse(std::getenv("MAYA_GLMD_MLA_DEC_RDNA")) == maya::glmenv::kOn) {
        hipDeviceProp_t prop{};
        bool arch = false;
        if (hipGetDeviceProperties(&prop, dev) == hipSuccess)
            for (const char* a : {"gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"})
                if (std::strncmp(prop.gcnArchName, a, 7) == 0 &&
                    (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':'))
                    arch = true;
        if (arch) {
            const long long sp = maya::glmenv::env_int("MAYA_GLMD_MLA_DEC_RDNA_SPLITS", kMaxSplits);
            const long long mt = maya::glmenv::env_int("MAYA_GLMD_MLA_DEC_RDNA_MIN_TILES", 2);
            c.s_max = (int) (sp < 1 ? 1 : sp > kMaxSplits ? kMaxSplits : sp);
            c.min_tiles = (int) (mt < 1 ? 1 : mt > 64 ? 64 : mt);
            c.state = 1;
            std::fprintf(stderr, "glmd_dec: mla_dec on the matrix cores (%s, splits %d, min tiles %d)\n",
                         prop.gcnArchName, c.s_max, c.min_tiles);
        } else {
            std::fprintf(stderr, "glmd_dec: MAYA_GLMD_MLA_DEC_RDNA=1 ignored (not gfx11)\n");
        }
    }
    return c;
}
#endif  // MAYA_USE_HIP

}  // namespace

bool mla_dec_ok(int n_head, int kv_lora, int qk_rope) {
    if (kv_lora != kLora || qk_rope != kRope || n_head <= 0) return false;
    const DevCfg& c = dev_cfg();
    return c.on && n_head % (kWarps * c.hpw) == 0;
}

int mla_dec_splits(int n_head) {
    const DevCfg& c = dev_cfg();
    if (c.hpw <= 0) return 0;
    const int groups = std::max(1, n_head / (kWarps * c.hpw));
    return std::max(1, std::min(kMaxSplits, c.per_sm * std::max(1, c.sms) / groups));
}

size_t mla_dec_part_bytes(int n_head) { return (size_t) kMaxSplits * n_head * kPart * sizeof(float); }

bool mla_dec(const float* q_abs, const float* q, int q_stride, int q_pe_off, const void* lat, int lat_q8,
             const unsigned short* k_pe, const int* cells, int n_sel, const int* n_sel_dev, int n_head, int kv_lora,
             int qk_rope, float scale, float* part, float* ctx, void* stream) {
    if (!mla_dec_ok(n_head, kv_lora, qk_rope) || !q_abs || !q || !lat || !k_pe || !part || !ctx || n_sel < 0 ||
        (n_sel > 0 && !cells) || q_stride % 4 || q_pe_off % 4 || ((uintptr_t) lat & 15u) || ((uintptr_t) k_pe & 15u) ||
        ((uintptr_t) q_abs & 15u) || ((uintptr_t) q & 15u))
        return false;
    cudaStream_t s = (cudaStream_t) stream;
#if defined(MAYA_USE_HIP)
    const RdnaDecCfg& rc = rdna_dec_cfg();
    if (rc.state == 1 && n_head % 16 == 0) {
        // the matrix-core path: a fixed grid (graph replay), the live cells' splits counted on the device
        if (n_sel > 0) {
            const dim3 grid(n_head / 16, rc.s_max);
            mla_dec_rdna_kernel<<< grid, kThreads, 0, s >>>(q_abs, q, q_stride, q_pe_off, lat, lat_q8 ? 1 : 0, k_pe,
                                                            cells, n_sel, n_sel_dev, n_head, scale, rc.s_max,
                                                            rc.min_tiles, part);
        }
        mla_dec_combine_kernel<<< n_head, kThreads, 0, s >>>(part, n_sel, n_sel_dev, n_head, rc.s_max, ctx,
                                                             rc.min_tiles);
        return launch_ok("mla_dec_rdna");
    }
#endif
    const DevCfg& c = dev_cfg();
    const int s_max = mla_dec_splits(n_head);
    if (n_sel > 0) {
        const dim3 grid(s_max, n_head / (kWarps * c.hpw));
        const size_t sm = smem_bytes(c.hpw);
        const unsigned short* kp = k_pe;
        if (c.hpw == 2) {
            if (lat_q8)
                mla_dec_kernel<1, 2><<<grid, kThreads, sm, s>>>(q_abs, q, q_stride, q_pe_off, lat, kp, cells, n_sel,
                                                                n_sel_dev, n_head, scale, s_max, part);
            else
                mla_dec_kernel<0, 2><<<grid, kThreads, sm, s>>>(q_abs, q, q_stride, q_pe_off, lat, kp, cells, n_sel,
                                                                n_sel_dev, n_head, scale, s_max, part);
        } else {
            if (lat_q8)
                mla_dec_kernel<1, 1><<<grid, kThreads, sm, s>>>(q_abs, q, q_stride, q_pe_off, lat, kp, cells, n_sel,
                                                                n_sel_dev, n_head, scale, s_max, part);
            else
                mla_dec_kernel<0, 1><<<grid, kThreads, sm, s>>>(q_abs, q, q_stride, q_pe_off, lat, kp, cells, n_sel,
                                                                n_sel_dev, n_head, scale, s_max, part);
        }
    }
    mla_dec_combine_kernel<<<n_head, kThreads, 0, s>>>(part, n_sel, n_sel_dev, n_head, s_max, ctx);
    return launch_ok("mla_dec");
}

}  // namespace glmd
