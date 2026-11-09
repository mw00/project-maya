// src/kernels/cuda/glmd_dec.cu - glm-dsa L1-4: the decode attention of mla2 shaped for the GPU (see
// include/strata/kernels/glmd_dec.hpp).  glm-dsa only: mla2 (glmd_fast.cu) calls it.  CUDA and HIP builds; on HIP the
// warps are hip_compat's 32-lane ones (wave64 cards: the half-wavefront mapping), and dev_cfg's device policy decides
// where it runs (S7b).
#include "strata/kernels/glmd_dec.hpp"

#include "strata/core/glm_env.hpp"

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
// the splits for n live cells: whole tiles, at most s_max (the combine computes the same)
__device__ __forceinline__ int n_splits(int n, int s_max) { return min((n + kTile - 1) / kTile, s_max); }

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

// one block a head: the S partials merged in f32 (mla2_combine_kernel's rule), ctx = O / l
__global__ void __launch_bounds__(kThreads) mla_dec_combine_kernel(const float* __restrict__ part, int n_sel,
                                                                   const int* __restrict__ n_sel_dev, int n_head,
                                                                   int s_max, float* __restrict__ ctx) {
    __shared__ float s_w[kMaxSplits];
    __shared__ float s_inv;
    const int h = blockIdx.x, tid = threadIdx.x;
    const int S = n_splits(n_live(n_sel, n_sel_dev), s_max);
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

// The device's default (STRATA_GLMD_MLA_DEC unset or auto).  A CUDA build: wherever it fits (L1-4's default,
// unchanged).  A HIP build: only on the archs where it was measured faster than mla2's own kernels, in the bench and in
// GLM-5.3's decode - gfx1151 (Radeon 8060S / 8065S, Strix / Gorgon Halo; SUN-NOTES S7b) - and mla2's kernels on every
// other AMD arch until someone measures it there (it builds and fits on all of them: 55 KB of LDS a block).
// STRATA_GLMD_MLA_DEC=1 takes it on any device it fits; =0 keeps mla2's kernels everywhere (decode_mla's legacy_attn).
bool auto_device(int dev) {
#if defined(STRATA_USE_HIP)
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
    const int sw = strata::glmenv::parse(std::getenv("STRATA_GLMD_MLA_DEC"));
    c.on = c.hpw > 0 && (sw == strata::glmenv::kOn || auto_device(dev));
    return c;
}

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
