// src/kernels/cuda/glmd_volta_mla.cu - glm-dsa B10.2 fused sparse-gather MLA decode for sm_70 and the B10.7 native
// headwise GEMV (W_kb / W_vb in BF16 or Q8_0).  See include/strata/kernels/glmd_volta.hpp for the algorithm;
// glmd::vtwin::mla_fused (src/core/glm_dsa_cpu_volta_twin.cpp) is the same algorithm on the host.
//
// WMMA on Volta: m16n16k16 with FP16 inputs and FP32 accumulators.  The accumulator fragment's element-to-row map
// is not documented, so the kernel reads it once (load_matrix_sync of a matrix whose entries are their row index)
// and rescales O rows through it; the k-quarter partial sums of S are combined element-wise through shared memory,
// which is valid because every warp's fragment uses the same lane/element map.
//
// HIP builds compile the headwise GEMV (portable) and leave the fused attention out (nvcuda::wmma: GLMD_WMMA,
// glmd_arch.cuh, is 0 there, and the fused attention's kernels are not built): there fused_mla_enabled() is false,
// mla_fused() refuses and mla_auto() runs mla2.
#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "glmd_arch.cuh"   // STRATA_GRID_CONSTANT, GLMD_WMMA (Pascal pass)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if GLMD_WMMA
#include <mma.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace glmd {
namespace {

#if GLMD_WMMA
using namespace nvcuda;
#endif

std::atomic<int> g_errors{0};

void refuse(const char* what) {
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_volta %s: shape refused\n", what);
}
bool launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_volta %s: %s\n", what, cudaGetErrorString(e));
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
__device__ __forceinline__ void q8_1_store_warp(float v, Q81* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) {
        dst->d = __half_as_ushort(__float2half(d));
        dst->s = __half_as_ushort(__float2half(fmaxf(-65504.0f, fminf(65504.0f, sum))));
    }
}
__device__ __forceinline__ __half half_sat(float v) { return __float2half(fmaxf(-65504.0f, fminf(65504.0f, v))); }

// ---------------------------------------------------------------- headwise GEMV (BF16 / Q8_0)
// one row against x (K floats, 16-B aligned): lane-strided groups, then the butterfly (the twin's lane_tree)
template <int WT>
__device__ __forceinline__ float head_row_dot(const void* W, size_t row, const float* __restrict__ x, int K, int lane) {
    float s = 0.0f;
    if constexpr (WT == kWTypeBF16) {
        const uint4* r8 = (const uint4*) ((const uint16_t*) W + row * (size_t) K);
        const float4* x4 = (const float4*) x;
        for (int k = lane; k < K / 8; k += 32) {
            const uint4 w = r8[k];
            const float4 a = x4[2 * k], b = x4[2 * k + 1];
            float acc = __uint_as_float(w.x << 16) * a.x;
            acc += __uint_as_float(w.x & 0xffff0000u) * a.y;
            acc += __uint_as_float(w.y << 16) * a.z;
            acc += __uint_as_float(w.y & 0xffff0000u) * a.w;
            acc += __uint_as_float(w.z << 16) * b.x;
            acc += __uint_as_float(w.z & 0xffff0000u) * b.y;
            acc += __uint_as_float(w.w << 16) * b.z;
            acc += __uint_as_float(w.w & 0xffff0000u) * b.w;
            s += acc;
        }
    } else {
        const uint8_t* r = (const uint8_t*) W + row * (size_t) (K / 32) * 34;
        const float4* x4 = (const float4*) x;
        for (int k = lane; k < K / 4; k += 32) {
            const uint8_t* blk = r + (size_t) (k >> 3) * 34;
            const float d = __half2float(*(const __half*) blk);
            const uint16_t* q16 = (const uint16_t*) (blk + 2 + 4 * (k & 7));   // 2-byte aligned (34-byte blocks)
            const uint32_t qq = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);
            const float4 a = x4[k];
            float acc = (float) (int8_t) (qq & 0xff) * a.x;
            acc += (float) (int8_t) ((qq >> 8) & 0xff) * a.y;
            acc += (float) (int8_t) ((qq >> 16) & 0xff) * a.z;
            acc += (float) (int8_t) (qq >> 24) * a.w;
            s += d * acc;
        }
    }
    return warp_sum(s);
}

// out[h * R + r] (f32), out_h[h * out_h_stride + r] (FP16, saturated), q8 (q8_1 blocks of the f32 values); 32 rows
// per block (4 per warp), R % 32 == 0 so a block never straddles heads.  Blocks past the n_head * R / 32 main
// blocks fill the rest of the FP16 Q tile (tail): rows h < n_head get their rotated q_pe at [R, R + n_rot), rows
// n_head <= h < kFusedHeads are zero.
struct HwArgs {
    const void* W;
    const float* in;
    int R, K, in_stride, n_head;
    float* out;
    __half* out_h;
    int out_h_stride;
    Q81* q8;
    // tail (out_h only)
    const float* q;
    int q_stride, q_pe_off, n_rot;
};
template <int WT>
__global__ void __launch_bounds__(256) headwise_kernel(const STRATA_GRID_CONSTANT HwArgs a) {
    __shared__ float s_o[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_main = a.n_head * a.R / 32;
    if ((int) blockIdx.x >= n_main) {   // the Q tile's tail: one block per head row of the tile
        const int h = blockIdx.x - n_main;
        __half* row = a.out_h + (size_t) h * a.out_h_stride;
        if (h < a.n_head) {
            for (int j = threadIdx.x; j < a.n_rot; j += blockDim.x)
                row[a.R + j] = half_sat(a.q[(size_t) a.q_stride * h + a.q_pe_off + j]);
        } else {
            for (int j = threadIdx.x; j < a.R + a.n_rot; j += blockDim.x) row[j] = __float2half(0.0f);
        }
        return;
    }
    const int row0 = blockIdx.x * 32;
    const int h = row0 / a.R;
    const float* x = a.in + (size_t) h * a.in_stride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const float s = head_row_dot<WT>(a.W, (size_t) row, x, a.K, lane);
        if (lane == 0) s_o[warp * 4 + j] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = s_o[lane];
        const int row = row0 + lane;
        if (a.out) a.out[row] = v;
        if (a.out_h) a.out_h[(size_t) h * a.out_h_stride + (row - h * a.R)] = half_sat(v);
        if (a.q8) q8_1_store_warp(v, a.q8 + row0 / 32, lane);
    }
}

bool launch_headwise(const HwArgs& a, int w_type, int extra_blocks, cudaStream_t s) {
    const int blocks = a.n_head * a.R / 32 + extra_blocks;
    if (w_type == kWTypeQ8_0)
        headwise_kernel<kWTypeQ8_0><<<blocks, 256, 0, s>>>(a);
    else
        headwise_kernel<kWTypeBF16><<<blocks, 256, 0, s>>>(a);
    return launch_check("headwise");
}

#if !defined(STRATA_USE_HIP)   // (the fused attention is nvcuda::wmma, CUDA builds only: HIP runs mla2)
// ---------------------------------------------------------------- the fused attention
#if GLMD_WMMA
using FragAcc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using FragBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
static_assert(FragAcc::num_elements == 8, "the row map packs 8 accumulator elements of 4 bits");
#endif

template <int R>
__global__ void __launch_bounds__(kFusedThreads, 1)
    fused_attn_kernel(const __half* __restrict__ Qg, const void* __restrict__ lat, int lat_q8,
                      const uint16_t* __restrict__ k_pe, const int* __restrict__ cells, int n_sel, int cps, float scale,
                      float* __restrict__ part_o, float* __restrict__ part_ml) {
#if !GLMD_WMMA
    __trap();   // Pascal (no tensor cores): MAYA_GLMD_FUSED_MLA=1 there stops loudly instead of computing nothing
#else
    constexpr int L = kFusedLora, K = L + R, LDK = K + 8, KS = K / 16, HP = kFusedHeads, SUB = kFusedSub;
    constexpr int VPC = K / 8;   // uint4 per staged cell
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __half* sQ = (__half*) smem_raw;          // HP x LDK
    __half* sL = sQ + HP * LDK;               // SUB x LDK
    float* sS = (float*) (sL + SUB * LDK);    // HP x SUB scores (the k-quarter slots, then S, then P as FP16)
    __half* sP = (__half*) sS;
    __shared__ float s_alpha[HP];
    __shared__ int s_cell[SUB];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int mt = warp & 3, grp = warp >> 2;
    const int c0 = blockIdx.x * cps, c1 = min(n_sel, c0 + cps);

    // the accumulator row map: element i of this lane's fragment lies in row (rowmap >> 4i) & 15
    for (int i = tid; i < 256; i += blockDim.x) sS[i] = (float) (i >> 4);
    for (int i = tid; i < HP * VPC; i += blockDim.x) {   // the Q tile
        const int h = i / VPC, j = i - h * VPC;
        *(uint4*) (sQ + h * LDK + 8 * j) = ((const uint4*) Qg)[i];
    }
    __syncthreads();
    unsigned rowmap = 0;
    {
        FragAcc f;
        wmma::load_matrix_sync(f, sS, 16, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < 8; ++i) rowmap |= ((unsigned) f.x[i] & 15u) << (4 * i);
    }
    FragAcc o[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) wmma::fill_fragment(o[nt], 0.0f);
    float m = -INFINITY, l = 0.0f;   // the softmax state of head tid >> 3 (8 identical copies)
    const int sh = tid >> 3, sj = tid & 7;

    for (int st = c0; st < c1; st += SUB) {
        __syncthreads();   // the previous sub-tile's readers of sL / sP / s_alpha (and the row map read) are done
        if (tid < SUB) s_cell[tid] = st + tid < c1 ? cells[st + tid] : -1;
        __syncthreads();
        for (int i = tid; i < SUB * VPC; i += blockDim.x) {
            const int s = i / VPC, j = i - s * VPC;
            const int cell = s_cell[s];
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (cell >= 0) {
                if (j < L / 8) {
                    if (!lat_q8) {
                        v = ((const uint4*) lat)[(size_t) cell * (L / 8) + j];
                    } else {
                        const uint8_t* rec = (const uint8_t*) lat + (size_t) cell * (L + L / 16);
                        const uint2 c8 = *(const uint2*) (rec + 8 * j);
                        const float d = __half2float(((const __half*) (rec + L))[(8 * j) >> 5]);
                        __half hv[8];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            hv[e] = __float2half((float) (int8_t) ((c8.x >> (8 * e)) & 0xff) * d);
                            hv[4 + e] = __float2half((float) (int8_t) ((c8.y >> (8 * e)) & 0xff) * d);
                        }
                        v = *(const uint4*) hv;
                    }
                } else {
                    v = ((const uint4*) k_pe)[(size_t) cell * (R / 8) + (j - L / 8)];
                }
            }
            *(uint4*) (sL + s * LDK + 8 * j) = v;
        }
        __syncthreads();
        // S partial of (head tile mt, k-quarter grp)
        FragAcc sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int k = grp * KS / 4; k < (grp + 1) * KS / 4; ++k) {
            FragA fa;
            FragBc fb;
            wmma::load_matrix_sync(fa, sQ + mt * 16 * LDK + k * 16, LDK);
            wmma::load_matrix_sync(fb, sL + k * 16, LDK);
            wmma::mma_sync(sacc, fa, fb, sacc);
        }
        // S = p0 + (p1 + (p2 + p3)), element-wise through the slot of head tile mt
        float* slot = sS + mt * 256 + lane * 8;
        if (grp == 3) {
#pragma unroll
            for (int i = 0; i < 8; ++i) slot[i] = sacc.x[i];
        }
        __syncthreads();
        if (grp == 2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) slot[i] = sacc.x[i] + slot[i];
        }
        __syncthreads();
        if (grp == 1) {
#pragma unroll
            for (int i = 0; i < 8; ++i) slot[i] = sacc.x[i] + slot[i];
        }
        __syncthreads();
        if (grp == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) sacc.x[i] = sacc.x[i] + slot[i];
            __syncwarp();
            wmma::store_matrix_sync(sS + mt * 256, sacc, 16, wmma::mem_row_major);
        }
        __syncthreads();
        // online softmax: thread (sh, sj) takes cells 2 sj, 2 sj + 1 of head sh
        {
            const bool v0 = s_cell[2 * sj] >= 0, v1 = s_cell[2 * sj + 1] >= 0;
            const float x0 = v0 ? sS[sh * SUB + 2 * sj] * scale : -INFINITY;
            const float x1 = v1 ? sS[sh * SUB + 2 * sj + 1] * scale : -INFINITY;
            float tmax = fmaxf(x0, x1);
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 4));
            const float mn = fmaxf(m, tmax);
            float al, p0, p1;
            if (mn == -INFINITY) {
                al = 1.0f;
                p0 = p1 = 0.0f;
            } else {
                al = m == -INFINITY ? 0.0f : expf(m - mn);
                p0 = v0 ? __half2float(__float2half(expf(x0 - mn))) : 0.0f;
                p1 = v1 ? __half2float(__float2half(expf(x1 - mn))) : 0.0f;
            }
            float ts = p0 + p1;
            ts += __shfl_xor_sync(0xffffffffu, ts, 1);
            ts += __shfl_xor_sync(0xffffffffu, ts, 2);
            ts += __shfl_xor_sync(0xffffffffu, ts, 4);
            l = l * al + ts;
            m = mn;
            __syncthreads();   // every S read before P overwrites it
            sP[sh * SUB + 2 * sj] = __float2half(p0);
            sP[sh * SUB + 2 * sj + 1] = __float2half(p1);
            if (sj == 0) s_alpha[sh] = al;
            __syncthreads();
        }
        // O = alpha . O + P . L_ckv over this warp's 8 column tiles
        {
            FragA fa;
            wmma::load_matrix_sync(fa, sP + mt * 16 * SUB, SUB);
            float al[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) al[i] = s_alpha[mt * 16 + ((rowmap >> (4 * i)) & 15u)];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[nt].x[i] *= al[i];
                FragBr fb;
                wmma::load_matrix_sync(fb, sL + grp * 128 + nt * 16, LDK);
                wmma::mma_sync(o[nt], fa, fb, o[nt]);
            }
        }
    }
    float* po = part_o + ((size_t) blockIdx.x * HP + mt * 16) * L + grp * 128;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) wmma::store_matrix_sync(po + nt * 16, o[nt], L, wmma::mem_row_major);
    if (sj == 0) {
        part_ml[((size_t) blockIdx.x * HP + sh) * 2] = m;
        part_ml[((size_t) blockIdx.x * HP + sh) * 2 + 1] = l;
    }
#endif
}

// ctx[h] = sum_b exp(m_b - M) O_b / sum_b exp(m_b - M) l_b (mla2_combine_kernel's arithmetic)
__global__ void __launch_bounds__(256) fused_combine_kernel(const float* __restrict__ part_o,
                                                            const float* __restrict__ part_ml, int ns,
                                                            float* __restrict__ ctx) {
    constexpr int L = kFusedLora, HP = kFusedHeads;
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int b = 0; b < ns; ++b) M = fmaxf(M, part_ml[((size_t) b * HP + h) * 2]);
    float Ls = 0.0f;
    for (int b = 0; b < ns; ++b) {
        const float mb = part_ml[((size_t) b * HP + h) * 2];
        if (mb != -INFINITY) Ls += expf(mb - M) * part_ml[((size_t) b * HP + h) * 2 + 1];
    }
    const float inv = Ls > 0.0f ? 1.0f / Ls : 0.0f;
    for (int c = threadIdx.x; c < L; c += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < ns; ++b) {
            const float mb = part_ml[((size_t) b * HP + h) * 2];
            if (mb != -INFINITY) acc += expf(mb - M) * part_o[((size_t) b * HP + h) * L + c];
        }
        ctx[(size_t) h * L + c] = acc * inv;
    }
}
#endif  // !STRATA_USE_HIP (the fused attention)

// an on/off switch, default off ("auto" counts as on here: these dispatchers have no auto rule of their own)
bool env_on(const char* name) { return strata::glmenv::env_switch(name, strata::glmenv::kOff, true) != strata::glmenv::kOff; }

#if !defined(STRATA_USE_HIP)
// the attention kernel's dynamic shared memory above 48 KB, set once per device and instantiation
template <int R>
bool fused_smem_attr() {
    static std::atomic<unsigned> done{0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev >= 32) return false;
    if (done.load(std::memory_order_acquire) & (1u << dev)) return true;
    if (cudaFuncSetAttribute(fused_attn_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) mla_fused_smem_bytes(R)) != cudaSuccess)
        return false;
    done.fetch_or(1u << dev, std::memory_order_acq_rel);
    return true;
}
#endif

}  // namespace

int volta_idx_launch_errors();   // glmd_volta_idx.cu
int volta_launch_errors() { return g_errors.load(std::memory_order_relaxed) + volta_idx_launch_errors(); }

bool fused_mla_enabled() {
#if defined(STRATA_USE_HIP)
    return false;   // the fused attention is a WMMA kernel of CUDA builds; mla_auto runs mla2 here
#else
    static const bool on = env_on("MAYA_GLMD_FUSED_MLA");
    return on;
#endif
}

int fused_split_default() {
    static const int n = [] {
        const long long x = strata::glmenv::env_int("MAYA_GLMD_FUSED_SPLIT", 0);
        return x > 0 ? (int) std::min<long long>(x, 1 << 20) : kFusedSplitDefault;
    }();
    return n;
}

bool headwise(const void* W, int w_type, const float* in, int n_head, int R, int K, int in_stride, float* out,
              void* q8, void* stream) {
    if (!W || !in || n_head <= 0 || R <= 0 || R % 32 || K <= 0 || in_stride % 4 || (!out && !q8) ||
        head_row_bytes(w_type, K) == 0 || (w_type == kWTypeBF16 && K % 8)) {
        refuse("headwise");
        return false;
    }
    HwArgs a{};
    a.W = W;
    a.in = in;
    a.R = R;
    a.K = K;
    a.in_stride = in_stride;
    a.n_head = n_head;
    a.out = out;
    a.q8 = (Q81*) q8;
    return launch_headwise(a, w_type, 0, (cudaStream_t) stream);
}

bool mla_fused(const FusedMla& a, void* stream) {
#if defined(STRATA_USE_HIP)
    (void) a;
    (void) stream;
    refuse("mla_fused (a CUDA WMMA kernel: HIP builds run mla2)");
    return false;
#else
    cudaStream_t s = (cudaStream_t) stream;
    const int req = a.n_split > 0 ? a.n_split : fused_split_default();
    if (!mla_fused_supported(a.n_head, a.qk_nope, a.qk_rope, a.kv_lora, a.v_head, a.w_type) || a.n_sel < 0 ||
        !a.q || a.q_stride % 4 || a.q_stride < a.qk_nope + a.qk_rope || !a.wk_b || !a.wv_b || !a.lat ||
        (a.qk_rope > 0 && !a.k_pe) || (a.n_sel > 0 && !a.cells) || !a.work ||
        a.work_bytes < mla_fused_work_bytes(a.n_head, a.kv_lora, a.qk_rope, a.n_sel, req) ||
        (!a.out && !a.out_q8_1)) {
        refuse("mla_fused");
        return false;
    }
    const int R = a.qk_rope, K = kFusedLora + R, ns = mla_fused_splits(a.n_sel, req);
    auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    char* w = (char*) a.work;
    __half* Qh = (__half*) w;
    w += al((size_t) kFusedHeads * K * 2);
    float* part_o = (float*) w;
    w += al((size_t) std::max(1, ns) * kFusedHeads * kFusedLora * 4);
    float* part_ml = (float*) w;
    w += al((size_t) std::max(1, ns) * kFusedHeads * 2 * 4);
    float* ctx = (float*) w;
    if (ns > 0) {
        HwArgs h{};
        h.W = a.wk_b;
        h.in = a.q;
        h.R = kFusedLora;
        h.K = a.qk_nope;
        h.in_stride = a.q_stride;
        h.n_head = a.n_head;
        h.out_h = Qh;
        h.out_h_stride = K;
        h.q = a.q;
        h.q_stride = a.q_stride;
        h.q_pe_off = a.qk_nope;
        h.n_rot = R;
        if (!launch_headwise(h, a.w_type, kFusedHeads, s)) return false;
        const int cps = mla_fused_split_cells(a.n_sel, req);
        if (R == 64) {
            if (!fused_smem_attr<64>()) {
                refuse("mla_fused smem attribute");
                return false;
            }
            fused_attn_kernel<64><<<ns, kFusedThreads, mla_fused_smem_bytes(64), s>>>(
                Qh, a.lat, a.lat_q8, a.k_pe, a.cells, a.n_sel, cps, a.scale, part_o, part_ml);
        } else {
            if (!fused_smem_attr<0>()) {
                refuse("mla_fused smem attribute");
                return false;
            }
            fused_attn_kernel<0><<<ns, kFusedThreads, mla_fused_smem_bytes(0), s>>>(
                Qh, a.lat, a.lat_q8, a.k_pe, a.cells, a.n_sel, cps, a.scale, part_o, part_ml);
        }
        if (!launch_check("mla_fused attention")) return false;
    }
    fused_combine_kernel<<<a.n_head, 256, 0, s>>>(part_o, part_ml, ns, ctx);
    if (!launch_check("mla_fused combine")) return false;
    HwArgs v{};
    v.W = a.wv_b;
    v.in = ctx;
    v.R = a.v_head;
    v.K = kFusedLora;
    v.in_stride = kFusedLora;
    v.n_head = a.n_head;
    v.out = a.out;
    v.q8 = (Q81*) a.out_q8_1;
    return launch_headwise(v, a.w_type, 0, s);
#endif
}

size_t mla_auto_work_bytes(int n_head, int kv_lora, int qk_rope, int max_sel) {
    const size_t a = mla2_work_bytes(n_head, kv_lora, max_sel);
    if (!fused_mla_enabled()) return a;
    return std::max(a, mla_fused_work_bytes(n_head, kv_lora, qk_rope, max_sel, fused_split_default()));
}

bool mla_auto(const Mla2& a, void* stream) {
    if (fused_mla_enabled() &&
        mla_fused_supported(a.n_head, a.qk_nope, a.qk_rope, a.kv_lora, a.v_head, a.w_type) &&
        a.work_bytes >= mla_fused_work_bytes(a.n_head, a.kv_lora, a.qk_rope, a.n_sel, fused_split_default())) {
        FusedMla f;
        f.q = a.q;
        f.q_stride = a.q_stride;
        f.wk_b = a.wk_b;
        f.wv_b = a.wv_b;
        f.w_type = a.w_type;
        f.lat = a.lat;
        f.lat_q8 = a.lat_q8;
        f.k_pe = a.k_pe;
        f.cells = a.cells;
        f.n_sel = a.n_sel;
        f.n_head = a.n_head;
        f.qk_nope = a.qk_nope;
        f.qk_rope = a.qk_rope;
        f.kv_lora = a.kv_lora;
        f.v_head = a.v_head;
        f.scale = a.scale;
        f.work = a.work;
        f.work_bytes = a.work_bytes;
        f.out = a.out;
        f.out_q8_1 = a.out_q8_1;
        return mla_fused(f, stream);
    }
    return mla2(a, stream);
}

}  // namespace glmd
