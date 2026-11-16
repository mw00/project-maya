// src/kernels/cuda/glmd_batch.cu - the glm-dsa batched prompt kernels (glm-dsa B9.2-B9.5); see
// include/strata/kernels/glmd_batch.hpp.
//
// The per-row arithmetic of the prep and rope kernels is glmd_fast.cu's (the same expressions in the same order), so
// a row written here equals the token path's bytes.  The rope is glmd_rope.cuh: no --use_fast_math (the header
// refuses).  Every index that multiplies rows by keys, heads or widths is size_t (B14.6).
//
// THE MULTI-ROW SELECTOR (B9.3).  glmd::topk keeps, of a row's n candidates, those whose order-preserving key is
// above the top-th largest key, then the lowest-indexed ones equal to it: the top `top` by the total order
// (key descending, cell ascending).  topk_merge_kernel applies the same rule to the concatenation [running set |
// tile cells]: the running set holds cells below the tile's first cell, in ascending order, so index order in the
// concatenation IS cell order and "lower index wins a tie" is "lower cell wins".  Induction over the tiles: after
// tile j the running set is the top `top` (by that total order) of the row's causal cells below the end of tile j -
// each of those is among the top of its own tile's cells or of the earlier set, since a cell beaten by `top` others
// in a subset is beaten by them in the whole.  After the last tile it is the whole row's selection, in ascending
// cell order, exactly glmd::topk's output (and the identity 0..n-1 when n <= top).  The CPU gate is
// tests/glm_dsa/glmd_batch_cpu_test.cpp (twin::topk_tiled vs twin::topk on every row).
//
// HIP builds compile every kernel here; B9.5's nvcuda::wmma one is GLMD_WMMA's (glmd_arch.cuh: 0 on HIP, its body
// traps) and is never launched there: mla_prompt_wmma_ok() is false on HIP, so the prompt attention is the RDNA WMMA
// kernel on gfx11 / gfx12 (mla_prompt_rdna, S7) and the F32 kernel (mla_prompt_f32) on every other AMD device.
#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_f16_guard.hpp"

#include "strata/kernels/glmd_route.hpp"

#include "glmd_arch.cuh"   // STRATA_GRID_CONSTANT, GLMD_WMMA (Pascal pass)
#include "glmd_had.cuh"
#include "glmd_rope.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if GLMD_WMMA
#include <mma.h>
#endif

#include <algorithm>
#include <type_traits>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace glmd {
namespace {

std::atomic<int> g_errors{0};

bool refuse(const char* what) {
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_batch %s: shape refused\n", what);
    return false;
}

bool launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_batch %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ---------------------------------------------------------------- helpers (glmd_fast.cu's arithmetic)
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
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
// B14.1: d and the sum clamped to +-65504 before the half store (as glmd_fast.cu)
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
__device__ __forceinline__ int lat8_rec(int kv_lora) { return kv_lora + kv_lora / 16; }
__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half(v)); }
__device__ __forceinline__ float lat8_f(const uint8_t* rec, int kv_lora, int c) {
    return (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
}
__device__ __forceinline__ void lat8_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half d16 = __float2half(amax / 127.0f);
    const float d = __half2float(d16);
    const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0;
    ((int8_t*) rec)[c] = (int8_t) q;
    if (lane == 0) ((__half*) (rec + kv_lora))[c >> 5] = d16;
}
// one latent value of a cell as FP16 bits
__device__ __forceinline__ uint16_t lat_h(const void* lat, int lat_q8, int kv_lora, int64_t cell, int c) {
    return lat_q8 ? f2h(lat8_f((const uint8_t*) lat + (size_t) lat8_rec(kv_lora) * cell, kv_lora, c))
                  : ((const uint16_t*) lat)[(size_t) kv_lora * cell + c];
}
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- B9.2 prep for T rows
// glmd_fast.cu dsa_prep_kernel with blockIdx.y the row: the same code per row (bytes equal the token path's)
__global__ void __launch_bounds__(512) dsa_prep_rows_kernel(const STRATA_GRID_CONSTANT DsaPrepRows r) {
    __shared__ float sred[32];
    __shared__ float s_y[512];
    const DsaPrep& a = r.a;
    const int tid = threadIdx.x, lane = tid & 31, t = blockIdx.y;
    const int p = a.p_dev ? a.p_dev[t] : a.p + t;
    if (blockIdx.x == 0) {
        const float* qr_raw = a.qr_raw + (size_t) t * a.q_lora;
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            vals[n] = qr_raw[e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            if (a.qr) a.qr[(size_t) t * a.q_lora + e] = y;
            if (r.qr16) r.qr16[(size_t) t * a.q_lora + e] = f2h(y);
            if (a.qr_q) q8_1_store_warp(y, (Q81*) a.qr_q + (size_t) t * (a.q_lora / 32) + e / 32, lane);
        }
        return;
    }
    if (blockIdx.x == 1) {
        const float* kv_raw = a.kv_raw + (size_t) t * r.kv_ld;
        const float v = tid < a.kv_lora ? kv_raw[tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (tid < a.kv_lora) {
            const float y = v * inv * a.kv_norm[tid];
            if (a.lat_q8)
                lat8_store_warp(y, (uint8_t*) a.lat + (size_t) lat8_rec(a.kv_lora) * p, a.kv_lora, tid, lane);
            else
                ((uint16_t*) a.lat)[(size_t) a.kv_lora * p + tid] = f2h(y);
        }
        if (tid < a.qk_rope / 2) {
            float x0 = kv_raw[a.kv_lora + 2 * tid], x1 = kv_raw[a.kv_lora + 2 * tid + 1];
            float c, s;
            rope_cs(p, a.inv_freq[tid], c, s);
            rope_pair(x0, x1, c, s);
            uint16_t* kp = a.k_pe + (size_t) a.qk_rope * p;
            kp[2 * tid] = f2h(x0);
            kp[2 * tid + 1] = f2h(x1);
        }
        return;
    }
    const int K = a.idx_dim;
    const float* ik_raw = a.ik_raw + (size_t) t * K;
    const float v = tid < K ? ik_raw[tid] : 0.0f;
    const float mu = block_sum(v, sred) / (float) K;
    const float d = tid < K ? v - mu : 0.0f;
    const float var = block_sum(d * d, sred) / (float) K;
    const float inv = rsqrtf(var + a.idx_ln_eps);
    if (tid < K) s_y[tid] = d * inv * a.ik_ln_w[tid] + a.ik_ln_b[tid];
    __syncthreads();
    if (tid < a.idx_rope / 2) {
        float c, s;
        rope_cs(p, a.idx_inv_freq[tid], c, s);
        rope_pair(s_y[2 * tid], s_y[2 * tid + 1], c, s);
    }
    __syncthreads();
    if (a.ik_had) had_block(s_y, K);   // B14.11: the rotated key, as the token path's dsa_prep
    if (tid < K) {
        if (a.ik_f16)   // B10.8: FP16 index keys (the token path's dsa_prep writes the same bits)
            ((uint16_t*) a.ik_cache)[(size_t) K * p + tid] = f2h(s_y[tid]);
        else
            a.ik_cache[(size_t) K * p + tid] = s_y[tid];
    }
}

__global__ void __launch_bounds__(256) dsa_rope_q_rows_kernel(const STRATA_GRID_CONSTANT DsaRopeQRows r) {
    const DsaRopeQ& a = r.a;
    const int t = blockIdx.y;
    const int p = a.p_dev ? a.p_dev[t] : a.p + t;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nq = a.n_head * (a.n_rot / 2);
    float* x;
    float f;
    if (i < nq) {
        const int h = i / (a.n_rot / 2), j = i - h * (a.n_rot / 2);
        x = a.q + (size_t) t * r.q_row + (size_t) a.q_stride * h + a.rot_off + 2 * j;
        f = a.inv_freq[j];
    } else {
        const int k = i - nq;
        if (a.iq == nullptr || k >= a.idx_heads * (a.idx_rope / 2)) return;
        const int h = k / (a.idx_rope / 2), j = k - h * (a.idx_rope / 2);
        x = a.iq + (size_t) t * r.iq_row + (size_t) a.idx_dim * h + 2 * j;
        f = a.idx_inv_freq[j];
    }
    float c, s;
    rope_cs(p, f, c, s);
    rope_pair(x[0], x[1], c, s);
}

// ---------------------------------------------------------------- B9.3 score tiles
// a block: 16 queries x 64 keys of the tile; keys staged once, the queries head by head
constexpr int kScQ = 16, kScK = 64, kScMaxDim = 128;
__device__ __forceinline__ float sc_key(const float* ik, size_t i, std::false_type) { return ik[i]; }
__device__ __forceinline__ float sc_key(const float* ik, size_t i, std::true_type) {
    return __half2float(__ushort_as_half(((const unsigned short*) ik)[i]));
}
// KF16: FP16 keys (B10.8), widened to f32 as they are staged (the products and sums stay f32, as the token scorer's)
template <bool KF16>
__global__ void __launch_bounds__(256) index_score_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                                          const float* __restrict__ ik, int H, int D, int T, int64_t k0,
                                                          int nk, float* __restrict__ score, int64_t ld) {
    __shared__ float sK[kScK][kScMaxDim + 1];
    __shared__ float sQ[kScQ][kScMaxDim];
    const int tid = threadIdx.x, tq = tid / 16, tk = tid % 16;
    const int q0 = blockIdx.y * kScQ, kb = blockIdx.x * kScK;
    for (int i = tid; i < kScK * D; i += blockDim.x) {
        const int j = i / D, e = i - j * D;
        sK[j][e] = kb + j < nk ? sc_key(ik, (size_t) (k0 + kb + j) * D + e, std::integral_constant<bool, KF16>()) : 0.0f;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int t = q0 + tq;
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        for (int i = tid; i < kScQ * D; i += blockDim.x) {
            const int qq = i / D, e = i - qq * D;
            sQ[qq][e] = q0 + qq < T ? iq[((size_t) (q0 + qq) * H + h) * D + e] : 0.0f;
        }
        __syncthreads();
        const float wh = t < T ? w[(size_t) t * H + h] : 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = tk + 16 * j;
            float d = 0.0f;
            for (int e = 0; e < D; ++e) d += sQ[tq][e] * sK[key][e];
            acc[j] += wh * fmaxf(d, 0.0f);
        }
    }
    if (t >= T) return;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int key = kb + tk + 16 * j;
        if (key < nk) score[(size_t) t * ld + key] = acc[j];
    }
}

// ---------------------------------------------------------------- the score tile on tensor cores
// index_score_tile_wmma: the same scores from FP16 queries and keys with F32 accumulation (WMMA 16 x 16 x 16), a
// block per 64 queries x 128 keys: the keys staged once ([8 k-blocks][128][16] FP16), the queries head by head
// (double buffered: the next head is read into registers while the current one is multiplied), each head's 64 x 128
// dot products relu'd, weighted and summed into F32 score fragments in head order (each element's row and column
// from probed fragments, as the H64 attention kernel).  B15.1: a (query, head) row whose largest magnitude passes
// 65504 is stored divided by a power of two qs and its weight multiplied by qs (relu commutes with a positive scale).
// nvcuda::wmma: CUDA builds only (HIP: index_score_wmma_fits is false there and the F32 scorer runs), and in CUDA the
// host pass and sm_70+ device passes only (<mma.h> has no nvcuda for a Pascal pass, where the kernel body traps - it is
// never launched there: the device cannot give its 80 KB of shared memory).
#ifndef STRATA_GRID_CONSTANT   // (glm_fast.cu's spelling: __grid_constant__ is compute_70+ only)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define STRATA_GRID_CONSTANT __grid_constant__
#else
#define STRATA_GRID_CONSTANT
#endif
#endif
#ifndef GLMD_WMMA
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define GLMD_WMMA 1
#else
#define GLMD_WMMA 0
#endif
#endif
namespace sw {
constexpr int kQ = 64, kN = 128, kD = 128;
constexpr size_t kKeyBytes = (size_t) 8 * kN * 16 * 2;   // 32,768
constexpr size_t kQBytes = (size_t) 8 * kQ * 16 * 2;     // 16,384 a buffer
constexpr size_t smem_bytes(int H) { return kKeyBytes + 2 * kQBytes + 2 * (size_t) kQ * 4 + (size_t) kQ * H * 4; }
}  // namespace sw
#if !defined(STRATA_USE_HIP)
namespace sw {
constexpr int kThreads = 256;
}  // namespace sw
#if GLMD_WMMA
namespace wm = nvcuda::wmma;
namespace sw {
using AccF = wm::fragment<wm::accumulator, 16, 16, 16, float>;
static_assert(AccF::num_elements == 8, "the row / column maps pack 8 elements x 4 bits");

// this thread's share of one head's query tile: query 8 warp + lane / 4, dims 16 kb + 4 (lane % 4), kb = 0..7
__device__ __forceinline__ void q_load(const float* iq, int H, int T, int q0, int h, int tid, float4 (&v)[8]) {
    const int q = q0 + (tid >> 5) * 8 + ((tid & 31) >> 2), e = 4 * (tid & 3);
#pragma unroll
    for (int kb = 0; kb < 8; ++kb)
        v[kb] = q < T ? *(const float4*) (iq + ((size_t) q * H + h) * kD + kb * 16 + e) : make_float4(0.f, 0.f, 0.f, 0.f);
}
__device__ __forceinline__ void q_store(const float4 (&v)[8], int tid, __half* sQ, float* sQs) {
    const int ql = (tid >> 5) * 8 + ((tid & 31) >> 2), e = 4 * (tid & 3);
    float m = 0.0f;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) m = fmaxf(m, fmaxf(fmaxf(fabsf(v[kb].x), fabsf(v[kb].y)), fmaxf(fabsf(v[kb].z), fabsf(v[kb].w))));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));   // the four lanes of this query hold its 128 values
    const float qs = glmd::f16_guard_scale(m), inv = 1.0f / qs;
#pragma unroll
    for (int kb = 0; kb < 8; ++kb) {
        __half2 h2[2];
        if (qs == 1.0f) {
            h2[0] = __floats2half2_rn(v[kb].x, v[kb].y);
            h2[1] = __floats2half2_rn(v[kb].z, v[kb].w);
        } else {
            h2[0] = __floats2half2_rn(v[kb].x * inv, v[kb].y * inv);
            h2[1] = __floats2half2_rn(v[kb].z * inv, v[kb].w * inv);
        }
        *(uint2*) (sQ + ((size_t) kb * kQ + ql) * 16 + e) = *(uint2*) h2;
    }
    if ((tid & 3) == 0) sQs[ql] = qs;
}
}  // namespace sw
#endif  // GLMD_WMMA

template <bool KF16>
__global__ void __launch_bounds__(sw::kThreads, 1) index_score_wmma_kernel(const float* __restrict__ iq,
                                                                           const float* __restrict__ w,
                                                                           const void* __restrict__ keys, int H, int T,
                                                                           int64_t k0, int nk, float* __restrict__ score,
                                                                           int64_t ld) {
#if !GLMD_WMMA
    __trap();   // a Pascal device pass: never launched (index_score_wmma_fits refuses the device)
#else
    using namespace sw;
    extern __shared__ __align__(128) unsigned char smem_s[];
    __half* sKey = (__half*) smem_s;
    __half* sQb[2] = {(__half*) (smem_s + kKeyBytes), (__half*) (smem_s + kKeyBytes + kQBytes)};
    float* sQs = (float*) (smem_s + kKeyBytes + 2 * kQBytes);   // [2][64]
    float* sW = sQs + 2 * kQ;                                    // [64][H]
    const int tid = threadIdx.x, warp = tid >> 5, rb = warp & 3, ct0 = 4 * (warp >> 2);
    const int q0 = blockIdx.y * kQ, n0 = blockIdx.x * kN;
    // the accumulator element map (rows and columns of a 16 x 16 tile), through sQb[0] before the queries land there
    float* probe = (float*) sQb[0];
    probe[tid] = (float) (tid >> 4);
    probe[256 + tid] = (float) (tid & 15);
    // keys: [8][128][16] FP16 (zeros past nk); a warp writes 256 contiguous bytes
    for (int u = tid; u < 8 * kN * 4; u += kThreads) {
        const int kb = u >> 9, key = (u >> 2) & (kN - 1), part = u & 3;
        uint2 v = make_uint2(0u, 0u);
        if ((int64_t) n0 + key < nk) {
            const size_t off = (size_t) (k0 + n0 + key) * kD + kb * 16 + part * 4;
            if constexpr (KF16) {
                v = *(const uint2*) ((const uint16_t*) keys + off);
            } else {   // F32 keys (the default layout), saturated to the FP16 range (LayerNorm outputs: far inside it)
                const float4 f = *(const float4*) ((const float*) keys + off);
                const float c = 65504.0f;
                __half2 h2[2] = {__floats2half2_rn(fminf(fmaxf(f.x, -c), c), fminf(fmaxf(f.y, -c), c)),
                                 __floats2half2_rn(fminf(fmaxf(f.z, -c), c), fminf(fmaxf(f.w, -c), c))};
                v = *(uint2*) h2;
            }
        }
        *(uint2*) (sKey + ((size_t) kb * kN + key) * 16 + part * 4) = v;
    }
    for (int i = tid; i < kQ * H; i += kThreads) {
        const int q = i / H;
        sW[i] = q0 + q < T ? w[(size_t) (q0 + q) * H + (i - q * H)] : 0.0f;
    }
    __syncthreads();
    unsigned rowmap = 0u, colmap = 0u;
    {
        AccF rf, cf;
        wm::load_matrix_sync(rf, probe, 16, wm::mem_row_major);
        wm::load_matrix_sync(cf, probe + 256, 16, wm::mem_row_major);
#pragma unroll
        for (int i = 0; i < AccF::num_elements; ++i) {
            rowmap |= ((unsigned) rf.x[i] & 15u) << (4 * i);
            colmap |= ((unsigned) cf.x[i] & 15u) << (4 * i);
        }
    }
    float4 qv[8];
    q_load(iq, H, T, q0, 0, tid, qv);
    __syncthreads();   // the probe is read
    q_store(qv, tid, sQb[0], sQs);
    __syncthreads();
    AccF acc[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) wm::fill_fragment(acc[c], 0.0f);
    for (int h = 0; h < H; ++h) {
        const int b = h & 1;
        if (h + 1 < H) q_load(iq, H, T, q0, h + 1, tid, qv);
        AccF s[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) wm::fill_fragment(s[c], 0.0f);
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> qa;
            wm::load_matrix_sync(qa, sQb[b] + ((size_t) kb * kQ + rb * 16) * 16, 16);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> kf;
                wm::load_matrix_sync(kf, sKey + ((size_t) kb * kN + (ct0 + c) * 16) * 16, 16);
                wm::mma_sync(s[c], qa, kf, s[c]);
            }
        }
        float wq[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rb * 16 + ((rowmap >> (4 * i)) & 15u);
            wq[i] = sW[r * H + h] * sQs[b * kQ + r];
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[c].x[i] += wq[i] * fmaxf(s[c].x[i], 0.0f);
        if (h + 1 < H) q_store(qv, tid, sQb[b ^ 1], sQs + (b ^ 1) * kQ);
        __syncthreads();
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int64_t q = q0 + rb * 16 + ((rowmap >> (4 * i)) & 15u);
            const int64_t j = (int64_t) n0 + (ct0 + c) * 16 + ((colmap >> (4 * i)) & 15u);
            if (q < T && j < nk) score[(size_t) q * ld + j] = acc[c].x[i];
        }
#endif  // GLMD_WMMA
}
#endif  // !STRATA_USE_HIP (the WMMA score tile)

// ---------------------------------------------------------------- B9.3 the multi-row merge (one row a block)
constexpr int kSelThreads = 1024;
__device__ __forceinline__ void scan_incl(int* s, int tid) {
    for (int off = 1; off < kSelThreads; off <<= 1) {
        const int a = tid >= off ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += a;
        __syncthreads();
    }
}
__global__ void __launch_bounds__(kSelThreads) topk_merge_kernel(const STRATA_GRID_CONSTANT TopkMerge m) {
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_eq[kSelThreads], s_w[kSelThreads];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int64_t pos = m.p0 + t;
    int64_t nt64 = pos + 1 - m.k0;
    nt64 = nt64 < 0 ? 0 : (nt64 > m.nk ? m.nk : nt64);
    const int nt = (int) nt64;
    const int r = m.in_n ? m.in_n[t] : 0;
    const int n = r + nt;
    const size_t ro = (size_t) t * m.top_k;
    const float* tile = m.score + (size_t) t * m.ld;
    const int* icell = m.in_cell ? m.in_cell + ro : nullptr;
    const float* iscore = m.in_score ? m.in_score + ro : nullptr;
    int* oc = m.out_cell + ro;
    float* os = m.out_score + ro;
    const auto sc = [&](int i) { return i < r ? iscore[i] : tile[i - r]; };
    const auto cell = [&](int i) { return i < r ? icell[i] : (int) (m.k0 + (i - r)); };
    if (n <= m.top_k) {
        for (int i = tid; i < m.top_k; i += kSelThreads) {
            oc[i] = i < n ? cell(i) : -1;
            os[i] = i < n ? sc(i) : 0.0f;
        }
        if (tid == 0) m.out_n[t] = n;
        return;
    }
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = m.top_k;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n; i += kSelThreads) {
            const unsigned k = score_key(sc(i));
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need;
            for (int b = 255; b >= 0; --b) {
                const int c = (int) s_hist[b];
                if (c >= need) {
                    s_prefix = prefix | ((unsigned) b << shift);
                    s_mask = mask | (255u << shift);
                    s_need = need;
                    break;
                }
                need -= c;
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    const int chunk = (n + kSelThreads - 1) / kSelThreads;
    const int lo = min(n, tid * chunk), hi = min(n, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc(i));
        gt += k > cut;
        eq += k == cut;
    }
    s_eq[tid] = eq;
    __syncthreads();
    scan_incl(s_eq, tid);
    const int eq_before = s_eq[tid] - eq;
    const int take_eq = max(0, min(eq, need_eq - eq_before));
    s_w[tid] = gt + take_eq;
    __syncthreads();
    scan_incl(s_w, tid);
    int o = s_w[tid] - (gt + take_eq);
    int eq_seen = eq_before;
    for (int i = lo; i < hi; ++i) {
        const float v = sc(i);
        const unsigned k = score_key(v);
        bool take = k > cut;
        if (k == cut) take = eq_seen++ < need_eq;
        if (take) {
            oc[o] = cell(i);
            os[o++] = v;
        }
    }
    if (tid == 0) m.out_n[t] = m.top_k;   // (n > top_k: exactly top_k taken; nothing to pad)
}

__global__ void sel_mask_kernel(const int* __restrict__ sel, int top, int T, int64_t n_kv, uint32_t* __restrict__ mask,
                                int64_t words) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * top) return;
    const int t = (int) (i / top);
    const int c = sel[i];
    if (c < 0 || c >= n_kv) return;
    atomicOr(mask + (size_t) t * words + c / 32, 1u << (c % 32));
}

// ---------------------------------------------------------------- B9.4 / B9.5 the per-query cell walk
// The block's chunk of 32 cells starting at slot c0: s_cell[i] = the cell or -1.  Returns false when the walk is over.
// Every thread calls it (block-uniform result).
__device__ __forceinline__ bool walk_chunk(const MlaPrompt& a, int t, int64_t& c0, int* s_cell, int* s_any) {
    const int64_t pos = a.p0 + t;
    const int tid = threadIdx.x;
    for (;;) {
        if (a.walk == AttnWalk::Gather) {
            if (c0 >= a.top_k) return false;
            const int* row = a.sel + (size_t) t * a.top_k;
            if (row[c0] < 0) return false;   // ascending, then -1 padding
            if (tid < 32) {
                const int64_t i = c0 + tid;
                const int c = i < a.top_k ? row[i] : -1;
                s_cell[tid] = c >= 0 && c <= pos ? c : -1;
            }
            __syncthreads();
            c0 += 32;
            return true;
        }
        if (c0 > pos) return false;
        uint32_t word = 0xffffffffu;
        if (a.walk == AttnWalk::DenseMasked) word = a.mask[(size_t) t * a.mask_words + c0 / 32];
        if (word == 0u) {   // (block-uniform: every thread read the same word)
            c0 += 32;
            continue;
        }
        if (tid < 32) {
            const int64_t c = c0 + tid;
            s_cell[tid] = c <= pos && ((word >> tid) & 1u) ? (int) c : -1;
        }
        (void) s_any;
        __syncthreads();
        c0 += 32;
        return true;
    }
}

// F32 arithmetic over the FP16 / INT8 cache: a block per (row, 16 heads), a warp per 2 heads, online softmax
template <int R>
__global__ void __launch_bounds__(256) mla_prompt_f32_kernel(const STRATA_GRID_CONSTANT MlaPrompt a) {
    constexpr int CH = 32, HPB = 16;
    extern __shared__ uint16_t smem_f32[];
    uint16_t* sL = smem_f32;              // CH x kv_lora
    uint16_t* sP = sL + CH * a.kv_lora;   // CH x R
    __shared__ int s_cell[CH];
    __shared__ int s_any;
    const int t = blockIdx.x, h0 = blockIdx.y * HPB, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int K = a.kv_lora, NJ = K / 32;
    float qa[2][16], qp[2][R > 0 ? R / 32 : 1], acc[2][16], m[2], l[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int h = h0 + warp * 2 + u;
        const float* qh = a.q_abs + ((size_t) t * a.n_head + h) * K;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            qa[u][j] = j < NJ ? qh[lane + 32 * j] : 0.0f;
            acc[u][j] = 0.0f;
        }
        if constexpr (R > 0) {
            const float* qph = a.q + ((size_t) t * a.n_head + h) * a.q_stride + a.q_pe_off;
#pragma unroll
            for (int j = 0; j < R / 32; ++j) qp[u][j] = qph[lane + 32 * j];
        }
        m[u] = -INFINITY;
        l[u] = 0.0f;
    }
    int64_t c0 = 0;
    for (;;) {
        __syncthreads();   // the previous chunk's readers are done
        if (!walk_chunk(a, t, c0, s_cell, &s_any)) break;
        for (int i = tid; i < CH * K; i += blockDim.x) {
            const int s = i / K, c = i - s * K;
            const int cell = s_cell[s];
            sL[i] = cell < 0 ? (uint16_t) 0 : lat_h(a.lat, a.lat_q8, K, cell, c);
        }
        if constexpr (R > 0)
            for (int i = tid; i < CH * R; i += blockDim.x) {
                const int s = i / R, j = i - s * R;
                const int cell = s_cell[s];
                sP[i] = cell < 0 ? (uint16_t) 0 : a.k_pe[(size_t) R * cell + j];
            }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            float mine = -INFINITY;   // lane s keeps logit s
            for (int s = 0; s < CH; ++s) {
                float d = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < NJ) d += qa[u][j] * h2f(sL[s * K + lane + 32 * j]);
                if constexpr (R > 0) {
#pragma unroll
                    for (int j = 0; j < R / 32; ++j) d += qp[u][j] * h2f(sP[s * R + lane + 32 * j]);
                }
                d = warp_sum(d);
                if (lane == s) mine = s_cell[s] >= 0 ? d * a.scale : -INFINITY;
            }
            const float cm = warp_max(mine);
            const float mn = fmaxf(m[u], cm);
            if (mn == -INFINITY) continue;   // nothing valid yet
            const float alpha = m[u] == -INFINITY ? 0.0f : expf(m[u] - mn);
            const float p = mine == -INFINITY ? 0.0f : expf(mine - mn);
            l[u] = l[u] * alpha + warp_sum(p);
            m[u] = mn;
#pragma unroll
            for (int j = 0; j < 16; ++j) acc[u][j] *= alpha;
            for (int s = 0; s < CH; ++s) {
                const float ps = __shfl_sync(0xffffffffu, p, s);
                if (ps == 0.0f) continue;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < NJ) acc[u][j] += ps * h2f(sL[s * K + lane + 32 * j]);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int h = h0 + warp * 2 + u;
        const float inv = l[u] > 0.0f ? 1.0f / l[u] : 0.0f;
        float* out = a.ctx + ((size_t) t * a.n_head + h) * K;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (j < NJ) out[lane + 32 * j] = acc[u][j] * inv;
    }
}

// ---- B9.5: WMMA (FP16 in, F32 accumulate) at width 576, O in shared memory
#if GLMD_WMMA
namespace wm = nvcuda::wmma;
#endif
constexpr int kWmK = 512, kWmW = 576, kWmCH = 32, kWmH = 16;
constexpr size_t kWmQ = (size_t) kWmH * kWmW * 2;     // sQ   18,432
constexpr size_t kWmKt = (size_t) kWmCH * kWmW * 2;   // sK   36,864
constexpr size_t kWmO = (size_t) kWmH * kWmK * 4;     // sO   32,768
constexpr size_t kWmS = (size_t) kWmH * kWmCH * 4;    // sS    2,048
constexpr size_t kWmP = (size_t) kWmH * kWmCH * 2;    // sP    1,024
constexpr size_t kWmML = (size_t) 3 * kWmH * 4;       // m, l, alpha   192
constexpr size_t kWmCell = (size_t) kWmCH * 4;        // cells   128
constexpr size_t kWmSmem = kWmQ + kWmKt + kWmO + kWmS + kWmP + kWmML + kWmCell;
static_assert(kWmSmem == kMlaWmmaSmem, "the header's constant is the kernel's dynamic shared memory");
static_assert(kWmSmem <= 98304, "V100: at most 96 KB of shared memory per block (opt-in)");
static_assert(kWmQ % 32 == 0 && kWmKt % 32 == 0 && kWmO % 32 == 0 && kWmS % 32 == 0, "WMMA tiles 256-bit aligned");

template <int R>
__global__ void __launch_bounds__(256) mla_prompt_wmma_kernel(const STRATA_GRID_CONSTANT MlaPrompt a) {
#if !GLMD_WMMA
    __trap();   // Pascal (no tensor cores): never launched - mla_prompt_wmma_ok() refuses its 89 KB (48 KB a block there)
#else
    extern __shared__ __align__(128) unsigned char smem_wm[];
    __half* sQ = (__half*) smem_wm;
    __half* sK = (__half*) (smem_wm + kWmQ);
    float* sO = (float*) (smem_wm + kWmQ + kWmKt);
    float* sS = (float*) (smem_wm + kWmQ + kWmKt + kWmO);
    __half* sPm = (__half*) (smem_wm + kWmQ + kWmKt + kWmO + kWmS);
    float* sM = (float*) (smem_wm + kWmQ + kWmKt + kWmO + kWmS + kWmP);
    float* sL = sM + kWmH;
    float* sA = sL + kWmH;
    int* s_cell = (int*) (sA + kWmH);
    __shared__ int s_any;
    __shared__ float s_qmax[8], s_qs;
    constexpr int KW = kWmK + R;   // the contraction width of S = Q K^T
    const int t = blockIdx.x, h0 = blockIdx.y * kWmH, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // Q (scale folded in: smaller FP16 magnitudes) and the zeroed accumulators
    float qmax = 0.0f;
    for (int i = tid; i < kWmH * kWmW; i += blockDim.x) {
        const int hh = i / kWmW, c = i - hh * kWmW;
        const size_t row = (size_t) t * a.n_head + h0 + hh;
        float v = 0.0f;
        if (c < kWmK) v = a.q_abs[row * kWmK + c];
        else if (c < KW) v = a.q[row * a.q_stride + a.q_pe_off + (c - kWmK)];
        sQ[i] = __float2half(v * a.scale);
        qmax = fmaxf(qmax, fabsf(v * a.scale));
    }
    // B15.1: the FP16 range guard of the Q tile (the block's 16 heads of row t): past 65504 the tile is stored divided
    // by a power of two qs and the scores are multiplied back (qs = 1, the normal case, changes no bit)
    qmax = warp_max(qmax);
    if (lane == 0) s_qmax[warp] = qmax;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) m = fmaxf(m, s_qmax[w]);
        s_qs = glmd::f16_guard_scale(m);
    }
    __syncthreads();
    const float qs = s_qs;
    if (qs != 1.0f) {
        const float inv = 1.0f / qs;
        for (int i = tid; i < kWmH * kWmW; i += blockDim.x) {
            const int hh = i / kWmW, c = i - hh * kWmW;
            const size_t row = (size_t) t * a.n_head + h0 + hh;
            float v = 0.0f;
            if (c < kWmK) v = a.q_abs[row * kWmK + c];
            else if (c < KW) v = a.q[row * a.q_stride + a.q_pe_off + (c - kWmK)];
            sQ[i] = __float2half((v * a.scale) * inv);
        }
    }
    for (int i = tid; i < kWmH * kWmK; i += blockDim.x) sO[i] = 0.0f;
    if (tid < kWmH) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.0f;
    }
    int64_t c0 = 0;
    for (;;) {
        __syncthreads();
        if (!walk_chunk(a, t, c0, s_cell, &s_any)) break;
        for (int i = tid; i < kWmCH * kWmW; i += blockDim.x) {
            const int s = i / kWmW, c = i - s * kWmW;
            const int cell = s_cell[s];
            uint16_t v = 0;
            if (cell >= 0) {
                if (c < kWmK) v = lat_h(a.lat, a.lat_q8, kWmK, cell, c);
                else if (c < KW) v = a.k_pe[(size_t) R * cell + (c - kWmK)];
            }
            sK[i] = __ushort_as_half(v);
        }
        __syncthreads();
        // S (16 heads x 32 cells) = Q K^T: warps 0 and 1, one 16-cell tile each
        if (warp < 2) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> sf;
            wm::fill_fragment(sf, 0.0f);
            for (int k = 0; k < KW; k += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> qf;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> kf;
                wm::load_matrix_sync(qf, sQ + k, kWmW);
                wm::load_matrix_sync(kf, sK + (size_t) warp * 16 * kWmW + k, kWmW);
                wm::mma_sync(sf, qf, kf, sf);
            }
            wm::store_matrix_sync(sS + warp * 16, sf, kWmCH, wm::mem_row_major);
        }
        __syncthreads();
        // online softmax: a warp per 2 heads, lane = cell; f16 P (each <= 1), f32 m / l
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int r = warp * 2 + u;
            const float v = s_cell[lane] >= 0 ? sS[r * kWmCH + lane] * qs : -INFINITY;   // (B15.1: qs 1 is exact)
            const float cm = warp_max(v);
            const float mo = sM[r];
            const float mn = fmaxf(mo, cm);
            float p = 0.0f, alpha = 1.0f;
            if (mn != -INFINITY) {
                alpha = mo == -INFINITY ? 0.0f : expf(mo - mn);
                p = v == -INFINITY ? 0.0f : expf(v - mn);
            }
            const float ps = warp_sum(p);
            sPm[r * kWmCH + lane] = __float2half(p);
            __syncwarp();
            if (lane == 0) {
                sL[r] = sL[r] * alpha + ps;
                sM[r] = mn;
                sA[r] = alpha;
            }
        }
        __syncthreads();
        for (int i = tid; i < kWmH * kWmK; i += blockDim.x) sO[i] *= sA[i / kWmK];
        __syncthreads();
        // O (16 x 512) += P (16 x 32) V (32 x 512): 32 tiles of 16 columns, 4 per warp
        for (int nt = warp * 4; nt < warp * 4 + 4; ++nt) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> of;
            wm::load_matrix_sync(of, sO + nt * 16, kWmK, wm::mem_row_major);
#pragma unroll
            for (int kk = 0; kk < kWmCH; kk += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> pf;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> vf;
                wm::load_matrix_sync(pf, sPm + kk, kWmCH);
                wm::load_matrix_sync(vf, sK + (size_t) kk * kWmW + nt * 16, kWmW);
                wm::mma_sync(of, pf, vf, of);
            }
            wm::store_matrix_sync(sO + nt * 16, of, kWmK, wm::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < kWmH * kWmK; i += blockDim.x) {
        const int hh = i / kWmK, c = i - hh * kWmK;
        const float inv = sL[hh] > 0.0f ? 1.0f / sL[hh] : 0.0f;
        a.ctx[((size_t) t * a.n_head + h0 + hh) * kWmK + c] = sO[i] * inv;
    }
#endif
}

// ---- H64: B9.5's arithmetic, one block per (query row, 64 heads).  The (row, 16 heads) kernel above stages every
// cache tile (INT8 decoded) once per 16 heads and keeps O in shared memory; here a 16-cell tile is staged once for
// all 64 heads (M = 64), O (64 x 512 F32) lives in WMMA accumulators (warp w owns value columns [64 w, 64 w + 64):
// 4 row blocks x 4 column tiles), and the next tile's cells are read into registers while the current one is
// multiplied.  Shared memory, every WMMA operand a contiguous 16 x 16 block (ldm 16):
//   sQ  [KB][64][16]  FP16 Q, the scale folded in (B15.1 guard as above)      73,728 B at width 576
//   sK  [KB][16][16]  the tile's 16 cells, FP16 (the latent then k_pe)        18,432 B
//   sS  [64][16] F32  S; P (f16) is written over the first half of its row      4,096 B
//   m / l / alpha per head, the tile's cells                                       832 B
// S = Q K^T: warps 0-3, a row block each (two accumulator chains); the online softmax: 4 threads per head row;
// O += P V: every warp.  The rescale by alpha and the final 1/l need each accumulator element's row: read once per
// block by loading a tile of row numbers into a fragment (the fragment layout is the hardware's, never assumed).
// nvcuda::wmma: CUDA builds only (HIP: mla_prompt_h64_fits is false there), the body in the host pass and sm_70+ device
// passes (a Pascal pass traps: never launched, mla_prompt_h64_fits refuses the device's 48 KB).
namespace h64 {
constexpr int kH = 64, kNC = 16;
constexpr size_t smem_bytes(int R) {
    return (size_t) kH * (512 + R) * 2 + (size_t) kNC * (512 + R) * 2 + (size_t) kH * kNC * 4 + (size_t) 3 * kH * 4 +
           (size_t) kNC * 4;
}
}  // namespace h64
static_assert(h64::smem_bytes(64) == kMlaH64Smem, "the header's constant is the H64 kernel's dynamic shared memory");
static_assert(h64::smem_bytes(64) + 64 <= 98304, "V100: at most 96 KB of shared memory per block (opt-in)");
#if !defined(STRATA_USE_HIP)
namespace h64 {
constexpr int kThreads = 256;
#if GLMD_WMMA
using AccF = wm::fragment<wm::accumulator, 16, 16, 16, float>;
static_assert(AccF::num_elements == 8, "the row map packs 8 elements x 4 bits");
#endif

// the registers of one tile: this thread's cell (slot j = tid & 15) and its share of the cell's values
struct Pre {
    int cell;
    uint4 v[4];   // INT8: v[0], v[1] = 16 codes of 16-value groups g and g + 16; FP16: 8-value parts g + 16 m
    uint4 kp;     // 8 k_pe values (threads 0-127: part tid >> 4)
    uint16_t d0, d1;   // INT8: the FP16 scales of the two groups
};

template <int R, bool LQ8>
__device__ __forceinline__ void pre_load(const MlaPrompt& a, int cell, int tid, Pre& p) {
    p.cell = cell;
    const int g = tid >> 4;
    if (cell < 0) {
        p.v[0] = p.v[1] = p.v[2] = p.v[3] = p.kp = make_uint4(0u, 0u, 0u, 0u);
        p.d0 = p.d1 = 0;
        return;
    }
    if constexpr (LQ8) {
        const uint8_t* rec = (const uint8_t*) a.lat + (size_t) lat8_rec(512) * cell;
        p.v[0] = *(const uint4*) (rec + 16 * g);
        p.v[1] = *(const uint4*) (rec + 16 * (g + 16));
        p.d0 = *(const uint16_t*) (rec + 512 + 2 * (g >> 1));
        p.d1 = *(const uint16_t*) (rec + 512 + 2 * ((g + 16) >> 1));
    } else {
        const uint16_t* row = (const uint16_t*) a.lat + (size_t) 512 * cell;
#pragma unroll
        for (int m = 0; m < 4; ++m) p.v[m] = *(const uint4*) (row + 8 * (g + 16 * m));
    }
    if constexpr (R > 0)
        if (tid < 128) p.kp = *(const uint4*) (a.k_pe + (size_t) R * cell + 8 * g);
}

__device__ __forceinline__ uint4 lat8_decode8(unsigned w0, unsigned w1, float d) {
    __half2 h[4];
    h[0] = __floats2half2_rn((float) (signed char) (w0 & 0xffu) * d, (float) (signed char) ((w0 >> 8) & 0xffu) * d);
    h[1] = __floats2half2_rn((float) (signed char) ((w0 >> 16) & 0xffu) * d, (float) (signed char) (w0 >> 24) * d);
    h[2] = __floats2half2_rn((float) (signed char) (w1 & 0xffu) * d, (float) (signed char) ((w1 >> 8) & 0xffu) * d);
    h[3] = __floats2half2_rn((float) (signed char) ((w1 >> 16) & 0xffu) * d, (float) (signed char) (w1 >> 24) * d);
    return *(uint4*) h;
}

// the registers into the tile: sK[kb][j][16]
template <int R, bool LQ8>
__device__ __forceinline__ void pre_store(const Pre& p, int tid, __half* sK, int* s_cell) {
    const int j = tid & 15, g = tid >> 4;
    if constexpr (LQ8) {
        const float d0 = __half2float(__ushort_as_half(p.d0)), d1 = __half2float(__ushort_as_half(p.d1));
        uint4* q0 = (uint4*) (sK + ((size_t) g * kNC + j) * 16);
        uint4* q1 = (uint4*) (sK + ((size_t) (g + 16) * kNC + j) * 16);
        q0[0] = lat8_decode8(p.v[0].x, p.v[0].y, d0);
        q0[1] = lat8_decode8(p.v[0].z, p.v[0].w, d0);
        q1[0] = lat8_decode8(p.v[1].x, p.v[1].y, d1);
        q1[1] = lat8_decode8(p.v[1].z, p.v[1].w, d1);
    } else {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int part = g + 16 * m;   // values [8 part, 8 part + 8): block part / 2, half part % 2
            *(uint4*) (sK + ((size_t) (part >> 1) * kNC + j) * 16 + (part & 1) * 8) = p.v[m];
        }
    }
    if constexpr (R > 0)
        if (tid < 128) *(uint4*) (sK + ((size_t) (32 + (g >> 1)) * kNC + j) * 16 + (g & 1) * 8) = p.kp;
    if (tid < kNC) s_cell[tid] = p.cell;
}

// Gather: the raw selection values of the tile at slot s (first = its slot 0, c = slot j; -1 past the row).  The
// kernel reads them a tile ahead of the tile's loads, so no warp waits on the index before the tile's own loads.
__device__ __forceinline__ void gather_idx(const MlaPrompt& a, int t, int j, int64_t s, int& first, int& c) {
    const int* row = a.sel + (size_t) t * a.top_k;
    first = s < a.top_k ? row[s] : -1;
    c = s + j < a.top_k ? row[s + j] : -1;
}

// the walk's next tile at slot s (Gather: s indexes the row's selection; Dense / DenseMasked: s is the first cell,
// empty masked windows skipped).  False when the walk is over; block-uniform (every thread reads the same words).
// cell = this thread's cell (slot j) or -1.
__device__ __forceinline__ bool tile_at(const MlaPrompt& a, int t, int64_t pos, int j, int64_t& s, int& cell) {
    if (a.walk == AttnWalk::Gather) {
        if (s >= a.top_k) return false;
        const int* row = a.sel + (size_t) t * a.top_k;
        const int first = row[s];
        if (first < 0 || first > pos) return false;   // ascending, then -1 padding
        const int c = s + j < a.top_k ? row[s + j] : -1;
        cell = c >= 0 && c <= pos ? c : -1;
        return true;
    }
    for (;;) {
        if (s > pos) return false;
        unsigned bits = 0xffffu;
        if (a.walk == AttnWalk::DenseMasked) bits = (a.mask[(size_t) t * a.mask_words + (s >> 5)] >> (s & 31)) & 0xffffu;
        if (bits == 0u) {
            s += kNC;
            continue;
        }
        const int64_t c = s + j;
        cell = c <= pos && ((bits >> j) & 1u) ? (int) c : -1;
        return true;
    }
}

}  // namespace h64

template <int R, bool LQ8>
__global__ void __launch_bounds__(h64::kThreads, 1) mla_prompt_h64_kernel(const STRATA_GRID_CONSTANT MlaPrompt a) {
#if !GLMD_WMMA
    __trap();   // a Pascal device pass: never launched (mla_prompt_h64_fits refuses the device)
#else
    using namespace h64;
    constexpr int W = 512 + R, KB = W / 16;
    extern __shared__ __align__(128) unsigned char smem_h[];
    __half* sQ = (__half*) smem_h;
    __half* sK = sQ + (size_t) KB * kH * 16;
    float* sS = (float*) (sK + (size_t) KB * kNC * 16);
    float* sM = sS + kH * kNC;
    float* sL = sM + kH;
    float* sA = sL + kH;
    int* s_cell = (int*) (sA + kH);
    __shared__ float s_qmax[kThreads / 32], s_qs;
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, j = tid & 15;
    const int64_t pos = a.p0 + t;
    const size_t row0 = (size_t) t * a.n_head + (size_t) blockIdx.y * kH;
    // Q: scale folded in, FP16 (as the 16-head kernel: v * scale, then / qs when the tile passes 65504)
    float qmax = 0.0f;
    for (int i = tid; i < kH * (W / 4); i += kThreads) {
        const int hh = i / (W / 4), c = (i - hh * (W / 4)) * 4;
        const float4 v = c < 512 ? *(const float4*) (a.q_abs + (row0 + hh) * 512 + c)
                                 : *(const float4*) (a.q + (row0 + hh) * a.q_stride + a.q_pe_off + (c - 512));
        const float x0 = v.x * a.scale, x1 = v.y * a.scale, x2 = v.z * a.scale, x3 = v.w * a.scale;
        qmax = fmaxf(qmax, fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
        __half2 h2[2] = {__floats2half2_rn(x0, x1), __floats2half2_rn(x2, x3)};
        *(uint2*) (sQ + ((size_t) (c >> 4) * kH + hh) * 16 + (c & 15)) = *(uint2*) h2;
    }
    qmax = warp_max(qmax);
    if (lane == 0) s_qmax[warp] = qmax;
    // the accumulator row map: a tile of row numbers through the fragment loader (sS is free until the walk)
    sS[tid] = (float) (tid >> 4);
    if (tid < kH) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) m = fmaxf(m, s_qmax[w]);
        s_qs = glmd::f16_guard_scale(m);
    }
    unsigned rowmap = 0u;
    {
        AccF rf;
        wm::load_matrix_sync(rf, sS, 16, wm::mem_row_major);
#pragma unroll
        for (int i = 0; i < AccF::num_elements; ++i) rowmap |= ((unsigned) rf.x[i] & 15u) << (4 * i);
    }
    __syncthreads();
    const float qs = s_qs;
    if (qs != 1.0f) {   // B15.1: rare; the tile again, divided by qs (exact powers of two)
        const float inv = 1.0f / qs;
        for (int i = tid; i < kH * (W / 4); i += kThreads) {
            const int hh = i / (W / 4), c = (i - hh * (W / 4)) * 4;
            const float4 v = c < 512 ? *(const float4*) (a.q_abs + (row0 + hh) * 512 + c)
                                     : *(const float4*) (a.q + (row0 + hh) * a.q_stride + a.q_pe_off + (c - 512));
            __half2 h2[2] = {__floats2half2_rn((v.x * a.scale) * inv, (v.y * a.scale) * inv),
                             __floats2half2_rn((v.z * a.scale) * inv, (v.w * a.scale) * inv)};
            *(uint2*) (sQ + ((size_t) (c >> 4) * kH + hh) * 16 + (c & 15)) = *(uint2*) h2;
        }
    }
    AccF o[4][4];
#pragma unroll
    for (int rb = 0; rb < 4; ++rb)
#pragma unroll
        for (int ci = 0; ci < 4; ++ci) wm::fill_fragment(o[rb][ci], 0.0f);
    int64_t slot = 0;
    int cell = -1;
    bool have = tile_at(a, t, pos, j, slot, cell);
    Pre pre;
    if (have) pre_load<R, LQ8>(a, cell, tid, pre);
    const bool gather = a.walk == AttnWalk::Gather;
    int g_first = -1, g_c = -1;   // Gather: the raw indices of the tile after the one in flight
    if (gather) gather_idx(a, t, j, slot + kNC, g_first, g_c);
    while (have) {
        __syncthreads();   // the previous tile's readers (softmax, P V) are done with sK / sS / s_cell
        pre_store<R, LQ8>(pre, tid, sK, s_cell);
        __syncthreads();
        // the next tile: its walk and its loads land while this one is multiplied
        slot += kNC;
        int ncell = -1;
        bool nhave;
        if (gather) {   // (its indices were read a tile ago; the tile after it is read now)
            nhave = g_first >= 0 && g_first <= pos;
            ncell = g_c >= 0 && g_c <= pos ? g_c : -1;
            gather_idx(a, t, j, slot + kNC, g_first, g_c);
        } else {
            nhave = tile_at(a, t, pos, j, slot, ncell);
        }
        if (nhave) pre_load<R, LQ8>(a, ncell, tid, pre);
        if (warp < 4) {
            // S (64 heads x 16 cells) = Q K^T: warp w < 4 takes heads [16 w, 16 w + 16), even / odd k-blocks in two
            // chains (S on all eight warps, k split in halves summed through sS, measured the same: 26.1 vs 26.0 ms)
            AccF s0, s1;
            wm::fill_fragment(s0, 0.0f);
            wm::fill_fragment(s1, 0.0f);
            const __half* qa = sQ + (size_t) warp * 16 * 16;
#pragma unroll 2
            for (int kb = 0; kb < KB; kb += 2) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> qf0, qf1;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> kf0, kf1;
                wm::load_matrix_sync(qf0, qa + (size_t) kb * kH * 16, 16);
                wm::load_matrix_sync(kf0, sK + (size_t) kb * kNC * 16, 16);
                wm::load_matrix_sync(qf1, qa + (size_t) (kb + 1) * kH * 16, 16);
                wm::load_matrix_sync(kf1, sK + (size_t) (kb + 1) * kNC * 16, 16);
                wm::mma_sync(s0, qf0, kf0, s0);
                wm::mma_sync(s1, qf1, kf1, s1);
            }
#pragma unroll
            for (int i = 0; i < AccF::num_elements; ++i) s0.x[i] += s1.x[i];
            wm::store_matrix_sync(sS + (size_t) warp * 16 * 16, s0, 16, wm::mem_row_major);
        }
        __syncthreads();
        // online softmax: 4 threads a head row (row tid / 4, cells 4 q .. 4 q + 3 with q = tid % 4), the whole 64 x 16
        // tile in one pass; P (f16) over the row's first 32 bytes (a half-warp a row in four passes: 9-19% slower)
        {
            const int r = tid >> 2, q4 = tid & 3;
            const float4 sv = *(const float4*) (sS + r * kNC + 4 * q4);
            const int4 cv = *(const int4*) (s_cell + 4 * q4);
            const float v[4] = {cv.x >= 0 ? sv.x * qs : -INFINITY, cv.y >= 0 ? sv.y * qs : -INFINITY,   // (B15.1: qs 1
                                cv.z >= 0 ? sv.z * qs : -INFINITY, cv.w >= 0 ? sv.w * qs : -INFINITY};  //  is exact)
            float mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mo = sM[r], mn = fmaxf(mo, mx);
            float p[4] = {0.0f, 0.0f, 0.0f, 0.0f}, alpha = 1.0f;
            if (mn != -INFINITY) {
                alpha = mo == -INFINITY ? 0.0f : expf(mo - mn);
#pragma unroll
                for (int u = 0; u < 4; ++u) p[u] = v[u] == -INFINITY ? 0.0f : expf(v[u] - mn);
            }
            float sum = (p[0] + p[1]) + (p[2] + p[3]);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();   // the row's four threads have read it before P overwrites its first half
            __half2 h2[2] = {__floats2half2_rn(p[0], p[1]), __floats2half2_rn(p[2], p[3])};
            *(uint2*) ((__half*) sS + r * 32 + 4 * q4) = *(uint2*) h2;
            if (q4 == 0) {
                sL[r] = sL[r] * alpha + sum;
                sM[r] = mn;
                sA[r] = alpha;
            }
        }
        __syncthreads();
        // O = alpha O + P V over this warp's 4 column tiles
#pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            float am[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) am[i] = sA[rb * 16 + ((rowmap >> (4 * i)) & 15u)];
#pragma unroll
            for (int ci = 0; ci < 4; ++ci)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[rb][ci].x[i] *= am[i];
        }
        wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> pf[4];
#pragma unroll
        for (int rb = 0; rb < 4; ++rb) wm::load_matrix_sync(pf[rb], (const __half*) sS + (size_t) rb * 16 * 32, 32);
#pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> vf;
            wm::load_matrix_sync(vf, sK + (size_t) (warp * 4 + ci) * kNC * 16, 16);
#pragma unroll
            for (int rb = 0; rb < 4; ++rb) wm::mma_sync(o[rb][ci], pf[rb], vf, o[rb][ci]);
        }
        have = nhave;
    }
    // ctx = O / l (sL was published by the last softmax's barrier; 0 when no cell was seen, as the 16-head kernel)
    float* out = a.ctx + row0 * 512;
#pragma unroll
    for (int rb = 0; rb < 4; ++rb) {
        float il[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float l = sL[rb * 16 + ((rowmap >> (4 * i)) & 15u)];
            il[i] = l > 0.0f ? 1.0f / l : 0.0f;
        }
#pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[rb][ci].x[i] *= il[i];
            wm::store_matrix_sync(out + (size_t) rb * 16 * 512 + (warp * 4 + ci) * 16, o[rb][ci], 512, wm::mem_row_major);
        }
    }
#endif  // GLMD_WMMA
}
#endif  // !STRATA_USE_HIP (the H64 kernel)

#if defined(STRATA_USE_HIP)
// ---- RDNA 3 / 3.5 / 4 (gfx11 / gfx12, wave32 WMMA): B9.5's arithmetic - Q in FP16 with the scale folded in and the
// B15.1 guard, the cells in FP16, F32 accumulation, the online softmax in F32, P in FP16 - within RDNA's 64 KiB of LDS
// a workgroup (the CUDA kernels take 89-97 KB).  A block per (16 heads, query row), the four head blocks of a row next
// to each other in the grid (they read the same cells: the L2 serves three of the four).  Q stays in registers: wave w
// holds its share of the 576-wide contraction for the block's 16 heads as WMMA A operands (latent dims [64 w, 64 w + 64)
// as four k-blocks, and rope k-block w for w < 4).  The cells go 16 at a time (latent 16 KiB + rope 2 KiB of LDS; a
// walk_chunk's 32 cells are two such tiles), the eight waves' partial scores meet in LDS (8 KiB), and O (16 heads x 512)
// stays in WMMA accumulators: wave w owns value columns [64 w, 64 w + 64), four 16-column tiles.  ~27 KiB of LDS.
// The WMMA builtins exist on the RDNA 3 / 3.5 / 4 device passes below only: any other pass compiles a trapping body,
// and mla_prompt_rdna_ok() takes the same list, so the kernel is never launched there (the F32 kernel runs).
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ &&                                                       \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) ||                  \
     defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))
#define GLMD_RDNA_WMMA 1
#else
#define GLMD_RDNA_WMMA 0
#endif
namespace rw {
using h16 = _Float16 __attribute__((ext_vector_type(16)));
using f8 = float __attribute__((ext_vector_type(8)));
// D = A B + C for 16 x 16 x 16 FP16 tiles, F32 accumulation.  The operands as gfx11 takes them in wave32: lane l holds
// row (l & 15) of A and column (l & 15) of B, all 16 k values (lanes 16-31 the same as 0-15); a gfx12 lane takes the
// half of them its own layout wants (k from 8 (l >> 4)).  D: element i of lane l is row row_of(i, l >> 4), column l & 15.
__device__ __forceinline__ f8 wmma(h16 a, h16 b, f8 c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (int) (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aa[i] = a[k0 + i];
        bb[i] = b[k0 + i];
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#elif GLMD_RDNA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ int row_of(int i, int hi) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
// eight latent values of a cell as FP16 (lat_h's arithmetic: an INT8 record's value times its block's scale, rounded)
__device__ __forceinline__ uint4 lat_h8(const void* lat, int lat_q8, int kv_lora, int64_t cell, int c8) {
    if (!lat_q8) return ((const uint4*) ((const uint16_t*) lat + (size_t) kv_lora * cell))[c8];
    const uint8_t* rec = (const uint8_t*) lat + (size_t) lat8_rec(kv_lora) * cell;
    const float d = h2f(((const uint16_t*) (rec + kv_lora))[(c8 * 8) >> 5]);
    const int8_t* q = (const int8_t*) rec + c8 * 8;
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        w[i] = (uint32_t) f2h((float) q[2 * i] * d) | ((uint32_t) f2h((float) q[2 * i + 1] * d) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
}  // namespace rw

template <int R>
__global__ void __launch_bounds__(256) mla_prompt_rdna_kernel(const STRATA_GRID_CONSTANT MlaPrompt a) {
#if !GLMD_RDNA_WMMA
    __builtin_trap();   // (no WMMA on this pass: never launched, mla_prompt_rdna_ok() refuses the device)
#else
    constexpr int K = 512, CH = 16, HB = 16, RW = R > 0 ? R : 16;
    constexpr int NKB = R > 0 ? 5 : 4;   // a wave's k-blocks: four latent ones, then its rope one (waves 0-3)
    __shared__ __align__(32) _Float16 sK[CH][K];     // the tile's 16 cells: latent
    __shared__ __align__(32) _Float16 sR[CH][RW];    // ... and k_pe
    __shared__ float sPart[8][HB][CH];               // each wave's partial scores
    __shared__ __align__(32) _Float16 sP[HB][CH];    // P = exp(S - m), FP16
    __shared__ float sM[HB], sL[HB], sA[HB];
    __shared__ int s_cell[32];
    __shared__ int s_any;
    __shared__ float s_qmax[8], s_qs;
    const int h0 = blockIdx.x * HB, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int l16 = lane & 15, hi = lane >> 4;
    const bool rope_kb = R > 0 && wave < 4;   // (wave-uniform)
    // this lane's A rows: head h0 + l16 of row t
    const size_t qrow = (size_t) t * a.n_head + h0 + l16;
    const float* qa = a.q_abs + qrow * K + wave * 64;
    const float* qp = R > 0 ? a.q + qrow * a.q_stride + a.q_pe_off + wave * 16 : nullptr;
    // B15.1: the FP16 range guard of the block's Q tile (its 16 heads), as the CUDA kernels
    float qmax = 0.0f;
#pragma unroll
    for (int k = 0; k < 64; ++k) qmax = fmaxf(qmax, fabsf(qa[k] * a.scale));
    if (rope_kb)
        for (int k = 0; k < 16; ++k) qmax = fmaxf(qmax, fabsf(qp[k] * a.scale));
    qmax = warp_max(qmax);
    if (lane == 0) s_qmax[wave] = qmax;
    if (tid < HB) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int w = 0; w < 8; ++w) m = fmaxf(m, s_qmax[w]);
        s_qs = glmd::f16_guard_scale(m);
    }
    __syncthreads();
    const float qs = s_qs, qinv = 1.0f / qs;
    rw::h16 qf[NKB];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float v = qa[kb * 16 + j] * a.scale;
            qf[kb][j] = (_Float16) (qs != 1.0f ? v * qinv : v);
        }
    if constexpr (R > 0) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float v = rope_kb ? qp[j] * a.scale : 0.0f;
            qf[4][j] = (_Float16) (qs != 1.0f ? v * qinv : v);
        }
    }
    rw::f8 acc[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    int64_t c0 = 0;
    for (;;) {
        __syncthreads();   // the previous tile's readers of s_cell are done
        if (!walk_chunk(a, t, c0, s_cell, &s_any)) break;
        for (int half = 0; half < 2; ++half) {
            const int* cl = s_cell + half * CH;
            bool any = false;
#pragma unroll
            for (int i = 0; i < CH; ++i) any = any || cl[i] >= 0;
            if (!any) continue;   // (block-uniform: every thread read the same cells)
            __syncthreads();      // the previous tile's readers of sK / sR / sP are done
            for (int i = tid; i < CH * K / 8; i += 256) {
                const int s = i / (K / 8), c8 = i - s * (K / 8);
                const int cell = cl[s];
                ((uint4*) &sK[s][0])[c8] = cell < 0 ? make_uint4(0u, 0u, 0u, 0u) : rw::lat_h8(a.lat, a.lat_q8, K, cell, c8);
            }
            if constexpr (R > 0) {
                for (int i = tid; i < CH * R / 8; i += 256) {
                    const int s = i / (R / 8), c8 = i - s * (R / 8);
                    const int cell = cl[s];
                    ((uint4*) &sR[s][0])[c8] =
                        cell < 0 ? make_uint4(0u, 0u, 0u, 0u) : ((const uint4*) (a.k_pe + (size_t) R * cell))[c8];
                }
            }
            __syncthreads();
            // this wave's share of S (16 heads x 16 cells) = Q K^T
            rw::f8 dot;
#pragma unroll
            for (int i = 0; i < 8; ++i) dot[i] = 0.0f;
#pragma unroll
            for (int kb = 0; kb < 4; ++kb) dot = rw::wmma(qf[kb], *(const rw::h16*) &sK[l16][wave * 64 + kb * 16], dot);
            if constexpr (R > 0) {
                if (rope_kb) dot = rw::wmma(qf[4], *(const rw::h16*) &sR[l16][wave * 16], dot);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) sPart[wave][rw::row_of(i, hi)][l16] = dot[i];
            __syncthreads();
            // the online softmax: 16 threads a head row (a row's threads are 16 lanes of one wave)
            {
                const int r = tid >> 4, col = tid & 15;
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < 8; ++w) v += sPart[w][r][col];
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
            const rw::h16 pa = *(const rw::h16*) &sP[l16][0];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[nt][i] *= sA[rw::row_of(i, hi)];
                rw::h16 b;
#pragma unroll
                for (int k = 0; k < 16; ++k) b[k] = sK[k][wave * 64 + nt * 16 + l16];
                acc[nt] = rw::wmma(pa, b, acc[nt]);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            asm volatile("" ::"v"(acc[nt][i]));   // (every matrix lane live before the stores: glm_batch.cu's wmma2)
            const int r = rw::row_of(i, hi);
            const float L = sL[r];
            const float inv = L > 0.0f ? 1.0f / L : 0.0f;
            a.ctx[((size_t) t * a.n_head + h0 + r) * K + wave * 64 + nt * 16 + l16] = acc[nt][i] * inv;
        }
#endif  // GLMD_RDNA_WMMA
}

// ---- the indexer's score tile on RDNA WMMA (S7): index_score_wmma_kernel's arithmetic - FP16 queries and keys, F32
// sums, each head's dot products relu'd, weighted and summed in head order, B15.1's per-(query, head) scale - in 56 KiB
// of LDS: a block per 64 queries x 128 keys, the keys staged once ([8 k-blocks][128][16] FP16), the queries head by head
// in one buffer (the next head is read into registers while the current one is multiplied and stored after a barrier),
// the weights [64][H <= 32] F32.  Wave w: query rows 16 (w & 3) .. + 15, key tiles 4 (w >> 2) .. + 3.
constexpr int kRsQ = 64, kRsN = 128, kRsD = 128, kRsMaxH = 32;
template <bool KF16>
__global__ void __launch_bounds__(256) index_score_rdna_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                                               const void* __restrict__ keys, int H, int T, int64_t k0,
                                                               int nk, float* __restrict__ score, int64_t ld) {
#if !GLMD_RDNA_WMMA
    __builtin_trap();   // (no WMMA on this pass: never launched, index_score_wmma_fits refuses the device)
#else
    __shared__ __align__(32) _Float16 sKey[8][kRsN][16];
    __shared__ __align__(32) _Float16 sQ[8][kRsQ][16];
    __shared__ float sW[kRsQ][kRsMaxH];
    __shared__ float sQs[kRsQ];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int rb = wave & 3, ct0 = 4 * (wave >> 2);
    const int q0 = blockIdx.y * kRsQ, n0 = blockIdx.x * kRsN;
    // keys: [8][128][16] FP16 (zeros past nk); F32 keys saturated to the FP16 range, as the CUDA kernel
    for (int u = tid; u < 8 * kRsN * 4; u += 256) {
        const int kb = u >> 9, key = (u >> 2) & (kRsN - 1), part = u & 3;
        uint2 v = make_uint2(0u, 0u);
        if ((int64_t) n0 + key < nk) {
            const size_t off = (size_t) (k0 + n0 + key) * kRsD + kb * 16 + part * 4;
            if constexpr (KF16) {
                v = *(const uint2*) ((const uint16_t*) keys + off);
            } else {
                const float4 f = *(const float4*) ((const float*) keys + off);
                const float c = 65504.0f;
                v.x = (uint32_t) f2h(fminf(fmaxf(f.x, -c), c)) | ((uint32_t) f2h(fminf(fmaxf(f.y, -c), c)) << 16);
                v.y = (uint32_t) f2h(fminf(fmaxf(f.z, -c), c)) | ((uint32_t) f2h(fminf(fmaxf(f.w, -c), c)) << 16);
            }
        }
        *(uint2*) &sKey[kb][key][part * 4] = v;
    }
    for (int i = tid; i < kRsQ * H; i += 256) {
        const int q = i / H;
        sW[q][i - q * H] = q0 + q < T ? w[(size_t) (q0 + q) * H + (i - q * H)] : 0.0f;
    }
    // this thread's share of a head's query tile: query ql, dims 16 kb + e (the CUDA kernel's q_load / q_store)
    const int ql = (tid >> 5) * 8 + ((tid & 31) >> 2), e = 4 * (tid & 3);
    float4 qv[8];
    const auto q_load = [&](int h) {
        const int q = q0 + ql;
#pragma unroll
        for (int kb = 0; kb < 8; ++kb)
            qv[kb] = q < T ? *(const float4*) (iq + ((size_t) q * H + h) * kRsD + kb * 16 + e)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    const auto q_store = [&]() {
        float m = 0.0f;
#pragma unroll
        for (int kb = 0; kb < 8; ++kb)
            m = fmaxf(m, fmaxf(fmaxf(fabsf(qv[kb].x), fabsf(qv[kb].y)), fmaxf(fabsf(qv[kb].z), fabsf(qv[kb].w))));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));   // the four lanes of this query hold its 128 values
        const float qs = glmd::f16_guard_scale(m), inv = 1.0f / qs;
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            float4 f = qv[kb];
            if (qs != 1.0f) f = make_float4(f.x * inv, f.y * inv, f.z * inv, f.w * inv);
            *(uint2*) &sQ[kb][ql][e] = make_uint2((uint32_t) f2h(f.x) | ((uint32_t) f2h(f.y) << 16),
                                                  (uint32_t) f2h(f.z) | ((uint32_t) f2h(f.w) << 16));
        }
        if ((tid & 3) == 0) sQs[ql] = qs;
    };
    q_load(0);
    __syncthreads();   // (the keys and the weights are staged)
    q_store();
    __syncthreads();
    rw::f8 acc[4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[c][i] = 0.0f;
    for (int h = 0; h < H; ++h) {
        if (h + 1 < H) q_load(h + 1);
        rw::f8 s[4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[c][i] = 0.0f;
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            const rw::h16 qa = *(const rw::h16*) &sQ[kb][rb * 16 + l16][0];
#pragma unroll
            for (int c = 0; c < 4; ++c) s[c] = rw::wmma(qa, *(const rw::h16*) &sKey[kb][(ct0 + c) * 16 + l16][0], s[c]);
        }
        float wq[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rb * 16 + rw::row_of(i, hi);
            wq[i] = sW[r][h] * sQs[r];
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] += wq[i] * fmaxf(s[c][i], 0.0f);
        __syncthreads();   // every wave is done with head h's queries and scales
        if (h + 1 < H) q_store();
        __syncthreads();
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            asm volatile("" ::"v"(acc[c][i]));
            const int64_t q = q0 + rb * 16 + rw::row_of(i, hi);
            const int64_t j = (int64_t) n0 + (ct0 + c) * 16 + l16;
            if (q < T && j < nk) score[(size_t) q * ld + j] = acc[c][i];
        }
#endif  // GLMD_RDNA_WMMA
}
#endif  // STRATA_USE_HIP (the RDNA kernels)

// ---------------------------------------------------------------- router, residual, packing
__global__ void __launch_bounds__(256) route_rows_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                         int E, int k, float w_scale, int norm_w, int norm_mode,
                                                         int* __restrict__ ids, float* __restrict__ w) {
    __shared__ float s_p[512], s_sel[512];
    __shared__ int bad;
    const int t = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) bad = 0;
    __syncthreads();
    for (int e = tid; e < E; e += blockDim.x) {
        const float lg = logits[(size_t) t * E + e];
        const float p = 1.0f / (1.0f + expf(-lg));
        s_p[e] = p;
        s_sel[e] = bias ? p + bias[e] : p;
        if (!isfinite(lg) || !isfinite(s_sel[e])) atomicExch(&bad, 1);
    }
    __syncthreads();
    if (tid != 0) return;
    int sel[8];
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int e = 0; e < E; ++e)
            if (s_sel[e] > bv) {   // ascending e: a tie keeps the lower id
                bv = s_sel[e];
                bi = e;
            }
        sel[r] = bi;
        if (bi < 0) bad = 1;
        else s_sel[bi] = -INFINITY;
    }
    if (bad) {
        for (int i = 0; i < k; ++i) {
            ids[(size_t) t * k + i] = -1;
            w[(size_t) t * k + i] = 0.0f;
        }
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += (double) s_p[sel[i]];
    const float inv = (float) (!norm_w ? 1.0 : norm_mode == kRouteNormHf ? 1.0 / (sum + 1e-20) : 1.0 / fmax(sum, 6.103515625e-5));
    for (int i = 0; i < k; ++i) {
        ids[(size_t) t * k + i] = sel[i];
        w[(size_t) t * k + i] = (float) ((double) s_p[sel[i]] * inv * w_scale);
    }
}

__global__ void __launch_bounds__(256) res_rows_kernel(float* __restrict__ R, const float* __restrict__ add,
                                                       const float* __restrict__ w, float eps, int E,
                                                       float* __restrict__ x, uint16_t* __restrict__ x16) {
    __shared__ float sred[32];
    const int t = blockIdx.x;
    float* Rt = R + (size_t) t * E;
    float ss = 0.0f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float v = Rt[e];
        if (add) {
            v += add[(size_t) t * E + e];
            Rt[e] = v;
        }
        ss += v * v;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) E + eps);
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float y = Rt[e] * inv * w[e];
        if (x) x[(size_t) t * E + e] = y;
        if (x16) x16[(size_t) t * E + e] = f2h(y);
    }
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] += x[i];
}

__global__ void pack_f16_kernel(const float* __restrict__ src, int64_t src_row, int src_head, int H, int K, int T,
                                uint16_t* __restrict__ dst) {
    const int64_t n = (int64_t) T * H * K;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int k = (int) (i % K);
        const int64_t r = i / K;
        const int h = (int) (r % H);
        const int64_t t = r / H;
        dst[i] = f2h(src[t * src_row + (int64_t) h * src_head + k]);
    }
}

__global__ void weight_f16_kernel(const void* __restrict__ w, int bf16, int64_t n, uint16_t* __restrict__ dst) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = f2h(bf16 ? __uint_as_float((uint32_t) ((const uint16_t*) w)[i] << 16) : ((const float*) w)[i]);
}

__global__ void q8_clamp_kernel(uint8_t* __restrict__ xq, int64_t n_blocks, int block_bytes, int nh) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks * nh) return;
    __half* h = (__half*) (xq + (size_t) (i / nh) * block_bytes) + (i % nh);
    const float v = __half2float(*h);
    if (isinf(v)) *h = __float2half(v > 0.0f ? 65504.0f : -65504.0f);
}

unsigned grid_for(int64_t n, int64_t per = 256, int64_t cap = 65535) {
    return (unsigned) std::max<int64_t>(1, std::min<int64_t>(cap, (n + per - 1) / per));
}

}  // namespace

int batch_launch_errors() { return g_errors.load(std::memory_order_relaxed); }

bool dsa_prep_rows(const DsaPrepRows& r, void* stream) {
    const DsaPrep& a = r.a;
    const bool full = a.ik_raw != nullptr;
    if (r.T <= 0) return true;
    if (r.T > 65535 || a.q_lora <= 0 || a.q_lora > 2048 || a.q_lora % 32 || a.kv_lora <= 0 || a.kv_lora > 512 ||
        a.kv_lora % 32 || a.qk_rope < 0 || a.qk_rope > 128 || a.qk_rope % 2 || r.kv_ld < a.kv_lora + a.qk_rope ||
        !a.qr_raw || !a.q_a_norm || !a.kv_raw || !a.kv_norm || !a.lat || (a.qk_rope > 0 && (!a.k_pe || !a.inv_freq)) ||
        (full && (a.idx_dim <= 0 || a.idx_dim > 512 || a.idx_rope < 0 || a.idx_rope > a.idx_dim || a.idx_rope % 2 ||
                  !a.ik_ln_w || !a.ik_ln_b || !a.ik_cache || (a.idx_rope > 0 && !a.idx_inv_freq) ||
                  (a.ik_had && !had_len_ok(a.idx_dim)))))
        return refuse("dsa_prep_rows");
    dsa_prep_rows_kernel<<<dim3(full ? 3 : 2, r.T), 512, 0, (cudaStream_t) stream>>>(r);
    return launch_check("dsa_prep_rows");
}

bool dsa_rope_q_rows(const DsaRopeQRows& r, void* stream) {
    const DsaRopeQ& a = r.a;
    if (r.T <= 0) return true;
    if (r.T > 65535 || !a.q || !a.inv_freq || a.n_rot % 2 || a.n_rot < 0 || a.rot_off + a.n_rot > a.q_stride ||
        (a.iq && (!a.idx_inv_freq || a.idx_rope % 2 || a.idx_rope < 0 || a.idx_rope > a.idx_dim ||
                  (a.iq_had && (!had_len_ok(a.idx_dim) || a.idx_heads <= 0)))))
        return refuse("dsa_rope_q_rows");
    const int n = a.n_head * (a.n_rot / 2) + (a.iq ? a.idx_heads * (a.idx_rope / 2) : 0);
    if (n == 0 && !(a.iq && a.iq_had)) return true;
    if (n > 0) dsa_rope_q_rows_kernel<<<dim3((n + 255) / 256, r.T), 256, 0, (cudaStream_t) stream>>>(r);
    if (a.iq && a.iq_had)   // B14.11: every row's indexer query heads, after their rope
        had_rows_kernel<<<dim3(a.idx_heads, r.T), 128, (size_t) a.idx_dim * sizeof(float), (cudaStream_t) stream>>>(
            a.iq, a.idx_dim, (long long) r.iq_row);
    return launch_check("dsa_rope_q_rows");
}

bool index_score_tile(const float* iq, const float* w, const float* ik_cache, int idx_heads, int idx_dim, int T,
                      int64_t k0, int nk, float* score, int64_t ld, void* stream) {
    return index_score_tile_keys(iq, w, ik_cache, 0, idx_heads, idx_dim, T, k0, nk, score, ld, stream);
}

bool index_score_tile_keys(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld, void* stream) {
    if (T <= 0 || nk <= 0) return true;
    if (idx_dim <= 0 || idx_dim > kScMaxDim || idx_heads <= 0 || ld < nk || k0 < 0 || !iq || !w || !keys || !score ||
        (T + kScQ - 1) / kScQ > 65535)
        return refuse("index_score_tile");
    const dim3 grid((unsigned) ((nk + kScK - 1) / kScK), (unsigned) ((T + kScQ - 1) / kScQ));
    if (key_f16)
        index_score_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(iq, w, (const float*) keys, idx_heads, idx_dim,
                                                                         T, k0, nk, score, ld);
    else
        index_score_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(iq, w, (const float*) keys, idx_heads,
                                                                          idx_dim, T, k0, nk, score, ld);
    return launch_check("index_score_tile");
}

// the WMMA scorer's opt-in, once per device and key dtype
#ifndef STRATA_USE_HIP
template <bool KF16>
static bool sw_attr_ok(size_t need) {
    int dev = 0;
    cudaGetDevice(&dev);
    static int checked[64] = {};   // 0 unknown, 1 ok, 2 refused (need is the H = 64 maximum, see index_score_wmma_fits)
    int& st = checked[dev & 63];
    if (st != 0) return st == 1;
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa{};
    bool ok = (size_t) optin >= need &&
              cudaFuncSetAttribute(index_score_wmma_kernel<KF16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int) need) == cudaSuccess &&
              cudaFuncGetAttributes(&fa, index_score_wmma_kernel<KF16>) == cudaSuccess;
    ok = ok && fa.sharedSizeBytes + need <= (size_t) optin;
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "glmd_batch index_score_wmma: CUDA%d cannot give %zu B of shared memory (opt-in %d) - the "
                             "F32 scorer runs\n", dev, need, optin);
    }
    st = ok ? 1 : 2;
    return ok;
}
#else
template <bool KF16>
static bool sw_attr_ok(size_t) { return false; }
#endif

#if defined(STRATA_USE_HIP)
bool rdna_wmma_device();   // (below: the RDNA kernels' devices)
#endif

bool index_score_wmma_fits(int key_f16, int idx_heads, int idx_dim) {
    if (idx_dim != sw::kD || idx_heads <= 0 || idx_heads > 64) return false;
#if defined(STRATA_USE_HIP)
    // HIP: the RDNA scorer (index_score_rdna_kernel, S7) on gfx11 / gfx12, up to 32 heads (its LDS holds 64 x 32 weights)
    (void) key_f16;
    return idx_heads <= kRsMaxH && rdna_wmma_device();
#else
    return key_f16 ? sw_attr_ok<true>(sw::smem_bytes(64)) : sw_attr_ok<false>(sw::smem_bytes(64));
#endif
}

bool index_score_tile_wmma(const float* iq, const float* w, const void* keys, int key_f16, int idx_heads, int idx_dim,
                           int T, int64_t k0, int nk, float* score, int64_t ld, void* stream) {
    if (T <= 0 || nk <= 0) return true;
    if (!index_score_wmma_fits(key_f16, idx_heads, idx_dim) || ld < nk || k0 < 0 || !iq || !w || !keys || !score ||
        ((uintptr_t) iq & 15u) || ((uintptr_t) keys & 15u) || (T + sw::kQ - 1) / sw::kQ > 65535)
        return refuse("index_score_tile_wmma");
#if !defined(STRATA_USE_HIP)
    const dim3 grid((unsigned) ((nk + sw::kN - 1) / sw::kN), (unsigned) ((T + sw::kQ - 1) / sw::kQ));
    const size_t smem = sw::smem_bytes(idx_heads);
    if (key_f16)
        index_score_wmma_kernel<true><<<grid, sw::kThreads, smem, (cudaStream_t) stream>>>(iq, w, keys, idx_heads, T, k0,
                                                                                          nk, score, ld);
    else
        index_score_wmma_kernel<false><<<grid, sw::kThreads, smem, (cudaStream_t) stream>>>(iq, w, keys, idx_heads, T,
                                                                                           k0, nk, score, ld);
    return launch_check("index_score_tile_wmma");
#else
    const dim3 grid((unsigned) ((nk + kRsN - 1) / kRsN), (unsigned) ((T + kRsQ - 1) / kRsQ));
    if (key_f16)
        index_score_rdna_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(iq, w, keys, idx_heads, T, k0, nk, score,
                                                                              ld);
    else
        index_score_rdna_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(iq, w, keys, idx_heads, T, k0, nk, score,
                                                                               ld);
    return launch_check("index_score_tile_wmma (RDNA)");
#endif
}

bool topk_merge_tile(const TopkMerge& m, void* stream) {
    if (m.T <= 0) return true;
    if (m.top_k <= 0 || m.nk < 0 || m.ld < m.nk || !m.out_cell || !m.out_score || !m.out_n ||
        (m.nk > 0 && !m.score) || (m.in_n && (!m.in_cell || !m.in_score)))
        return refuse("topk_merge_tile");
    topk_merge_kernel<<<m.T, kSelThreads, 0, (cudaStream_t) stream>>>(m);
    return launch_check("topk_merge_tile");
}

bool sel_to_mask(const int* sel, int top_k, int T, int64_t n_kv, uint32_t* mask, int64_t words, void* stream) {
    if (T <= 0) return true;
    if (!sel || !mask || words * 32 < n_kv) return refuse("sel_to_mask");
    cudaMemsetAsync(mask, 0, (size_t) T * words * sizeof(uint32_t), (cudaStream_t) stream);
    const int64_t n = (int64_t) T * top_k;
    sel_mask_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(sel, top_k, T, n_kv, mask, words);
    return launch_check("sel_to_mask");
}

static bool mla_args_ok(const MlaPrompt& a, int max_kv) {
    return a.T > 0 && a.n_head > 0 && a.n_head % 16 == 0 && a.kv_lora > 0 && a.kv_lora <= max_kv &&
           a.kv_lora % 32 == 0 && (a.qk_rope == 0 || a.qk_rope == 64) && a.q_abs && a.lat && a.ctx &&
           (a.qk_rope == 0 || (a.q && a.k_pe && a.q_pe_off + a.qk_rope <= a.q_stride)) &&
           (a.walk != AttnWalk::Gather || (a.sel && a.top_k > 0)) &&
           (a.walk != AttnWalk::DenseMasked || (a.mask && a.mask_words * 32 >= a.p0 + a.T)) && a.T <= 2147483647;
}

bool mla_prompt_f32(const MlaPrompt& a, void* stream) {
    if (a.T <= 0) return true;
    if (!mla_args_ok(a, 512)) return refuse("mla_prompt_f32");
    const dim3 grid((unsigned) a.T, (unsigned) (a.n_head / 16));
    const size_t smem = (size_t) 32 * (a.kv_lora + a.qk_rope) * 2;   // 36,864 B at 512 + 64: no opt-in
    if (a.qk_rope == 64)
        mla_prompt_f32_kernel<64><<<grid, 256, smem, (cudaStream_t) stream>>>(a);
    else
        mla_prompt_f32_kernel<0><<<grid, 256, smem, (cudaStream_t) stream>>>(a);
    return launch_check("mla_prompt_f32");
}

// the opt-in and the attribute check, once per device (R6: a missing cudaFuncSetAttribute is a silent no-attention bug)
#ifndef STRATA_USE_HIP   // the WMMA kernel is CUDA's (HIP: never ok, the F32 kernel runs); keeps check_hip_compat exact
template <int R>
static bool wmma_attr_ok() {
    int dev = 0;
    cudaGetDevice(&dev);
    static int checked[64] = {};   // 0 unknown, 1 ok, 2 refused
    int& st = checked[dev & 63];
    if (st != 0) return st == 1;
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa{};
    bool ok = cudaFuncSetAttribute(mla_prompt_wmma_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int) kWmSmem) == cudaSuccess &&
              cudaFuncGetAttributes(&fa, mla_prompt_wmma_kernel<R>) == cudaSuccess;
    ok = ok && fa.maxDynamicSharedSizeBytes >= (int) kWmSmem && fa.sharedSizeBytes + kWmSmem <= (size_t) optin &&
         fa.sharedSizeBytes + kWmSmem <= 98304;
    if (!ok) {
        cudaGetLastError();
        g_errors.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "glmd_batch mla_prompt_wmma: CUDA%d cannot give %zu B dynamic + %zu B static shared memory "
                             "(opt-in %d, attribute %d) - the F32 kernel runs\n",
                     dev, kWmSmem, (size_t) fa.sharedSizeBytes, optin, fa.maxDynamicSharedSizeBytes);
    }
    st = ok ? 1 : 2;
    return ok;
}
#else
template <int R>
static bool wmma_attr_ok() { return false; }
#endif

bool mla_prompt_wmma_ok() { return wmma_attr_ok<64>() && wmma_attr_ok<0>(); }

bool mla_prompt_wmma(const MlaPrompt& a, void* stream) {
    if (a.T <= 0) return true;
    if (!mla_args_ok(a, 512) || a.kv_lora != kWmK) return refuse("mla_prompt_wmma");
    const bool ok = a.qk_rope == 64 ? wmma_attr_ok<64>() : wmma_attr_ok<0>();
    if (!ok) return false;
#if !defined(STRATA_USE_HIP)
    const dim3 grid((unsigned) a.T, (unsigned) (a.n_head / kWmH));
    if (a.qk_rope == 64)
        mla_prompt_wmma_kernel<64><<<grid, 256, kWmSmem, (cudaStream_t) stream>>>(a);
    else
        mla_prompt_wmma_kernel<0><<<grid, 256, kWmSmem, (cudaStream_t) stream>>>(a);
    return launch_check("mla_prompt_wmma");
#else
    return false;   // (unreachable: wmma_attr_ok is false on HIP)
#endif
}

// the H64 kernel's opt-in, once per device and instance (as wmma_attr_ok)
#ifndef STRATA_USE_HIP
template <int R, bool LQ8>
static bool h64_attr_ok() {
    int dev = 0;
    cudaGetDevice(&dev);
    static int checked[64] = {};   // 0 unknown, 1 ok, 2 refused
    int& st = checked[dev & 63];
    if (st != 0) return st == 1;
    const size_t need = h64::smem_bytes(R);
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa{};
    bool ok = (size_t) optin >= need &&
              cudaFuncSetAttribute(mla_prompt_h64_kernel<R, LQ8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int) need) == cudaSuccess &&
              cudaFuncGetAttributes(&fa, mla_prompt_h64_kernel<R, LQ8>) == cudaSuccess;
    ok = ok && fa.maxDynamicSharedSizeBytes >= (int) need && fa.sharedSizeBytes + need <= (size_t) optin;
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "glmd_batch mla_prompt_h64: CUDA%d cannot give %zu B dynamic + %zu B static shared memory "
                             "(opt-in %d) - the 16-head kernel runs\n",
                     dev, need, (size_t) fa.sharedSizeBytes, optin);
    }
    st = ok ? 1 : 2;
    return ok;
}
#else
template <int R, bool LQ8>
static bool h64_attr_ok() { return false; }
#endif

static bool a16(const void* p) { return ((uintptr_t) p & 15u) == 0; }

bool mla_prompt_h64_fits(const MlaPrompt& a) {
    if (!mla_args_ok(a, 512) || a.kv_lora != 512 || a.n_head % h64::kH != 0 || !a16(a.q_abs) || !a16(a.lat) ||
        !a16(a.ctx) || (a.qk_rope > 0 && (!a16(a.k_pe) || !a16(a.q + a.q_pe_off) || a.q_stride % 4 != 0)))
        return false;
    if (a.qk_rope == 64) return a.lat_q8 ? h64_attr_ok<64, true>() : h64_attr_ok<64, false>();
    return a.lat_q8 ? h64_attr_ok<0, true>() : h64_attr_ok<0, false>();
}

bool mla_prompt_h64(const MlaPrompt& a, void* stream) {
    if (a.T <= 0) return true;
    if (!mla_prompt_h64_fits(a)) return refuse("mla_prompt_h64");
#if !defined(STRATA_USE_HIP)
    const dim3 grid((unsigned) a.T, (unsigned) (a.n_head / h64::kH));
    const cudaStream_t s = (cudaStream_t) stream;
    if (a.qk_rope == 64) {
        if (a.lat_q8)
            mla_prompt_h64_kernel<64, true><<<grid, h64::kThreads, h64::smem_bytes(64), s>>>(a);
        else
            mla_prompt_h64_kernel<64, false><<<grid, h64::kThreads, h64::smem_bytes(64), s>>>(a);
    } else {
        if (a.lat_q8)
            mla_prompt_h64_kernel<0, true><<<grid, h64::kThreads, h64::smem_bytes(0), s>>>(a);
        else
            mla_prompt_h64_kernel<0, false><<<grid, h64::kThreads, h64::smem_bytes(0), s>>>(a);
    }
    return launch_check("mla_prompt_h64");
#else
    (void) stream;
    return false;   // (unreachable: mla_prompt_h64_fits is false on HIP)
#endif
}

#if defined(STRATA_USE_HIP)
// the RDNA WMMA kernels' devices: the gfx11 / gfx12 archs whose device pass has their bodies (GLMD_RDNA_WMMA's list),
// once per device
bool rdna_wmma_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    static std::atomic<int> cache[64]{};   // 0 not probed, 1 yes, 2 no
    int c = cache[dev].load(std::memory_order_relaxed);
    if (c == 0) {
        c = 2;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            for (const char* arch : {"gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"})
                if (std::strncmp(prop.gcnArchName, arch, 7) == 0 &&
                    (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':'))
                    c = 1;
        cache[dev].store(c, std::memory_order_relaxed);
    }
    return c == 1;
}

// MAYA_GLMD_PF_RDNA=0 keeps every device on the F32 attention kernel (A/B; the scorer: MAYA_GLMD_IDX_WMMA=0)
bool mla_prompt_rdna_ok() {
    static const bool off = [] {
        const char* v = std::getenv("MAYA_GLMD_PF_RDNA");
        return v != nullptr && v[0] == '0';
    }();
    return !off && rdna_wmma_device();
}

bool mla_prompt_rdna(const MlaPrompt& a, void* stream) {
    if (a.T <= 0) return true;
    if (!mla_args_ok(a, 512) || a.kv_lora != 512 || a.T > 65535 || !mla_prompt_rdna_ok()) return false;   // F32 runs
    const dim3 grid((unsigned) (a.n_head / 16), (unsigned) a.T);
    if (a.qk_rope == 64)
        mla_prompt_rdna_kernel<64><<<grid, 256, 0, (cudaStream_t) stream>>>(a);
    else
        mla_prompt_rdna_kernel<0><<<grid, 256, 0, (cudaStream_t) stream>>>(a);
    return launch_check("mla_prompt_rdna");
}
#else
bool mla_prompt_rdna_ok() { return false; }
bool mla_prompt_rdna(const MlaPrompt&, void*) { return false; }
#endif

bool route_rows(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int norm_mode,
                int T, int* ids, float* w, void* stream) {
    if (T <= 0) return true;
    if (n_expert < 1 || n_expert > 512 || k < 1 || k > 8 || k > n_expert || !logits || !ids || !w)
        return refuse("route_rows");
    route_rows_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, norm_mode,
                                                            ids, w);
    return launch_check("route_rows");
}

bool res_rows(float* R, const float* add, const float* w, float eps, int T, int n_embd, float* x, uint16_t* x16,
              void* stream) {
    if (T <= 0) return true;
    if (n_embd <= 0 || n_embd > 8192 || n_embd % 32 || !R || !w) return refuse("res_rows");
    res_rows_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(R, add, w, eps, n_embd, x, x16);
    return launch_check("res_rows");
}

bool add_rows(float* y, const float* x, int64_t n, void* stream) {
    if (n <= 0) return true;
    add_kernel<<<grid_for(n), 256, 0, (cudaStream_t) stream>>>(y, x, n);
    return launch_check("add_rows");
}

bool pack_f16(const float* src, int64_t src_row, int src_head, int n_head, int K, int T, uint16_t* dst, void* stream) {
    if (T <= 0) return true;
    pack_f16_kernel<<<grid_for((int64_t) T * n_head * K), 256, 0, (cudaStream_t) stream>>>(src, src_row, src_head,
                                                                                             n_head, K, T, dst);
    return launch_check("pack_f16");
}

bool weight_f16(const void* w, int type, int64_t n, uint16_t* dst, void* stream) {
    if (n <= 0) return true;
    if (type != 30 && type != 0) return refuse("weight_f16");
    weight_f16_kernel<<<grid_for(n), 256, 0, (cudaStream_t) stream>>>(w, type == 30 ? 1 : 0, n, dst);
    return launch_check("weight_f16");
}

bool q8_mmq_clamp(void* xq, int64_t n_blocks, int block_bytes, int ds_halves_per_block, void* stream) {
    if (n_blocks <= 0 || ds_halves_per_block <= 0) return true;
    const int64_t n = n_blocks * ds_halves_per_block;
    q8_clamp_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((uint8_t*) xq, n_blocks, block_bytes,
                                                                                     ds_halves_per_block);
    return launch_check("q8_mmq_clamp");
}

}  // namespace glmd
