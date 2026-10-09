// src/kernels/cuda/glm_batch.cu - see include/strata/kernels/glm_batch.hpp.
//
// Every kernel here is the batched twin of a one-token kernel in glm_fast.cu with the same per-token arithmetic
// (same operation order where the order is observable), so a prompt read in chunks leaves the state the token path
// would have left up to float rounding - the projections around them (FP16 tensor-core GEMMs instead of q8_1 dot
// products) are where the two paths really differ, as llama.cpp's batched and one-token paths do.
#if defined(STRATA_USE_HIP)
// The HIP bf16/fp8 headers rocWMMA pulls in define the __shfl_*_sync names that the force-included hip_compat macros
// remap; hide the macros while they are read.
#pragma push_macro("__shfl_xor_sync")
#undef __shfl_xor_sync
#pragma push_macro("__shfl_down_sync")
#undef __shfl_down_sync
#pragma push_macro("__shfl_up_sync")
#undef __shfl_up_sync
#pragma push_macro("__shfl_sync")
#undef __shfl_sync
#include <rocwmma/rocwmma.hpp>
#pragma pop_macro("__shfl_xor_sync")
#pragma pop_macro("__shfl_down_sync")
#pragma pop_macro("__shfl_up_sync")
#pragma pop_macro("__shfl_sync")
#endif
#include "strata/kernels/glm_batch.hpp"
#include "dsa_topk.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::glmb {
namespace {

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
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
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
// the DSA latent cache is FP16 (uint16_t bits): read / written in F32
__device__ __forceinline__ float lat_f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t lat_h(float v) { return __half_as_ushort(__float2half(v)); }
// the INT8 latent (STRATA_GLM_LAT8; glm_fast.cu has the same layout): a row is kv_lora int8 codes, then one FP16 scale
// per 32 values (kv_lora * 17 / 16 bytes).  The prompt attention reads 4 or 8 values at a time as FP16 bits - what an
// FP16 row's uint2 / uint4 would hold (KV = 512 rows, the only width the attention takes)
__device__ __forceinline__ size_t lat8_row(int kv_lora) { return (size_t) kv_lora + (size_t) kv_lora / 16; }
__device__ __forceinline__ uint32_t lat8_pair(int8_t a, int8_t b, float sc) {
    return (uint32_t) lat_h((float) a * sc) | ((uint32_t) lat_h((float) b * sc) << 16);
}
__device__ __forceinline__ uint2 lat_ld4(const uint16_t* lat, int lat8, int cell, int c4) {
    if (!lat8) return ((const uint2*) (lat + (size_t) 512 * cell))[c4];
    const int8_t* row = (const int8_t*) lat + lat8_row(512) * (size_t) cell;
    const char4 q = ((const char4*) row)[c4];
    const float sc = __half2float(((const __half*) (row + 512))[c4 >> 3]);
    return make_uint2(lat8_pair(q.x, q.y, sc), lat8_pair(q.z, q.w, sc));
}
__device__ __forceinline__ uint4 lat_ld8(const uint16_t* lat, int lat8, int cell, int c8) {
    if (!lat8) return ((const uint4*) (lat + (size_t) 512 * cell))[c8];
    const int8_t* row = (const int8_t*) lat + lat8_row(512) * (size_t) cell;
    const uint2 w = ((const uint2*) row)[c8];
    const float sc = __half2float(((const __half*) (row + 512))[c8 >> 2]);
    const char4 a = *(const char4*) &w.x, b = *(const char4*) &w.y;
    return make_uint4(lat8_pair(a.x, a.y, sc), lat8_pair(a.z, a.w, sc), lat8_pair(b.x, b.y, sc), lat8_pair(b.z, b.w, sc));
}
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_batch %s: %s\n", what, cudaGetErrorString(e));
        g_errors.fetch_add(1, std::memory_order_relaxed);
    }
}
unsigned nblk(int64_t n, int b = 256) { return (unsigned) ((n + b - 1) / b); }

// ---------------------------------------------------------------- conversions
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float((uint32_t) src[i] << 16);
}
__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(__uint_as_float((uint32_t) src[i] << 16));
}
__global__ void embed_rows_kernel(const float* __restrict__ emb, float* __restrict__ R, int T, int E) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * 4 * E) return;
    const int64_t t = i / (4 * (int64_t) E);
    const int e = (int) (i % E);
    R[i] = emb[t * E + e];
}

// ---------------------------------------------------------------- mHC
__global__ void __launch_bounds__(256) hc_update_kernel(const float* __restrict__ block_out, float* __restrict__ R,
                                                        const float* __restrict__ post, const float* __restrict__ comb,
                                                        float* __restrict__ ss, int E) {
    const int t = blockIdx.x;
    float* Rt = R + (size_t) t * 4 * E;
    __shared__ float sred[32];
    __shared__ float sp[4], sc[16];
    if (block_out != nullptr) {
        if (threadIdx.x < 4) sp[threadIdx.x] = post[t * 4 + threadIdx.x];
        if (threadIdx.x < 16) sc[threadIdx.x] = comb[t * 16 + threadIdx.x];
    }
    __syncthreads();
    float acc = 0.0f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float r[4] = {Rt[e], Rt[E + e], Rt[2 * E + e], Rt[3 * E + e]};
        if (block_out != nullptr) {
            const float b = block_out[(size_t) t * E + e];
            float n[4];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float v = b * sp[d];
#pragma unroll
                for (int s = 0; s < 4; ++s) v += sc[d * 4 + s] * r[s];
                n[d] = v;
            }
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                r[d] = n[d];
                Rt[d * E + e] = n[d];
            }
        }
        acc += r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
    }
    if (ss == nullptr) return;
    acc = block_sum(acc, sred);
    if (threadIdx.x == 0) ss[t] = acc;
}

__global__ void __launch_bounds__(256) hc_finish_kernel(const HcFinishArgs a) {
    const int hc = 4;
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int E = a.n_embd, hc_dim = hc * E;
    __shared__ float smix[25], s_pre[4], sred[32];
    if (tid < 24) smix[tid] = a.mix[(size_t) t * 24 + tid];
    if (tid == 24) smix[24] = a.ss[t];
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            a.pre[t * 4 + lane] = pre_v;
            a.post[t * 4 + lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (lane < 16) a.comb[t * 16 + d * hc + s] = c;
    }
    __syncthreads();
    const float* Rt = a.R + (size_t) t * hc * E;
    float mixed[16];   // E <= 16 * blockDim.x
    float ss2 = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * Rt[s * E + e];
        mixed[i] = acc;
        ss2 += acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    const float inv2 = rsqrtf(ss2 / (float) E + a.norm_eps);
    __half* x16 = (__half*) a.x16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        const float xv = mixed[i] * inv2 * a.norm_w[e];
        a.x[(size_t) t * E + e] = xv;
        x16[(size_t) t * E + e] = __float2half(xv);
    }
}

// ---------------------------------------------------------------- KDA
struct Ptr3 {
    const float* p[3];
};
struct WPtr3 {
    float* p[3];
};

// one thread a channel and a run of kConvRun tokens, the window sliding in registers: each input is read once (one
// thread an output read it dconv times and split a 64-bit index twice - 6.6 ms for a 4096-token sub-batch of a
// 3090, ~7x its traffic; now ~1 ms).  The same taps in the same order: bitwise the same outputs.
constexpr int kConvRun = 16;
template <int HIST>
__global__ void __launch_bounds__(256) kda_conv_kernel(Ptr3 proj, Ptr3 w, const float* __restrict__ conv_state,
                                                       WPtr3 out, int T, int DI) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= DI) return;
    const int role = blockIdx.z, t0 = blockIdx.y * kConvRun, t1 = min(T, t0 + kConvRun);
    const float* in = proj.p[role];
    const float* st = conv_state + (size_t) role * DI * HIST + (size_t) HIST * c;
    const float* wc = w.p[role] + (size_t) (HIST + 1) * c;
    float wr[HIST + 1], win[HIST > 0 ? HIST : 1];
#pragma unroll
    for (int j = 0; j <= HIST; ++j) wr[j] = wc[j];
#pragma unroll
    for (int j = 0; j < HIST; ++j) {
        const int tt = t0 - HIST + j;
        win[j] = tt >= 0 ? in[(size_t) tt * DI + c] : st[HIST + tt];
    }
    float* o = out.p[role];
    for (int t = t0; t < t1; ++t) {
        const float cur = in[(size_t) t * DI + c];
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < HIST; ++tap) acc += wr[tap] * win[tap];
        acc += wr[HIST] * cur;
        o[(size_t) t * DI + c] = acc / (1.0f + __expf(-acc));
#pragma unroll
        for (int j = 0; j + 1 < HIST; ++j) win[j] = win[j + 1];
        if (HIST > 0) win[HIST > 0 ? HIST - 1 : 0] = cur;
    }
}

__global__ void kda_conv_state_kernel(Ptr3 proj, float* __restrict__ conv_state, int T, int DI, int dconv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * DI) return;
    const int role = i / DI, c = i - role * DI;
    const int hist = dconv - 1;
    float* st = conv_state + (size_t) role * DI * hist + (size_t) hist * c;
    float old[8];
    for (int j = 0; j < hist && j < 8; ++j) old[j] = st[j];
    for (int j = 0; j < hist && j < 8; ++j) {
        const int tt = T - hist + j;
        st[j] = tt >= 0 ? proj.p[role][(size_t) tt * DI + c] : old[hist + tt];
    }
}

__global__ void __launch_bounds__(256) kda_rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ g1_raw,
                                                      const float* __restrict__ dt_bias, const float* __restrict__ ssm_a,
                                                      float lb, const float* __restrict__ beta_raw,
                                                      float* __restrict__ state, const float* __restrict__ g2_raw,
                                                      const float* __restrict__ norm_w, float eps, int n_head, int T,
                                                      __half* __restrict__ out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, j = t & (HD - 1), half = t >> 7;
    const int DI = n_head * HD;
    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[32];
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) reg[ii] = Sh[(half * 64 + ii) * HD + j];
    const float a_h = ssm_a[h];
    const float nw = norm_w[j];
    const float dtb = t < HD ? dt_bias[h * HD + t] : 0.0f;
    for (int tok = 0; tok < T; ++tok) {
        const size_t base = (size_t) tok * DI + (size_t) h * HD;
        float qv = 0.0f, kv = 0.0f, gr = 0.0f;
        if (t < HD) {
            qv = q[base + t];
            kv = k[base + t];
            gr = g1_raw[base + t];
        }
        const float b = 1.0f / (1.0f + expf(-beta_raw[(size_t) tok * n_head + h]));
        const float vj = half == 0 ? v[base + j] : 0.0f;
        const float g2 = half == 0 ? g2_raw[base + j] : 0.0f;
        const float ssq = block_sum(qv * qv, sred);
        const float ssk = block_sum(kv * kv, sred);
        if (t < HD) {
            sq[t] = qv * rsqrtf(ssq + 1e-6f);
            sk[t] = kv * rsqrtf(ssk + 1e-6f);
            sdec[t] = __expf(lb * dsigmoid(-(gr + dtb) * a_h));
        }
        __syncthreads();
        float A = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float s = reg[ii] * sdec[i];
            reg[ii] = s;
            A += s * sk[i];
        }
        sacc[half][j] = A;
        __syncthreads();
        if (half == 0) sdel[j] = b * (vj - (sacc[0][j] + sacc[1][j]));
        __syncthreads();
        const float dl = sdel[j];
        float O = 0.0f;
#pragma unroll
        for (int ii = 0; ii < 64; ++ii) {
            const int i = half * 64 + ii;
            const float s2 = reg[ii] + sk[i] * dl;
            reg[ii] = s2;
            O += s2 * sq[i];
        }
        __syncthreads();
        sacc[half][j] = O;
        __syncthreads();
        const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
        float ss = half == 0 ? o * o : 0.0f;
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) HD + eps);
        if (half == 0) out[base + j] = __float2half(o * inv * nw * dsigmoid(g2));
    }
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) Sh[(half * 64 + ii) * HD + j] = reg[ii];
}

// ---- kda_rec in three parts (kda_rec_split): the prologue and the output gate as parallel kernels, the
//      recurrence with KJ value columns a block - each head over 128 / KJ blocks, 3 barriers a token, the next
//      token's key-dim values loaded while this one computes
__global__ void __launch_bounds__(128) kda_prep_kernel(float* __restrict__ q, float* __restrict__ k,
                                                       float* __restrict__ g1, const float* __restrict__ dt_bias,
                                                       const float* __restrict__ ssm_a, float lb, int n_head) {
    __shared__ float sred[32];
    const int h = blockIdx.y, i = threadIdx.x;
    const size_t base = ((size_t) blockIdx.x * n_head + h) * 128 + i;
    const float qv = q[base], kv = k[base];
    const float ssq = block_sum(qv * qv, sred);
    const float ssk = block_sum(kv * kv, sred);
    q[base] = qv * rsqrtf(ssq + 1e-6f);
    k[base] = kv * rsqrtf(ssk + 1e-6f);
    g1[base] = __expf(lb * dsigmoid(-(g1[base] + dt_bias[h * 128 + i]) * ssm_a[h]));
}

template <int KJ>
__global__ void __launch_bounds__(256) kda_rec_split_kernel(const float* __restrict__ qn, const float* __restrict__ kn,
                                                            float* v, const float* __restrict__ dec,
                                                            const float* __restrict__ beta_raw,
                                                            float* __restrict__ state, int n_head, int T) {
    constexpr int HD = 128, NG = 256 / KJ, IPT = HD / NG;   // NG groups of IPT key dims, a thread a column of each
    const int h = blockIdx.x, t = threadIdx.x, jj = t % KJ, gi = t / KJ;
    const int j = blockIdx.y * KJ + jj;
    const int DI = n_head * HD;
    __shared__ float sq[HD], sk[HD], sd[HD];
    __shared__ float sA[NG][KJ], sO[NG][KJ];
    float* Sh = state + (size_t) h * HD * HD;
    float reg[IPT];
#pragma unroll
    for (int ii = 0; ii < IPT; ++ii) reg[ii] = Sh[(gi * IPT + ii) * HD + j];
    float pq = 0.0f, pk = 0.0f, pd = 0.0f;
    if (t < HD && T > 0) {
        pq = qn[(size_t) h * HD + t];
        pk = kn[(size_t) h * HD + t];
        pd = dec[(size_t) h * HD + t];
    }
    for (int tok = 0; tok < T; ++tok) {
        const size_t base = (size_t) tok * DI + (size_t) h * HD;
        // (every read of sq/sk/sd for the token before ended before its third barrier)
        if (t < HD) {
            sq[t] = pq;
            sk[t] = pk;
            sd[t] = pd;
            if (tok + 1 < T) {
                pq = qn[base + DI + t];
                pk = kn[base + DI + t];
                pd = dec[base + DI + t];
            }
        }
        const float vj = v[base + j];
        const float b = 1.0f / (1.0f + expf(-beta_raw[(size_t) tok * n_head + h]));
        __syncthreads();
        float A = 0.0f;
#pragma unroll
        for (int ii = 0; ii < IPT; ++ii) {
            const int i = gi * IPT + ii;
            const float s = reg[ii] * sd[i];
            reg[ii] = s;
            A += s * sk[i];
        }
        sA[gi][jj] = A;
        __syncthreads();
        float At = 0.0f;
#pragma unroll
        for (int g = 0; g < NG; ++g) At += sA[g][jj];
        const float dl = b * (vj - At);
        float O = 0.0f;
#pragma unroll
        for (int ii = 0; ii < IPT; ++ii) {
            const int i = gi * IPT + ii;
            const float s2 = reg[ii] + sk[i] * dl;
            reg[ii] = s2;
            O += s2 * sq[i];
        }
        sO[gi][jj] = O;
        __syncthreads();
        if (gi == 0) {   // the raw output over v: only this column's threads read v[base + j], before the barriers
            float Ot = 0.0f;
#pragma unroll
            for (int g = 0; g < NG; ++g) Ot += sO[g][jj];
            v[base + j] = Ot;
        }
    }
#pragma unroll
    for (int ii = 0; ii < IPT; ++ii) Sh[(gi * IPT + ii) * HD + j] = reg[ii];
}

__global__ void __launch_bounds__(128) kda_out_kernel(const float* __restrict__ o_raw, const float* __restrict__ g2_raw,
                                                      const float* __restrict__ norm_w, float eps, int n_head,
                                                      __half* __restrict__ out) {
    __shared__ float sred[32];
    const int j = threadIdx.x;
    const size_t base = ((size_t) blockIdx.x * n_head + blockIdx.y) * 128 + j;
    const float o = o_raw[base] * rsqrtf(128.0f);
    const float ss = block_sum(o * o, sred);
    const float inv = rsqrtf(ss / 128.0f + eps);
    out[base] = __float2half(o * inv * norm_w[j] * dsigmoid(g2_raw[base]));
}

// ---------------------------------------------------------------- DSA
__global__ void __launch_bounds__(512) dsa_prep_kernel(const DsaPrepArgs a) {
    __shared__ float sred[32];
    const int t = blockIdx.x, role = blockIdx.y, tid = threadIdx.x;
    const int p = a.p0 + t;
    if (role == 0) {
        if (a.qr_raw == nullptr) return;   // the NextN block's cache fill needs no queries
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora && n < 4; e += 512, ++n) {
            vals[n] = a.qr_raw[(size_t) t * a.q_lora + e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        __half* q16 = (__half*) a.qr16;
        for (int e = tid; e < a.q_lora && n < 4; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            a.qr[(size_t) t * a.q_lora + e] = y;
            q16[(size_t) t * a.q_lora + e] = __float2half(y);
        }
        return;
    }
    if (role == 1) {
        const float v = tid < a.kv_lora ? a.kv_raw[(size_t) t * a.kv_lora + tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (a.lat8) {   // INT8: one scale per warp's 32 values (kv_lora % 32 == 0: a warp is all in or all out)
            if (tid < a.kv_lora) {
                const float y = v * inv * a.kv_norm[tid];
                int8_t* row = (int8_t*) a.lat + lat8_row(a.kv_lora) * (size_t) p;
                const __half hs = __float2half(warp_max(fabsf(y)) / 127.0f);
                const float sc = __half2float(hs);
                row[tid] = (int8_t) fmaxf(-127.0f, fminf(127.0f, sc > 0.0f ? rintf(y / sc) : 0.0f));
                if ((tid & 31) == 0) ((__half*) (row + a.kv_lora))[tid >> 5] = hs;
            }
        } else if (tid < a.kv_lora) {
            a.lat[(size_t) a.kv_lora * p + tid] = lat_h(v * inv * a.kv_norm[tid]);
        }
        return;
    }
    const int K = a.idx_key;
    const float v = tid < K ? a.ik_raw[(size_t) t * K + tid] : 0.0f;
    const float s1 = block_sum(v, sred);
    const float s2 = block_sum(v * v, sred);
    const float mu = s1 / (float) K;
    const float inv = rsqrtf(s2 / (float) K - mu * mu + a.eps);
    const size_t row = (size_t) (a.ring > 0 ? p % a.ring : p);   // a ring of a.ring rows (0: row p)
    if (tid < K) {
        a.ik_cache[(size_t) K * row + tid] = (v - mu) * inv * a.k_norm_w[tid] + a.k_norm_b[tid];
        a.ig_cache[(size_t) K * row + tid] = a.ig_raw[(size_t) t * K + tid];
    }
}

__global__ void dsa_pool_kernel(const float* __restrict__ ik, const float* __restrict__ ig, const float* __restrict__ ape,
                                float* __restrict__ pooled, int K, int kpool, int pool0, int ring) {
    const int pi = pool0 + blockIdx.x, tid = threadIdx.x;
    if (tid >= K) return;
    // the pool's first member's row (ring a multiple of kpool: the members stay contiguous)
    const size_t row0 = (size_t) (ring > 0 ? ((int64_t) pi * kpool) % ring : (int64_t) pi * kpool);
    float lg[16];
    float mx = -INFINITY;
    for (int m = 0; m < kpool && m < 16; ++m) {
        lg[m] = ig[(size_t) tid + (size_t) K * (row0 + m)] + ape[tid + K * m];
        mx = fmaxf(mx, lg[m]);
    }
    float den = 0.0f;
    for (int m = 0; m < kpool && m < 16; ++m) {
        lg[m] = expf(lg[m] - mx);
        den += lg[m];
    }
    float acc = 0.0f;
    for (int m = 0; m < kpool && m < 16; ++m) acc += (lg[m] / den) * ik[(size_t) tid + (size_t) K * (row0 + m)];
    pooled[(size_t) tid + (size_t) K * pi] = acc;
}

constexpr int SCORE_POOLS = 64;   // pools per block (8 warps x 8)
__global__ void __launch_bounds__(256) dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                        const float* __restrict__ iw, int key_dim, int idx_heads,
                                                        int p0, int kpool, float* __restrict__ score, int score_ld) {
    extern __shared__ float s_iq[];   // idx_heads * key_dim, then idx_heads weights
    const int t = blockIdx.y;
    const int n_vis = (p0 + t + 1) / kpool;
    const int pbeg = blockIdx.x * SCORE_POOLS;
    if (pbeg >= n_vis) return;
    float* s_iw = s_iq + idx_heads * key_dim;
    for (int i = threadIdx.x; i < idx_heads * key_dim; i += blockDim.x) s_iq[i] = iq[(size_t) t * idx_heads * key_dim + i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_iw[i] = iw[(size_t) t * idx_heads + i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int pend = min(n_vis, pbeg + SCORE_POOLS);
    for (int p = pbeg + warp; p < pend; p += 8) {
        const float* pk = pooled + (size_t) key_dim * p;
        float acc = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            float dot = 0.0f;
            for (int e = lane; e < key_dim; e += 32) dot += s_iq[h * key_dim + e] * pk[e];
            dot = warp_sum(dot);
            acc += fmaxf(dot, 0.0f) * s_iw[h];
        }
        if (lane == 0) score[(size_t) t * score_ld + p] = acc;
    }
}

// The same scores as a register-tiled FP32 GEMM (idx_heads 32, key_dim 128): a block takes 4 tokens' 128 (token, head)
// query rows against 64 pools, k in chunks of 32 through shared memory, 8 x 4 accumulators a thread; the epilogue sums
// relu(dot) * w over each token's 32 heads.  FP32 throughout - only the summation order differs from dsa_score_kernel
// (which split each 128-long dot across a warp and shuffled it down once a head and pool: ~0.9 TFLOPS on a 3090,
// 111 ms for 4096 tokens at position 10240; this one ~15x faster).
constexpr int TS_T = 4, TS_P = 64, TS_K = 32;
__global__ void __launch_bounds__(256) dsa_score_tiled_kernel(const float* __restrict__ iq,
                                                              const float* __restrict__ pooled,
                                                              const float* __restrict__ iw, int p0, int kpool, int T,
                                                              int max_vis, float* __restrict__ score, int score_ld) {
    constexpr int H = 32, K = 128, R = TS_T * H;   // 128 query rows
    __shared__ float As[TS_K][R + 4];              // [k][row]
    __shared__ float Bs[TS_K][TS_P + 4];           // [k][pool]
    __shared__ float red[4][TS_T][TS_P];           // the 4 head quarters' partial sums
    const int t0 = blockIdx.y * TS_T, pbeg = blockIdx.x * TS_P;
    const int t_last = min(T, t0 + TS_T) - 1;
    if (pbeg >= (p0 + t_last + 1) / kpool) return;   // no pool of this tile is visible to any of its tokens
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;   // rows ty*8 .. +7, pools tx*4 .. +3
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS_K) {
        // A: 128 rows x 32 k = 1024 float4, 4 a thread (row = token-in-tile * 32 + head)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, r = idx >> 3, kk = (idx & 7) * 4;
            const int t = t0 + (r >> 5);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < T) v = *(const float4*) (iq + ((size_t) t * H + (r & 31)) * K + k0 + kk);
            As[kk + 0][r] = v.x;
            As[kk + 1][r] = v.y;
            As[kk + 2][r] = v.z;
            As[kk + 3][r] = v.w;
        }
        // B: 64 pools x 32 k = 512 float4, 2 a thread
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, pp = idx >> 3, kk = (idx & 7) * 4;
            const int p = pbeg + pp;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < max_vis) v = *(const float4*) (pooled + (size_t) p * K + k0 + kk);
            Bs[kk + 0][pp] = v.x;
            Bs[kk + 1][pp] = v.y;
            Bs[kk + 2][pp] = v.z;
            Bs[kk + 3][pp] = v.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TS_K; ++kk) {
            const float4 a0 = *(const float4*) &As[kk][ty * 8], a1 = *(const float4*) &As[kk][ty * 8 + 4];
            const float4 b = *(const float4*) &Bs[kk][tx * 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
        }
        __syncthreads();
    }
    // epilogue: this thread's 8 heads (quarter ty & 3 of token ty >> 2) -> relu * w, summed; then the 4 quarters
    const int tt = ty >> 2, qh = ty & 3, t = t0 + tt;
    float wv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = t < T ? iw[(size_t) t * H + qh * 8 + i] : 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum += fmaxf(acc[i][j], 0.0f) * wv[i];
        red[qh][tt][tx * 4 + j] = sum;
    }
    __syncthreads();
    // 256 threads = 4 tokens x 64 pools
    {
        const int ot = tid >> 6, op = tid & 63, tok = t0 + ot, p = pbeg + op;
        if (tok < T && p < (p0 + tok + 1) / kpool)
            score[(size_t) tok * score_ld + p] = red[0][ot][op] + red[1][ot][op] + red[2][ot][op] + red[3][ot][op];
    }
}

__global__ void __launch_bounds__(1024) dsa_select_kernel(const float* __restrict__ score, int score_ld, int p0,
                                                          int kpool, int top_pools_max, int tail, int n_sel_max,
                                                          int* __restrict__ cells_all, int* __restrict__ n_sel_out) {
    const int t = blockIdx.x;
    const int pos = p0 + t;
    const int n_vis = (pos + 1) / kpool;
    const int top = min(top_pools_max, n_vis);
    const int n_sel = kpool * top + (tail ? kpool - 1 : 0);
    int* cells = cells_all + (size_t) t * n_sel_max;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) cells[i] = -1;
    if (threadIdx.x == 0) n_sel_out[t] = n_sel;
    __syncthreads();
    if (n_vis <= top_pools_max) {
        for (int i = threadIdx.x; i < n_vis * kpool; i += blockDim.x) cells[i] = i;
    } else {
        // the top pools in rank order, O(n_vis) - src/kernels/cuda/dsa_topk.cuh
        strata::kernels::dsa::select_top_pools(score + (size_t) t * score_ld, n_vis, top, kpool, cells);
    }
    __syncthreads();
    if (threadIdx.x == 0 && tail) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos) cells[top * kpool + m] = cell;
        }
    }
}

// MLA attention for a chunk: block = (token, 16 heads).  The token's cells are walked in chunks of 32 latent rows
// (loaded into shared memory once for the 16 heads), with an online softmax per head; thread c owns the context
// columns c and c + 256 of all 16 heads.  The rows stay FP16 in shared memory, as in the cache (read as F32, the
// same values): 34 KB, under the 48 KB every card gives without an opt-in - F32 rows (66 KB) were over Turing's
// 64 KB, and every launch failed there (issue #8).
constexpr int MB_HG = 16, MB_CH = 32;
constexpr size_t kF32Smem = (size_t) MB_CH * 512 * sizeof(uint16_t) + (size_t) MB_HG * MB_CH * sizeof(float);
static_assert(kF32Smem <= 48 * 1024, "the F32 prompt attention must fit the default 48 KB of shared memory");
__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                       const int* __restrict__ cells_all, const int* __restrict__ n_sel_arr,
                                                       int n_sel_max, int n_head, float scale, float* __restrict__ ctx,
                                                       int lat8) {
    constexpr int KV = 512;
    extern __shared__ float sm[];
    uint16_t* sL = (uint16_t*) sm;   // MB_CH x KV, FP16
    float* sP = sm + MB_CH * KV / 2; // MB_HG x MB_CH
    __shared__ float s_m[MB_HG], s_l[MB_HG], s_scale[MB_HG];
    __shared__ int s_cell[MB_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * MB_HG, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    float q0[16], q1[16];
    {
        const float* qa = q_abs + ((size_t) t * n_head + h0 + 2 * warp) * KV;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            q0[j] = qa[lane + 32 * j];
            q1[j] = qa[KV + lane + 32 * j];
        }
    }
    float acc0[MB_HG], acc1[MB_HG];
#pragma unroll
    for (int hh = 0; hh < MB_HG; ++hh) acc0[hh] = acc1[hh] = 0.0f;
    if (tid < MB_HG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    for (int s0 = 0; s0 < ns; s0 += MB_CH) {
        __syncthreads();
        if (tid < MB_CH) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
        __syncthreads();
        for (int i = tid; i < MB_CH * KV / 4; i += blockDim.x) {
            const int s = i / (KV / 4), c4 = i - s * (KV / 4);
            const int cell = s_cell[s];
            ((uint2*) sL)[i] = cell >= 0 ? lat_ld4(lat, lat8, cell, c4) : make_uint2(0u, 0u);
        }
        __syncthreads();
        for (int s = 0; s < MB_CH; ++s) {
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float l = lat_f(sL[s * KV + lane + 32 * j]);
                a0 += q0[j] * l;
                a1 += q1[j] * l;
            }
            a0 = warp_sum(a0);
            a1 = warp_sum(a1);
            if (lane == 0) {
                const bool ok = s_cell[s] >= 0;
                sP[(2 * warp) * MB_CH + s] = ok ? a0 * scale : -INFINITY;
                sP[(2 * warp + 1) * MB_CH + s] = ok ? a1 * scale : -INFINITY;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            const float v = sP[h * MB_CH + lane];
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            sP[h * MB_CH + lane] = e;
            __syncwarp();
            if (lane == 0) {
                const float sc = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                s_scale[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < MB_HG; ++hh) {
            const float sc = s_scale[hh];
            acc0[hh] *= sc;
            acc1[hh] *= sc;
        }
        for (int s = 0; s < MB_CH; ++s) {
            const float l0 = lat_f(sL[s * KV + tid]), l1 = lat_f(sL[s * KV + tid + 256]);
#pragma unroll
            for (int hh = 0; hh < MB_HG; ++hh) {
                const float p = sP[hh * MB_CH + s];
                acc0[hh] += p * l0;
                acc1[hh] += p * l1;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < MB_HG; ++hh) {
        const float L = s_l[hh];
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        float* o = ctx + ((size_t) t * n_head + h0 + hh) * KV;
        o[tid] = acc0[hh] * inv;
        o[tid + 256] = acc1[hh] * inv;
    }
}

#if !defined(STRATA_USE_HIP)
// The same attention on the tensor cores (sm_70+): block = (token, 16 heads), 8 warps.  The token's cells are walked
// in chunks of 32: their latent rows go to FP16 in shared memory, the scores S = Q16 . L^T come from the tensor cores
// (two 16-cell tiles, the K = 512 split in four quarters across the warps and summed), the softmax is the online one
// above in F32, and the context O (F32, in shared memory) is rescaled and O += P . L (32 column tiles, 4 a warp).
// FP16 operands with F32 accumulation, as flash attention: the context is within ~1e-3 of the F32 kernel's.
// 91 KB of shared memory a block - more than Turing allows (64 KB): there the register variant below runs.
constexpr int TC_HG = 16, TC_CH = 32;
constexpr size_t kTcSmem = (size_t) TC_HG * 512 * 4 + (size_t) TC_HG * 512 * 2 + (size_t) TC_CH * 512 * 2 +
                           (size_t) 4 * TC_HG * TC_CH * 4 + (size_t) TC_HG * TC_CH * 2;
__global__ void __launch_bounds__(256) mla_attn_tc_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                          const int* __restrict__ cells_all,
                                                          const int* __restrict__ n_sel_arr, int n_sel_max, int n_head,
                                                          float scale, float* __restrict__ ctx, int lat8) {
    using namespace nvcuda;
    constexpr int KV = 512;
    extern __shared__ __align__(128) unsigned char tc_sm[];
    float* sO = (float*) tc_sm;                          // TC_HG x KV, F32
    __half* sQ = (__half*) (sO + TC_HG * KV);            // TC_HG x KV
    __half* sL = sQ + TC_HG * KV;                        // TC_CH x KV
    float* sPart = (float*) (sL + TC_CH * KV);           // 4 x TC_HG x TC_CH: the scores' K quarters
    __half* sP = (__half*) (sPart + 4 * TC_HG * TC_CH);  // TC_HG x TC_CH: exp(s - m)
    __shared__ float s_m[TC_HG], s_l[TC_HG], s_sc[TC_HG];
    __shared__ int s_cell[TC_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * TC_HG, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    const float* qa = q_abs + ((size_t) t * n_head + h0) * KV;
    for (int i = tid; i < TC_HG * KV; i += blockDim.x) {
        sQ[i] = __float2half(qa[i]);
        sO[i] = 0.0f;
    }
    if (tid < TC_HG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // a chunk's latent rows travel through registers (32 rows x 64 uint4 = 8 a thread): the next chunk's loads are
    // issued before this chunk's arithmetic, so they arrive while it runs
    constexpr int PRE = TC_CH * (KV / 8) / 256;
    uint4 pre[PRE];
    const auto issue = [&](int c0) {
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            const int cell = c0 + s < ns ? cl[c0 + s] : -1;
            pre[k] = cell >= 0 ? lat_ld8(lat, lat8, cell, c8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    issue(0);
    for (int s0 = 0; s0 < ns; s0 += TC_CH) {
        __syncthreads();   // the previous chunk's product has read sL and sP
        if (tid < TC_CH) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            ((uint4*) (sL + s * KV))[c8] = pre[k];
        }
        if (s0 + TC_CH < ns) issue(s0 + TC_CH);
        __syncthreads();
        {
            const int n0 = (warp & 1) * 16, kq = (warp >> 1) * (KV / 4);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < KV / 4; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, sQ + kq + k, KV);
                wmma::load_matrix_sync(b, sL + (size_t) n0 * KV + kq + k, KV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sPart + (warp >> 1) * TC_HG * TC_CH + n0, acc, TC_CH, wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            float v = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) v += sPart[q * TC_HG * TC_CH + h * TC_CH + lane];
            v = s_cell[lane] >= 0 ? v * scale : -INFINITY;
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            sP[h * TC_CH + lane] = __float2half(e);
            if (lane == 0) {
                const float sc = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                s_sc[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        for (int i = tid; i < TC_HG * KV; i += blockDim.x) sO[i] *= s_sc[i / KV];
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
            wmma::load_matrix_sync(a0, sP, TC_CH);
            wmma::load_matrix_sync(a1, sP + 16, TC_CH);
#pragma unroll
            for (int j = 0; j < KV / 16 / 8; ++j) {
                const int c0 = (warp * (KV / 16 / 8) + j) * 16;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
                wmma::load_matrix_sync(acc, sO + c0, KV, wmma::mem_row_major);
                wmma::load_matrix_sync(b0, sL + c0, KV);
                wmma::load_matrix_sync(b1, sL + 16 * KV + c0, KV);
                wmma::mma_sync(acc, a0, b0, acc);
                wmma::mma_sync(acc, a1, b1, acc);
                wmma::store_matrix_sync(sO + c0, acc, KV, wmma::mem_row_major);
            }
        }
    }
    __syncthreads();
    float* o = ctx + ((size_t) t * n_head + h0) * KV;
    for (int i = tid; i < TC_HG * KV; i += blockDim.x) {
        const float L = s_l[i / KV];
        o[i] = L > 0.0f ? sO[i] / L : 0.0f;
    }
}

// The same tensor-core attention with the context O in the wmma accumulator registers (four 16-wide column tiles a
// warp) instead of shared memory: 58368 bytes a block, so Turing (64 KB) runs it - by @dummerjindabin (#10).  The
// per-chunk rescale multiplies the fragments' elements by their rows, which assumes the sm_75+ accumulator layout
// (lane l holds rows l/4 and l/4 + 8); Volta's differs (a V100 measured rel L2 0.8-7.9 against the F32 kernel), so
// mla_attn picks it only where the shared-memory kernel does not fit, or with STRATA_GLM_PREFILL_ATTN=tcreg.
constexpr size_t kTcRegSmem = (size_t) TC_HG * 512 * 2 + (size_t) TC_CH * 512 * 2 +
                              (size_t) 4 * TC_HG * TC_CH * 4 + (size_t) TC_HG * TC_CH * 2;
__global__ void __launch_bounds__(256) mla_attn_tc_reg_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                              const int* __restrict__ cells_all,
                                                              const int* __restrict__ n_sel_arr, int n_sel_max, int n_head,
                                                              float scale, float* __restrict__ ctx, int lat8) {
    using namespace nvcuda;
    constexpr int KV = 512;
    extern __shared__ __align__(128) unsigned char tc_sm[];
    __half* sQ = (__half*) tc_sm;                        // TC_HG x KV
    __half* sL = sQ + TC_HG * KV;                        // TC_CH x KV
    float* sPart = (float*) (sL + TC_CH * KV);           // 4 x TC_HG x TC_CH: the scores' K quarters
    __half* sP = (__half*) (sPart + 4 * TC_HG * TC_CH);  // TC_HG x TC_CH: exp(s - m)
    __shared__ float s_m[TC_HG], s_l[TC_HG], s_sc[TC_HG];
    __shared__ int s_cell[TC_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * TC_HG, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    const float* qa = q_abs + ((size_t) t * n_head + h0) * KV;
    for (int i = tid; i < TC_HG * KV; i += blockDim.x) sQ[i] = __float2half(qa[i]);
    // the context O: four 16-wide column tiles a warp, held in registers for the whole cell walk
    constexpr int TC_OT = KV / 16 / 8;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accO[TC_OT];
#pragma unroll
    for (int j = 0; j < TC_OT; ++j) wmma::fill_fragment(accO[j], 0.0f);
    if (tid < TC_HG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // a chunk's latent rows travel through registers (32 rows x 64 uint4 = 8 a thread): the next chunk's loads are
    // issued before this chunk's arithmetic, so they arrive while it runs
    constexpr int PRE = TC_CH * (KV / 8) / 256;
    uint4 pre[PRE];
    const auto issue = [&](int c0) {
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            const int cell = c0 + s < ns ? cl[c0 + s] : -1;
            pre[k] = cell >= 0 ? lat_ld8(lat, lat8, cell, c8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    issue(0);
    for (int s0 = 0; s0 < ns; s0 += TC_CH) {
        __syncthreads();   // the previous chunk's product has read sL and sP
        if (tid < TC_CH) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            ((uint4*) (sL + s * KV))[c8] = pre[k];
        }
        if (s0 + TC_CH < ns) issue(s0 + TC_CH);
        __syncthreads();
        {
            const int n0 = (warp & 1) * 16, kq = (warp >> 1) * (KV / 4);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < KV / 4; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, sQ + kq + k, KV);
                wmma::load_matrix_sync(b, sL + (size_t) n0 * KV + kq + k, KV);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sPart + (warp >> 1) * TC_HG * TC_CH + n0, acc, TC_CH, wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            float v = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) v += sPart[q * TC_HG * TC_CH + h * TC_CH + lane];
            v = s_cell[lane] >= 0 ? v * scale : -INFINITY;
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            sP[h * TC_CH + lane] = __float2half(e);
            if (lane == 0) {
                const float sc = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                s_sc[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        // rescale the register context by this chunk's factor: lane l holds rows (l>>2) and (l>>2)+8
        {
            const int grp = lane >> 2;
            const float lo = s_sc[grp], hi = s_sc[grp + 8];
#pragma unroll
            for (int j = 0; j < TC_OT; ++j) {
                float* x = accO[j].x;
                x[0] *= lo; x[1] *= lo; x[4] *= lo; x[5] *= lo;
                x[2] *= hi; x[3] *= hi; x[6] *= hi; x[7] *= hi;
            }
        }
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
            wmma::load_matrix_sync(a0, sP, TC_CH);
            wmma::load_matrix_sync(a1, sP + 16, TC_CH);
#pragma unroll
            for (int j = 0; j < TC_OT; ++j) {
                const int c0 = (warp * TC_OT + j) * 16;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b0, b1;
                wmma::load_matrix_sync(b0, sL + c0, KV);
                wmma::load_matrix_sync(b1, sL + 16 * KV + c0, KV);
                wmma::mma_sync(accO[j], a0, b0, accO[j]);
                wmma::mma_sync(accO[j], a1, b1, accO[j]);
            }
        }
    }
    float* o = ctx + ((size_t) t * n_head + h0) * KV;
#pragma unroll
    for (int j = 0; j < TC_OT; ++j) {
        const int c0 = (warp * TC_OT + j) * 16;
        wmma::store_matrix_sync(o + c0, accO[j], KV, wmma::mem_row_major);
    }
    __syncthreads();   // every warp's tiles must be in ctx before the row normalization reads them
    for (int i = tid; i < TC_HG * KV; i += blockDim.x) {
        const float L = s_l[i / KV];
        o[i] = L > 0.0f ? o[i] / L : 0.0f;
    }
}

// The same attention on Ampere and newer (sm_80+: mma.sync m16n8k16 and ldmatrix), FlashAttention-2 style: block =
// (token, 32 heads), 8 warps.  The 32 heads' queries and each tile of 32 selected latent rows sit in shared memory as
// FP16 (rows padded by 16 bytes: the ldmatrix row addresses then fall in distinct banks); S = Q . L^T and O += P . L
// run as mma.sync with F32 accumulators and O in registers (72 KB of shared memory a block); each gathered latent
// tile serves twice the heads of the kernel above.  The softmax is online per head row, its rescale applied to the O
// fragments (m16n8's C layout: a thread's rows are lane / 4 and lane / 4 + 8).  The absorbed queries are not bounded,
// so each head's row is scaled by a power of two to a largest magnitude near 2^14 (exact; undone on its scores) -
// FP16's mantissa then keeps the result within ~1e-3 of the F32 kernel (rel L2 1.1e-4 over a 7.8K-token prompt).
// RTX 3090, one GPU: prompts of 14.8K / 26.6K tokens 674 / 832 tok/s with the 91 KB kernel, 759 / 966 with this.
constexpr int MT_H = 32, MT_C = 32, MT_KV = 512, MT_LD = MT_KV + 8;   // heads, cells a tile, latent, padded row (fp16)

// ldmatrix needs sm_75 and this mma sm_80: below sm_80 (a build for Volta or Turing) the helpers compile to nothing -
// mla_attn picks this kernel only on Ampere and newer
__device__ __forceinline__ void mt_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    (void) p;
    r0 = r1 = r2 = r3 = 0u;
#endif
}
__device__ __forceinline__ void mt_ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
#else
    (void) p;
    r0 = r1 = 0u;
#endif
}
__device__ __forceinline__ void mt_ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
#else
    (void) p;
    r0 = r1 = 0u;
#endif
}
__device__ __forceinline__ void mt_mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

constexpr size_t kMtSmem = ((size_t) (MT_H + MT_C) * MT_LD + (size_t) MT_H * (MT_C + 8)) * sizeof(__half) +
                           (size_t) MT_H * MT_C * sizeof(float);
__global__ void __launch_bounds__(256) mla_attn_mma_kernel(const float* __restrict__ q_abs,
                                                           const uint16_t* __restrict__ lat,
                                                           const int* __restrict__ cells_all,
                                                           const int* __restrict__ n_sel_arr, int n_sel_max, int n_head,
                                                           float scale, float* __restrict__ ctx, int lat8) {
    extern __shared__ __align__(16) unsigned char mt_sm[];
    __half* sQ = (__half*) mt_sm;                 // MT_H x MT_LD
    __half* sL = sQ + MT_H * MT_LD;               // MT_C x MT_LD
    __half* sP = sL + MT_C * MT_LD;               // MT_H x (MT_C + 8)
    float* sS = (float*) (sP + MT_H * (MT_C + 8));              // MT_H x MT_C
    __shared__ float s_m[MT_H], s_l[MT_H], s_sc[MT_H], s_qs[MT_H];
    __shared__ int s_cell[MT_C];
    const int t = blockIdx.x, h0 = blockIdx.y * MT_H, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    // the 32 heads' queries -> FP16, each row scaled by 2^e to a largest magnitude in [2^13, 2^14) (warp w: rows 4w..)
    for (int rr = 0; rr < MT_H / 8; ++rr) {
        const int r = warp * (MT_H / 8) + rr;
        const float* qr = q_abs + ((size_t) t * n_head + h0 + r) * MT_KV;
        float mx = 0.f;
        for (int c = lane; c < MT_KV; c += 32) mx = fmaxf(mx, fabsf(qr[c]));
        mx = warp_max(mx);
        int ex = 0;
        if (mx > 0.f) {
            frexpf(mx, &ex);                       // mx in [2^(ex-1), 2^ex)
            ex = max(-60, min(60, 14 - ex));      // mx * 2^ex in [2^13, 2^14)
        }
        const float f = ldexpf(1.0f, ex);
        for (int c = 2 * lane; c < MT_KV; c += 64)
            *(__half2*) (sQ + r * MT_LD + c) = __floats2half2_rn(qr[c] * f, qr[c + 1] * f);
        if (lane == 0) s_qs[r] = ldexpf(1.0f, -ex);
    }
    if (tid < MT_H) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // O fragments: warp w owns columns [64 w, 64 w + 64) - 8 n-tiles - of both m-tiles (rows 0-15, 16-31)
    float acc[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.0f;
    const int g = lane >> 2, q2 = (lane & 3) * 2;
    for (int s0 = 0; s0 < ns; s0 += MT_C) {
        __syncthreads();
        if (tid < MT_C) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
        __syncthreads();
        for (int i = tid; i < MT_C * MT_KV / 8; i += blockDim.x) {
            const int r = i / (MT_KV / 8), c8 = i - r * (MT_KV / 8);
            const int cell = s_cell[r];
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (cell >= 0) v = lat_ld8(lat, lat8, cell, c8);
            *(uint4*) (sL + r * MT_LD + 8 * c8) = v;
        }
        __syncthreads();
        // S = Q . L^T: warp w -> m-tile w / 4 (16 heads), n-tile w % 4 (8 cells); 32 k-steps of 16
        {
            const int mt = warp >> 2, nt = warp & 3;
            float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
            for (int ks = 0; ks < MT_KV / 16; ++ks) {
                uint32_t a0, a1, a2, a3, b0, b1;
                mt_ldsm_x4(a0, a1, a2, a3, sQ + (mt * 16 + (lane & 15)) * MT_LD + ks * 16 + (lane >> 4) * 8);
                mt_ldsm_x2(b0, b1, sL + (nt * 8 + (lane & 7)) * MT_LD + ks * 16 + ((lane >> 3) & 1) * 8);
                mt_mma(sacc, a0, a1, a2, a3, b0, b1);
            }
            const int r0 = mt * 16 + g, c0 = nt * 8 + q2;
            const float fa = scale * s_qs[r0], fb = scale * s_qs[r0 + 8];
            sS[r0 * MT_C + c0] = sacc[0] * fa;
            sS[r0 * MT_C + c0 + 1] = sacc[1] * fa;
            sS[(r0 + 8) * MT_C + c0] = sacc[2] * fb;
            sS[(r0 + 8) * MT_C + c0 + 1] = sacc[3] * fb;
        }
        __syncthreads();
        // the online softmax: 8 threads a head row, 4 cells each
        {
            const int r = tid >> 3, j0 = (tid & 7) * 4;
            float v[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = s_cell[j0 + j] >= 0 ? sS[r * MT_C + j0 + j] : -INFINITY;
                mx = fmaxf(mx, v[j]);
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = s_m[r], m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = v[j] == -INFINITY ? 0.f : __expf(v[j] - m_new);
                sum += e;
                sP[r * (MT_C + 8) + j0 + j] = __float2half(e);
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if ((tid & 7) == 0) {
                const float sc = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
                s_sc[r] = sc;
                s_l[r] = s_l[r] * sc + sum;
                s_m[r] = m_new;
            }
        }
        __syncthreads();
        // O = O * scale(row) + P . L: warp w's 8 n-tiles of both m-tiles, 2 k-steps of 16 cells
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const float sa = s_sc[mt * 16 + g], sb = s_sc[mt * 16 + g + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mt][nt][0] *= sa;
                acc[mt][nt][1] *= sa;
                acc[mt][nt][2] *= sb;
                acc[mt][nt][3] *= sb;
            }
        }
#pragma unroll
        for (int ks = 0; ks < MT_C / 16; ++ks) {
            uint32_t pa[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                mt_ldsm_x4(pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3],
                           sP + (mt * 16 + (lane & 15)) * (MT_C + 8) + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                uint32_t b0, b1;
                mt_ldsm_x2_t(b0, b1, sL + (ks * 16 + (lane & 15)) * MT_LD + warp * 64 + nt * 8);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) mt_mma(acc[mt][nt], pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], b0, b1);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int ra = mt * 16 + g, rb = ra + 8;
        const float ia = s_l[ra] > 0.f ? 1.f / s_l[ra] : 0.f, ib = s_l[rb] > 0.f ? 1.f / s_l[rb] : 0.f;
        float* oa = ctx + ((size_t) t * n_head + h0 + ra) * MT_KV;
        float* ob = ctx + ((size_t) t * n_head + h0 + rb) * MT_KV;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int c = warp * 64 + nt * 8 + q2;
            *(float2*) (oa + c) = make_float2(acc[mt][nt][0] * ia, acc[mt][nt][1] * ia);
            *(float2*) (ob + c) = make_float2(acc[mt][nt][2] * ib, acc[mt][nt][3] * ib);
        }
    }
}
#endif

#if defined(STRATA_USE_HIP)
// mla_attn_tc_kernel on RDNA3 WMMA (rocWMMA).  RDNA allows 64 KiB of LDS a workgroup, the CUDA kernel takes 91,
// so Q (FP16, from the q_abs product) stays in registers - each warp holds its 64-wide K slice of the 16 heads as four
// A fragments - and the cells go in chunks of 16: O 32 KiB (F32) + L 16 KiB + score partials 8 KiB + P 0.5 KiB.
// Same arithmetic as the CUDA kernel: FP16 operands, F32 accumulation, the online softmax in F32.
constexpr int HW_HG = 16, HW_CH = 16;
constexpr size_t kHwSmem = (size_t) HW_HG * 512 * 4 + (size_t) HW_CH * 512 * 2 + (size_t) 8 * HW_HG * HW_CH * 4 +
                           (size_t) HW_HG * HW_CH * 2;
__global__ void __launch_bounds__(256) mla_attn_wmma_kernel(const _Float16* __restrict__ q16,
                                                            const uint16_t* __restrict__ lat,
                                                            const int* __restrict__ cells_all,
                                                            const int* __restrict__ n_sel_arr, int n_sel_max,
                                                            int n_head, float scale, float* __restrict__ ctx, int lat8) {
    namespace wm = rocwmma;
    constexpr int KV = 512;
    extern __shared__ __align__(128) unsigned char hw_sm[];
    float* sO = (float*) hw_sm;                              // HW_HG x KV, F32
    _Float16* sL = (_Float16*) (sO + HW_HG * KV);            // HW_CH x KV
    float* sPart = (float*) (sL + HW_CH * KV);               // 8 x HW_HG x HW_CH: the scores' K eighths
    _Float16* sP = (_Float16*) (sPart + 8 * HW_HG * HW_CH);  // HW_HG x HW_CH: exp(s - m)
    __shared__ float s_m[HW_HG], s_l[HW_HG], s_sc[HW_HG];
    __shared__ int s_cell[HW_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * HW_HG, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ns = n_sel_arr[t];
    const int* cl = cells_all + (size_t) t * n_sel_max;
    const _Float16* qa = q16 + ((size_t) t * n_head + h0) * KV;
    for (int i = tid; i < HW_HG * KV; i += blockDim.x) sO[i] = 0.0f;
    if (tid < HW_HG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    wm::fragment<wm::matrix_a, 16, 16, 16, _Float16, wm::row_major> qf[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) wm::load_matrix_sync(qf[k], qa + warp * 64 + k * 16, KV);
    constexpr int PRE = HW_CH * (KV / 8) / 256;
    uint4 pre[PRE];
    const auto issue = [&](int c0) {
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            const int cell = c0 + s < ns ? cl[c0 + s] : -1;
            pre[k] = cell >= 0 ? lat_ld8(lat, lat8, cell, c8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    issue(0);
    for (int s0 = 0; s0 < ns; s0 += HW_CH) {
        __syncthreads();   // the previous chunk's product has read sL and sP
        if (tid < HW_CH) s_cell[tid] = s0 + tid < ns ? cl[s0 + tid] : -1;
#pragma unroll
        for (int k = 0; k < PRE; ++k) {
            const int i = tid + k * 256, s = i / (KV / 8), c8 = i - s * (KV / 8);
            ((uint4*) (sL + s * KV))[c8] = pre[k];
        }
        if (s0 + HW_CH < ns) issue(s0 + HW_CH);
        __syncthreads();
        {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                wm::fragment<wm::matrix_b, 16, 16, 16, _Float16, wm::col_major> b;
                wm::load_matrix_sync(b, sL + warp * 64 + k * 16, KV);
                wm::mma_sync(acc, qf[k], b, acc);
            }
            wm::store_matrix_sync(sPart + warp * HW_HG * HW_CH, acc, HW_CH, wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            float v = -INFINITY;
            if (lane < HW_CH) {
                float a = 0.0f;
#pragma unroll
                for (int q = 0; q < 8; ++q) a += sPart[q * HW_HG * HW_CH + h * HW_CH + lane];
                v = s_cell[lane] >= 0 ? a * scale : -INFINITY;
            }
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = (v == -INFINITY) ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            if (lane < HW_CH) sP[h * HW_CH + lane] = (_Float16) e;
            if (lane == 0) {
                const float sc = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                s_sc[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        for (int i = tid; i < HW_HG * KV; i += blockDim.x) sO[i] *= s_sc[i / KV];
        __syncthreads();
        {
            wm::fragment<wm::matrix_a, 16, 16, 16, _Float16, wm::row_major> a;
            wm::load_matrix_sync(a, sP, HW_CH);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = (warp * 4 + j) * 16;
                wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
                wm::fragment<wm::matrix_b, 16, 16, 16, _Float16, wm::row_major> b;
                wm::load_matrix_sync(acc, sO + c0, KV, wm::mem_row_major);
                wm::load_matrix_sync(b, sL + c0, KV);
                wm::mma_sync(acc, a, b, acc);
                wm::store_matrix_sync(sO + c0, acc, KV, wm::mem_row_major);
            }
        }
    }
    __syncthreads();
    float* o = ctx + ((size_t) t * n_head + h0) * KV;
    for (int i = tid; i < HW_HG * KV; i += blockDim.x) {
        const float L = s_l[i / KV];
        o[i] = L > 0.0f ? sO[i] / L : 0.0f;
    }
}
#endif

// ---------------------------------------------------------------- the NextN block's caches over a prompt
// h[t] = rms(mean_s R[t][s]) * w (the final hidden state the draft block reads; head_prep's arithmetic)
__global__ void __launch_bounds__(256) head_rows_kernel(const float* __restrict__ R, const float* __restrict__ w,
                                                        float eps, int E, float* __restrict__ h) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* Rt = R + (size_t) t * 4 * E;
    float v[16];
    float ss = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        float acc = 0.0f;
        for (int s = 0; s < 4; ++s) acc += Rt[e + (size_t) E * s];
        acc = acc / 4.0f;
        v[i] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        h[(size_t) t * E + e] = v[i] * inv * w[e];
    }
}
// out16[t] = [rms(emb[t]) * enorm, rms(h[t]) * hnorm] (FP16, 2E per row)
__global__ void __launch_bounds__(256) mtp_in_rows_kernel(const float* __restrict__ emb, const float* __restrict__ h,
                                                          const float* __restrict__ enorm,
                                                          const float* __restrict__ hnorm, float eps, int E,
                                                          __half* __restrict__ out) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    float a[16], b[16];
    float sa = 0.0f, sb = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        a[i] = emb[(size_t) t * E + e];
        b[i] = h[(size_t) t * E + e];
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sred);
    sb = block_sum(sb, sred);
    const float ia = rsqrtf(sa / (float) E + eps), ib = rsqrtf(sb / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        out[(size_t) t * 2 * E + e] = __float2half(a[i] * ia * enorm[e]);
        out[(size_t) t * 2 * E + E + e] = __float2half(b[i] * ib * hnorm[e]);
    }
}
// x[t] = rms(in[t]) * w, F32 and FP16
__global__ void __launch_bounds__(256) rms_rows_kernel(const float* __restrict__ in, const float* __restrict__ w, float eps,
                                                       int E, float* __restrict__ x, __half* __restrict__ x16) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    float v[16];
    float ss = 0.0f;
    const int per = E / (int) blockDim.x;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        v[i] = in[(size_t) t * E + e];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) E + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= per) break;
        const int e = tid + i * (int) blockDim.x;
        const float xv = v[i] * inv * w[e];
        x[(size_t) t * E + e] = xv;
        x16[(size_t) t * E + e] = __float2half(xv);
    }
}

// ---------------------------------------------------------------- the routed experts
__device__ void topk_argmax(float* sel, int E, int k, int* out, float* s_bv, int* s_bi) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = s_bv[0];
            int i = s_bi[0];
            for (int w = 1; w < nw; ++w)
                if (s_bv[w] > v || (s_bv[w] == v && s_bi[w] < i)) {
                    v = s_bv[w];
                    i = s_bi[w];
                }
            out[r] = i;
            if (i >= 0 && i < E) sel[i] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int E, int k, float w_scale, int norm_w, int* __restrict__ ids,
                                                    float* __restrict__ w) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float s_p[512], s_sel[512];
    __shared__ int s_ids[8];
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    for (int e = tid; e < E; e += blockDim.x) {
        const float p = 1.0f / (1.0f + expf(-logits[(size_t) t * E + e]));
        s_p[e] = p;
        s_sel[e] = bias ? p + bias[e] : p;
    }
    __syncthreads();
    topk_argmax(s_sel, E, k, s_ids, s_bv, s_bi);
    if (tid == 0) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += (double) s_p[s_ids[i]];
        const float inv = (float) (norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        for (int i = 0; i < k; ++i) {
            ids[(size_t) t * k + i] = s_ids[i];
            w[(size_t) t * k + i] = (float) ((double) s_p[s_ids[i]] * inv * w_scale);
        }
    }
}

__global__ void __launch_bounds__(256) expert_count_kernel(const int* __restrict__ ids, int n, int* __restrict__ counts,
                                                           int* __restrict__ rank) {
    const int e = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ int s_wc[8];
    int base = 0;
    for (int c0 = 0; c0 < n; c0 += 256) {
        const int i = c0 + tid;
        const bool m = i < n && ids[i] == e;
        const unsigned ballot = __ballot_sync(0xffffffffu, m);
        if (lane == 0) s_wc[warp] = __popc(ballot);
        __syncthreads();
        int off = base;
        for (int w2 = 0; w2 < warp; ++w2) off += s_wc[w2];
        if (m) rank[i] = off + __popc(ballot & ((1u << lane) - 1u));
        int tot = 0;
        for (int w2 = 0; w2 < 8; ++w2) tot += s_wc[w2];
        base += tot;
        __syncthreads();
    }
    if (tid == 0) counts[e] = base;
}

__global__ void expert_scatter_kernel(const int* __restrict__ ids, const int* __restrict__ rank,
                                      const int* __restrict__ base, int n, int k, int* __restrict__ row_tok,
                                      int* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = base[ids[i]] + rank[i];
    row_tok[row] = i / k;
    pos[i] = row;
}

// (h may be gu: each thread reads its gate and up values before it writes the one place they came from)
__global__ void swiglu_rows_kernel(const float* gu, float* h, int rows, int n_ff, int ld_h, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n_ff) return;
    const int64_t r = i / n_ff;
    const int c = (int) (i - r * n_ff);
    const float* row = gu + r * 2 * n_ff;
    const float g = fminf(row[c], limit);
    const float u = fminf(fmaxf(row[n_ff + c], -limit), limit);
    h[r * ld_h + c] = (g / (1.0f + expf(-g))) * u;
}

__global__ void swiglu_f16_kernel(const float* __restrict__ gate, const float* __restrict__ up, __half* __restrict__ h,
                                  int64_t n, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    h[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// ffn[t] += sum over t's routes j whose sorted row is in [lo, hi) of w[j] * out[row - base] (in route order)
__global__ void moe_combine_add_kernel(const float* __restrict__ out, int base, int lo, int hi,
                                       const int* __restrict__ pos, const float* __restrict__ w, int k, int E,
                                       float* __restrict__ ffn) {
    const int t = blockIdx.x;
    __shared__ int s_pos[8];
    __shared__ float s_w[8];
    __shared__ int s_n;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int j = 0; j < k; ++j) {
            const int p = pos[t * k + j];
            if (p >= lo && p < hi) {
                s_pos[n] = p - base;
                s_w[n] = w[t * k + j];
                ++n;
            }
        }
        s_n = n;
    }
    __syncthreads();
    const int n = s_n;
    if (n == 0) return;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float m = 0.0f;
        for (int j = 0; j < n; ++j) m += s_w[j] * out[(size_t) s_pos[j] * E + e];
        ffn[(size_t) t * E + e] += m;
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ out, const int* __restrict__ pos, const float* __restrict__ w,
                                   const float* __restrict__ sh, int k, int E, float* __restrict__ ffn) {
    const int t = blockIdx.x;
    __shared__ int s_pos[8];
    __shared__ float s_w[8];
    if (threadIdx.x < k) {
        s_pos[threadIdx.x] = pos[t * k + threadIdx.x];
        s_w[threadIdx.x] = w[t * k + threadIdx.x];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float m = 0.0f;
        for (int j = 0; j < k; ++j) m += s_w[j] * out[(size_t) s_pos[j] * E + e];
        ffn[(size_t) t * E + e] = m + (sh ? sh[(size_t) t * E + e] : 0.0f);
    }
}

}  // namespace

int launch_errors() { return g_errors.load(std::memory_order_relaxed); }

void bf16_to_f32(const uint16_t* src, float* dst, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<nblk(n), 256, 0, s>>>(src, dst, n);
    check("bf16_to_f32");
}

void bf16_to_f16(const uint16_t* src, void* dst, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<nblk(n), 256, 0, s>>>(src, (__half*) dst, n);
    check("bf16_to_f16");
}

void f32_to_f16(const float* src, void* dst, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    f32_to_f16_kernel<<<nblk(n), 256, 0, s>>>(src, (__half*) dst, n);
    check("f32_to_f16");
}

void embed_rows(const float* emb, float* R, int T, int n_embd, cudaStream_t s) {
    if (T <= 0) return;
    embed_rows_kernel<<<nblk((int64_t) T * 4 * n_embd), 256, 0, s>>>(emb, R, T, n_embd);
    check("embed_rows");
}

void hc_update(const float* block_out, float* R, const float* post, const float* comb, float* ss, int T, int n_embd,
               cudaStream_t s) {
    if (T <= 0) return;
    hc_update_kernel<<<T, 256, 0, s>>>(block_out, R, post, comb, ss, n_embd);
    check("hc_update");
}

void hc_finish(const HcFinishArgs& a, cudaStream_t s) {
    if (a.T <= 0) return;
    if (a.n_embd > 16 * 256 || a.n_embd % 256 != 0) {
        std::fprintf(stderr, "glm_batch hc_finish: n_embd %d unsupported\n", a.n_embd);
        return;
    }
    hc_finish_kernel<<<a.T, 256, 0, s>>>(a);
    check("hc_finish");
}

void kda_conv(const float* const proj[3], const float* const conv_w[3], const float* conv_state, float* const out[3],
              int T, int d_inner, int d_conv, cudaStream_t s) {
    if (T <= 0) return;
    Ptr3 p{{proj[0], proj[1], proj[2]}}, w{{conv_w[0], conv_w[1], conv_w[2]}};
    WPtr3 o{{out[0], out[1], out[2]}};
    const dim3 grid(nblk(d_inner), (unsigned) ((T + kConvRun - 1) / kConvRun), 3);
    switch (d_conv - 1) {
        case 0: kda_conv_kernel<0><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 1: kda_conv_kernel<1><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 2: kda_conv_kernel<2><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 3: kda_conv_kernel<3><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 4: kda_conv_kernel<4><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 5: kda_conv_kernel<5><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 6: kda_conv_kernel<6><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        case 7: kda_conv_kernel<7><<<grid, 256, 0, s>>>(p, w, conv_state, o, T, d_inner); break;
        default: std::fprintf(stderr, "glm batch: kda_conv: d_conv %d unsupported (1..8)\n", d_conv); break;
    }
    check("kda_conv");
}

void kda_conv_state(const float* const proj[3], float* conv_state, int T, int d_inner, int d_conv, cudaStream_t s) {
    if (T <= 0) return;
    Ptr3 p{{proj[0], proj[1], proj[2]}};
    kda_conv_state_kernel<<<nblk(3 * (int64_t) d_inner), 256, 0, s>>>(p, conv_state, T, d_inner, d_conv);
    check("kda_conv_state");
}

void kda_rec(const float* q, const float* k, const float* v, const float* g1_raw, const float* dt_bias,
             const float* ssm_a, float lower_bound, const float* beta_raw, float* state, const float* g2_raw,
             const float* norm_w, float eps, int n_head, int T, void* out16, cudaStream_t s) {
    if (T <= 0) return;
    kda_rec_kernel<<<n_head, 256, 0, s>>>(q, k, v, g1_raw, dt_bias, ssm_a, lower_bound, beta_raw, state, g2_raw,
                                          norm_w, eps, n_head, T, (__half*) out16);
    check("kda_rec");
}

void kda_rec_split(float* q, float* k, float* v, float* g1_raw, const float* dt_bias, const float* ssm_a,
                   float lower_bound, const float* beta_raw, float* state, const float* g2_raw, const float* norm_w,
                   float eps, int n_head, int T, void* out16, cudaStream_t s, int kv_split) {
    if (T <= 0) return;
    kda_prep_kernel<<<dim3((unsigned) T, (unsigned) n_head), 128, 0, s>>>(q, k, g1_raw, dt_bias, ssm_a, lower_bound,
                                                                         n_head);
    if (kv_split == 8)
        kda_rec_split_kernel<16><<<dim3((unsigned) n_head, 8), 256, 0, s>>>(q, k, v, g1_raw, beta_raw, state, n_head, T);
    else
        kda_rec_split_kernel<32><<<dim3((unsigned) n_head, 4), 256, 0, s>>>(q, k, v, g1_raw, beta_raw, state, n_head, T);
    kda_out_kernel<<<dim3((unsigned) T, (unsigned) n_head), 128, 0, s>>>(v, g2_raw, norm_w, eps, n_head, (__half*) out16);
    check("kda_rec_split");
}

void dsa_prep(const DsaPrepArgs& a, cudaStream_t s) {
    if (a.T <= 0) return;
    dsa_prep_kernel<<<dim3((unsigned) a.T, 3), 512, 0, s>>>(a);
    check("dsa_prep");
}

void dsa_pool(const float* ik_cache, const float* ig_cache, const float* ape, float* pooled, int idx_key, int kpool,
              int pool0, int n, cudaStream_t s, int ring) {
    if (n <= 0) return;
    dsa_pool_kernel<<<n, ((idx_key + 31) / 32) * 32, 0, s>>>(ik_cache, ig_cache, ape, pooled, idx_key, kpool, pool0,
                                                             ring);
    check("dsa_pool");
}

void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int p0, int kpool,
               int T, int max_vis, float* score, int score_ld, cudaStream_t s) {
    if (T <= 0 || max_vis <= 0) return;
    // STRATA_GLM_DSA_SCORE=0: the warp-per-pool kernel (the tiled one needs idx_heads 32 and key_dim 128)
    static const bool tiled = [] {
        const char* v = getenv("STRATA_GLM_DSA_SCORE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (tiled && idx_heads == 32 && key_dim == 128) {
        dsa_score_tiled_kernel<<<dim3((unsigned) ((max_vis + TS_P - 1) / TS_P), (unsigned) ((T + TS_T - 1) / TS_T)), 256,
                                 0, s>>>(iq, pooled, iw, p0, kpool, T, max_vis, score, score_ld);
        check("dsa_score_tiled");
        return;
    }
    const size_t smem = ((size_t) idx_heads * key_dim + idx_heads) * sizeof(float);
    dsa_score_kernel<<<dim3((unsigned) ((max_vis + SCORE_POOLS - 1) / SCORE_POOLS), (unsigned) T), 256, smem, s>>>(
        iq, pooled, iw, key_dim, idx_heads, p0, kpool, score, score_ld);
    check("dsa_score");
}

void dsa_select(const float* score, int score_ld, int p0, int kpool, int top_pools_max, int tail, int T,
                int n_sel_max, int* cells, int* n_sel, cudaStream_t s) {
    if (T <= 0) return;
    dsa_select_kernel<<<T, 1024, 0, s>>>(score, score_ld, p0, kpool, top_pools_max, tail, n_sel_max, cells, n_sel);
    check("dsa_select");
}

void mla_attn(const float* q_abs, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head,
              int kv_lora, float scale, int T, float* ctx, cudaStream_t s, int lat8) {
    if (T <= 0) return;
    if (kv_lora != 512 || n_head % MB_HG != 0) {
        std::fprintf(stderr, "glm_batch mla_attn: kv_lora %d / n_head %d unsupported\n", kv_lora, n_head);
        return;
    }
    const dim3 grid((unsigned) T, (unsigned) (n_head / MB_HG));
#if !defined(STRATA_USE_HIP)
    // per device, the prompt attention kernel: 0 not chosen yet, 1 tensor cores with O in shared memory (91 KB: Volta,
    // and Ampere+ with STRATA_GLM_PREFILL_ATTN=wmma), 2 tensor cores with O in registers (58 KB: Turing, sm_75 - its
    // rescale assumes the sm_75+ fragment layout), 3 mma.sync, 32 heads a block (Ampere and newer), -1 the F32 kernel
    static int tc_ok[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    // STRATA_GLM_PREFILL_ATTN=f32: the F32 kernel; =tcreg: the register kernel on sm_75+ (A/B, e.g. on Ampere); =wmma:
    // the 91 KB kernel on Ampere+ instead of the mma.sync one;
    // STRATA_GLM_PREFILL_ATTN_CHECK=1 (debug): the chosen tensor-core kernel and the F32 one, compared
    static const char* attn_mode = getenv("STRATA_GLM_PREFILL_ATTN");
    static const bool f32_only = attn_mode != nullptr && std::strcmp(attn_mode, "f32") == 0;
    static const bool want_reg = attn_mode != nullptr && std::strcmp(attn_mode, "tcreg") == 0;
    static const bool want_wmma = attn_mode != nullptr && std::strcmp(attn_mode, "wmma") == 0;
    static const bool check_tc = getenv("STRATA_GLM_PREFILL_ATTN_CHECK") != nullptr;
    if (tc_ok[dev] == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        const bool reg_layout = major * 10 + minor >= 75;   // the register kernel's fragment layout
        const auto fits = [](const void* k, size_t bytes) {
            const bool ok = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) ==
                            cudaSuccess;
            cudaGetLastError();
            return ok;
        };
        tc_ok[dev] = -1;
        if (major >= 8 && !want_reg && !want_wmma && n_head % MT_H == 0 &&
            fits((const void*) mla_attn_mma_kernel, kMtSmem))
            tc_ok[dev] = 3;
        else if (major >= 7 && !(want_reg && reg_layout) && fits((const void*) mla_attn_tc_kernel, kTcSmem))
            tc_ok[dev] = 1;
        else if (reg_layout && fits((const void*) mla_attn_tc_reg_kernel, kTcRegSmem))
            tc_ok[dev] = 2;
        std::fprintf(stderr, "glm_batch: CUDA%d prompt attention on the %s\n", dev,
                     f32_only || tc_ok[dev] < 0 ? "F32 cores"
                     : tc_ok[dev] == 2          ? "tensor cores (context in registers)"
                     : tc_ok[dev] == 3          ? "tensor cores (mma.sync, 32 heads a block)"
                                                : "tensor cores");
    }
    if (tc_ok[dev] > 0 && !f32_only) {
        if (tc_ok[dev] == 3)
            mla_attn_mma_kernel<<<dim3((unsigned) T, (unsigned) (n_head / MT_H)), 256, kMtSmem, s>>>(
                q_abs, lat, cells, n_sel, n_sel_max, n_head, scale, ctx, lat8);
        else if (tc_ok[dev] == 2)
            mla_attn_tc_reg_kernel<<<grid, 256, kTcRegSmem, s>>>(q_abs, lat, cells, n_sel, n_sel_max, n_head, scale, ctx, lat8);
        else
            mla_attn_tc_kernel<<<grid, 256, kTcSmem, s>>>(q_abs, lat, cells, n_sel, n_sel_max, n_head, scale, ctx, lat8);
        check("mla_attn_tc");
        if (check_tc) {
            const size_t n = (size_t) T * n_head * 512;
            float* ref = nullptr;
            if (cudaMalloc(&ref, n * sizeof(float)) == cudaSuccess) {
                mla_attn_kernel<<<grid, 256, kF32Smem, s>>>(q_abs, lat, cells, n_sel, n_sel_max, n_head, scale, ref, lat8);
                std::vector<float> A(n), B(n);
                cudaStreamSynchronize(s);
                cudaMemcpy(A.data(), ctx, n * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(B.data(), ref, n * sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(ref);
                double num = 0, den = 0, worst = 0;
                for (size_t r = 0; r < (size_t) T * n_head; ++r) {
                    double rn = 0, rd = 0;
                    for (int j = 0; j < 512; ++j) {
                        const double d = (double) A[r * 512 + j] - B[r * 512 + j];
                        rn += d * d;
                        rd += (double) B[r * 512 + j] * B[r * 512 + j];
                    }
                    num += rn;
                    den += rd;
                    worst = std::max(worst, std::sqrt(rn / std::max(1e-30, rd)));
                }
                std::fprintf(stderr, "glm_batch attention check: %d tokens, rel L2 %.3e, worst head %.3e\n", T,
                             std::sqrt(num / std::max(1e-30, den)), worst);
            }
            cudaGetLastError();
        }
        return;
    }
#endif
    mla_attn_kernel<<<grid, 256, kF32Smem, s>>>(q_abs, lat, cells, n_sel, n_sel_max, n_head, scale, ctx, lat8);
    check("mla_attn");
}

#if defined(STRATA_USE_HIP)
void mla_attn_f16q(const uint16_t* q16, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max,
                   int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s) {
    if (T <= 0) return;
    if (kv_lora != 512 || n_head % HW_HG != 0) {
        std::fprintf(stderr, "glm_batch mla_attn_f16q: kv_lora %d / n_head %d unsupported\n", kv_lora, n_head);
        return;
    }
    // kHwSmem is below the 64 KiB a workgroup may take without opting in
    const dim3 grid((unsigned) T, (unsigned) (n_head / HW_HG));
    mla_attn_wmma_kernel<<<grid, 256, kHwSmem, s>>>((const _Float16*) q16, lat, cells, n_sel, n_sel_max, n_head, scale, ctx,
                                                    0);   // (FP16 latents only: an INT8 cache takes mla_attn)
    check("mla_attn_wmma");
}
#endif

void head_rows(const float* R, const float* w, float eps, int T, int n_embd, float* h, cudaStream_t s) {
    if (T <= 0) return;
    head_rows_kernel<<<T, 256, 0, s>>>(R, w, eps, n_embd, h);
    check("head_rows");
}

void mtp_in_rows(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int T, int n_embd,
                 void* out16, cudaStream_t s) {
    if (T <= 0) return;
    mtp_in_rows_kernel<<<T, 256, 0, s>>>(emb, h, enorm, hnorm, eps, n_embd, (__half*) out16);
    check("mtp_in_rows");
}

void rms_rows(const float* in, const float* w, float eps, int T, int n_embd, float* x, void* x16, cudaStream_t s) {
    if (T <= 0) return;
    rms_rows_kernel<<<T, 256, 0, s>>>(in, w, eps, n_embd, x, (__half*) x16);
    check("rms_rows");
}

void route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int T, int* ids,
           float* w, cudaStream_t s) {
    if (T <= 0) return;
    route_kernel<<<T, 256, 0, s>>>(logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, ids, w);
    check("route");
}

__global__ void copy_i32_kernel(const int* __restrict__ src, int* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

void copy_i32(const int* src, int* dst, int n, cudaStream_t s) {
    if (n <= 0) return;
    copy_i32_kernel<<<nblk(n), 256, 0, s>>>(src, dst, n);
    check("copy_i32");
}

void expert_count(const int* ids, int n, int n_expert, int* counts, int* rank, cudaStream_t s) {
    expert_count_kernel<<<n_expert, 256, 0, s>>>(ids, n, counts, rank);
    check("expert_count");
}

void expert_scatter(const int* ids, const int* rank, const int* base, int n, int k, int* row_tok, int* pos,
                    cudaStream_t s) {
    if (n <= 0) return;
    expert_scatter_kernel<<<nblk(n), 256, 0, s>>>(ids, rank, base, n, k, row_tok, pos);
    check("expert_scatter");
}

void swiglu_rows(const float* gu, float* h, int rows, int n_ff, float limit, cudaStream_t s, int ld_h) {
    if (rows <= 0) return;
    swiglu_rows_kernel<<<nblk((int64_t) rows * n_ff), 256, 0, s>>>(gu, h, rows, n_ff, ld_h > 0 ? ld_h : n_ff, limit);
    check("swiglu_rows");
}

void swiglu_f16(const float* gate, const float* up, void* h16, int64_t n, float limit, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_f16_kernel<<<nblk(n), 256, 0, s>>>(gate, up, (__half*) h16, n, limit);
    check("swiglu_f16");
}

void moe_combine_add(const float* out, int base, int lo, int hi, const int* pos, const float* w, int T, int k, int n_embd,
                     float* ffn, cudaStream_t s) {
    if (T <= 0 || hi <= lo) return;
    moe_combine_add_kernel<<<T, 256, 0, s>>>(out, base, lo, hi, pos, w, k, n_embd, ffn);
    check("moe_combine_add");
}

void moe_combine(const float* out, const int* pos, const float* w, const float* sh, int T, int k, int n_embd,
                 float* ffn, cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_kernel<<<T, 256, 0, s>>>(out, pos, w, sh, k, n_embd, ffn);
    check("moe_combine");
}

}  // namespace strata::kernels::glmb
