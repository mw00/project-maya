// src/hy4/hy4_kernels.cu - maya-hy4's own kernels (hy4_kernels.cuh).  Correctness first: plain F32 arithmetic on the
// CUDA cores, the latent cache in F16 as the reference keeps it; the reference is AngelSlim's hy4-flash graph
// (src/models/hy-v4-flash.cpp of their hy4-flash patch).
#include "hy4_kernels.cuh"
#include "strata/kernels/fresh_load.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <map>
#include <mutex>
#include <cstdio>
#include <cstdlib>

namespace hy4::k {

namespace {

std::atomic<int> g_err{0};
float* g_mix_out = nullptr;   // (tests: hc_pre's mixes, set_mix_out)

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "maya-hy4 kernel %s: %s\n", what, cudaGetErrorString(e));
        g_err.fetch_add(1, std::memory_order_relaxed);
    }
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
// a block's sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.0f;
    r = warp_sum(r);
    return r;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// ---------------------------------------------------------------- int8 rows
__global__ void q8_kernel(const float* __restrict__ x, size_t ldx, int k, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * ldx + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}

__global__ void q8_heads_kernel(const float* __restrict__ x, size_t ldt, size_t ldh, int T, int k,
                                int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, h = r / T, t = r % T, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) t * ldt + (size_t) h * ldh + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}
__global__ void copy_cols_kernel(const float* __restrict__ src, size_t lds, float* __restrict__ dst, size_t ldd, int n,
                                 size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    dst[r * ldd + c] = src[r * lds + c];
}
template <typename D>
__global__ void store_rows_kernel(const float* __restrict__ src, int n, D* __restrict__ ring, int R,
                                  const int* __restrict__ pos) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        ring[(size_t) (pos[t] % R) * n + i] = (D) src[(size_t) t * n + i];
}

// ---------------------------------------------------------------- F32 / BF16 GEMV: a warp a row, up to 8 columns a pass
template <typename W, int NC>
__global__ void gemv_kernel(const W* __restrict__ w, size_t ld, int k, int n, const float* __restrict__ x, size_t ldx,
                            int ncol, float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    if (row >= n) return;
    const W* wr = w + (size_t) row * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = lane * 2; i < k; i += 64) {
        float w0, w1;
        if constexpr (sizeof(W) == 2) {
            const uint32_t p = *(const uint32_t*) (wr + i);
            w0 = __uint_as_float(p << 16);
            w1 = __uint_as_float(p & 0xffff0000u);
        } else {
            const float2 p = *(const float2*) (wr + i);
            w0 = p.x;
            w1 = p.y;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c0 + c < ncol) {
                const float2 xv = *(const float2*) (x + (size_t) (c0 + c) * ldx + i);
                acc[c] = fmaf(w0, xv.x, fmaf(w1, xv.y, acc[c]));
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * v;
    }
}
template <typename W>
__global__ void __launch_bounds__(256) gemv_rows_kernel(const W* __restrict__ w, size_t ld, int k,
                                                        const float* __restrict__ x, size_t ldx, int ncol,
                                                        float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x, c = blockIdx.y;
    const W* wr = w + (size_t) row * ld;
    const float* xc = x + (size_t) c * ldx;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float wv;
        if constexpr (sizeof(W) == 2) wv = __uint_as_float((uint32_t) wr[i] << 16);
        else wv = (float) wr[i];
        acc = fmaf(wv, xc[i], acc);
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) y[(size_t) c * ldy + row] = alpha * acc;
}
// ... 16-byte loads (4 values a lane a step, rows and columns 16-byte aligned, k a multiple of 128): a column's sums in
// the same order whatever NC, so a window's rows are a token's bits
template <typename W, int NC>
__global__ void gemv4_kernel(const W* __restrict__ w, size_t ld, int k, int n, const float* __restrict__ x, size_t ldx,
                             int ncol, float* __restrict__ y, size_t ldy, float alpha) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    if (row >= n) return;
    const W* wr = w + (size_t) row * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = lane * 4; i < k; i += 128) {
        float4 wv;
        if constexpr (sizeof(W) == 2) {
            const uint2 p = *(const uint2*) (wr + i);
            wv.x = __uint_as_float(p.x << 16);
            wv.y = __uint_as_float(p.x & 0xffff0000u);
            wv.z = __uint_as_float(p.y << 16);
            wv.w = __uint_as_float(p.y & 0xffff0000u);
        } else {
            wv = *(const float4*) (wr + i);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c0 + c < ncol) {
                const float4 xv = *(const float4*) (x + (size_t) (c0 + c) * ldx + i);
                acc[c] = fmaf(wv.w, xv.w, fmaf(wv.z, xv.z, fmaf(wv.y, xv.y, fmaf(wv.x, xv.x, acc[c]))));
            }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = alpha * v;
    }
}
template <typename W>
void gemv(const W* w, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy, float alpha,
          cudaStream_t s) {
    if (ncol <= 0 || n <= 0) return;
    static const bool old = std::getenv("HY4_GEMV_OLD") != nullptr;
    const bool v4 = !old && n >= 128 && k % 128 == 0 && ld % 8 == 0 && ldx % 4 == 0 && ((uintptr_t) w & 15) == 0 &&
                    ((uintptr_t) x & 15) == 0;
    if (v4) {
        const dim3 b(256), g1((unsigned) ((n + 7) / 8), 1);
        if (ncol == 1) gemv4_kernel<W, 1><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else if (ncol <= 4) gemv4_kernel<W, 4><<<g1, b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        else
            gemv4_kernel<W, 8><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(
                w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
        check("gemv4");
        return;
    }
    if (n < 128) {   // (few rows: a block each, every one of its warps on a part of k)
        gemv_rows_kernel<W><<<dim3((unsigned) n, (unsigned) ncol), 256, 0, s>>>(w, ld, k, x, ldx, ncol, y, ldy, alpha);
        check("gemv_rows");
        return;
    }
    const dim3 b(256);
    if (ncol == 1) {
        gemv_kernel<W, 1><<<dim3((unsigned) ((n + 7) / 8), 1), b, 0, s>>>(w, ld, k, n, x, ldx, ncol, y, ldy, alpha);
    } else {
        gemv_kernel<W, 8><<<dim3((unsigned) ((n + 7) / 8), (unsigned) ((ncol + 7) / 8)), b, 0, s>>>(w, ld, k, n, x, ldx,
                                                                                                   ncol, y, ldy, alpha);
    }
    check("gemv");
}

// ---------------------------------------------------------------- norms
__global__ void rms_kernel(const float* __restrict__ x, size_t ldx, int n, const float* __restrict__ w, float eps,
                           float* __restrict__ y, size_t ldy) {
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) n + eps);
    float* yr = y + (size_t) blockIdx.x * ldy;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
}

// ---------------------------------------------------------------- hyper-connections
// a block a token: rmsnorm of the hc E values, the 2 hc (head: hc) mixes, pre / post, cur
// Two passes so a token's 12288 values are read by many blocks: hc_part (grid (kHcBlocks, T)) - each block a slice's
// sum of squares and its 2 hc (head: hc) partial dots with fn's rows; hc_fin (a block a token) - the sums, the mixes,
// pre / post, cur
constexpr int kHcBlocks = 24, kMaxM = 16;
__device__ __forceinline__ void hc_part_body(int t, int b, const float* __restrict__ x, int HE, int nm,
                                             const float* __restrict__ fn, float* __restrict__ part) {
    __shared__ float sh[8][kMaxM + 1];
    const int per = (HE + kHcBlocks - 1) / kHcBlocks, i0 = b * per, i1 = min(HE, i0 + per);
    const float* xt = x + (size_t) t * HE;
    float acc[kMaxM + 1];
#pragma unroll
    for (int j = 0; j <= kMaxM; ++j) acc[j] = 0.0f;
    for (int i = i0 + (int) threadIdx.x; i < i1; i += blockDim.x) {
        const float v = xt[i];
        acc[kMaxM] = fmaf(v, v, acc[kMaxM]);
#pragma unroll
        for (int j = 0; j < kMaxM; ++j)
            if (j < nm) acc[j] = fmaf(fn[(size_t) j * HE + i], v, acc[j]);
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j <= kMaxM; ++j) {
        const float v = warp_sum(acc[j]);
        if (l == 0) sh[w][j] = v;
    }
    __syncthreads();
    if ((int) threadIdx.x <= kMaxM) {
        float v = 0.0f;
        for (int q = 0; q < (int) (blockDim.x >> 5); ++q) v += sh[q][threadIdx.x];
        part[((size_t) t * kHcBlocks + b) * (kMaxM + 1) + threadIdx.x] = v;
    }
}
__global__ void __launch_bounds__(256) hc_part_kernel(const float* __restrict__ x, int HE, int nm,
                                                      const float* __restrict__ fn, float* __restrict__ part) {
    hc_part_body(blockIdx.y, blockIdx.x, x, HE, nm, fn, part);
}
// hc_fin's work for token t, KPER values a thread (E <= blockDim KPER); part read past L1 (another block's writes)
template <bool HEAD, int KPER>
__device__ __forceinline__ void hc_fin_body(int t, const float* __restrict__ x, int hc, int E,
                                            const float* __restrict__ part, const float* __restrict__ scale,
                                            const float* __restrict__ base, float eps, float hc_eps, float mag,
                                            float* __restrict__ cur, float* __restrict__ post,
                                            float* __restrict__ mix_out, const float* __restrict__ nw, float neps) {
    __shared__ float s_mix[kMaxM + 1], s_pre[8];
    const int nm = HEAD ? hc : 2 * hc, HE = hc * E;
    if ((int) threadIdx.x <= kMaxM) {
        float v = 0.0f;
        for (int b = 0; b < kHcBlocks; ++b)
            v += strata::kernels::ld_l2(part + ((size_t) t * kHcBlocks + b) * (kMaxM + 1) + threadIdx.x);
        s_mix[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float r = 1.0f / sqrtf(s_mix[kMaxM] / (float) HE + eps);
        for (int j = 0; j < nm; ++j) s_mix[j] *= r;
        if (mix_out)
            for (int j = 0; j < nm; ++j) mix_out[(size_t) t * nm + j] = s_mix[j];
        for (int j = 0; j < hc; ++j) s_pre[j] = sigmoidf_(s_mix[j] * scale[0] + base[j]) + hc_eps;
        if (!HEAD)
            for (int j = 0; j < hc; ++j)
                post[(size_t) t * hc + j] = sigmoidf_(s_mix[hc + j] * scale[1] + base[hc + j]) * mag + hc_eps;
    }
    __syncthreads();
    const float* xt = x + (size_t) t * HE;
    constexpr int kPer = KPER;   // (E <= blockDim kPer: a thread's values kept for the norm)
    float vals[kPer], ss = 0.0f;
#pragma unroll
    for (int q = 0; q < kPer; ++q) {
        const int d = threadIdx.x + q * blockDim.x;
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = d < E && j < hc ? xt[(size_t) j * E + d] : 0.0f;
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) v = fmaf(j < hc ? s_pre[j] : 0.0f, a[j], v);
        vals[q] = v;
        ss += v * v;
    }
    float r = 1.0f;
    if (nw) {
        ss = block_sum(ss);
        r = 1.0f / sqrtf(ss / (float) E + neps);
    }
#pragma unroll
    for (int q = 0; q < kPer; ++q) {
        const int d = threadIdx.x + q * blockDim.x;
        if (d < E) cur[(size_t) t * E + d] = nw ? vals[q] * r * nw[d] : vals[q];
    }
}
template <bool HEAD>
__global__ void __launch_bounds__(512) hc_fin_kernel(const float* __restrict__ x, int hc, int E,
                                                     const float* __restrict__ part, const float* __restrict__ scale,
                                                     const float* __restrict__ base, float eps, float hc_eps, float mag,
                                                     float* __restrict__ cur, float* __restrict__ post,
                                                     float* __restrict__ mix_out, const float* __restrict__ nw,
                                                     float neps) {
    hc_fin_body<HEAD, 8>(blockIdx.x, x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, mix_out, nw, neps);
}
// ... in one launch: the slices' blocks as hc_part, the last of a token's to finish its hc_fin (a counter a token)
__global__ void __launch_bounds__(256) hc_pre1_kernel(const float* __restrict__ x, int hc, int E,
                                                      const float* __restrict__ fn, float* __restrict__ part,
                                                      unsigned* __restrict__ cnt, const float* __restrict__ scale,
                                                      const float* __restrict__ base, float eps, float hc_eps, float mag,
                                                      float* __restrict__ cur, float* __restrict__ post,
                                                      float* __restrict__ mix_out, const float* __restrict__ nw,
                                                      float neps) {
    const int t = blockIdx.y;
    hc_part_body(t, blockIdx.x, x, hc * E, 2 * hc, fn, part);
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&cnt[t], 1u) == (unsigned) kHcBlocks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    hc_fin_body<false, 12>(t, x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, mix_out, nw, neps);
    if (threadIdx.x == 0) cnt[t] = 0u;
}
// hc_pre1's counters (a token's, back to 0 after each launch), a zeroed buffer a device, grown as needed
unsigned* hc_cnt_buf(int T) {
    int dev = 0;
    cudaGetDevice(&dev);
    static std::mutex mu;
    static std::map<int, std::pair<unsigned*, size_t>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto& b = bufs[dev];
    if (b.second < (size_t) T) {
        if (b.first) cudaFree(b.first);
        b.first = nullptr;
        b.second = 0;
        const size_t n = std::max<size_t>((size_t) T, 1024);
        if (cudaMalloc(&b.first, n * sizeof(unsigned)) != cudaSuccess || cudaMemset(b.first, 0, n * sizeof(unsigned)) != cudaSuccess)
            return nullptr;
        b.second = n;
    }
    return b.first;
}
// hc_part's sums (kHcBlocks x (kMaxM + 1) floats a token), a buffer a device, grown as needed
float* hc_part_buf(int T) {
    int dev = 0;
    cudaGetDevice(&dev);
    static std::mutex mu;
    static std::map<int, std::pair<float*, size_t>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto& b = bufs[dev];
    const size_t need = (size_t) T * kHcBlocks * (kMaxM + 1);
    if (b.second < need) {
        if (b.first) cudaFree(b.first);
        b.first = nullptr;
        b.second = 0;
        if (cudaMalloc(&b.first, need * sizeof(float)) != cudaSuccess) return nullptr;
        b.second = need;
    }
    return b.first;
}
// grid (kPostBlocks, T): every block the RMS of y's row (12 KB from L2), then its slice: x[s] += y r w post[s]
// (yo: the normed y written there too - the debug read-outs)
constexpr int kPostBlocks = 12;
__global__ void __launch_bounds__(256) hc_post_norm_kernel(float* __restrict__ x, int hc, int E,
                                                           const float* __restrict__ y, const float* __restrict__ post,
                                                           const float* __restrict__ w, float eps,
                                                           float* __restrict__ yo) {
    const int t = blockIdx.y, b = blockIdx.x;
    const float* yt = y + (size_t) t * E;
    float ss = 0.0f;
    for (int d = threadIdx.x * 4; d < E; d += blockDim.x * 4) {
        const float4 v = *(const float4*) (yt + d);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) E + eps);
    const int per = (E + kPostBlocks - 1) / kPostBlocks, d0 = b * per, d1 = min(E, d0 + per);
    float* xt = x + (size_t) t * hc * E;
    float pw[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) pw[j] = j < hc ? post[(size_t) t * hc + j] : 0.0f;
    for (int d = d0 + (int) threadIdx.x; d < d1; d += blockDim.x) {
        const float v = yt[d] * r * w[d];
        if (yo) yo[(size_t) t * E + d] = v;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < hc) xt[(size_t) j * E + d] += v * pw[j];
    }
}
__global__ void hc_post_kernel(float* __restrict__ x, int hc, int E, const float* __restrict__ y,
                               const float* __restrict__ post, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t t = i / ((size_t) hc * E), s = (i / E) % hc, d = i % E;
    x[i] += y[t * E + d] * post[t * hc + s];
}

// ---------------------------------------------------------------- RoPE (NORM: consecutive pairs)
__device__ __forceinline__ void rope_pair(float& a, float& b, float pos, int i, int n_rot, float base) {
    const float theta_scale = powf(base, -2.0f / (float) n_rot);
    const float theta = pos * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = a, x1 = b;
    a = x0 * c - x1 * s;
    b = x0 * s + x1 * c;
}
__global__ void rope_kernel(float* __restrict__ x, int nh, size_t ld, size_t ldt, int off, int n_rot,
                            const int* __restrict__ pos, float base) {
    const int t = blockIdx.y, half = n_rot / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (head, pair)
    if (i >= nh * half) return;
    const int h = i / half, p = i % half;
    float* v = x + (size_t) t * ldt + (size_t) h * ld + off + 2 * p;
    rope_pair(v[0], v[1], (float) pos[t], p, n_rot, base);
}

// ---------------------------------------------------------------- the latent cache
__global__ void kv_store_kernel(const float* __restrict__ kv, int kv_lora, int n_rot, const float* __restrict__ w,
                                float eps, const int* __restrict__ pos, float base, __half* __restrict__ cache,
                                int ring) {
    const int t = blockIdx.x, row = kv_lora + n_rot;
    const float* x = kv + (size_t) t * row;
    const int p = pos[t];
    __half* dst = cache + (size_t) (ring > 0 ? p % ring : p) * row;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) kv_lora + eps);
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) dst[i] = __float2half_rn(x[i] * r * w[i]);
    for (int i = threadIdx.x; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i], b = x[kv_lora + 2 * i + 1];
        rope_pair(a, b, (float) p, i, n_rot, base);
        dst[kv_lora + 2 * i] = __float2half_rn(a);
        dst[kv_lora + 2 * i + 1] = __float2half_rn(b);
    }
}

// ... in Q8_0 rows (kv_q8_row_bytes): the row's values staged in shared memory, then each block of 32 of the latent by
// a warp - its largest magnitude / 127 the F16 scale, every value rounded to an int8 of it (ggml's Q8_0 rounding; the
// values the F16 kernel stores, before their F16 rounding); the rotated part as F16 after the scales
__global__ void kv_store_q8_kernel(const float* __restrict__ kv, int kv_lora, int n_rot, const float* __restrict__ w,
                                   float eps, const int* __restrict__ pos, float base, uint8_t* __restrict__ cache,
                                   int ring, int rb) {
    __shared__ float v[1024];
    const int t = blockIdx.x, row = kv_lora + n_rot;
    const float* x = kv + (size_t) t * row;
    const int p = pos[t];
    uint8_t* dst = cache + (size_t) (ring > 0 ? p % ring : p) * rb;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = 1.0f / sqrtf(ss / (float) kv_lora + eps);
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) v[i] = x[i] * r * w[i];
    for (int i = threadIdx.x; i < n_rot / 2; i += blockDim.x) {
        float a = x[kv_lora + 2 * i], b = x[kv_lora + 2 * i + 1];
        rope_pair(a, b, (float) p, i, n_rot, base);
        v[kv_lora + 2 * i] = a;
        v[kv_lora + 2 * i + 1] = b;
    }
    __syncthreads();
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __half* sc = (__half*) (dst + kv_lora);
    __half* rp = sc + kv_lora / 32;
    for (int i = threadIdx.x; i < n_rot; i += blockDim.x) rp[i] = __float2half_rn(v[kv_lora + i]);
    for (int b = wp; b < kv_lora / 32; b += nw) {
        const float x0 = v[32 * b + lane];
        float am = fabsf(x0);
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const __half dh = __float2half_rn(am / 127.0f);
        const float d = __half2float(dh);
        const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(x0 / d))) : 0;
        dst[32 * b + lane] = (uint8_t) (int8_t) q;
        if (lane == 0) sc[b] = dh;
    }
    for (int i = kv_lora + kv_lora / 16 + 2 * n_rot + threadIdx.x; i < rb; i += blockDim.x) dst[i] = 0;   // (its pad)
}

// ---------------------------------------------------------------- MLA attention over the latent cache
// grid (chunks, ceil(H / 8), T); a warp a head, the chunk's keys in tiles of 32 rows through shared memory.  A row is
// 576 F16 = 288 words; lane l reads the words l + 32 i (bank l: no conflicts): q's dims 2 (l + 32 i) and the next, the
// value's (i < 8) the same.  Partial (m, l, acc[512]) a (t, head, chunk) -> attn_combine.
constexpr int kKC = 128, kKCmin = 32, kTile = 32;   // (a chunk's keys: kKC, or kKCmin for a few tokens - decode)
constexpr int kRowW = 288;   // words a row (kv_lora 512 + rope 64, F16)
// (Q8: the cache's Q8_0 rows of rb bytes - the latent's int8 values, its 16 blocks' F16 scales, the rotated part's F16
// values -, dequantized to F16 as the tile is loaded: the same arithmetic after it)
template <bool Q8>
__global__ void __launch_bounds__(256) attn_kernel(const float* __restrict__ q, int H, const void* __restrict__ cache_v,
                                                   int ring, const int* __restrict__ keys, int ldk,
                                                   const int* __restrict__ lo, const int* __restrict__ nk, float scale,
                                                   int nchunk, int kc, float* __restrict__ ws, int rb) {
    __shared__ __half2 tile[kTile][kRowW];
    __shared__ int rows[kTile];
    const int c = blockIdx.x, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * 8 + w;
    const int n = nk[t], k0 = c * kc, k1 = min(n, k0 + kc);
    float qr[18];
    if (h < H) {
        const float* qh = q + ((size_t) t * H + h) * (2 * kRowW);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            qr[2 * i] = qh[2 * (lane + 32 * i)];
            qr[2 * i + 1] = qh[2 * (lane + 32 * i) + 1];
        }
    }
    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int kt = k0; kt < k1; kt += kTile) {
        const int nt = min(kTile, k1 - kt);
        __syncthreads();
        if ((int) threadIdx.x < nt) {
            const int p = keys ? keys[(size_t) t * ldk + kt + threadIdx.x] : lo[t] + kt + (int) threadIdx.x;
            rows[threadIdx.x] = ring > 0 ? p % ring : p;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nt * kRowW; i += blockDim.x) {
            const int r = i / kRowW, col = i % kRowW;
            if (Q8) {
                const uint8_t* rp = (const uint8_t*) cache_v + (size_t) rows[r] * rb;
                if (col < 256) {
                    const char2 qv = *(const char2*) (rp + 2 * col);
                    const float d = __half2float(*(const __half*) (rp + 512 + 2 * (col >> 4)));
                    tile[r][col] = __floats2half2_rn(d * (float) qv.x, d * (float) qv.y);
                } else {
                    tile[r][col] = *(const __half2*) (rp + 544 + 4 * (col - 256));
                }
            } else {
                tile[r][col] = ((const __half2*) cache_v)[(size_t) rows[r] * kRowW + col];
            }
        }
        __syncthreads();
        if (h >= H) continue;
        for (int j = 0; j < nt; ++j) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const float2 kv = __half22float2(tile[j][lane + 32 * i]);
                s = fmaf(qr[2 * i], kv.x, fmaf(qr[2 * i + 1], kv.y, s));
            }
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s);
            const float corr = expf(m - mn), pj = expf(s - mn);
            l = l * corr + pj;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 v = __half22float2(tile[j][lane + 32 * i]);
                acc[2 * i] = fmaf(pj, v.x, acc[2 * i] * corr);
                acc[2 * i + 1] = fmaf(pj, v.y, acc[2 * i + 1] * corr);
            }
            m = mn;
        }
    }
    if (h >= H) return;
    float* o = ws + (((size_t) t * H + h) * nchunk + c) * (512 + 2);
    if (lane == 0) {
        o[0] = m;
        o[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[2 + 2 * (lane + 32 * i)] = acc[2 * i];
        o[2 + 2 * (lane + 32 * i) + 1] = acc[2 * i + 1];
    }
}
// a block a (t, head): the chunks merged with the sink, 512 threads (a value dim each)
__global__ void attn_combine_kernel(int H, int nchunk, int kc, const int* __restrict__ nk,
                                    const float* __restrict__ sinks, const float* __restrict__ ws,
                                    float* __restrict__ out) {
    const int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    const int used = (nk[t] + kc - 1) / kc;
    const float* base = ws + ((size_t) t * H + h) * nchunk * (512 + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int c = 0; c < used; ++c) M = fmaxf(M, base[(size_t) c * 514]);
    float den = sinks ? expf(sinks[h] - M) : 0.0f, num = 0.0f;
    for (int c = 0; c < used; ++c) {
        const float* o = base + (size_t) c * 514;
        if (o[0] == -INFINITY) continue;
        const float f = expf(o[0] - M);
        den += o[1] * f;
        num += o[2 + d] * f;
    }
    out[((size_t) t * H + h) * 512 + d] = den > 0.0f ? num / den : 0.0f;
}

// ---------------------------------------------------------------- elementwise
__global__ void sigmoid_mul_kernel(float* x, const float* g, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf_(g[i]);
}
__global__ void swiglu_kernel(const float* g, const float* u, size_t ldg, int n, float* h, size_t ldh, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    const float gv = g[r * ldg + c];
    h[r * ldh + c] = gv / (1.0f + expf(-gv)) * u[r * ldg + c];
}
__global__ void add_kernel(float* y, const float* x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// ---------------------------------------------------------------- the router and the experts' bookkeeping
// a block (1024 threads) a token: n_exp <= 1024 candidates (sigmoid + bias, the index) bitonic-sorted in shared memory
// (larger first, ties: the lower index), the first n_used the picks
__global__ void __launch_bounds__(1024) router_kernel(const float* __restrict__ logits, int n_exp,
                                                      const float* __restrict__ bias, int n_used, int norm, float scale,
                                                      int* __restrict__ ids, float* __restrict__ wout) {
    __shared__ float sv[1024], sp[1024];
    __shared__ int si[1024];
    const int t = blockIdx.x, i = threadIdx.x;
    const float* lr = logits + (size_t) t * n_exp;
    if (i < n_exp) {
        const float p = sigmoidf_(lr[i]);
        sp[i] = p;
        sv[i] = p + (bias ? bias[i] : 0.0f);
        si[i] = i;
    } else {
        sv[i] = -INFINITY;
        si[i] = 0x7fffffff;
    }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = i ^ j;
            if (ixj > i) {
                const float a = sv[i], b = sv[ixj];
                const int ia = si[i], ib = si[ixj];
                // "first" in the descending order: larger value, then lower index
                const bool b_first = b > a || (b == a && ib < ia);
                const bool desc = (i & k) == 0;
                if (desc ? b_first : !b_first) {
                    sv[i] = b;
                    sv[ixj] = a;
                    si[i] = ib;
                    si[ixj] = ia;
                }
            }
            __syncthreads();
        }
    if (i < 32) {
        const float p = i < n_used ? sp[si[i]] : 0.0f;
        const float sum = warp_sum(p);
        const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        if (i < n_used) {
            ids[(size_t) t * n_used + i] = si[i];
            wout[(size_t) t * n_used + i] = p / den * scale;
        }
    }
}
// ... a warp a token: each lane holds every 32nd candidate in registers, n_used rounds of a warp-wide argmax (larger value,
// then the lower index - the sort's order) pick them; the weights summed as above (pick r in lane r)
template <int PER>
__global__ void __launch_bounds__(32) router_warp_kernel(const float* __restrict__ logits, int n_exp,
                                                        const float* __restrict__ bias, int n_used, int norm,
                                                        float scale, int* __restrict__ ids, float* __restrict__ wout) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* lr = logits + (size_t) t * n_exp;
    float v[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = lane + 32 * k;
        v[k] = i < n_exp ? sigmoidf_(lr[i]) + (bias ? bias[i] : 0.0f) : -INFINITY;
    }
    int mine = -1;   // (the pick this lane reports: round `lane`'s)
    for (int r = 0; r < n_used; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = lane + 32 * k;
            if (v[k] > bv || (v[k] == bv && i < bi)) {
                bv = v[k];
                bi = i;
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
        if ((bi & 31) == lane) {
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (lane + 32 * k == bi) v[k] = -INFINITY;
        }
        if (lane == r) mine = bi;
    }
    const float p = lane < n_used ? sigmoidf_(lr[mine]) : 0.0f;
    const float sum = warp_sum(p);
    const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
    if (lane < n_used) {
        ids[(size_t) t * n_used + lane] = mine;
        wout[(size_t) t * n_used + lane] = p / den * scale;
    }
}
// ... 512 candidates at most, 16 picks at most: a thread a candidate, each warp sorting its 32 (a register bitonic
// sort, larger first, then the lower index), the warps' top 16s merged pairwise (16 lists, 8, 4, 2, 1) - the same picks
// in the same order, the weights summed as above
__device__ __forceinline__ bool rk_before(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }
__device__ __forceinline__ void rk_step(float& v, int& i, int j, bool up, int lane) {
    const float pv = __shfl_xor_sync(0xffffffffu, v, j);
    const int pi = __shfl_xor_sync(0xffffffffu, i, j);
    const bool lower = (lane & j) == 0, pb = rk_before(pv, pi, v, i);
    if (lower == up ? pb : !pb) {
        v = pv;
        i = pi;
    }
}
__global__ void __launch_bounds__(512) router_bitonic_kernel(const float* __restrict__ logits, int n_exp,
                                                             const float* __restrict__ bias, int n_used, int norm,
                                                             float scale, int* __restrict__ ids, float* __restrict__ wout) {
    __shared__ float lv[16][16];
    __shared__ int li[16][16];
    const int t = blockIdx.x, e = threadIdx.x, w = e >> 5, lane = e & 31;
    const float* lr = logits + (size_t) t * n_exp;
    float v = e < n_exp ? sigmoidf_(lr[e]) + (bias ? bias[e] : 0.0f) : -INFINITY;
    int i = e < n_exp ? e : 0x7fffffff;
    for (int k = 2; k <= 32; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) rk_step(v, i, j, (lane & k) == 0, lane);
    if (lane < 16) {
        lv[w][lane] = v;
        li[w][lane] = i;
    }
    __syncthreads();
    for (int lists = 16; lists > 1; lists >>= 1) {
        // warp q < lists / 2 merges lists 2q and 2q + 1: the first ascending, the second reversed - a bitonic 32
        float mv = -INFINITY;
        int mi = 0x7fffffff;
        if (w < lists / 2) {
            mv = lane < 16 ? lv[2 * w][lane] : lv[2 * w + 1][31 - lane];
            mi = lane < 16 ? li[2 * w][lane] : li[2 * w + 1][31 - lane];
            for (int j = 16; j > 0; j >>= 1) rk_step(mv, mi, j, true, lane);
        }
        __syncthreads();
        if (w < lists / 2 && lane < 16) {
            lv[w][lane] = mv;
            li[w][lane] = mi;
        }
        __syncthreads();
    }
    if (w == 0) {
        const int pick = lane < n_used ? li[0][lane] : 0;
        const float p = lane < n_used ? sigmoidf_(lr[pick]) : 0.0f;
        const float sum = warp_sum(p);
        const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        if (lane < n_used) {
            ids[(size_t) t * n_used + lane] = pick;
            wout[(size_t) t * n_used + lane] = p / den * scale;
        }
    }
}
// one thread (a window has at most 16 x 16 entries): counting sort of the entries by expert
__global__ void group_kernel(const int* __restrict__ ids, int T, int n_used, int n_exp, const uint8_t* base,
                             size_t stride, unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups,
                             const int* __restrict__ slot, float* __restrict__ w) {
    __shared__ int cnt[1024], at[1024];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_used; ++j) {
            const int e = ids[t * n_used + j];
            if (slot && slot[e] < 0) w[t * n_used + j] = 0.0f;   // (the CPU lane's)
            else ++cnt[e];
        }
    int g = 0, off = 0;
    for (int e = 0; e < n_exp; ++e) {
        at[e] = off;
        if (cnt[e]) {
            grp_ptr[g] = (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
            grp_start[g] = off;
            ++g;
            off += cnt[e];
        }
    }
    grp_start[g] = off;
    *n_groups = g;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_used; ++j) {
            const int e = ids[t * n_used + j];
            if (!slot || slot[e] >= 0) grp_ent[at[e]++] = t * 16 + j;
        }
}
// ... the same groups and the same entry order (by expert, then t 16 + j), a thread an expert: the window's entries
// staged in shared memory, each thread counting and then listing its expert's, a block scan for the offsets and the
// group numbers
__global__ void __launch_bounds__(1024) group_par_kernel(const int* __restrict__ ids, int T, int n_used, int n_exp,
                                                         const uint8_t* base, size_t stride,
                                                         unsigned long long* grp_ptr, int* grp_start, int* grp_ent,
                                                         int* n_groups, const int* __restrict__ slot,
                                                         float* __restrict__ w) {
    __shared__ int ent[16 * 16];
    __shared__ int sc[1024], sg[1024];
    const int e = threadIdx.x, n_ent = T * n_used;
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int x = ids[i];
        const bool out = slot && slot[x] < 0;
        if (out) w[i] = 0.0f;   // (not this GPU's)
        ent[i] = out ? -1 : x;
    }
    __syncthreads();
    int c = 0;
    if (e < n_exp)
        for (int i = 0; i < n_ent; ++i) c += ent[i] == e;
    sc[e] = c;
    sg[e] = c > 0;
    __syncthreads();
    // inclusive scans of the counts and of the non-empty flags (Hillis-Steele over the block)
    for (int o = 1; o < 1024; o <<= 1) {
        const int a = e >= o ? sc[e - o] : 0, b = e >= o ? sg[e - o] : 0;
        __syncthreads();
        sc[e] += a;
        sg[e] += b;
        __syncthreads();
    }
    const int off = sc[e] - c, gi = sg[e] - (c > 0);
    if (e < n_exp && c > 0) {
        grp_ptr[gi] = (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
        grp_start[gi] = off;
        int k = off;
        for (int i = 0; i < n_ent; ++i)
            if (ent[i] == e) grp_ent[k++] = (i / n_used) * 16 + i % n_used;
    }
    if (e == 1023 || e == n_exp - 1) {
        if (e == n_exp - 1) {
            grp_start[sg[e]] = sc[e];
            *n_groups = sg[e];
        }
    }
}
__global__ void plan_kernel(const int* ids, int n_used, const uint8_t* base, size_t stride, unsigned long long* plan,
                            const int* __restrict__ slot, float* __restrict__ w) {
    const int j = threadIdx.x;
    if (j >= n_used) return;
    const int s = slot ? slot[ids[j]] : ids[j];
    plan[j] = s >= 0 ? (unsigned long long) (base + (size_t) s * stride) : 0ull;   // (0: the CPU lane's)
    if (s < 0) w[j] = 0.0f;
}
__global__ void moe_sum_kernel(const float* __restrict__ part, int n_used, int n, const float* __restrict__ w,
                               float* __restrict__ out, int stride16) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < n_used; ++j) {
        const float wj = w[(size_t) t * n_used + j];
        if (wj != 0.0f) v = fmaf(wj, part[((size_t) t * stride16 + j) * n + i], v);   // (0: the CPU lane's entry)
    }
    out[(size_t) t * n + i] = v;
}
__global__ void dup_q8_kernel(const int8_t* q, const float* d, int k, int8_t* q2, float* d2) {
    const int r = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const int8_t v = q[(size_t) r * k + i];
        q2[(size_t) (2 * r) * k + i] = v;
        q2[(size_t) (2 * r + 1) * k + i] = v;
        if (i < k / 32) {
            const float dv = d[(size_t) r * (k / 32) + i];
            d2[(size_t) (2 * r) * (k / 32) + i] = dv;
            d2[(size_t) (2 * r + 1) * (k / 32) + i] = dv;
        }
    }
}

// ---------------------------------------------------------------- the prompt path's attention (around two GEMMs)
// a block (256 threads) a (token, head) row
__global__ void __launch_bounds__(256) attn_softmax_kernel(const float* __restrict__ S, size_t ldS, int H, int nkr,
                                                           int k0, const int* __restrict__ pos, int window,
                                                           const float* __restrict__ mask, size_t ldm, float scale,
                                                           const float* __restrict__ sinks, __half* __restrict__ P,
                                                           size_t ldP) {
    __shared__ float sh[32];
    const int r = blockIdx.x, t = r / H, h = r % H;
    const int p = pos[t];
    const float* sr = S + (size_t) r * ldS;
    __half* pr = P + (size_t) r * ldP;
    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    auto seen = [&](int j) {
        const int q = k0 + j;
        if (q > p || q < lo) return false;
        return window > 0 || mask[(size_t) t * ldm + j] == 0.0f;
    };
    float m = sinks ? sinks[h] : -INFINITY;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        if (seen(j)) m = fmaxf(m, sr[j] * scale);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : -INFINITY;
        v = warp_max(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    m = sh[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        if (seen(j)) sum += expf(sr[j] * scale - m);
    sum = block_sum(sum) + (sinks ? expf(sinks[h] - m) : 0.0f);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x)
        pr[j] = __float2half_rn(seen(j) ? expf(sr[j] * scale - m) * inv : 0.0f);
}
__global__ void dsa_mask_kernel(const int* __restrict__ keys, int ldk, const int* __restrict__ nk, int k0, int nkr,
                                float* __restrict__ mask, size_t ldm) {
    const int t = blockIdx.x;
    float* mr = mask + (size_t) t * ldm;
    for (int j = threadIdx.x; j < nkr; j += blockDim.x) mr[j] = -INFINITY;
    __syncthreads();
    for (int i = threadIdx.x; i < nk[t]; i += blockDim.x) {
        const int j = keys[(size_t) t * ldk + i] - k0;
        if (j >= 0 && j < nkr) mr[j] = 0.0f;
    }
}
// Q8_0 rows (rb bytes) -> F16 rows of kv_lora + n_rot values, a thread a value pair
__global__ void kv_deq_kernel(const uint8_t* __restrict__ cache, int ring, int kv_lora, int n_rot, int rb, int k0, int n,
                              __half2* __restrict__ dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int half_row = (kv_lora + n_rot) / 2;
    if (i >= (size_t) n * half_row) return;
    const int j = (int) (i / half_row), c = (int) (i % half_row);
    const int q = k0 + j;
    const uint8_t* rp = cache + (size_t) (ring > 0 ? q % ring : q) * rb;
    if (2 * c < kv_lora) {
        const char2 qv = *(const char2*) (rp + 2 * c);
        const float d = __half2float(*(const __half*) (rp + kv_lora + 2 * (c >> 4)));
        dst[i] = __floats2half2_rn(d * (float) qv.x, d * (float) qv.y);
    } else {
        dst[i] = *(const __half2*) (rp + kv_lora + kv_lora / 16 + 2 * (2 * c - kv_lora));
    }
}
__global__ void kv_gather_kernel(const uint4* __restrict__ cache, int ring, int row16, int k0, int n,
                                 uint4* __restrict__ dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * row16) return;
    const int j = (int) (i / row16), c = (int) (i % row16);
    const int q = k0 + j;
    dst[i] = cache[(size_t) (ring > 0 ? q % ring : q) * row16 + c];
}

// ---------------------------------------------------------------- the prompt path
__global__ void to_f16_kernel(const float* __restrict__ src, size_t lds, int n, __half* __restrict__ dst, size_t ldd,
                              size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n, c = i % n;
    dst[r * ldd + c] = __float2half_rn(src[r * lds + c]);
}
// one block: the counts, their scan, the scatter (the order within an expert does not change a row's sums)
__global__ void __launch_bounds__(1024) moe_sort_kernel(const int* __restrict__ ids, int n_ent, int n_exp,
                                                        int* __restrict__ bounds, int* __restrict__ tok,
                                                        int* __restrict__ row_of, int n_used,
                                                        const int* __restrict__ slot, float* __restrict__ w) {
    __shared__ int cnt[1024];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s >= 0) atomicAdd(&cnt[s], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int off = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int c = cnt[e];
            bounds[e] = off;
            cnt[e] = off;
            off += c;
        }
        bounds[n_exp] = off;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s < 0) {   // (not in this set: no row here)
            row_of[i] = -1;
            if (w) w[i] = 0.0f;
            continue;
        }
        const int r = atomicAdd(&cnt[s], 1);
        tok[r] = i / n_used;
        row_of[i] = r;
    }
#if defined(STRATA_USE_HIP)
    // AMD: the scatter's order within an expert is the order its threads' atomics land in, which differs from one
    // run to the next there - and the expert kernels' sums of a row depend a little on the rows beside it (their
    // tiles), so two starts gave two answers.  Each expert's rows put in token order (a token routes to an expert
    // once: its rank among the expert's tokens is its row) - the same rows every run
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s < 0) continue;
        const int t = i / n_used, b0 = bounds[s], b1 = cnt[s];   // (cnt: each expert's end after the scatter)
        int rank = 0;
        for (int r = b0; r < b1; ++r) rank += tok[r] < t ? 1 : 0;
        row_of[i] = b0 + rank;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x)
        if (row_of[i] >= 0) tok[row_of[i]] = i / n_used;
#endif
}
__global__ void moe_combine_kernel(const float* __restrict__ y, const int* __restrict__ row_of,
                                   const float* __restrict__ w, int n_used, int n, float* __restrict__ out, int acc) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < n_used; ++j) {
        const int r = row_of[(size_t) t * n_used + j];
        if (r >= 0) v = fmaf(w[(size_t) t * n_used + j], y[(size_t) r * n + i], v);
    }
    out[(size_t) t * n + i] = acc ? out[(size_t) t * n + i] + v : v;
}

// ---------------------------------------------------------------- a draft's top token
__global__ void __launch_bounds__(1024) top1_kernel(const float* __restrict__ row, int n, float* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (row[i] > best) best = row[i], bi = i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) sv[w] = best, si[w] = bi;
    __syncthreads();
    if (w == 0) {
        best = l < (int) (blockDim.x >> 5) ? sv[l] : -INFINITY;
        bi = l < (int) (blockDim.x >> 5) ? si[l] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
        }
        if (l == 0) sv[0] = best, si[0] = bi;
    }
    __syncthreads();
    const float m = sv[0];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(row[i] - m);
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        out[0] = __int_as_float(si[0]);
        out[1] = 1.0f / sum;
    }
}

// ... over several blocks: each its slice's top value (ties: the lower id) and sum of exp(v - its top), the last block to
// finish combining them in block order (one launch)
constexpr int kTop1Blocks = 64;
__device__ unsigned g_top1_done = 0;
__device__ float g_top1_part[kTop1Blocks * 3];
__global__ void __launch_bounds__(256) top1_multi_kernel(const float* __restrict__ row, int n, float* __restrict__ out) {
    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ bool last;
    const int per = (n + gridDim.x - 1) / gridDim.x, a0 = blockIdx.x * per, a1 = min(n, a0 + per);
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = a0 + threadIdx.x; i < a1; i += blockDim.x)
        if (row[i] > best) best = row[i], bi = i;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    if (l == 0) sv[w] = best, si[w] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 8; ++k)
            if (sv[k] > sv[0] || (sv[k] == sv[0] && si[k] < si[0])) sv[0] = sv[k], si[0] = si[k];
    }
    __syncthreads();
    const float m = sv[0];
    float sum = 0.0f;
    for (int i = a0 + threadIdx.x; i < a1; i += blockDim.x) sum += expf(row[i] - m);
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        g_top1_part[blockIdx.x * 3 + 0] = m;
        g_top1_part[blockIdx.x * 3 + 1] = __int_as_float(si[0]);
        g_top1_part[blockIdx.x * 3 + 2] = sum;
        __threadfence();
        last = atomicAdd(&g_top1_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float M = -INFINITY;
    int I = 0x7fffffff;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float v = ((volatile float*) g_top1_part)[b * 3];
        const int ix = __float_as_int(((volatile float*) g_top1_part)[b * 3 + 1]);
        if (v > M || (v == M && ix < I)) M = v, I = ix;
    }
    float S = 0.0f;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float v = ((volatile float*) g_top1_part)[b * 3];
        if (v > -INFINITY) S += ((volatile float*) g_top1_part)[b * 3 + 2] * expf(v - M);
    }
    out[0] = __int_as_float(I);
    out[1] = 1.0f / S;
    g_top1_done = 0;
}

// ---------------------------------------------------------------- Engram
__device__ __forceinline__ float f8e4m3(uint8_t x) {
    const int ax = x & 0x7f;
    if (ax == 0x7f) return __int_as_float(0x7fc00000);
    const int e = (ax >> 3) & 0xf, m = ax & 7;
    const float v = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return (x & 0x80) ? -v : v;
}
// a block (dim threads, dim <= 1024) a (column, token)
__global__ void engram_rows_kernel(const uint8_t* __restrict__ rows, const float* __restrict__ valid, int cols, int dim,
                                   const float* __restrict__ ln, float eps, float* __restrict__ r,
                                   const float* __restrict__ scale) {
    const int col = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float v = i < dim ? f8e4m3(rows[((size_t) t * cols + col) * dim + i]) : 0.0f;
    if (scale != nullptr) v *= scale[(size_t) t * cols + col];
    const float ss = block_sum(v * v);
    const float rr = 1.0f / sqrtf(ss / (float) dim + eps);
    if (i < dim) r[((size_t) t * cols + col) * dim + i] = v * rr * ln[(size_t) col * dim + i] * valid[t];
}
__global__ void engram_mix_kernel(float* __restrict__ x, const float* __restrict__ kv, const float* __restrict__ kv_b,
                                  int hc, int E, const float* __restrict__ egate, const float* __restrict__ gamma,
                                  const float* __restrict__ hcs, const float* __restrict__ valid, float eps) {
    __shared__ float s_gate[8];
    __shared__ float s_rv;
    const int t = blockIdx.x;
    if (valid[t] == 0.0f) return;   // (uniform: the whole block)
    const float* kt = kv + (size_t) t * (hc + 1) * E;
    float* xt = x + (size_t) t * hc * E;
    for (int s = 0; s < hc; ++s) {
        float sx = 0.0f, sk = 0.0f, sd = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) {
            const float xv = xt[(size_t) s * E + d], kvv = kt[(size_t) s * E + d] + kv_b[(size_t) s * E + d];
            sx += xv * xv;
            sk += kvv * kvv;
            sd += xv * kvv * egate[(size_t) s * E + d];
        }
        sx = block_sum(sx);
        sk = block_sum(sk);
        sd = block_sum(sd);
        if (threadIdx.x == 0) {
            const float rx = 1.0f / sqrtf(sx / (float) E + eps), rk = 1.0f / sqrtf(sk / (float) E + eps);
            const float dot = sd * rx * rk / sqrtf((float) E);
            const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            s_gate[s] = sigmoidf_(dot < 0.0f ? -mag : mag);
        }
    }
    float sv = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        const float v = kt[(size_t) hc * E + d] + kv_b[(size_t) hc * E + d];
        sv += v * v;
    }
    sv = block_sum(sv);
    if (threadIdx.x == 0) s_rv = 1.0f / sqrtf(sv / (float) E + eps);
    __syncthreads();
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        const float sig = (kt[(size_t) hc * E + d] + kv_b[(size_t) hc * E + d]) * s_rv * gamma[d];
        for (int s = 0; s < hc; ++s) xt[(size_t) s * E + d] += sig * (s_gate[s] + hcs[(size_t) s * E + d]);
    }
}

// ---------------------------------------------------------------- the indexer
// a block (dim threads) a new group
template <typename P, typename V>   // (the pooled keys' and the value | gate ring's types: float, or __half with --kv q8)
__global__ void ix_pool_kernel(const V* __restrict__ ivg, int ring, int g0, int kpool, int dim,
                               const float* __restrict__ ape, const float* __restrict__ w, float eps, int n_rot,
                               float base, P* __restrict__ pooled) {
    __shared__ float sv[1024];
    const int g = g0 + blockIdx.x, c = threadIdx.x;
    float lg[8], mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) {
        const int p = g * kpool + k;
        lg[k] = (float) ivg[(size_t) (p % ring) * 2 * dim + dim + c] + ape[(size_t) k * dim + c];
        mx = fmaxf(mx, lg[k]);
    }
    float den = 0.0f, num = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const int p = g * kpool + k;
        const float e = expf(lg[k] - mx);
        den += e;
        num += e * (float) ivg[(size_t) (p % ring) * 2 * dim + c];
    }
    const float v = num / den;
    const float ss = block_sum(v * v);
    sv[c] = v / sqrtf(ss / (float) dim + eps) * w[c];
    __syncthreads();
    const int nope = dim - n_rot;
    if (c >= nope && ((c - nope) & 1) == 0) {
        float a = sv[c], b = sv[c + 1];
        rope_pair(a, b, (float) (g * kpool), (c - nope) / 2, n_rot, base);
        sv[c] = a;
        sv[c + 1] = b;
    }
    __syncthreads();
    pooled[(size_t) g * dim + c] = (P) sv[c];
}
// grid (ceil(n_pool / 256), T): a thread a pool
__device__ __forceinline__ float4 pooled4(const float* p, int i) { return ((const float4*) p)[i]; }
__device__ __forceinline__ float4 pooled4(const __half* p, int i) {
    const uint2 u = ((const uint2*) p)[i];
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
template <typename P>
__global__ void __launch_bounds__(256) ix_score_kernel(const float* __restrict__ q, const float* __restrict__ wt,
                                                       int heads, int dim, const P* __restrict__ pooled,
                                                       const int* __restrict__ pos, int kpool, int n_pool,
                                                       float* __restrict__ score, size_t ld) {
    extern __shared__ float sq[];   // [heads][dim], then wt[heads]
    const int t = blockIdx.y, g = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) sq[i] = q[(size_t) t * heads * dim + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) sq[heads * dim + i] = wt[(size_t) t * heads + i];
    __syncthreads();
    if (g >= n_pool) return;
    const int nv = pos[t] / kpool;
    float out = -INFINITY;
    if (g < nv) {
        float acc[16];
#pragma unroll
        for (int h = 0; h < 16; ++h) acc[h] = 0.0f;
        const P* pk = pooled + (size_t) g * dim;
        for (int i = 0; i < dim / 4; ++i) {
            const float4 kk = pooled4(pk, i);
#pragma unroll
            for (int h = 0; h < 16; ++h)
                if (h < heads) {
                    const float* qh = sq + h * dim + 4 * i;
                    acc[h] = fmaf(qh[0], kk.x, fmaf(qh[1], kk.y, fmaf(qh[2], kk.z, fmaf(qh[3], kk.w, acc[h]))));
                }
        }
        out = 0.0f;
#pragma unroll
        for (int h = 0; h < 16; ++h)
            if (h < heads) out = fmaf(fmaxf(acc[h], 0.0f), sq[heads * dim + h], out);
    }
    score[(size_t) t * ld + g] = out;
}
__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// a block (1024 threads) a token: the top `top` of its nv scores (radix select on the order-preserving keys, then the
// picks in index order - the ties at the threshold the lowest indices), their kpool positions each, then the tail
__global__ void __launch_bounds__(1024) ix_select_kernel(const float* __restrict__ score, size_t ld,
                                                         const int* __restrict__ pos, int kpool, int top,
                                                         int* __restrict__ keys, int ldk, int* __restrict__ nk) {
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[1024], s_eq[1024];
    const int t = blockIdx.x, nth = blockDim.x, tid = threadIdx.x;
    const int p = pos[t], nv = p / kpool;
    const float* sc = score + (size_t) t * ld;
    int* kt = keys + (size_t) t * ldk;
    const int nsel = min(nv, top);
    uint32_t thr = 0;   // the threshold key: every key > thr picked, then ties at thr
    int take_eq = 0;    // ... how many of the ties
    if (nv > top) {
        if (tid == 0) {
            s_prefix = 0;
            s_mask = 0;
            s_need = top;
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nth) hist[i] = 0;
            __syncthreads();
            const uint32_t pre = s_prefix, msk = s_mask;
            for (int g = tid; g < nv; g += nth) {
                const uint32_t kk = ord_key(sc[g]);
                if ((kk & msk) == pre) atomicAdd(&hist[(kk >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int need = s_need, b = 255;
                for (; b > 0; --b) {
                    if (hist[b] >= need) break;
                    need -= hist[b];
                }
                s_need = need;
                s_prefix = pre | ((uint32_t) b << shift);
                s_mask = msk | (255u << shift);
            }
            __syncthreads();
        }
        thr = s_prefix;
        take_eq = s_need;   // (the picks among the keys equal to thr)
    }
    // the picks in index order: each thread a contiguous range, its counts, a block scan, then the writes
    const int per = (nv + nth - 1) / nth, b0 = min(nv, tid * per), b1 = min(nv, b0 + per);
    int cg = 0, ce = 0;
    if (nv > top)
        for (int g = b0; g < b1; ++g) {
            const uint32_t kk = ord_key(sc[g]);
            cg += kk > thr;
            ce += kk == thr;
        }
    else
        cg = b1 - b0;
    s_gt[tid] = cg;
    s_eq[tid] = ce;
    __syncthreads();
    if (tid == 0) {   // (exclusive scans; 1024 entries)
        int a = 0, e = 0;
        for (int i = 0; i < nth; ++i) {
            const int x = s_gt[i], y = s_eq[i];
            s_gt[i] = a;
            s_eq[i] = e;
            a += x;
            e += y;
        }
    }
    __syncthreads();
    // a pick's slot: the picks before it in index order = the greater keys before it + the ties taken before it (the
    // first take_eq ties in index order)
    int ng = s_gt[tid], ne = s_eq[tid];
    for (int g = b0; g < b1; ++g) {
        int slot = -1;
        if (nv <= top) {
            slot = ng++;
        } else {
            const uint32_t kk = ord_key(sc[g]);
            if (kk > thr) {
                slot = ng + min(ne, take_eq);
                ++ng;
            } else if (kk == thr) {
                if (ne < take_eq) slot = ng + ne;
                ++ne;
            }
        }
        if (slot < 0) continue;
        for (int k = 0; k < kpool; ++k) kt[slot * kpool + k] = g * kpool + k;
    }
    __syncthreads();
    const int ntail = p % kpool + 1;
    for (int k = tid; k < ntail; k += nth) kt[nsel * kpool + k] = nv * kpool + k;
    if (tid == 0) nk[t] = nsel * kpool + ntail;
}

}  // namespace

// ---------------------------------------------------------------- launches
void q8_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    if (rows <= 0) return;
    q8_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) rows), 256, 0, s>>>(x, ldx, k, q, d);
    check("q8_rows");
}
void q8_heads(const float* x, size_t ldt, size_t ldh, int T, int H, int k, int8_t* q, float* d, cudaStream_t s) {
    if (T <= 0 || H <= 0) return;
    q8_heads_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) (T * H)), 256, 0, s>>>(x, ldt, ldh, T, k, q, d);
    check("q8_heads");
}
void copy_cols(const float* src, size_t lds, float* dst, size_t ldd, int rows, int n, cudaStream_t s) {
    const size_t total = (size_t) rows * n;
    if (total == 0) return;
    copy_cols_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(src, lds, dst, ldd, n, total);
    check("copy_cols");
}
void store_rows(const float* src, int T, int n, float* ring, int R, const int* pos, cudaStream_t s, bool f16) {
    if (T <= 0) return;
    if (f16)
        store_rows_kernel<__half><<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(src, n, (__half*) ring,
                                                                                                R, pos);
    else
        store_rows_kernel<float><<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(src, n, ring, R, pos);
    check("store_rows");
}
void gemv_f32w(const float* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
               float alpha, cudaStream_t s) {
    gemv<float>(W, ld, k, n, x, ldx, ncol, y, ldy, alpha, s);
}
void gemv_bf16w(const uint16_t* W, size_t ld, int k, int n, const float* x, size_t ldx, int ncol, float* y, size_t ldy,
                float alpha, cudaStream_t s) {
    gemv<uint16_t>(W, ld, k, n, x, ldx, ncol, y, ldy, alpha, s);
}
void rmsnorm(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s) {
    if (rows <= 0) return;
    rms_kernel<<<rows, 256, 0, s>>>(x, ldx, n, w, eps, y, ldy);
    check("rmsnorm");
}
void hc_pre(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
            float hc_eps, float mag, float* cur, float* post, cudaStream_t s, const float* norm_w, float norm_eps) {
    float* part = hc_part_buf(T);
    if (part == nullptr || 2 * hc > kMaxM || hc > 8 || E > 512 * 8) {
        std::fprintf(stderr, "maya-hy4 hc_pre: no room for its sums (%d streams)\n", hc);
        g_err.fetch_add(1);
        return;
    }
    // (HY4_HC_ONE=1: one launch, the last block of a token finishing it - measured no faster: the launches queue ahead)
    static const bool two = std::getenv("HY4_HC_ONE") == nullptr;
    unsigned* cnt = two || E > 256 * 12 ? nullptr : hc_cnt_buf(T);
    if (cnt) {
        hc_pre1_kernel<<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc, E, fn, part, cnt, scale, base, eps, hc_eps,
                                                                     mag, cur, post, g_mix_out, norm_w, norm_eps);
    } else {
        hc_part_kernel<<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, 2 * hc, fn, part);
        hc_fin_kernel<false><<<T, 512, 0, s>>>(x, hc, E, part, scale, base, eps, hc_eps, mag, cur, post, g_mix_out,
                                               norm_w, norm_eps);
    }
    check("hc_pre");
}
void hc_head(const float* x, int T, int hc, int E, const float* fn, const float* scale, const float* base, float eps,
             float hc_eps, float* cur, cudaStream_t s) {
    float* part = hc_part_buf(T);
    if (part == nullptr || hc > kMaxM) {
        std::fprintf(stderr, "maya-hy4 hc_head: no room for its sums\n");
        g_err.fetch_add(1);
        return;
    }
    hc_part_kernel<<<dim3(kHcBlocks, (unsigned) T), 256, 0, s>>>(x, hc * E, hc, fn, part);
    hc_fin_kernel<true><<<T, 512, 0, s>>>(x, hc, E, part, scale, base, eps, hc_eps, 1.0f, cur, nullptr, nullptr, nullptr,
                                          0.0f);
    check("hc_head");
}
void hc_post(float* x, int T, int hc, int E, const float* y, const float* post, cudaStream_t s) {
    const size_t n = (size_t) T * hc * E;
    hc_post_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, hc, E, y, post, n);
    check("hc_post");
}
void hc_post_norm(float* x, int T, int hc, int E, const float* y, const float* post, const float* w, float eps,
                  cudaStream_t s, float* yo) {
    if (hc > 8 || E % 4) {
        std::fprintf(stderr, "maya-hy4 hc_post_norm: %d streams of %d (8 / a multiple of 4)\n", hc, E);
        g_err.fetch_add(1);
        return;
    }
    hc_post_norm_kernel<<<dim3(kPostBlocks, (unsigned) T), 256, 0, s>>>(x, hc, E, y, post, w, eps, yo);
    check("hc_post_norm");
}
void rope(float* x, int T, int nh, size_t ld, size_t ldt, int off, int n_rot, const int* pos, float base, cudaStream_t s) {
    const int n = nh * n_rot / 2;
    rope_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(x, nh, ld, ldt, off, n_rot, pos, base);
    check("rope");
}
size_t kv_q8_row_bytes(int kv_lora, int n_rot) {
    return ((size_t) kv_lora + (size_t) kv_lora / 16 + 2 * (size_t) n_rot + 15) / 16 * 16;
}
void kv_store(const float* kv, int T, int kv_lora, int n_rot, const float* w, float eps, const int* pos, float base,
              uint16_t* cache, int ring, cudaStream_t s, bool q8) {
    const int row = kv_lora + n_rot;
    if (q8 && (kv_lora % 32 || n_rot % 2 || row > 1024)) {
        std::fprintf(stderr, "maya-hy4 kv_store: a Q8_0 row of %d + %d values (whole blocks of 32, at most 1024)\n",
                     kv_lora, n_rot);
        g_err.fetch_add(1);
        return;
    }
    if (q8)
        kv_store_q8_kernel<<<T, 256, 0, s>>>(kv, kv_lora, n_rot, w, eps, pos, base, (uint8_t*) cache, ring,
                                             (int) kv_q8_row_bytes(kv_lora, n_rot));
    else
        kv_store_kernel<<<T, 256, 0, s>>>(kv, kv_lora, n_rot, w, eps, pos, base, (__half*) cache, ring);
    check("kv_store");
}
size_t attn_ws_floats(int T, int H, int kv_lora, int max_keys) {
    // (the larger of the two chunkings: T tokens at kKC, or the few-token calls' kKCmin)
    return std::max((size_t) T * H * ((max_keys + kKC - 1) / kKC),
                    (size_t) kMaxWin * H * ((max_keys + kKCmin - 1) / kKCmin)) * (kv_lora + 2);
}
void attention(const float* q, int T, int H, int kv_lora, int n_rot, const uint16_t* cache, int ring, const int* keys,
               int ldk, const int* lo, const int* nk, int max_keys, float scale, const float* sinks, float* ws,
               float* out, cudaStream_t s, bool q8) {
    if (kv_lora != 512 || n_rot != 64) {
        std::fprintf(stderr, "maya-hy4 attention: kv_lora %d + rope %d (512 + 64 only)\n", kv_lora, n_rot);
        g_err.fetch_add(1);
        return;
    }
    // (every window of the token path - a decode step, a draft's verify window - in chunks of kKCmin: a token's sums are
    // the same whichever window it is in, so drafting does not change a greedy answer)
    const int kc = T <= kMaxWin ? kKCmin : kKC, nchunk = (max_keys + kc - 1) / kc;
    const dim3 grid((unsigned) nchunk, (unsigned) ((H + 7) / 8), (unsigned) T);
    if (q8)
        attn_kernel<true><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws,
                                               (int) kv_q8_row_bytes(kv_lora, n_rot));
    else
        attn_kernel<false><<<grid, 256, 0, s>>>(q, H, cache, ring, keys, ldk, lo, nk, scale, nchunk, kc, ws, 0);
    check("attention");
    attn_combine_kernel<<<dim3((unsigned) H, (unsigned) T), 512, 0, s>>>(H, nchunk, kc, nk, sinks, ws, out);
    check("attn_combine");
}
void sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s) {
    sigmoid_mul_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, g, n);
    check("sigmoid_mul");
}
void swiglu(const float* g, const float* u, size_t ldg, int rows, int n, float* h, size_t ldh, cudaStream_t s) {
    const size_t total = (size_t) rows * n;
    swiglu_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(g, u, ldg, n, h, ldh, total);
    check("swiglu");
}
void add(float* y, const float* x, size_t n, cudaStream_t s) {
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, x, n);
    check("add");
}
void router(const float* logits, int T, int n_exp, const float* bias, int n_used, bool norm, float scale, int* ids,
            float* w, cudaStream_t s) {
    if (n_exp > 1024 || n_used > 16) {
        std::fprintf(stderr, "maya-hy4 router: %d experts, %d a token (1024 / 16 at most)\n", n_exp, n_used);
        g_err.fetch_add(1);
        return;
    }
    static const bool sort = std::getenv("HY4_ROUTER_SORT") != nullptr;   // (the block-wide sort: the earlier kernel)
    static const bool warp = std::getenv("HY4_ROUTER_WARP") != nullptr;   // (the one-warp picks)
    if (!sort && !warp && n_exp <= 512 && n_used <= 16)
        router_bitonic_kernel<<<(unsigned) T, 512, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else if (!sort && n_exp <= 512 && n_used <= 32)
        router_warp_kernel<16><<<(unsigned) T, 32, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else if (!sort && n_used <= 32)
        router_warp_kernel<32><<<(unsigned) T, 32, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    else
        router_kernel<<<(unsigned) T, 1024, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w);
    check("router");
}
void group_experts(const int* ids, int T, int n_used, int n_exp, const uint8_t* base, size_t stride,
                   unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups, cudaStream_t s,
                   const int* slot, float* w) {
    static const bool serial = std::getenv("HY4_GROUP_SERIAL") != nullptr;   // (the earlier one-thread sort)
    if (!serial && n_exp <= 1024)
        group_par_kernel<<<1, 1024, 0, s>>>(ids, T, n_used, n_exp, base, stride, grp_ptr, grp_start, grp_ent, n_groups,
                                            slot, w);
    else
        group_kernel<<<1, 256, 0, s>>>(ids, T, n_used, n_exp, base, stride, grp_ptr, grp_start, grp_ent, n_groups, slot, w);
    check("group_experts");
}
void plan_experts(const int* ids, int n_used, const uint8_t* base, size_t stride, unsigned long long* plan,
                  cudaStream_t s, const int* slot, float* w) {
    plan_kernel<<<1, 32, 0, s>>>(ids, n_used, base, stride, plan, slot, w);
    check("plan_experts");
}
void moe_sum(const float* part, int T, int n_used, int n, const float* w, float* out, cudaStream_t s) {
    moe_sum_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(part, n_used, n, w, out, 16);
    check("moe_sum");
}
void dup_rows_q8(const int8_t* q, const float* d, int rows, int k, int8_t* q2, float* d2, cudaStream_t s) {
    if (rows <= 0) return;
    dup_q8_kernel<<<dim3((unsigned) ((k + 255) / 256), (unsigned) rows), 256, 0, s>>>(q, d, k, q2, d2);
    check("dup_rows_q8");
}
void engram_rows(const uint8_t* rows, const float* valid, int T, int cols, int dim, const float* lookup_norm, float eps,
                 float* r, cudaStream_t s, const float* scale) {
    const int th = (dim + 31) / 32 * 32;
    engram_rows_kernel<<<dim3((unsigned) cols, (unsigned) T), th, 0, s>>>(rows, valid, cols, dim, lookup_norm, eps, r,
                                                                         scale);
    check("engram_rows");
}
void engram_mix(float* x, const float* kv, const float* kv_b, int T, int hc, int E, const float* egate,
                const float* gamma, const float* hc_scale, const float* valid, float eps, cudaStream_t s) {
    engram_mix_kernel<<<T, 512, 0, s>>>(x, kv, kv_b, hc, E, egate, gamma, hc_scale, valid, eps);
    check("engram_mix");
}
void ix_pool(const float* ivg, int ring, int g0, int n, int kpool, int dim, const float* ape, const float* w, float eps,
             int n_rot, float base, float* pooled, cudaStream_t s, bool f16) {
    if (n <= 0) return;
    if (f16)
        ix_pool_kernel<__half, __half><<<n, dim, 0, s>>>((const __half*) ivg, ring, g0, kpool, dim, ape, w, eps, n_rot,
                                                         base, (__half*) pooled);
    else
        ix_pool_kernel<float, float><<<n, dim, 0, s>>>(ivg, ring, g0, kpool, dim, ape, w, eps, n_rot, base, pooled);
    check("ix_pool");
}
void ix_score(const float* q, const float* wt, int T, int heads, int dim, const float* pooled, const int* pos, int kpool,
              int n_pool, float* score, size_t ld, cudaStream_t s, bool f16) {
    if (n_pool <= 0) return;
    const size_t sh = (size_t) (heads * dim + heads) * sizeof(float);
    const dim3 grid((unsigned) ((n_pool + 255) / 256), (unsigned) T);
    if (f16)
        ix_score_kernel<__half><<<grid, 256, sh, s>>>(q, wt, heads, dim, (const __half*) pooled, pos, kpool, n_pool,
                                                      score, ld);
    else
        ix_score_kernel<float><<<grid, 256, sh, s>>>(q, wt, heads, dim, pooled, pos, kpool, n_pool, score, ld);
    check("ix_score");
}
void ix_select(const float* score, size_t ld, int T, const int* pos, int kpool, int top, int* keys, int ldk, int* nk,
               cudaStream_t s) {
    ix_select_kernel<<<T, 1024, 0, s>>>(score, ld, pos, kpool, top, keys, ldk, nk);
    check("ix_select");
}
void attn_softmax(const float* S, size_t ldS, int T, int H, int nkr, int k0, const int* pos, int window,
                  const float* mask, size_t ldm, float scale, const float* sinks, uint16_t* P, size_t ldP,
                  cudaStream_t s) {
    if (T <= 0) return;
    attn_softmax_kernel<<<T * H, 256, 0, s>>>(S, ldS, H, nkr, k0, pos, window, mask, ldm, scale, sinks, (__half*) P,
                                             ldP);
    check("attn_softmax");
}
void dsa_mask(const int* keys, int ldk, const int* nk, int T, int k0, int nkr, float* mask, size_t ldm, cudaStream_t s) {
    if (T <= 0) return;
    dsa_mask_kernel<<<T, 256, 0, s>>>(keys, ldk, nk, k0, nkr, mask, ldm);
    check("dsa_mask");
}
void kv_deq(const uint16_t* cache, int ring, int kv_lora, int n_rot, int k0, int n, uint16_t* dst, cudaStream_t s) {
    const size_t total = (size_t) n * ((kv_lora + n_rot) / 2);
    if (total == 0) return;
    kv_deq_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>((const uint8_t*) cache, ring, kv_lora, n_rot,
                                                                 (int) kv_q8_row_bytes(kv_lora, n_rot), k0, n,
                                                                 (__half2*) dst);
    check("kv_deq");
}
void kv_gather(const uint16_t* cache, int ring, int row, int k0, int n, uint16_t* dst, cudaStream_t s) {
    const int row16 = row * 2 / 16;   // (uint4s a row: 576 F16 = 72)
    const size_t total = (size_t) n * row16;
    if (total == 0) return;
    kv_gather_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>((const uint4*) cache, ring, row16, k0, n,
                                                                    (uint4*) dst);
    check("kv_gather");
}
void to_f16(const float* src, size_t lds, int rows, int n, uint16_t* dst, size_t ldd, cudaStream_t s) {
    const size_t total = (size_t) rows * n;
    if (total == 0) return;
    to_f16_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(src, lds, n, (__half*) dst, ldd, total);
    check("to_f16");
}
void moe_sort(const int* ids, int T, int n_used, int n_exp, int* bounds, int* tok, int* row_of, cudaStream_t s,
              const int* slot, float* w, int n_slots) {
    if (n_exp > 1024) {
        std::fprintf(stderr, "maya-hy4 moe_sort: %d experts (1024 at most)\n", n_exp);
        g_err.fetch_add(1);
        return;
    }
    moe_sort_kernel<<<1, 1024, 0, s>>>(ids, T * n_used, slot ? n_slots : n_exp, bounds, tok, row_of, n_used, slot, w);
    check("moe_sort");
}
void moe_combine(const float* y, const int* row_of, const float* w, int T, int n_used, int n, float* out,
                 cudaStream_t s, bool accumulate) {
    if (T <= 0) return;
    moe_combine_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(y, row_of, w, n_used, n, out,
                                                                                       accumulate ? 1 : 0);
    check("moe_combine");
}
namespace {
constexpr unsigned long long kWaitLimitNs = 60ull * 1000000000ull;
__device__ unsigned g_wait_timeouts = 0;
// thread 0 of the block waits for *flag >= v (wrapping); every thread after it
__device__ __forceinline__ void wait_for(const unsigned* flag, unsigned v) {
    if (threadIdx.x == 0) {
        const unsigned long long t0 = strata::kernels::clock_ns();
        while ((int) (*(volatile const unsigned*) flag - v) < 0) {
            __nanosleep(64);
            if (strata::kernels::clock_ns() - t0 > kWaitLimitNs) {
                atomicAdd(&g_wait_timeouts, 1u);
                break;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
}
}  // namespace
__global__ void publish_kernel(const float4* __restrict__ src, float4* dst, size_t n4, unsigned* counter, unsigned* flag,
                               unsigned v) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (last && threadIdx.x == 0) {
        *counter = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = v;
        __threadfence_system();
    }
}
__global__ void wait_pull_kernel(const unsigned* flag, unsigned v, float4* __restrict__ dst, const float4* src,
                                 size_t n4) {
    wait_for(flag, v);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = strata::kernels::ld_fresh_v(src + i);
}
__global__ void wait_flag_kernel(const unsigned* flag, unsigned v) { wait_for(flag, v); }
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
    __threadfence_system();
}
// (a few tokens' rows go through the kernels; a prompt chunk's through the copy engines, the flag after them)
constexpr size_t kViaKernel = 64 * 1024;
void publish(const float* src, float* dst, size_t n, unsigned* counter, unsigned* flag, unsigned v, cudaStream_t s) {
    if (n % 4) {
        std::fprintf(stderr, "maya-hy4 publish: %zu values (a multiple of 4 expected)\n", n);
        g_err.fetch_add(1);
        return;
    }
    if (n > kViaKernel) {
        if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess) {
            std::fprintf(stderr, "maya-hy4 publish: the copy failed\n");
            g_err.fetch_add(1);
            return;
        }
        set_flag_kernel<<<1, 1, 0, s>>>(flag, v);
        check("publish");
        return;
    }
    const size_t n4 = n / 4;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(16, (n4 + 255) / 256));
    publish_kernel<<<blocks, 256, 0, s>>>((const float4*) src, (float4*) dst, n4, counter, flag, v);
    check("publish");
}
void wait_pull(const unsigned* flag, unsigned v, float* dst, const float* src, size_t n, cudaStream_t s) {
    if (n % 4) {
        std::fprintf(stderr, "maya-hy4 wait_pull: %zu values (a multiple of 4 expected)\n", n);
        g_err.fetch_add(1);
        return;
    }
    if (n > kViaKernel) {
        wait_flag_kernel<<<1, 32, 0, s>>>(flag, v);
        if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) {
            std::fprintf(stderr, "maya-hy4 wait_pull: the copy failed\n");
            g_err.fetch_add(1);
        }
        check("wait_pull");
        return;
    }
    const size_t n4 = n / 4;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(8, (n4 + 255) / 256));
    wait_pull_kernel<<<blocks, 256, 0, s>>>(flag, v, (float4*) dst, (const float4*) src, n4);
    check("wait_pull");
}
void wait_flag(const unsigned* flag, unsigned v, cudaStream_t s) {
    wait_flag_kernel<<<1, 32, 0, s>>>(flag, v);
    check("wait_flag");
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) {
    set_flag_kernel<<<1, 1, 0, s>>>(flag, v);
    check("set_flag");
}
__global__ void pack3_kernel(const unsigned* a, size_t na, const unsigned* b, size_t nb, const unsigned* c, size_t nc,
                             unsigned* dst, int split) {
    const size_t n = na + nb + nc;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const unsigned* src = i < na ? a + i : i < na + nb ? b + (i - na) : c + (i - na - nb);
        if (!split) dst[i] = *src;
        else *(unsigned*) src = dst[i];
    }
}
void pack3(const void* a, size_t na, const void* b, size_t nb, const void* c, size_t nc, void* dst, cudaStream_t s) {
    const size_t n = na + nb + nc;
    pack3_kernel<<<(unsigned) std::max<size_t>(1, std::min<size_t>(1024, (n + 255) / 256)), 256, 0, s>>>(
        (const unsigned*) a, na, (const unsigned*) b, nb, (const unsigned*) c, nc, (unsigned*) dst, 0);
    check("pack3");
}
void split3(const void* src, void* a, size_t na, void* b, size_t nb, void* c, size_t nc, cudaStream_t s) {
    const size_t n = na + nb + nc;
    pack3_kernel<<<(unsigned) std::max<size_t>(1, std::min<size_t>(1024, (n + 255) / 256)), 256, 0, s>>>(
        (const unsigned*) a, na, (const unsigned*) b, nb, (const unsigned*) c, nc, (unsigned*) src, 1);
    check("split3");
}
__global__ void __launch_bounds__(256) pack16_kernel(const float* __restrict__ x, int E, uint16_t* __restrict__ h,
                                                     float* __restrict__ sc) {
    const int r = blockIdx.x;
    const float* xr = x + (size_t) r * E;
    float am = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) am = fmaxf(am, fabsf(xr[i]));
    __shared__ float s_am[8];
    am = warp_max(am);
    if ((threadIdx.x & 31) == 0) s_am[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = s_am[0];
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w) a = fmaxf(a, s_am[w]);
        float f = 1.0f;
        while (a / f > 32768.0f && f < 1e30f) f *= 2.0f;
        s_am[0] = f;
        sc[r] = f;
    }
    __syncthreads();
    const float inv = 1.0f / s_am[0];
    for (int i = threadIdx.x; i < E; i += blockDim.x) h[(size_t) r * E + i] = __half_as_ushort(__float2half_rn(xr[i] * inv));
}
__global__ void unpack16_kernel(const uint16_t* __restrict__ h, const float* __restrict__ sc, int E, size_t n,
                                float* __restrict__ y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__ushort_as_half(h[i])) * sc[i / (size_t) E];
}
void pack16_rows(const float* x, int rows, int E, uint16_t* h, float* sc, cudaStream_t s) {
    if (rows <= 0) return;
    pack16_kernel<<<(unsigned) rows, 256, 0, s>>>(x, E, h, sc);
    check("pack16_rows");
}
void unpack16_rows(const uint16_t* h, const float* sc, int rows, int E, float* y, cudaStream_t s) {
    const size_t n = (size_t) rows * E;
    if (n == 0) return;
    unpack16_kernel<<<(unsigned) std::min<size_t>(4096, (n + 255) / 256), 256, 0, s>>>(h, sc, E, n, y);
    check("unpack16_rows");
}
__global__ void expand_kernel(const float* __restrict__ e, int hc, int E, size_t n, float* __restrict__ x) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const size_t t = i / ((size_t) hc * E), d = i % (size_t) E;
        x[i] = e[t * (size_t) E + d];
    }
}
void expand_streams(const float* e, int T, int hc, int E, float* x, cudaStream_t s) {
    const size_t n = (size_t) T * hc * E;
    if (n == 0) return;
    expand_kernel<<<(unsigned) std::min<size_t>(4096, (n + 255) / 256), 256, 0, s>>>(e, hc, E, n, x);
    check("expand_streams");
}
unsigned wait_timeouts() {
    unsigned n = 0;
    if (cudaMemcpyFromSymbol(&n, g_wait_timeouts, sizeof n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
}

void top1(const float* row, int n, float* out, cudaStream_t s) {
    static const bool one = std::getenv("HY4_TOP1_ONE") != nullptr;   // (the earlier one-block kernel)
    if (one) {
        top1_kernel<<<1, 1024, 0, s>>>(row, n, out);
    } else {
        const int blocks = std::max(1, std::min(kTop1Blocks, (n + 4095) / 4096));
        top1_multi_kernel<<<blocks, 256, 0, s>>>(row, n, out);
    }
    check("top1");
}
int errors() { return g_err.exchange(0); }
void set_mix_out(float* p) { g_mix_out = p; }

}  // namespace hy4::k
