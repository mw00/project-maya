// src/ds4/ds4_kernels.cu - DeepSeek-V4.1's device kernels (see ds4_kernels.cuh).  No fast math here: the rotary
// angles, the softmaxes and the roundings follow the reference's F32 arithmetic.  A window's token is blockIdx.y.
#include "ds4_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ __forceinline__ uint16_t f2bf16(float f) {   // round to nearest even (the values here are exact anyway)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

__device__ __forceinline__ float fp8_rt(float v) {   // E4M3, nearest even (the input is already clamped)
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}

__device__ __forceinline__ float e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

// ---------------------------------------------------------------- embed
__global__ void embed_kernel(const uint16_t* __restrict__ rows, float* __restrict__ h, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    const float v = bf16f(rows[(size_t) t * dim + e]);
    for (int s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}

// ---------------------------------------------------------------- finish
constexpr int kFinMax = 32;   // values a thread holds: len <= 32 * 256

__global__ void __launch_bounds__(kThreads) finish_kernel(float* x, int len, long long stride, Finish f) {
    float* v = x + (long long) blockIdx.x * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = f.pos + (int) (blockIdx.x / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + (long long) blockIdx.x * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else v[e] = r[j];
    }
}

// ---------------------------------------------------------------- hyper-connections
__global__ void __launch_bounds__(kThreads) hc_mix_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                          int hcdim, float eps, float* __restrict__ mixes) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
}

// a warp a token: lane j * hc + k holds comb[j][k] (hc <= 4: lanes 0..15; wider streams fall back to lane 0 alone)
__global__ void hc_split_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                const float* __restrict__ base, int hc, int iters, float eps, float* pre, float* post,
                                float* comb) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* mx = mixes + (size_t) t * 32;
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + expf(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + expf(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    if (hc == 4) {
        // rows are the 4-lane groups (j = lane / 4), columns the lanes 4 apart (k = lane % 4)
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto col_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        c = on ? expf(c - m) : 0.0f;
        c = c / row_sum(c) + eps;                       // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);                     // each destination column over its sources
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float m = -FLT_MAX;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            m = fmaxf(m, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = expf(c[j][k] - m);
            s += c[j][k];
        }
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) s += c[j][k];
        for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
        }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}

__global__ void __launch_bounds__(1024) hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                                           const float* __restrict__ w, int dim, int hc, float eps,
                                                           float* __restrict__ x) {
    const int t = blockIdx.x;
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = rsqrtf(block_sum(ss) / (float) dim + eps);
    n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

__global__ void hc_post_kernel(const float* __restrict__ y, const float* __restrict__ res,
                               const float* __restrict__ post, const float* __restrict__ comb, int dim, int hc,
                               float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
    for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
__global__ void __launch_bounds__(kThreads) gemv_f16_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float alpha, float* __restrict__ y) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            acc += f.x * xv[c + 2 * i] + f.y * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}

// ---------------------------------------------------------------- the router
__global__ void __launch_bounds__(512) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int n, int k, float route_scale, int* idx, float* w,
                                                    const float* __restrict__ bias_vl, const VlBits vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl.w[t >> 5] >> (t & 31)) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float lim,
                              float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

// ---------------------------------------------------------------- the compressor (ratio > 1), a window in order
__global__ void __launch_bounds__(512) compress_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                                       float* kv_state, float* score_state, int ratio, int p0, int T,
                                                       int d, const float* __restrict__ w, float eps,
                                                       float* __restrict__ latent) {
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        __syncthreads();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) {
            float m = -FLT_MAX;
            for (int s = 0; s < ratio; ++s) m = fmaxf(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += expf(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (expf(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) d + eps);
        n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) latent[(size_t) g * d + e] = val[n] * inv * w[e];
        ++g;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- per-query sizes
__device__ __forceinline__ void rows_at(const Rows& r, int t, int& n, int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = (int) ((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn, k = min(r.kcap, nn); break;
    case 2: n = nn, k = cb; break;
    case 3: n = cb * r.blk, k = min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
    }
}

// ---------------------------------------------------------------- the indexer
// a thread a position: its key row (BF16) against every head's query (shared memory), relu, weighted sum
template <int HD>
__global__ void __launch_bounds__(kThreads) index_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                               const uint16_t* __restrict__ kc, int heads, int n,
                                                               const int* __restrict__ cand, int cand_stride, int blk,
                                                               Rows lim, float* __restrict__ out,
                                                               long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then heads weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    int p = i;
    if (cand != nullptr) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const uint16_t* kr = kc + (size_t) p * HD;
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        const uint4 raw = *(const uint4*) (kr + c);
        const uint16_t* b = (const uint16_t*) &raw;
        float kv[8];
#pragma unroll
        for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h >= heads) break;
            const float* qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties
__global__ void __launch_bounds__(1024) topk_kernel(const float* __restrict__ vals, long long stride, Rows r,
                                                    const int* __restrict__ cand, int cand_stride, int blk,
                                                    int* __restrict__ out, int out_stride) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int cnt_gt, cnt_eq;
    const int t = blockIdx.x;
    int n, k;
    rows_at(r, t, n, k);
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t u = ord_key(vals[i]);
        int slot = -1;
        if (u > thr) slot = atomicAdd(&cnt_gt, 1);
        else if (u == thr) {
            const int e = atomicAdd(&cnt_eq, 1);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
    }
}

__global__ void block_max_kernel(const float* __restrict__ score, long long stride, Rows r, int blk,
                                 float* __restrict__ bm, long long bm_stride) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    int n, k_;
    rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmaxf(m, score[(long long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- sparse attention, a block a (head, query)
constexpr int kAttTile = 64;

__global__ void __launch_bounds__(kThreads) sparse_attn_kernel(const float* __restrict__ q,
                                                               const uint16_t* __restrict__ win,
                                                               const uint16_t* __restrict__ ckv,
                                                               const int* __restrict__ sel, int sel_stride,
                                                               AttnRows ar, const float* __restrict__ sink, int heads,
                                                               int d, float scale, float* __restrict__ out) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    const int h = blockIdx.x, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);   // the window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    __shared__ float qs[512];
    __shared__ float sc[kAttTile];
    __shared__ const uint16_t* rows[kAttTile];
    __shared__ float m_run_s, tile_scale_s, den_s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[((size_t) t * heads + h) * d + i];
    if (threadIdx.x == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? wbase + (size_t) j * d : ckv + (size_t) sl[j - n_win] * d;
        }
        __syncthreads();
        for (int jj = wp; jj < nt; jj += kThreads / 32) {
            const uint16_t* rr = rows[jj];
            const uint4 a = *(const uint4*) (rr + lane * 16);
            const uint4 b = *(const uint4*) (rr + lane * 16 + 8);
            const uint16_t* pa = (const uint16_t*) &a;
            const uint16_t* pb = (const uint16_t*) &b;
            float s = 0.0f;
#pragma unroll
            for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
            for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
            s = warp_sum(s);
            if (lane == 0) sc[jj] = s * scale;
        }
        __syncthreads();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s = expf(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        __syncthreads();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = threadIdx.x * 2;
        for (int j = 0; j < nt; ++j) {
            const uint32_t pr = *(const uint32_t*) (rows[j] + c);
            o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
            o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
        }
        __syncthreads();
    }
    const float den = den_s + expf(sink[h] - m_run_s);
    const int c = threadIdx.x * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

__global__ void ring_to_ext_kernel(const uint16_t* __restrict__ ring, uint16_t* __restrict__ ext, long long p0, int n,
                                   int d) {
    const int r = blockIdx.x;
    const long long p = p0 - n + r;
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        ext[(size_t) r * d + e] = p >= 0 ? ring[(size_t) (p % n) * d + e] : (uint16_t) 0;
}

__global__ void ext_to_ring_kernel(const uint16_t* __restrict__ ext, uint16_t* __restrict__ ring, long long p0, int n,
                                   int T, int d, uint16_t* __restrict__ backup) {
    const int t = blockIdx.x;
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    uint16_t* row = ring + (size_t) ((p0 + t) % n) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) {
        if (backup != nullptr) backup[(size_t) t * d + e] = row[e];
        row[e] = ext[(size_t) (n + t) * d + e];
    }
}

__global__ void ring_restore_kernel(uint16_t* __restrict__ ring, const uint16_t* __restrict__ backup, long long p0,
                                    int n, int keep, int d) {
    const int t = keep + blockIdx.x;
    uint16_t* row = ring + (size_t) ((p0 + t) % n) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x) row[e] = backup[(size_t) t * d + e];
}

__global__ void hc_mean_kernel(const float* __restrict__ h, int hc, int dim, float* __restrict__ out,
                               long long out_stride) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * hc + s) * dim + e];
    out[(long long) t * out_stride + e] = a / (float) hc;
}

__global__ void iota_rows_kernel(int* out, int n, int stride) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[(size_t) t * stride + i] = i;
}

// a warp a vocabulary row: E[*tok] (shared memory) . H[v]
__global__ void __launch_bounds__(kThreads) markov_kernel(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                          const uint16_t* __restrict__ H, const int* __restrict__ tok,
                                                          int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

__global__ void __launch_bounds__(1024) confidence_kernel(const float* __restrict__ x, const uint16_t* __restrict__ E,
                                                          const int* __restrict__ tok, const float* __restrict__ w,
                                                          int dim, int R, float* __restrict__ out) {
    const int t = blockIdx.x;
    const float* xt = x + (size_t) t * dim;
    const uint16_t* er = E + (size_t) tok[t] * R;
    float a = 0.0f;
    for (int i = threadIdx.x; i < dim + R; i += blockDim.x) a += w[i] * (i < dim ? xt[i] : bf16f(er[i - dim]));
    a = block_sum(a);
    if (threadIdx.x == 0) out[t] = 1.0f / (1.0f + expf(-a));
}

__global__ void __launch_bounds__(1024) draft_probs_kernel(const float* __restrict__ logits,
                                                           const int* __restrict__ draft, int V, float temp,
                                                           float* __restrict__ p) {
    const int t = blockIdx.x;
    const float* z = logits + (size_t) t * V;
    __shared__ float red[32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, z[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        v = warp_max(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) s += expf((z[i] - m) / temp);
    s = block_sum(s);
    if (threadIdx.x == 0) p[t] = expf((z[draft[t]] - m) / temp) / s;
}

// ---------------------------------------------------------------- engram
__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ scales, int hd,
                                      float* __restrict__ out) {
    const int r = blockIdx.x;
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const __half_raw hv = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) rows[(size_t) r * hd + e], __NV_E4M3);
        const uint8_t sb = scales[(size_t) r * (hd / 32) + e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = __half2float(__half(hv)) * s;
    }
}

__global__ void __launch_bounds__(1024) engram_gate_kernel(float* __restrict__ h, const float* __restrict__ kv,
                                                           const float* __restrict__ w, int hc, int dim, float eps) {
    const int k = blockIdx.x, t = blockIdx.y;
    float* hk = h + ((size_t) t * hc + k) * dim;
    const float* kvt = kv + (size_t) t * (hc + 1) * dim;
    const float* key = kvt + (size_t) k * dim;
    const float* val = kvt + (size_t) hc * dim;
    const float* wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd = rsqrtf(sh / (float) dim + eps) * rsqrtf(sk / (float) dim + eps);
    const float dot = sd * rstd * rsqrtf((float) dim);
    const float sq = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + expf(-sq));
    for (int e = threadIdx.x; e < dim; e += blockDim.x) hk[e] = hk[e] + gate * val[e];
}

__global__ void gather_kernel(const float* __restrict__ x, Gather g, int cols, float* __restrict__ out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

__global__ void combine_kernel(const float* __restrict__ ed, const float* __restrict__ cpu,
                               const float* __restrict__ sh, Combine cm, int dim, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int j = 0; j < cm.n[t]; ++j) a += cm.w[t][j] * ed[(size_t) cm.row[t][j] * dim + e];
    if (cpu != nullptr && cm.cpu[t]) a += cpu[(size_t) t * dim + e];
    if (sh != nullptr) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

__global__ void __launch_bounds__(512) route_publish_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias, int n, int k,
                                                            float route_scale, const unsigned long long* __restrict__ vram,
                                                            const float* __restrict__ x, int dim, int* idx, float* w,
                                                            unsigned long long* plan_ptr, Req* req,
                                                            const float* __restrict__ bias_vl, unsigned vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    if (req != nullptr)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) req->x[t][i] = x[(size_t) t * dim + i];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
    }
}

__global__ void publish_kernel(Req* req, int T, unsigned seq) {
    __threadfence_system();
    req->T = T;
    __threadfence_system();
    *(volatile unsigned*) &req->seq = seq;
}

__global__ void wait_seq_kernel(const unsigned* seq_ptr, unsigned seq) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) seq_ptr != seq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

__global__ void init_pre_kernel(float* pre, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
}


// the hand-offs through host-mapped memory: a grid-stride copy of the segments (16 bytes a thread), then (put, flag
// given) the last block to finish sets the flag after a system fence; a get first waits for the flag
__device__ __forceinline__ void xfer_copy(const Xfer& x, bool from_host) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    for (size_t q = (size_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (size_t) gridDim.x * blockDim.x) {
        size_t r = q;
        int i = 0;
        while (r >= x.bytes[i] / 16) r -= x.bytes[i++] / 16;
        const uint4* src = (const uint4*) x.src[i] + r;
        ((uint4*) x.dst[i])[r] = from_host ? __ldcv(src) : *src;
    }
}

__global__ void __launch_bounds__(kThreads) xfer_put_kernel(const Xfer x, unsigned* flag, unsigned seq, unsigned* cnt) {
    xfer_copy(x, false);
    if (flag == nullptr) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *cnt = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}

__device__ __forceinline__ void wait_at_least(const unsigned* flag, unsigned seq) {
    while ((int) (*(volatile const unsigned*) flag - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
}

__global__ void __launch_bounds__(kThreads) xfer_get_kernel(const Xfer x, const unsigned* flag, unsigned seq) {
    if (threadIdx.x == 0) wait_at_least(flag, seq);
    __syncthreads();
    xfer_copy(x, true);
}

__global__ void wait_flags_kernel(const unsigned* flags, int n, int stride, unsigned seq) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) wait_at_least(flags + (size_t) i * stride, seq);
    __threadfence_system();
}
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
}

// x[i] = the members' partials summed in member order (this member's from x, the others' from their host rows)
__global__ void __launch_bounds__(kThreads) ar_sum_kernel(float* x, int n4, const float* peers, size_t stride,
                                                         const unsigned* flags, int nm, int mi, int slot, int slots,
                                                         unsigned seq) {
    if (threadIdx.x == 0)
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    __syncthreads();
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? ((const float4*) x)[q]
                                     : __ldcv((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        ((float4*) x)[q] = v;
    }
}


// a thread a (token, value): every stream's value read before any is written
__global__ void hc_post_inplace_kernel(const float* __restrict__ y, float* h, const float* __restrict__ post,
                                       const float* __restrict__ comb, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (int j = 0; j < hc; ++j) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (int k = 0; k < hc; ++k) {
        float a = post[(size_t) t * 8 + k] * yv;
        for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

__global__ void __launch_bounds__(kThreads) hc_mix_scale_kernel(const float* __restrict__ h, int hcdim, int n_mix,
                                                                float eps, float* mixes) {
    const int t = blockIdx.x;
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) hcdim + eps);
    if (threadIdx.x < n_mix) mixes[(size_t) t * 32 + threadIdx.x] *= inv;
}

__global__ void engram_dequant_tok_kernel(const uint8_t* __restrict__ src, size_t per, int cols, int hd,
                                          float* __restrict__ out) {
    const int r = blockIdx.x, t = r / cols, c = r - t * cols;
    const uint8_t* rows = src + (size_t) t * per + (size_t) c * hd;
    const uint8_t* scales = src + (size_t) t * per + (size_t) cols * hd + (size_t) c * (hd / 32);
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const __half_raw hv = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) rows[e], __NV_E4M3);
        const uint8_t sb = scales[e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = __half2float(__half(hv)) * s;
    }
}

// a block a virtual slot: its picks counted (sort pass 1), or (fill) placed in token order by a block-wide scan
template <bool FILL>
__global__ void __launch_bounds__(kThreads) moe_sort_kernel(const int* __restrict__ idx, int T, int k,
                                                            const int* __restrict__ vs, int* bounds, int* row_tok,
                                                            int* pos) {
    const int slot = blockIdx.x;
    __shared__ int warp_n[kThreads / 32];
    __shared__ int base_s;
    if (threadIdx.x == 0) base_s = FILL ? bounds[slot] : 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int n = T * k;
    for (int q0 = 0; q0 < n; q0 += kThreads) {
        const int q = q0 + threadIdx.x;
        const int t = q / k, i = q - t * k;
        const bool hit = q < n && vs[idx[(size_t) t * kPicks + i]] == slot;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) warp_n[wp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wp) before += warp_n[w];
            total += warp_n[w];
        }
        if (FILL && hit) {
            const int r = base_s + before + __popc(m & ((1u << lane) - 1u));
            row_tok[r] = t;
            pos[(size_t) t * kPicks + i] = r;
        }
        __syncthreads();
        if (threadIdx.x == 0) base_s += total;
        __syncthreads();
    }
    if (!FILL && threadIdx.x == 0) bounds[slot + 1] = base_s;   // (the counts; the scan makes them bounds)
}

__global__ void bounds_scan_kernel(int* bounds, int n_vs) {   // one thread: bounds[0] = 0, bounds[s + 1] += bounds[s]
    if (threadIdx.x != 0) return;
    bounds[0] = 0;
    for (int s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

__global__ void moe_combine_kernel(float* acc, const float* __restrict__ y, const int* __restrict__ pos,
                                   const float* __restrict__ w, int k, int r0, int r1, int dim) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        if (r < r0 || r >= r1) continue;
        a = __fmaf_rn(w[(size_t) t * kPicks + i], y[(size_t) (r - r0) * dim + e], a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl,
                   unsigned vl) {
    route_publish_kernel<<<T, 512, 0, s>>>(logits, bias, n, k, route_scale, vram, x, dim, idx, w, plan_ptr, req,
                                           vl != 0 ? bias_vl : nullptr, vl);
    if (req != nullptr) publish_kernel<<<1, 1, 0, s>>>(req, T, seq);
}
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s) { wait_seq_kernel<<<1, 32, 0, s>>>(seq_ptr, seq); }
void init_pre(float* pre, int T, cudaStream_t s) { init_pre_kernel<<<cdiv(T * 8, 128), 128, 0, s>>>(pre, T); }

void embed(const uint16_t* rows, int T, float* h, int dim, int hc, cudaStream_t s) {
    embed_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(rows, h, dim, hc);
}

void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s) {
    if (len > kFinMax * kThreads) {
        std::fprintf(stderr, "ds4 finish: a vector of %d values is more than %d\n", len, kFinMax * kThreads);
        return;
    }
    if (n_vec <= 0) return;
    finish_kernel<<<n_vec, kThreads, 0, s>>>(x, len, stride, f);
}

void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
    hc_mix_kernel<<<dim3(n_mix, T), kThreads, 0, s>>>(h, fn, hcdim, eps, mixes);
}
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s) {
    hc_split_kernel<<<T, 32, 0, s>>>(mixes, scale, base, hc, iters, hc_eps, pre, post, comb);
}
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s) {
    hc_pre_norm_kernel<<<T, 1024, 0, s>>>(h, mix, w, dim, hc, eps, x);
}
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s) {
    hc_post_kernel<<<dim3(cdiv(dim * hc, kThreads), T), kThreads, 0, s>>>(y, res, post, comb, dim, hc, out);
}

void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s) {
    hc_post_inplace_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(y, h, post, comb, dim, hc);
}
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
    hc_mix_scale_kernel<<<T, kThreads, 0, s>>>(h, hcdim, n_mix, eps, mixes);
}
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s) {
    if (T > 0) engram_dequant_tok_kernel<<<T * cols, kThreads, 0, s>>>(src, per, cols, hd, out);
}
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s) {
    moe_sort_kernel<false><<<n_vs, kThreads, 0, s>>>(idx, T, k, vs, bounds, row_tok, pos);
    bounds_scan_kernel<<<1, 32, 0, s>>>(bounds, n_vs);
    moe_sort_kernel<true><<<n_vs, kThreads, 0, s>>>(idx, T, k, vs, bounds, row_tok, pos);
}
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s) {
    if (r1 > r0 && T > 0) moe_combine_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(acc, y, pos, w, k, r0, r1, dim);
}

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s) {
    gemv_f16_kernel<<<dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s>>>(W, x, n, k, alpha, y);
}

void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl, const VlBits* vl) {
    static const VlBits none = {};
    route_kernel<<<T, 512, 0, s>>>(logits, bias, n, k, route_scale, idx, w, vl != nullptr ? bias_vl : nullptr,
                                   vl != nullptr ? *vl : none);
}

void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s) {
    if (n > 0) swiglu_kernel<<<cdiv(n, kThreads), kThreads, 0, s>>>(g, u, n, lim, h);
}

void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s) {
    compress_kernel<<<1, 512, 0, s>>>(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent);
}

void index_score(const float* q, const float* w, const uint16_t* k_bf16, int heads, int hd, int T, int n,
                 const int* cand, int cand_stride, int blk, const Rows& lim, float* out, long long out_stride,
                 cudaStream_t s) {
    if (hd != 128 || heads > 32) {
        std::fprintf(stderr, "ds4 index_score: %d heads of %d not covered\n", heads, hd);
        return;
    }
    if (n <= 0) return;
    const size_t smem = (size_t) (heads * hd + heads) * sizeof(float);
    index_score_kernel<128><<<dim3(cdiv(n, kThreads), T), kThreads, smem, s>>>(q, w, k_bf16, heads, n, cand,
                                                                                cand_stride, blk, lim, out,
                                                                                out_stride);
}

void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s) {
    topk_kernel<<<T, 1024, 0, s>>>(vals, stride, r, cand, cand_stride, blk, out, out_stride);
}

void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s) {
    int nb_max = 1;
    Rows rb = r;
    if (r.gen != 0) {   // (a chunk: the widest query's blocks, made on the device per query)
        rb.gen = 4;
        rb.blk = blk;
        nb_max = cdiv((int) ((r.p0 + T) / r.R), blk);
    } else
        for (int t = 0; t < T; ++t) {
            rb.n[t] = cdiv(r.n[t], blk);
            rb.k[t] = r.k[t];
            nb_max = rb.n[t] > nb_max ? rb.n[t] : nb_max;
        }
    block_max_kernel<<<dim3(cdiv(nb_max, kThreads), T), kThreads, 0, s>>>(score, stride, r, blk, bm, bm_stride);
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s);
}

void sparse_attn(const float* q, const uint16_t* win, const uint16_t* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 cudaStream_t s) {
    if (d != 512) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d not covered\n", d);
        return;
    }
    sparse_attn_kernel<<<dim3(heads, T), kThreads, 0, s>>>(q, win, ckv, sel, sel_stride, a, sink, heads, d, scale, out);
}

void ring_to_ext(const uint16_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s) {
    ring_to_ext_kernel<<<n, kThreads, 0, s>>>(ring, ext, (long long) p0, n, d);
}
void ext_to_ring(const uint16_t* ext, uint16_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s,
                 uint16_t* backup) {
    ext_to_ring_kernel<<<T, kThreads, 0, s>>>(ext, ring, (long long) p0, n, T, d, backup);
}
void ring_restore(uint16_t* ring, const uint16_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
    if (T > keep) ring_restore_kernel<<<T - keep, kThreads, 0, s>>>(ring, backup, (long long) p0, n, keep, d);
}
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s) {
    hc_mean_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(h, hc, dim, out, out_stride);
}
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s) { iota_rows_kernel<<<T, 64, 0, s>>>(out, n, stride); }
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s) {
    markov_kernel<<<cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s>>>(logits, E, H, tok, V, R);
}
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s) {
    confidence_kernel<<<T, 1024, 0, s>>>(x, E, tok, w, dim, R, out);
}
void draft_probs(const float* logits, const int* draft, int n, int V, float temp, float* p, cudaStream_t s) {
    if (n > 0) draft_probs_kernel<<<n, 1024, 0, s>>>(logits, draft, V, temp, p);
}

void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s) {
    engram_dequant_kernel<<<n, kThreads, 0, s>>>(rows, scales, hd, out);
}

void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s) {
    engram_gate_kernel<<<dim3(hc, T), 1024, 0, s>>>(h, kv, w, hc, dim, eps);
}

void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s) {
    if (n > 0) gather_kernel<<<dim3(cdiv(cols, kThreads), n), kThreads, 0, s>>>(x, g, cols, out);
}

void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s) {
    combine_kernel<<<dim3(cdiv(dim, kThreads), T), kThreads, 0, s>>>(ed, cpu, sh, c, dim, out);
}

void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    xfer_put_kernel<<<blocks, kThreads, 0, s>>>(x, flag, seq, cnt);
}
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    xfer_get_kernel<<<blocks, kThreads, 0, s>>>(x, flag, seq);
}
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s) {
    if (n > 0) wait_flags_kernel<<<1, 32, 0, s>>>(flags, n, stride, seq);
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) { set_flag_kernel<<<1, 1, 0, s>>>(flag, v); }
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s) {
    const int slot = (int) (seq % (unsigned) a.slots);
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    ar_sum_kernel<<<blocks, kThreads, 0, s>>>(x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots, seq);
}

}  // namespace strata::ds4
