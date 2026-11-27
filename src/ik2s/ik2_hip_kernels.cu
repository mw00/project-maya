// src/ik2s/ik2_hip_kernels.cu - maya-ik2s's own kernels (ik2_hip_kernels.cuh).  Written for clarity first
// (the decode path's small vectors: one launch a step of the reference graph); the sums are F32 in a fixed order.
#include "ik2_hip_kernels.cuh"

#include <cuda_fp16.h>

#include "maya/kernels/fresh_load.hpp"
#include "maya/kernels/rdna_wmma.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ik2hip::k {

namespace {

std::atomic<int> g_err{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "maya-ik2s kernel %s: %s\n", what, cudaGetErrorString(e));
        g_err.fetch_add(1, std::memory_order_relaxed);
    }
}

// a block's sum of v over its threads (blockDim.x <= 256, a power of two); every thread gets it
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + o]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// ---- the 8-bit cache (--kv q8, the global layers): a slot's row is kv8_row_bytes(Hkv, D) bytes, the Hkv heads' D
// int8 values side by side, then their Hkv F32 scales.  A value x is stored from its F16 rounding x16 (the F16 cache's
// value): the scale of its (slot, head) row sc = max |x16| / 127, q = rint(x16 / sc) in -127 .. 127; it is read back as
// F16(q sc), so every kernel downstream of the read sees F16 values as it does with the F16 cache (IK2_HIP_KV_SIM=1
// rounds the F16 cache's rows the same way: the same bits) ---------------------------------------------------------
// a block's (slot, head) row from the threads' values x (thread i < D: value i; every thread of the block calls it)
__device__ void kv8_put(void* cache, size_t slot, int Hkv, int D, int h, float x, float* sh) {
    const int i = threadIdx.x;
    const float x16 = i < D ? __half2float(__float2half_rn(x)) : 0.0f;
    const float sc = block_max(fabsf(x16), sh) / 127.0f;
    uint8_t* row = (uint8_t*) cache + slot * kv8_row_bytes(Hkv, D);
    if (i < D) ((int8_t*) row)[(size_t) h * D + i] = (int8_t) (sc > 0.0f ? fminf(fmaxf(rintf(x16 / sc), -127.0f), 127.0f) : 0.0f);
    if (i == 0) ((float*) (row + (size_t) Hkv * D))[h] = sc;
}
__device__ __forceinline__ uint32_t kv8_h2(int a, int b, float sc) {
    const __half2 v = __floats2half2_rn((float) a * sc, (float) b * sc);
    return *(const uint32_t*) &v;
}
// values c .. c + 7 of head g's row in slot (c a multiple of 8) as eight F16s
__device__ __forceinline__ uint4 kv8_get8(const void* cache, size_t slot, int Hkv, int D, int g, int c) {
    const uint8_t* row = (const uint8_t*) cache + slot * kv8_row_bytes(Hkv, D);
    const uint2 b = *(const uint2*) (row + (size_t) g * D + c);
    const float sc = ((const float*) (row + (size_t) Hkv * D))[g];
    auto q = [](uint32_t w, int k) { return (int) (int8_t) (uint8_t) (w >> (8 * k)); };
    return make_uint4(kv8_h2(q(b.x, 0), q(b.x, 1), sc), kv8_h2(q(b.x, 2), q(b.x, 3), sc), kv8_h2(q(b.y, 0), q(b.y, 1), sc),
                      kv8_h2(q(b.y, 2), q(b.y, 3), sc));
}
// values c .. c + 3 (c a multiple of 4) as two F16 pairs
__device__ __forceinline__ uint2 kv8_get4(const void* cache, size_t slot, int Hkv, int D, int g, int c) {
    const uint8_t* row = (const uint8_t*) cache + slot * kv8_row_bytes(Hkv, D);
    const uint32_t b = *(const uint32_t*) (row + (size_t) g * D + c);
    const float sc = ((const float*) (row + (size_t) Hkv * D))[g];
    auto q = [](uint32_t w, int k) { return (int) (int8_t) (uint8_t) (w >> (8 * k)); };
    return make_uint2(kv8_h2(q(b, 0), q(b, 1), sc), kv8_h2(q(b, 2), q(b, 3), sc));
}

// block s = a state: stat[4 s] = its 1 / rms, stat[4 s + 1 + q] = (state / rms) . query q
__global__ void mix_stat_kernel(const float* __restrict__ blocks, int nb, const float* __restrict__ cur, int E,
                                const float* __restrict__ query, int nq, float eps, float* __restrict__ stat) {
    __shared__ float sh[256];
    const int st = blockIdx.x;
    const float* x = st < nb ? blocks + (size_t) st * E : cur;
    float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
        for (int q = 0; q < nq; ++q) d[q] += v * query[(size_t) q * E + i];
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) E + eps);
    for (int q = 0; q < nq; ++q) {
        const float dq = block_sum(d[q], sh);
        if (threadIdx.x == 0) stat[4 * st + 1 + q] = dq * inv;
    }
    if (threadIdx.x == 0) stat[4 * st] = inv;
}
__global__ void mix_out_kernel(const float* __restrict__ blocks, int nb, const float* __restrict__ cur, int E, int nq,
                               const float* __restrict__ stat, float* __restrict__ out) {
    const int q = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int ns = nb + 1;
    const float sc = rsqrtf((float) E);
    float m = -INFINITY;
    for (int st = 0; st < ns; ++st) m = fmaxf(m, stat[4 * st + 1 + q] * sc);
    float den = 0.0f, v = 0.0f;
    for (int st = 0; st < ns; ++st) {
        const float w = expf(stat[4 * st + 1 + q] * sc - m);
        den += w;
        v += w * stat[4 * st] * (st < nb ? blocks[(size_t) st * E + i] : cur[i]);
    }
    out[(size_t) q * E + i] = v / den;
}

__global__ void rms_kernel(const float* x, size_t ldx, int n, const float* __restrict__ w, float eps,
                           float* y, size_t ldy) {
    __shared__ float sh[256];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * (w ? w[i] : 1.0f);
}

__global__ void conv_kernel(float* __restrict__ y, float* __restrict__ hist, const float* __restrict__ w, int C,
                            const float* __restrict__ dyn, int heads, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
    float o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
    if (dyn) {
        const float* d = dyn + 4 * (c / (C / heads));
        o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
    }
    y[c] = (v + o) * scale;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = v;
}

__global__ void head_clip_kernel(float* __restrict__ x, int n, const float* __restrict__ scale, float thr, float mul) {
    __shared__ float sh[256];
    const int i = threadIdx.x;
    float* xr = x + (size_t) blockIdx.x * n;
    float v = 0.0f;
    if (i < n) v = xr[i] * (scale ? scale[(size_t) blockIdx.x * n + i] : 1.0f);
    const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
    const float f = fminf(1.0f, thr / norm);
    if (i < n) xr[i] = v * f * mul;
}

// thread = (head, key): its score
__global__ void attn_score_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                  const float* __restrict__ sink, int H, int Hkv, int D, int n_rel,
                                  const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K, int ring,
                                  long long p, long long lo, int nk, float scale, float* __restrict__ S) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) H * nk) return;
    const int h = (int) (idx / (size_t) nk), j = (int) (idx % (size_t) nk);
    const long long pos = lo + j;
    const size_t slot = (size_t) (ring > 0 ? pos % ring : pos);
    const __half* kr = K + (slot * Hkv + (size_t) (h / (H / Hkv))) * D;
    const float* qr = q + (size_t) h * D;
    float a = 0.0f;
    for (int i = 0; i < D; ++i) a += qr[i] * __half2float(kr[i]);
    a *= scale;
    const long long d = p - pos;
    if (d < extent) {
        const float* b = rel_bias + (size_t) d * n_rel;
        const float* r = rel + (size_t) h * n_rel;
        float bs = 0.0f;
        for (int i = 0; i < n_rel; ++i) bs += r[i] * b[i];
        a += bs;
    }
    S[idx] = a - sink[h];
}
// block = a head: its scores -> the probabilities with the zero sink logit in the sum
__global__ void attn_softmax_kernel(float* __restrict__ S, int nk) {
    __shared__ float sh[256];
    float* sr = S + (size_t) blockIdx.x * nk;
    float m = 0.0f;   // (the sink's logit)
    for (int j = threadIdx.x; j < nk; j += blockDim.x) m = fmaxf(m, sr[j]);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        const float e = expf(sr[j] - m);
        sr[j] = e;
        sum += e;
    }
    sum = block_sum(sum, sh) + expf(-m);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) sr[j] *= inv;
}
// block = (a tile of kTile keys, a kv head): the tile's K rows and the kv head's queries in shared memory, then a
// thread a (query head, key) - its score (a thread reading its own K row from the cache took most of a decode step's
// attention time)
constexpr int kTile = 32, kSeg = 128;   // (a tile's shared memory against blocks: measurable with IK2_HIP_PHASES)
__global__ void attn_score_tile_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                       const float* __restrict__ sink, int H, int Hkv, int D, int n_rel,
                                       const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K, int ring,
                                       long long p, long long lo, int nk, float scale, float* __restrict__ S) {
    extern __shared__ float sm[];   // [kTile][D] keys, then [per][D] queries
    const int g = blockIdx.y, j0 = blockIdx.x * kTile, per = H / Hkv;
    float* sk = sm;
    float* sq = sm + (size_t) kTile * D;
    for (int i = threadIdx.x; i < kTile * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {
            const long long pos = lo + j;
            sk[i] = __half2float(K[((size_t) (ring > 0 ? pos % ring : pos) * Hkv + g) * D + i % D]);
        }
    }
    for (int i = threadIdx.x; i < per * D; i += blockDim.x) sq[i] = q[(size_t) g * per * D + i];
    __syncthreads();
    for (int e = threadIdx.x; e < per * kTile; e += blockDim.x) {
        const int r = e / kTile, jj = e % kTile, j = j0 + jj, h = g * per + r;
        if (j >= nk) continue;
        float a = 0.0f;
        for (int i = 0; i < D; ++i) a += sq[r * D + i] * sk[jj * D + i];
        a *= scale;
        const long long d = p - (lo + j);
        if (d < extent) {
            const float* bb = rel_bias + (size_t) d * n_rel;
            const float* rr = rel + (size_t) h * n_rel;
            float bs = 0.0f;
            for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
            a += bs;
        }
        S[(size_t) h * nk + j] = a - sink[h];
    }
}
// block = (a segment of kSeg keys, a kv head): part = the segment's sum of P_j V_j for each of the kv head's query
// heads, a V row read once for them - then the segments' sum
__global__ void attn_part_kernel(const float* __restrict__ S, int H, int Hkv, int D, const __half* __restrict__ V, int ring,
                                 long long lo, int nk, int nseg, float* __restrict__ part) {
    const int g = blockIdx.y, seg = blockIdx.x, i = threadIdx.x, per = H / Hkv;
    if (i >= D || per > 8) return;
    const size_t col = (size_t) g * D + i, row = (size_t) Hkv * D;
    const int j1 = min(nk, (seg + 1) * kSeg);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = seg * kSeg; j < j1; ++j) {
        const long long pos = lo + j;
        const float x = __half2float(V[(size_t) (ring > 0 ? pos % ring : pos) * row + col]);
        for (int r = 0; r < per; ++r) v[r] += S[(size_t) (g * per + r) * nk + j] * x;
    }
    for (int r = 0; r < per; ++r) part[((size_t) (g * per + r) * nseg + seg) * D + i] = v[r];
}
__global__ void attn_sum_kernel(const float* __restrict__ part, int D, int nseg, float* __restrict__ out) {
    const int h = blockIdx.x, i = threadIdx.x;
    if (i >= D) return;
    float v = 0.0f;
    for (int seg = 0; seg < nseg; ++seg) v += part[((size_t) h * nseg + seg) * D + i];
    out[(size_t) h * D + i] = v;
}

__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ g, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}
__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void situ_kernel(const float* __restrict__ gu, int ff, float beta, float lb, float* __restrict__ h) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff) return;
    const float g = gu[(size_t) r * 2 * ff + i];
    float u = gu[(size_t) r * 2 * ff + ff + i];
    if (lb > 0.0f) u = lb * tanhf(u / lb);
    h[(size_t) r * ff + i] = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g))) * u;
}
__global__ void wsum_kernel(const float* __restrict__ part, int m, int n, const float* __restrict__ w,
                            float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < m; ++j)
        if (w[j] != 0.0f) v = fmaf(w[j], part[(size_t) j * n + i], v);   // (0: not this side's entry)
    out[i] = v;
}
__global__ void softcap_kernel(float* __restrict__ x, size_t n, float cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
__global__ void argmax_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    __shared__ float bv[256];
    __shared__ int bi[256];
    float v = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) {
            v = x[i];
            at = i;
        }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = at;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            const float v2 = bv[threadIdx.x + o];
            const int i2 = bi[threadIdx.x + o];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bi[threadIdx.x])) {
                bv[threadIdx.x] = v2;
                bi[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = (float) bi[0];
        out[1] = bv[0];
    }
}

// block = a kv head, 1024 threads: its query heads' softmax (the zero sink logit in the sum), then the output - a
// thread a (segment of the keys, value dim), the segments' sums added in shared memory.  One launch for what were
// three (the softmax, the segments' parts, their sum); the scores stay the tiles' launch, which needs its many blocks
// (a single block a kv head that also computed the scores measured twice the four launches' time).
__global__ void attn_soft_out_kernel(float* __restrict__ S, int H, int Hkv, int D, const __half* __restrict__ V, int ring,
                                     long long lo, int nk, float* __restrict__ out) {
    __shared__ float vr[6144];   // [per][threads]: the heads' reductions side by side (a tree for all, not one a head)
    __shared__ float ps[6144];   // [per][segments][D]
    const int g = blockIdx.x, per = H / Hkv, tid = threadIdx.x, B = blockDim.x;
    for (int r = 0; r < per; ++r) {
        const float* sr = S + (size_t) (g * per + r) * nk;
        float m = 0.0f;   // (the sink's logit)
        for (int j = tid; j < nk; j += B) m = fmaxf(m, sr[j]);
        vr[r * B + tid] = m;
    }
    __syncthreads();
    for (int o = B / 2; o > 0; o >>= 1) {
        if (tid < o)
            for (int r = 0; r < per; ++r) vr[r * B + tid] = fmaxf(vr[r * B + tid], vr[r * B + tid + o]);
        __syncthreads();
    }
    float mx[8];
    for (int r = 0; r < per; ++r) mx[r] = vr[r * B];
    __syncthreads();
    for (int r = 0; r < per; ++r) {
        float* sr = S + (size_t) (g * per + r) * nk;
        float sum = 0.0f;
        for (int j = tid; j < nk; j += B) {
            const float e = expf(sr[j] - mx[r]);
            sr[j] = e;
            sum += e;
        }
        vr[r * B + tid] = sum;
    }
    __syncthreads();
    for (int o = B / 2; o > 0; o >>= 1) {
        if (tid < o)
            for (int r = 0; r < per; ++r) vr[r * B + tid] += vr[r * B + tid + o];
        __syncthreads();
    }
    float inv[8];
    for (int r = 0; r < per; ++r) inv[r] = 1.0f / (vr[r * B] + expf(-mx[r]));
    __syncthreads();
    // (the probabilities stay unnormalised in S: the output takes the factor)
    const int nseg = blockDim.x / D, seg = tid / D, i = tid % D, len = (nk + nseg - 1) / nseg;
    const size_t col = (size_t) g * D + i, row = (size_t) Hkv * D;
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int j1 = min(nk, (seg + 1) * len);
    for (int j = seg * len; j < j1; ++j) {
        const long long pos = lo + j;
        const float x = __half2float(V[(size_t) (ring > 0 ? pos % ring : pos) * row + col]);
        for (int r = 0; r < per; ++r) v[r] += S[(size_t) (g * per + r) * nk + j] * x;
    }
    for (int r = 0; r < per; ++r) ps[(r * nseg + seg) * D + i] = v[r] * inv[r];
    __syncthreads();
    if (seg == 0)
        for (int r = 0; r < per; ++r) {
            float o = 0.0f;
            for (int q = 0; q < nseg; ++q) o += ps[(r * nseg + q) * D + i];
            out[(size_t) (g * per + r) * D + i] = o;
        }
}
// Split-K attention.  block = (a tile of TK keys, a kv head): the tile's K rows and the head's queries in shared
// memory, a thread a (query head, key) for the scores, each query head's softmax within the tile (its own max), the
// tile's part of the output - st [H][tiles][2] = (the tile's max, its sum of exp(s - max)), part [H][tiles][D]
__global__ void attn_tile_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ sink,
                                 int H, int Hkv, int D, int n_rel, const float* __restrict__ rel_bias, int extent,
                                 const __half* __restrict__ K, const __half* __restrict__ V, int ring, long long base,
                                 int d0, int nk, int TK, float scale, float* __restrict__ st,
                                 float* __restrict__ part) {
    extern __shared__ float sm[];   // [TK][D] keys, [per][D] queries, [per][TK] scores / weights, [per] maxima
    const int g = blockIdx.y, tile = blockIdx.x, ntile = gridDim.x, j0 = tile * TK, per = H / Hkv, tid = threadIdx.x;
    float* sk = sm;
    float* sq = sk + (size_t) TK * D;
    float* sw = sq + (size_t) per * D;
    float* smb = sw + (size_t) per * TK;
    for (int i = tid; i < TK * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {   // (key j's row: base + j, wrapped once - no 64-bit modulo a value, which cost most of the launch)
            long long sl = base + j;
            if (ring > 0 && sl >= ring) sl -= ring;
            sk[i] = __half2float(K[((size_t) sl * Hkv + g) * D + i % D]);
        }
    }
    for (int i = tid; i < per * D; i += blockDim.x) sq[i] = q[(size_t) g * per * D + i];
    __syncthreads();
    for (int e = tid; e < per * TK; e += blockDim.x) {
        const int r = e / TK, jj = e % TK, j = j0 + jj, h = g * per + r;
        float a = -INFINITY;
        if (j < nk) {
            a = 0.0f;
            for (int i = 0; i < D; ++i) a += sq[r * D + i] * sk[jj * D + i];
            a *= scale;
            const int d = d0 - j;   // (the query's position less the key's)
            if (d < extent) {
                const float* bb = rel_bias + (size_t) d * n_rel;
                const float* rr = rel + (size_t) h * n_rel;
                float bs = 0.0f;
                for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
                a += bs;
            }
            a -= sink[h];
        }
        sw[e] = a;
    }
    __syncthreads();
    if (tid < per) {
        float m = -INFINITY;
        for (int jj = 0; jj < TK; ++jj) m = fmaxf(m, sw[tid * TK + jj]);
        smb[tid] = m;
    }
    __syncthreads();
    for (int e = tid; e < per * TK; e += blockDim.x) sw[e] = j0 + e % TK < nk ? expf(sw[e] - smb[e / TK]) : 0.0f;
    __syncthreads();
    if (tid < per) {
        float sum = 0.0f;
        for (int jj = 0; jj < TK; ++jj) sum += sw[tid * TK + jj];
        float* o = st + ((size_t) (g * per + tid) * ntile + tile) * 2;
        o[0] = smb[tid];
        o[1] = sum;
    }
    if (tid < D && per <= 8) {
        const size_t col = (size_t) g * D + tid, row = (size_t) Hkv * D;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const int n = min(TK, nk - j0);
        for (int jj = 0; jj < n; ++jj) {
            long long sl = base + j0 + jj;
            if (ring > 0 && sl >= ring) sl -= ring;
            const float x = __half2float(V[(size_t) sl * row + col]);
            for (int r = 0; r < per; ++r) v[r] += sw[r * TK + jj] * x;
        }
        for (int r = 0; r < per; ++r) part[((size_t) (g * per + r) * ntile + tile) * D + tid] = v[r];
    }
}
// ... for a window's T rows: block = (a tile of TK keys, a kv head) - the tile's K rows in shared memory once for the
// T x per queries (read where they lie), each query's softmax within the tile over the keys its row sees, then the
// tile's V rows in the same shared memory for the parts.  st [T H][tiles][2], part [T H][tiles][D]
constexpr int kATK = 16;      // keys a tile of the window attention (split_tk's measured best below, 16)
// kAU: keys a thread of the scores takes (the query's values read once for its kAU dots; each dot summed in the
// dimensions' order whatever kAU is, so the instances give the same bits).  Measured, ms of attention a pass at 2.7k
// context for T = 1 / 2 / 4 / 8 / 9 rows: 1 key 3.8 / 5.8 / 9.7 / 18.0 / 20.0, 2 keys 3.4 / 4.0 / 5.6 / 10.0 / 11.4,
// 4 keys 3.2 / 3.6 / 4.3 / 7.0 / 7.7, 8 keys 3.9 / 4.0 / 4.3 / 5.8 / 6.4 - 4 up to four rows, 8 from five.
template <int kAU>
__global__ void attn_tile_rows_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                      const float* __restrict__ sink, int T, int H, int Hkv, int D, int n_rel,
                                      const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K,
                                      const __half* __restrict__ V, int ring, long long base, int d0, int window, int nk,
                                      int TK, float scale, float* __restrict__ st, float* __restrict__ part) {
    extern __shared__ float sm[];   // [TK][D] keys (then values), [T per][TK] scores / weights, [T per] maxima
    const int g = blockIdx.y, tile = blockIdx.x, ntile = gridDim.x, j0 = tile * TK, per = H / Hkv, tid = threadIdx.x;
    const int TP = T * per;
    float* sk = sm;
    float* sw = sk + (size_t) TK * D;
    float* smb = sw + (size_t) TP * TK;
    for (int i = tid; i < TK * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {
            long long sl = base + j;
            if (ring > 0 && sl >= ring) sl -= ring;
            sk[i] = __half2float(K[((size_t) sl * Hkv + g) * D + i % D]);
        }
    }
    __syncthreads();
    // (a thread: a query and kAU of the tile's keys - the query's values read once for the kAU dots, each dot summed in
    //  the dimensions' order)
    for (int e = tid; e < TP * (TK / kAU); e += blockDim.x) {
        const int tr = e / (TK / kAU), jg = (e % (TK / kAU)) * kAU, t = tr / per, h = g * per + tr % per;
        const float* qr = q + ((size_t) t * H + h) * D;
        const float* kr = sk + (size_t) jg * D;
        float acc[kAU];
        for (int u = 0; u < kAU; ++u) acc[u] = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float qv = qr[i];
            for (int u = 0; u < kAU; ++u) acc[u] += qv * kr[u * D + i];
        }
        for (int u = 0; u < kAU; ++u) {
            const int j = j0 + jg + u;
            const int d = d0 + t - j;   // (row t's position less the key's)
            float a = -INFINITY;
            if (j < nk && d >= 0 && (window == 0 || d < window)) {
                a = acc[u] * scale;
                if (d < extent) {
                    const float* bb = rel_bias + (size_t) d * n_rel;
                    const float* rr = rel + ((size_t) t * H + h) * n_rel;
                    float bs = 0.0f;
                    for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
                    a += bs;
                }
                a -= sink[(size_t) t * H + h];
            }
            sw[tr * TK + jg + u] = a;
        }
    }
    __syncthreads();
    if (tid < TP) {
        float m = -INFINITY;
        for (int jj = 0; jj < TK; ++jj) m = fmaxf(m, sw[tid * TK + jj]);
        smb[tid] = m;
    }
    __syncthreads();
    for (int e = tid; e < TP * TK; e += blockDim.x) sw[e] = sw[e] > -INFINITY ? expf(sw[e] - smb[e / TK]) : 0.0f;
    // (the keys are done with: the tile's values in their place)
    for (int i = tid; i < TK * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {
            long long sl = base + j;
            if (ring > 0 && sl >= ring) sl -= ring;
            sk[i] = __half2float(V[((size_t) sl * Hkv + g) * D + i % D]);
        }
    }
    __syncthreads();
    if (tid < TP) {
        float sum = 0.0f;
        for (int jj = 0; jj < TK; ++jj) sum += sw[tid * TK + jj];
        float* o = st + (((size_t) (tid / per) * H + g * per + tid % per) * ntile + tile) * 2;
        o[0] = smb[tid];
        o[1] = sum;
    }
    if (tid < D) {   // (the tile's values of this dimension in registers; the keys past the last: zeros)
        const int n = min(TK, nk - j0);
        float vc[kATK];
        for (int jj = 0; jj < kATK; ++jj) vc[jj] = jj < n && jj < TK ? sk[jj * D + tid] : 0.0f;
        for (int tr = 0; tr < TP; ++tr) {
            float v = 0.0f;
            for (int jj = 0; jj < kATK; ++jj) v += sw[tr * TK + jj] * vc[jj];
            part[(((size_t) (tr / per) * H + g * per + tr % per) * ntile + tile) * D + tid] = v;
        }
    }
}
// block = a query head: the tiles' parts under one maximum (the sink's zero logit in it and in the sum), the gate
__global__ void attn_combine_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                    const float* __restrict__ gate, float* __restrict__ out) {
    __shared__ float sh[256];
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* sth = st + (size_t) h * ntile * 2;
    float m = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) m = fmaxf(m, sth[2 * b]);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) sum += sth[2 * b + 1] * expf(sth[2 * b] - m);
    sum = block_sum(sum, sh) + expf(-m);
    if (tid >= D) return;
    float v = 0.0f;
    for (int b = 0; b < ntile; ++b) v += part[((size_t) h * ntile + b) * D + tid] * expf(sth[2 * b] - m);
    v /= sum;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * D + tid]));
    out[(size_t) h * D + tid] = v;
}
// ... the window step's: the tiles summed in their order by every thread (no tree over the threads), so a row's
// result does not depend on how many tiles the launch has before or after the row's own keys - empty tiles add zeros
__device__ __forceinline__ float fz_wave_max2(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__global__ void attn_combine_seq_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                        const float* __restrict__ gate, float* __restrict__ out,
                                        int8_t* __restrict__ q8 = nullptr, float* __restrict__ d8 = nullptr) {
    extern __shared__ float sm[];   // [256] the maximum's tree, [ntile] each tile's factor (an exp a tile, not a thread)
    float* sh = sm;
    float* se = sm + 256;
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* sth = st + (size_t) h * ntile * 2;
    float m = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) m = fmaxf(m, sth[2 * b]);
    m = block_max(m, sh);
    for (int b = tid; b < ntile; b += blockDim.x) se[b] = expf(sth[2 * b] - m);
    __syncthreads();
    if (tid >= D) return;
    float v = 0.0f, sum = 0.0f;
    for (int b = 0; b < ntile; ++b) {
        v += part[((size_t) h * ntile + b) * D + tid] * se[b];
        sum += sth[2 * b + 1] * se[b];
    }
    sum += expf(-m);
    v /= sum;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * D + tid]));
    out[(size_t) h * D + tid] = v;
    if (q8) {   // (q8_kernel's int8 rows of out, a row's heads side by side: D a multiple of 32, whole waves live)
        const float am = fz_wave_max2(fabsf(v));
        const float scq = am / 127.0f;
        const size_t i = (size_t) h * D + tid;
        q8[i] = (int8_t) (am > 0.0f ? __float2int_rn(v / scq) : 0);
        if ((tid & 31) == 0) d8[i / 32] = scq;
    }
}
// ... the same sums for more tiles than a block's shared memory holds factors for (contexts past kCombTiles * kATK
// keys: 131072): the tiles' factors a chunk of kCombTiles at a time, summed in the same order
constexpr int kCombTiles = 8192;   // (8192 floats = 32 KB of a block's shared memory)
__global__ void attn_combine_seq_big_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                            const float* __restrict__ gate, float* __restrict__ out,
                                            int8_t* __restrict__ q8 = nullptr, float* __restrict__ d8 = nullptr) {
    __shared__ float sh[256];
    __shared__ float se[kCombTiles];
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* sth = st + (size_t) h * ntile * 2;
    float m = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) m = fmaxf(m, sth[2 * b]);
    m = block_max(m, sh);
    float v = 0.0f, sum = 0.0f;
    for (int b0 = 0; b0 < ntile; b0 += kCombTiles) {
        const int nb = ntile - b0 < kCombTiles ? ntile - b0 : kCombTiles;
        for (int b = tid; b < nb; b += blockDim.x) se[b] = expf(sth[2 * (b0 + b)] - m);
        __syncthreads();
        if (tid < D)
            for (int b = 0; b < nb; ++b) {
                v += part[((size_t) h * ntile + b0 + b) * D + tid] * se[b];
                sum += sth[2 * (b0 + b) + 1] * se[b];
            }
        __syncthreads();
    }
    if (tid >= D) return;
    sum += expf(-m);
    v /= sum;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * D + tid]));
    out[(size_t) h * D + tid] = v;
    if (q8) {   // (q8_kernel's int8 rows of out, a row's heads side by side: D a multiple of 32, whole waves live)
        const float am = fz_wave_max2(fabsf(v));
        const float scq = am / 127.0f;
        const size_t i = (size_t) h * D + tid;
        q8[i] = (int8_t) (am > 0.0f ? __float2int_rn(v / scq) : 0);
        if ((tid & 31) == 0) d8[i / 32] = scq;
    }
}
// ---- decode attention in chunks (own runtime, 2026-11-24): the same results as attention_split_rows (within float
// rounding: its tiles are 16 keys, these kACK), with every thread of a block busy. A block = (a chunk of kACK keys, a kv
// head): the chunk's keys read once (F16, rows padded so the score threads don't share banks) for all T x (H / Hkv)
// query rows, the rows' queries in shared memory. Each (row, key) dot is summed in the dimensions' order and each
// row's chunk maximum, sum and partial values in the keys' order, at chunk boundaries that are multiples of kACK
// positions whatever the first key is: a row's result does not depend on T (the token step and a verify window give the
// same bits). The chunks' parts go through the same combine as attention_split_rows.
template <int CK, int RU, bool Q8 = false>   // (Q8: the 8-bit cache's rows, read as F16 into the same LDS)
__global__ void attn_chunk_rows_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                       const float* __restrict__ sink, int T, int H, int Hkv, int D, int n_rel,
                                       const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K,
                                       const __half* __restrict__ V, int ring, long long base, int d0, int window, int nk,
                                       float scale, int rpb, float* __restrict__ st, float* __restrict__ part) {
    extern __shared__ float sm[];   // [TP][D] queries, [TP][CK] scores / weights, [TP] maxima, [CK][D + 2] F16 keys / values
    const int g = blockIdx.y, tile = blockIdx.x, ntile = gridDim.x, j0 = tile * CK, per = H / Hkv, tid = threadIdx.x;
    // (blockIdx.z: a slice of rpb of the T x per query rows - fewer rows a block, more blocks a CU)
    const int rz = blockIdx.z * rpb, TP = min(rpb, T * per - rz), nt = blockDim.x, DP = D + 2;
    float* sq = sm;
    float* sw = sq + (size_t) TP * D;
    float* smx = sw + (size_t) TP * CK;
    __half* skv = (__half*) (smx + ((TP + 3) & ~3));
    for (int i = tid; i < TP * D; i += nt) {
        const int tr = rz + i / D, t = tr / per, h = g * per + tr % per;
        sq[i] = q[((size_t) t * H + h) * D + i % D];
    }
    const int n = min(CK, nk - j0);
    if (Q8) {   // (four values a thread: D a multiple of 4)
        for (int i = tid; i < CK * (D / 4); i += nt) {
            const int jj = i / (D / 4), c = (i % (D / 4)) * 4;
            uint2 v = make_uint2(0u, 0u);
            if (jj < n) {
                long long sl = base + j0 + jj;
                if (ring > 0 && sl >= ring) sl -= ring;
                v = kv8_get4(K, (size_t) sl, Hkv, D, g, c);
            }
            *(uint32_t*) (skv + (size_t) jj * DP + c) = v.x;
            *(uint32_t*) (skv + (size_t) jj * DP + c + 2) = v.y;
        }
    } else
    for (int i = tid; i < CK * (D / 2); i += nt) {
        const int jj = i / (D / 2), c = (i % (D / 2)) * 2;
        __half2 v = __floats2half2_rn(0.0f, 0.0f);
        if (jj < n) {
            long long sl = base + j0 + jj;
            if (ring > 0 && sl >= ring) sl -= ring;
            v = *(const __half2*) (K + ((size_t) sl * Hkv + g) * D + c);
        }
        *(__half2*) (skv + (size_t) jj * DP + c) = v;
    }
    __syncthreads();
    // (a thread: a key and RU consecutive rows - the key's values read once for the RU dots, each dot summed in the
    //  dimensions' order whatever RU is, so the instances give the same bits)
    for (int e = tid; e < (TP / RU) * CK; e += nt) {
        const int r0 = (e / CK) * RU, jj = e % CK, j = j0 + jj;
        const __half* kr = skv + (size_t) jj * DP;
        float acc[RU];
        for (int r = 0; r < RU; ++r) acc[r] = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float kv = __half2float(kr[i]);
            for (int r = 0; r < RU; ++r) acc[r] += sq[(size_t) (r0 + r) * D + i] * kv;
        }
        for (int r = 0; r < RU; ++r) {
            const int tl = r0 + r, tr = rz + tl, t = tr / per, h = g * per + tr % per, d = d0 + t - j;
            float a = -INFINITY;
            if (jj < n && d >= 0 && (window == 0 || d < window)) {
                a = acc[r] * scale;
                if (d < extent) {
                    const float* bb = rel_bias + (size_t) d * n_rel;
                    const float* rr = rel + ((size_t) t * H + h) * n_rel;
                    float bs = 0.0f;
                    for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
                    a += bs;
                }
                a -= sink[(size_t) t * H + h];
            }
            sw[(size_t) tl * CK + jj] = a;
        }
    }
    __syncthreads();
    if (tid < TP) {
        float m = -INFINITY;
        for (int jj = 0; jj < CK; ++jj) m = fmaxf(m, sw[(size_t) tid * CK + jj]);
        smx[tid] = m;
    }
    __syncthreads();
    for (int e = tid; e < TP * CK; e += nt) sw[e] = sw[e] > -INFINITY ? expf(sw[e] - smx[e / CK]) : 0.0f;
    if (Q8) {
        for (int i = tid; i < CK * (D / 4); i += nt) {
            const int jj = i / (D / 4), c = (i % (D / 4)) * 4;
            uint2 v = make_uint2(0u, 0u);
            if (jj < n) {
                long long sl = base + j0 + jj;
                if (ring > 0 && sl >= ring) sl -= ring;
                v = kv8_get4(V, (size_t) sl, Hkv, D, g, c);
            }
            *(uint32_t*) (skv + (size_t) jj * DP + c) = v.x;
            *(uint32_t*) (skv + (size_t) jj * DP + c + 2) = v.y;
        }
    } else
    for (int i = tid; i < CK * (D / 2); i += nt) {   // (the keys are done with: the chunk's values in their place)
        const int jj = i / (D / 2), c = (i % (D / 2)) * 2;
        __half2 v = __floats2half2_rn(0.0f, 0.0f);
        if (jj < n) {
            long long sl = base + j0 + jj;
            if (ring > 0 && sl >= ring) sl -= ring;
            v = *(const __half2*) (V + ((size_t) sl * Hkv + g) * D + c);
        }
        *(__half2*) (skv + (size_t) jj * DP + c) = v;
    }
    __syncthreads();
    if (tid < TP) {
        float sum = 0.0f;
        for (int jj = 0; jj < CK; ++jj) sum += sw[(size_t) tid * CK + jj];
        const int tr = rz + tid;
        float* o = st + (((size_t) (tr / per) * H + g * per + tr % per) * ntile + tile) * 2;
        o[0] = smx[tid];
        o[1] = sum;
    }
    for (int e = tid; e < (TP / RU) * D; e += nt) {   // (a thread: a dimension and RU rows, the value read once)
        const int r0 = (e / D) * RU, dd = e % D;
        float v[RU];
        for (int r = 0; r < RU; ++r) v[r] = 0.0f;
        for (int jj = 0; jj < CK; ++jj) {
            const float x = __half2float(skv[(size_t) jj * DP + dd]);
            for (int r = 0; r < RU; ++r) v[r] += sw[(size_t) (r0 + r) * CK + jj] * x;
        }
        for (int r = 0; r < RU; ++r) {
            const int tr = rz + r0 + r;
            part[(((size_t) (tr / per) * H + g * per + tr % per) * ntile + tile) * D + dd] = v[r];
        }
    }
}
// ---- the window step's fused forms: the decode step's launches over T rows (T 1: a token) --------------------------
// block = (a kv head, K or V, a row): the row's conv from the window's inputs and the histories before it, the clip /
// the norm, the F16 cache row; the last row's blocks write the histories after the window into h1
__global__ void kv_finish_rows_kernel(const float* __restrict__ in_k, const float* __restrict__ in_v,
                                      const float* __restrict__ hk0, const float* __restrict__ hv0,
                                      float* __restrict__ hk1, float* __restrict__ hv1, const float* __restrict__ k_conv,
                                      const float* __restrict__ v_conv, const float* __restrict__ dk,
                                      const float* __restrict__ dv, int T, int Hkv, int D, float thr, float eps,
                                      __half* __restrict__ K, __half* __restrict__ V, int ring, long long p0, int kvq) {
    __shared__ float sh[256];
    const int h = blockIdx.x, isv = blockIdx.y, t = blockIdx.z, i = threadIdx.x, C = Hkv * D, c = h * D + i;
    const float* y = isv ? in_v : in_k;
    const float* hist = isv ? hv0 : hk0;
    float* hnew = isv ? hv1 : hk1;
    const float* w = isv ? v_conv : k_conv;
    const float* dyn = isv ? dv : dk;
    float o = 0.0f;
    if (i < D) {
        float in[4];   // (in[k]: the input at t - 3 + k)
        for (int k = 0; k < 4; ++k) {
            const int tt = t - 3 + k;
            in[k] = tt >= 0 ? y[(size_t) tt * C + c] : hist[(size_t) (3 + tt) * C + c];
        }
        const float h0 = in[0], h1 = in[1], h2 = in[2], v = in[3];
        o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
        if (dyn) {
            const float* d = dyn + ((size_t) t * Hkv + h) * 4;
            o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        o = (v + o) * (dyn ? 0.333333343f : 0.44721359f);   // (1 / sqrt(9), 1 / sqrt(5))
        if (t == T - 1)
            for (int r = 0; r < 3; ++r) {
                const int q = T + r;
                hnew[(size_t) r * C + c] = q < 3 ? hist[(size_t) q * C + c] : y[(size_t) (q - 3) * C + c];
            }
    }
    const float ss = block_sum(o * o, sh);
    const float f = isv ? rsqrtf(ss / (float) D + eps) : fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
    if (kvq) {
        const long long p = p0 + t;
        kv8_put(isv ? V : K, (size_t) (ring > 0 ? p % ring : p), Hkv, D, h, o * f, sh);
    } else if (i < D) {
        const long long p = p0 + t;
        (isv ? V : K)[(size_t) (ring > 0 ? p % ring : p) * C + c] = __float2half_rn(o * f);
    }
}
struct RowMul {
    float m[16];
};
__global__ void clip2_rows_kernel(float* __restrict__ q, int H, int D, const float* __restrict__ qs, float qthr,
                                  float* __restrict__ rel, int n_rel, const float* __restrict__ rs, float rthr, RowMul mul) {
    __shared__ float sh[256];
    const int i = threadIdx.x, h = blockIdx.x, t = blockIdx.y;
    float* qr = q + ((size_t) t * H + h) * D;
    float* rr = rel + ((size_t) t * H + h) * n_rel;
    const float v = i < D ? qr[i] * qs[(size_t) h * D + i] : 0.0f;
    const float r = i < n_rel ? rr[i] * rs[(size_t) h * n_rel + i] : 0.0f;
    const float fq = fminf(1.0f, qthr / (sqrtf(block_sum(v * v, sh)) + 1e-6f));
    const float fr = fminf(1.0f, rthr / (sqrtf(block_sum(r * r, sh)) + 1e-6f));
    if (i < D) qr[i] = v * fq * mul.m[t];
    if (i < n_rel) rr[i] = r * fr * mul.m[t];
}
// block = (channels, a row): the row's conv (no taps) from the window's inputs y and the histories h0, then cur += it
// - or, with blk, blk = cur and cur = it (a block's first layer: the stream saved, a new one); the last row's blocks
// write the histories after the window into h1
__global__ void conv_add_rows_kernel(const float* __restrict__ y, const float* __restrict__ h0p, float* __restrict__ h1p,
                                     const float* __restrict__ w, int C, int T, float* __restrict__ cur,
                                     float* __restrict__ blk) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float in[4];
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        in[k] = tt >= 0 ? y[(size_t) tt * C + c] : h0p[(size_t) (3 + tt) * C + c];
    }
    const float h0 = in[0], h1 = in[1], h2 = in[2], v = in[3];
    const float o = (v + w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v) * 0.44721359f;
    float* cr = cur + (size_t) t * C + c;
    if (blk) {
        blk[(size_t) t * C + c] = *cr;
        *cr = o;
    } else {
        *cr += o;
    }
    if (t == T - 1)
        for (int r = 0; r < 3; ++r) {
            const int q = T + r;
            h1p[(size_t) r * C + c] = q < 3 ? h0p[(size_t) q * C + c] : y[(size_t) (q - 3) * C + c];
        }
}
// a window's taps for the drafter: dst [T][n][E] (rows ld apart) from n sources [T][E] each (src.p[k]; host memory the
// device maps where src.vol bit k is set - another GPU wrote it before this GPU's hand-off came)
struct TapSrc {
    const float* p[8];
    unsigned vol;
};
__global__ void taps_gather_kernel(TapSrc src, int n, int T, int E, float* __restrict__ dst, size_t ld) {
    const size_t total = (size_t) T * n * E;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
        const int e = (int) (i % (size_t) E), k = (int) (i / (size_t) E % (size_t) n), t = (int) (i / ((size_t) E * n));
        const float* sp = src.p[k] + (size_t) t * E + e;
        dst[(size_t) t * ld + (size_t) k * E + e] = (src.vol >> k) & 1u ? *(const volatile float*) sp : *sp;
    }
}
// a request's logit bias (Model::set_logit_bias): entry i of row blockIdx.y; the ids are distinct
__global__ void bias_rows_kernel(float* __restrict__ x, size_t ld, const int* __restrict__ ids,
                                 const float* __restrict__ vals, int n) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[(size_t) blockIdx.y * ld + (size_t) ids[i]] += vals[i];
}
// each row's argmax: out[2 t] the index as a float (ties: the lower), out[2 t + 1] its value; rows ld apart
__global__ void argmax_rows_kernel(const float* __restrict__ x, size_t ld, int n, float* __restrict__ out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* xr = x + (size_t) blockIdx.x * ld;
    float best = -INFINITY;
    int at = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (xr[i] > best) {
            best = xr[i];
            at = i;
        }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = at;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            const float o = bv[threadIdx.x + st];
            const int oi = bi[threadIdx.x + st];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[2 * blockIdx.x] = (float) bi[0];
        out[2 * blockIdx.x + 1] = bv[0];
    }
}
// one block: the states' norms and dots, then each query's mix, its own norm and gain
__global__ void mix_rms_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur, int T,
                               int E,
                               const float* __restrict__ query, int nq, const float* __restrict__ g0,
                               const float* __restrict__ g1, const float* __restrict__ g2, float eps,
                               float* __restrict__ out) {
    __shared__ float sh[1024];
    __shared__ float st[kMaxStates * 4];
    const int ns = nb + 1, tid = threadIdx.x, t = blockIdx.x;   // (a block a token: each the one-token arithmetic)
    const size_t to = (size_t) t * E;
    for (int s = 0; s < ns; ++s) {
        const float* x = (s < nb ? blocks + (size_t) s * ldb : cur) + to;
        float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
        for (int i = tid; i < E; i += blockDim.x) {
            const float v = x[i];
            ss += v * v;
            for (int qi = 0; qi < nq; ++qi) d[qi] += v * query[(size_t) qi * E + i];
        }
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float) E + 1e-6f);
        for (int qi = 0; qi < nq; ++qi) {
            const float dq = block_sum(d[qi], sh);
            if (tid == 0) st[4 * s + 1 + qi] = dq * inv;
        }
        if (tid == 0) st[4 * s] = inv;
        __syncthreads();
    }
    const float sc = rsqrtf((float) E);
    for (int qi = 0; qi < nq; ++qi) {
        const float* gain = qi == 0 ? g0 : qi == 1 ? g1 : g2;
        float m = -INFINITY, w[kMaxStates], den = 0.0f;
        for (int s = 0; s < ns; ++s) m = fmaxf(m, st[4 * s + 1 + qi] * sc);
        for (int s = 0; s < ns; ++s) {
            w[s] = expf(st[4 * s + 1 + qi] * sc - m);
            den += w[s];
        }
        float vv[8], ss = 0.0f;
        int n = 0;
        for (int i = tid; i < E && n < 8; i += blockDim.x, ++n) {
            float v = 0.0f;
            for (int s = 0; s < ns; ++s) v += w[s] * st[4 * s] * (s < nb ? blocks[(size_t) s * ldb + to + i] : cur[to + i]);
            vv[n] = v / den;
            ss += vv[n] * vv[n];
        }
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float) E + eps);
        n = 0;
        for (int i = tid; i < E && n < 8; i += blockDim.x, ++n) out[((size_t) qi * T + t) * E + i] = vv[n] * inv * (gain ? gain[i] : 1.0f);
    }
}
// ---- mix + norms with few barriers (own runtime, 2026-11-24): mix_rms_kernel's results (within float rounding) with the
// sums of four states at a time in one pass over the states and one reduction (wave shuffles, then the waves' partials
// in shared memory), and the queries' norms in one reduction - instead of a tree with barriers for every sum
template <int NV>
__device__ __forceinline__ void block_sum_v(float (&v)[NV], float* sh) {   // (sh: [32 waves][NV] + [NV]; every thread gets the sums)
#pragma unroll
    for (int o = warpSize / 2; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < NV; ++i) v[i] += __shfl_xor(v[i], o);
    const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < NV; ++i) sh[w * NV + i] = v[i];
    __syncthreads();
    if ((int) threadIdx.x < NV) {
        float s = 0.0f;
        for (int k2 = 0; k2 < nw; ++k2) s += sh[k2 * NV + threadIdx.x];
        sh[32 * NV + threadIdx.x] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = sh[32 * NV + i];
    __syncthreads();
}
__global__ void mix_rms2_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur, int T,
                                int E, const float* __restrict__ query, int nq, const float* __restrict__ g0,
                                const float* __restrict__ g1, const float* __restrict__ g2, float eps,
                                float* __restrict__ out) {
    __shared__ float sh[33 * 16];
    __shared__ float st[kMaxStates * 4];
    const int ns = nb + 1, tid = threadIdx.x, t = blockIdx.x;   // (a block a token: each the one-token arithmetic)
    const size_t to = (size_t) t * E;
    for (int s0 = 0; s0 < ns; s0 += 4) {   // (four states a pass: their squares and their dots with the queries)
        float a[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) a[i] = 0.0f;
        for (int i = tid; i < E; i += blockDim.x) {
            float qv[3];
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) qv[qi] = qi < nq ? query[(size_t) qi * E + i] : 0.0f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = s0 + u;
                if (s < ns) {
                    const float v = (s < nb ? blocks + (size_t) s * ldb : cur)[to + i];
                    a[4 * u] += v * v;
#pragma unroll
                    for (int qi = 0; qi < 3; ++qi) a[4 * u + 1 + qi] += v * qv[qi];
                }
            }
        }
        block_sum_v<16>(a, sh);
        if (tid == 0)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = s0 + u;
                if (s < ns) {
                    const float inv = rsqrtf(a[4 * u] / (float) E + 1e-6f);
                    st[4 * s] = inv;
                    for (int qi = 0; qi < nq; ++qi) st[4 * s + 1 + qi] = a[4 * u + 1 + qi] * inv;
                }
            }
    }
    __syncthreads();
    const float sc = rsqrtf((float) E);
    float vv[3][8], ss[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) continue;
        float m = -INFINITY, w[kMaxStates], den = 0.0f;
        for (int s = 0; s < ns; ++s) m = fmaxf(m, st[4 * s + 1 + qi] * sc);
        for (int s = 0; s < ns; ++s) {
            w[s] = expf(st[4 * s + 1 + qi] * sc - m);
            den += w[s];
        }
        int n = 0;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            const int i = tid + k2 * blockDim.x;
            vv[qi][k2] = 0.0f;
            if (i >= E) continue;
            float v = 0.0f;
            for (int s = 0; s < ns; ++s) v += w[s] * st[4 * s] * (s < nb ? blocks[(size_t) s * ldb + to + i] : cur[to + i]);
            vv[qi][k2] = v / den;
            ss[qi] += vv[qi][k2] * vv[qi][k2];
            ++n;
        }
    }
    block_sum_v<3>(ss, sh);
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) continue;
        const float* gain = qi == 0 ? g0 : qi == 1 ? g1 : g2;
        const float inv = rsqrtf(ss[qi] / (float) E + eps);
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            const int i = tid + k2 * blockDim.x;
            if (i < E) out[((size_t) qi * T + t) * E + i] = vv[qi][k2] * inv * (gain ? gain[i] : 1.0f);
        }
    }
}
__global__ void hist_set4_kernel(float* h0p, float* h1p, float* h2p, float* h3p, const float* s0, const float* s1,
                                 const float* s2, const float* s3, const float* i0, const float* i1, const float* i2,
                                 const float* i3, int C0, int C1, int C2, int C3, int keep) {
    float* const hist[4] = {h0p, h1p, h2p, h3p};
    const float* const h0[4] = {s0, s1, s2, s3};
    const float* const in[4] = {i0, i1, i2, i3};
    const int C[4] = {C0, C1, C2, C3};
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    for (int q = 0; q < 4; ++q) {
        if (c >= C[q]) continue;
        for (int r = 0; r < 3; ++r) {
            const int i = keep + r;
            hist[q][(size_t) r * C[q] + c] = i < 3 ? h0[q][(size_t) i * C[q] + c] : in[q][(size_t) (i - 3) * C[q] + c];
        }
    }
}
__global__ void lane_add_kernel(const unsigned* flag, unsigned v, const int* __restrict__ ids, const int* __restrict__ slot,
                                int nu, const float* src, float* __restrict__ y, int n) {
    __shared__ int need;
    if (threadIdx.x == 0) {
        int nd = 0;
        for (int j = 0; j < nu; ++j) nd |= slot[ids[j]] < 0;
        if (nd) {
            const unsigned long long t0 = maya::kernels::clock_ns();
            while ((int) (*(volatile const unsigned*) flag - v) < 0)
                if (maya::kernels::clock_ns() - t0 > 60ull * 1000000000ull) {   // (a minute: the lane is gone)
                    nd = 0;
                    break;
                }
        }
        need = nd;
    }
    __syncthreads();
    __threadfence_system();
    if (!need) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += *(const volatile float*) (src + i);
}

// ---- the decode step's fused forms ----------------------------------------------------------------------------------
// block = (a kv head, K or V): its D channels' conv, then the clip / the norm, the F16 row
__global__ void kv_finish_kernel(float* __restrict__ ky, float* __restrict__ vy, float* __restrict__ hist_k,
                                 float* __restrict__ hist_v, const float* __restrict__ k_conv,
                                 const float* __restrict__ v_conv, const float* __restrict__ dk,
                                 const float* __restrict__ dv, int Hkv, int D, float thr, float eps,
                                 __half* __restrict__ K, __half* __restrict__ V, size_t row, int kvq) {
    __shared__ float sh[256];
    const int h = blockIdx.x, isv = blockIdx.y, i = threadIdx.x, C = Hkv * D, c = h * D + i;
    float* y = isv ? vy : ky;
    float* hist = isv ? hist_v : hist_k;
    const float* w = isv ? v_conv : k_conv;
    const float* dyn = isv ? dv : dk;
    float o = 0.0f;
    if (i < D) {
        const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
        o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
        if (dyn) {
            const float* d = dyn + 4 * h;
            o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        o = (v + o) * (dyn ? 0.333333343f : 0.44721359f);   // (1 / sqrt(9), 1 / sqrt(5))
        hist[c] = h1;
        hist[(size_t) C + c] = h2;
        hist[(size_t) 2 * C + c] = v;
    }
    const float ss = block_sum(o * o, sh);
    const float f = isv ? rsqrtf(ss / (float) D + eps) : fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
    if (i < D) {
        o *= f;
        y[c] = o;
        if (!kvq) (isv ? V : K)[row * C + c] = __float2half_rn(o);
    }
    if (kvq) kv8_put(isv ? V : K, row, Hkv, D, h, o, sh);
}
__global__ void clip2_kernel(float* __restrict__ q, int D, const float* __restrict__ qs, float qthr,
                             float* __restrict__ rel, int n_rel, const float* __restrict__ rs, float rthr, float mul) {
    __shared__ float sh[256];
    const int i = threadIdx.x, h = blockIdx.x;
    float* qr = q + (size_t) h * D;
    float* rr = rel + (size_t) h * n_rel;
    const float v = i < D ? qr[i] * qs[(size_t) h * D + i] : 0.0f;
    const float r = i < n_rel ? rr[i] * rs[(size_t) h * n_rel + i] : 0.0f;
    const float fq = fminf(1.0f, qthr / (sqrtf(block_sum(v * v, sh)) + 1e-6f));
    const float fr = fminf(1.0f, rthr / (sqrtf(block_sum(r * r, sh)) + 1e-6f));
    if (i < D) qr[i] = v * fq * mul;
    if (i < n_rel) rr[i] = r * fr * mul;
}
__global__ void conv_add_kernel(float* __restrict__ y, float* __restrict__ hist, const float* __restrict__ w, int C,
                                float* __restrict__ cur) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
    const float o = (v + w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v) * 0.44721359f;
    y[c] = o;
    cur[c] += o;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = v;
}

// ---- the prompt path ------------------------------------------------------------------------------------------------
__global__ void mix_stat_rows_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur,
                                     int E, const float* __restrict__ query, int nq, float eps, float* __restrict__ stat) {
    __shared__ float sh[256];
    const int st = blockIdx.x, t = blockIdx.y, ns = nb + 1;
    const float* x = (st < nb ? blocks + (size_t) st * ldb : cur) + (size_t) t * E;
    float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
        for (int q = 0; q < nq; ++q) d[q] += v * query[(size_t) q * E + i];
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) E + eps);
    float* o = stat + ((size_t) t * ns + st) * 4;
    for (int q = 0; q < nq; ++q) {
        const float dq = block_sum(d[q], sh);
        if (threadIdx.x == 0) o[1 + q] = dq * inv;
    }
    if (threadIdx.x == 0) o[0] = inv;
}
__global__ void mix_out_rows_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur,
                                    int T, int E, int nq, const float* __restrict__ stat, float* __restrict__ out) {
    const int q = blockIdx.z, t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int ns = nb + 1;
    const float* stt = stat + (size_t) t * ns * 4;
    const float sc = rsqrtf((float) E);
    float m = -INFINITY;
    for (int st = 0; st < ns; ++st) m = fmaxf(m, stt[4 * st + 1 + q] * sc);
    float den = 0.0f, v = 0.0f;
    for (int st = 0; st < ns; ++st) {
        const float w = expf(stt[4 * st + 1 + q] * sc - m);
        den += w;
        v += w * stt[4 * st] * (st < nb ? blocks[(size_t) st * ldb + (size_t) t * E + i] : cur[(size_t) t * E + i]);
    }
    out[((size_t) q * T + t) * E + i] = v / den;
}
__global__ void conv_rows_kernel(const float* __restrict__ y, const float* __restrict__ hist, const float* __restrict__ w,
                                 int C, const float* __restrict__ dyn, int heads, float scale, float* __restrict__ out) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float in[4];   // (in[k]: the input at t - 3 + k)
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        in[k] = tt >= 0 ? y[(size_t) tt * C + c] : hist[(size_t) (3 + tt) * C + c];
    }
    float o = w[4 * c] * in[0] + w[4 * c + 1] * in[1] + w[4 * c + 2] * in[2] + w[4 * c + 3] * in[3];
    if (dyn) {
        const float* d = dyn + ((size_t) t * heads + (size_t) (c / (C / heads))) * 4;
        o += d[0] * in[3] + d[1] * in[2] + d[2] * in[1] + d[3] * in[0];
    }
    out[(size_t) t * C + c] = (in[3] + o) * scale;
}
__global__ void conv_hist_kernel(const float* __restrict__ y, int T, float* hist, int C, float* __restrict__ save) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    if (save)
        for (int r = 0; r < 3; ++r) save[(size_t) r * C + c] = hist[(size_t) r * C + c];
    for (int r = 0; r < 3; ++r) {   // (the row T + r of hist's three, then the chunk's)
        const int i = T + r;
        hist[(size_t) r * C + c] = i < 3 ? hist[(size_t) i * C + c] : y[(size_t) (i - 3) * C + c];
    }
}
__global__ void head_clip_rows_kernel(float* __restrict__ x, int H, int n, const float* __restrict__ scale, float thr,
                                      float alpha, float inv_floor, long long p0) {
    __shared__ float sh[256];
    const int i = threadIdx.x, h = blockIdx.x % H, t = blockIdx.x / H;
    float* xr = x + (size_t) blockIdx.x * n;
    float v = 0.0f;
    if (i < n) v = xr[i] * (scale ? scale[(size_t) h * n + i] : 1.0f);
    const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
    const float f = fminf(1.0f, thr / norm);
    const float mul = alpha != 0.0f ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (p0 + t + 1) * inv_floor)) : 1.0f;
    if (i < n) xr[i] = v * f * mul;
}
// (--kv q8) a block a (row, head): store_rows' values as kv8_put stores them
__global__ void store_rows8_kernel(const float* __restrict__ src, int Hkv, int D, void* cache, int ring, long long p0) {
    __shared__ float sh[256];
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const long long p = p0 + t;
    kv8_put(cache, (size_t) (ring > 0 ? p % ring : p), Hkv, D, h, i < D ? src[((size_t) t * Hkv + h) * D + i] : 0.0f, sh);
}
__global__ void store_rows_kernel(const float* __restrict__ src, int n, __half* __restrict__ cache, int ring, long long p0) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long p = p0 + t;
    cache[(size_t) (ring > 0 ? p % ring : p) * n + i] = __float2half_rn(src[(size_t) t * n + i]);
}
__global__ void perm_heads_kernel(const float* __restrict__ src, float* __restrict__ dst, int T, int Hkv, int per, int D,
                                  int to_p) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) T * Hkv * per * D;
    if (idx >= total) return;
    // idx walks x [T][Hkv][per][D]
    const int d = (int) (idx % D), r = (int) (idx / D % per), g = (int) (idx / ((size_t) D * per) % Hkv);
    const int t = (int) (idx / ((size_t) D * per * Hkv));
    const size_t pi = (((size_t) g * T + t) * per + r) * D + d;
    if (to_p) dst[pi] = src[idx];
    else dst[idx] = src[pi];
}
// block = a query
__global__ void attn_softmax_rows_kernel(float* __restrict__ S, int T, int Hkv, int per, int nk, long long klo,
                                         long long p0, int window, int extent, const float* __restrict__ tbl,
                                         const float* __restrict__ sink, float scale) {
    __shared__ float sh[256];
    const int qi = blockIdx.x, g = qi / (T * per), t = qi % (T * per) / per, r = qi % per;
    const int H = Hkv * per, h = g * per + r;
    float* sr = S + (size_t) qi * nk;
    const float* tb = tbl + ((size_t) t * H + h) * extent;
    const float snk = sink[(size_t) t * H + h];
    const long long p = p0 + t;
    // the keys the token sees: positions [lo, p]
    long long lo = window > 0 ? p - (window - 1) : 0;
    if (lo < klo) lo = klo;
    const int j0 = (int) (lo - klo), j1 = (int) (p - klo);   // (inclusive)
    float m = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) {
        const long long d = p - (klo + j);
        const float v = sr[j] * scale + (d < extent ? tb[d] : 0.0f) - snk;
        sr[j] = v;
        m = fmaxf(m, v);
    }
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) {
        const float e = expf(sr[j] - m);
        sr[j] = e;
        sum += e;
    }
    sum = block_sum(sum, sh) + expf(-m);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) sr[j] = j >= j0 && j <= j1 ? sr[j] * inv : 0.0f;
}

// attn_softmax_rows_kernel's probabilities straight to F16 (to_f16's __float2half_rn), S read three times and never
// written: each pass's logit made from S by the same expression, the product rounded to F32 before its F16 - the
// same bits (smtest: attn_softmax_rows + the conversion against this, rows of 300 / 1100 / 5000 keys).  Without the
// barriers the compiler rounds the product straight to F16, once: other bits in about 1 value in 16,000, which the
// model's last logits feel (KL 0.25 on a 1.7K prompt)
// (relq non-null: the distance table made here, in LDS, instead of read from tbl - row (t, h)'s entry d the sum over
//  i < d_rel of rel_bias[d d_rel + i] relq[(t H + h) d_rel + i] in i's order, the GEMM's own form; extent <= 1024)
__global__ void attn_softmax_p16_kernel(float* __restrict__ S, int T, int Hkv, int per, int nk, long long klo,
                                        long long p0, int window, int extent, const float* __restrict__ tbl,
                                        const float* __restrict__ sink, float scale, __half* __restrict__ P,
                                        const float* __restrict__ relq, const float* __restrict__ rel_bias, int d_rel) {
    __shared__ float sh[256];
    __shared__ float tsh[1024];
    const int qi = blockIdx.x, g = qi / (T * per), t = qi % (T * per) / per, r = qi % per;
    const int H = Hkv * per, h = g * per + r;
    float* sr = S + (size_t) qi * nk;
    __half* pr = P + (size_t) qi * nk;
    const float* tb = tbl + ((size_t) t * H + h) * extent;
    if (relq != nullptr) {
        const float* q = relq + ((size_t) t * H + h) * d_rel;
        for (int d = threadIdx.x; d < extent; d += blockDim.x) {
            const float* b = rel_bias + (size_t) d * d_rel;
            float a = 0.0f;
            for (int i = 0; i < d_rel; ++i) a = fmaf(b[i], q[i], a);
            tsh[d] = a;
        }
        __syncthreads();
        tb = tsh;
    }
    const float snk = sink[(size_t) t * H + h];
    const long long p = p0 + t;
    long long lo = window > 0 ? p - (window - 1) : 0;
    if (lo < klo) lo = klo;
    const int j0 = (int) (lo - klo), j1 = (int) (p - klo);   // (inclusive)
    auto val = [&](int j) {   // (each pass's logit from S: the same expression, kept from fusing into what follows)
        const long long d = p - (klo + j);
        float v = sr[j] * scale + (d < extent ? tb[d] : 0.0f) - snk;
        asm volatile("" : "+v"(v));
        return v;
    };
    float m = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) m = fmaxf(m, val(j));
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) sum += expf(val(j) - m);
    sum = block_sum(sum, sh) + expf(-m);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        float x = j >= j0 && j <= j1 ? expf(val(j) - m) * inv : 0.0f;
        asm volatile("" : "+v"(x));   // (the F32 product first, then its F16: not one fused rounding - to_f16's bits)
        pr[j] = __float2half_rn(x);
    }
}

// ---- the prompt attention as one kernel (IK2_HIP_ATTN_FLASH, off by default): flash attention on RDNA 4's matrix
// cores.  Block = kv head g and 64 query rows q = t per + r of the pass (Q's F16 rows [Hkv][T per][D], as the GEMMs
// take them); 4 waves, wave w the rows w 16 .. + 15, each with its own running max and denominator.  A tile of 64 keys
// (and their values) staged once in LDS for the 4 waves: S = Q K^T by v_wmma_f32_16x16x16_f16 (F16 in, F32 sums, as
// the GEMM), the logits as attn_softmax_rows makes them (scale, the distance table, the sink; masked: outside the
// window, after the token), the probabilities in F16 (as P16) for O += P V.  The sink's zero logit is the running
// sums' start (m 0, denominator 1).  O [Hkv][T per][D] F32, as the second GEMM leaves it.  Not the same bits: the sums
// in other orders and the probabilities scaled by running maxima, not the row's.
// (the exponentials: the hardware's exp2 of x log2 e - IK2_FLASH_EXACT_EXP builds expf's instead)
__device__ __forceinline__ float fl_exp(float x) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(IK2_FLASH_EXACT_EXP)
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
#else
    return expf(x);
#endif
}
namespace fl {
constexpr int D = 128, KT = 64, RB = 64, KS = D + 8, VS = KT + 8;   // (KS: K's rows' stride in LDS, VS: V's columns')
}
__global__ void __launch_bounds__(128) attn_flash_kernel(const uint16_t* __restrict__ qh16, int Q, int per, int Hkv,
                                                         const uint16_t* __restrict__ K16, const uint16_t* __restrict__ V16,
                                                         int ring, long long klo, int nk, long long pf, int window,
                                                         int extent, const float* __restrict__ tbl,
                                                         const float* __restrict__ sink, float scale,
                                                         float* __restrict__ out) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
    // gfx12's v_wmma_f32_16x16x16_f16: lane l gives row (A) / column (B) l % 16's k-values 8 (l / 16) .. + 7 - each
    // lane loads just those 8 (one 16-byte read from rows laid out along k); result value i: row 8 (l / 16) + i
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    using f8 = float __attribute__((ext_vector_type(8)));
    const _Float16* qh = (const _Float16*) qh16;
    const _Float16* K = (const _Float16*) K16;
    const _Float16* V = (const _Float16*) V16;
    using fl::D;
    using fl::KT;
    using fl::KS;
    using fl::VS;
    __shared__ __align__(16) _Float16 ks[KT * KS];      // K: a key's 128 values a row
    __shared__ __align__(16) _Float16 vs[D * VS];       // V transposed: a dimension's 64 keys a row
    __shared__ __align__(16) _Float16 qs[4][16 * KS];   // the waves' queries
    __shared__ __align__(16) _Float16 ps[4][16 * VS];   // the waves' probabilities, a row's 64 keys a row
    __shared__ float tsc_[4][16];
    const int g = blockIdx.y, lt = threadIdx.x, lane = lt & 31, wp = lt >> 5, m16 = lane & 15, hi = lane >> 4;
    const int k0 = 8 * hi;
    const int H = Hkv * per, qb = blockIdx.x * fl::RB, q0 = qb + wp * 16;
    for (int e = lane; e < 16 * (D / 8); e += 32) {   // (the wave's 16 rows, 16 uint4s each)
        const int rr = e / (D / 8), part = e % (D / 8), q = q0 + rr;
        *(uint4*) (qs[wp] + rr * KS + part * 8) =
            q < Q ? ((const uint4*) (qh + ((size_t) g * Q + q) * D))[part] : make_uint4(0u, 0u, 0u, 0u);
    }
    int rpj[8], rloj[8], rto[8];
    float rs[8], m_run[8], den[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = q0 + k0 + i, qq = q < Q ? q : Q - 1, t = qq / per, r = qq % per, h = g * per + r;
        rpj[i] = q < Q ? (int) (pf + t - klo) : -1;   // (a row past Q: no keys)
        const long long lo = window > 0 ? pf + t - (window - 1) : 0;
        rloj[i] = lo < klo ? 0 : (int) (lo - klo);
        rto[i] = (t * H + h) * extent;
        rs[i] = sink[(size_t) t * H + h];
        m_run[i] = 0.0f;
        den[i] = 1.0f;
    }
    f8 acc[D / 16];
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) acc[nt] = f8{};
    const int qlast = min(qb + fl::RB, Q) - 1;
    const long long p_lo = pf + qb / per, p_hi = pf + qlast / per;
    long long k_lo = window > 0 ? p_lo - (window - 1) : 0;
    if (k_lo < klo) k_lo = klo;
    const int j_begin = (int) (k_lo - klo), j_end = (int) min((long long) nk, p_hi - klo + 1);
    for (int j0 = j_begin; j0 < j_end; j0 += KT) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {   // (64 keys' rows: 16 uint4s each, 8 a thread; V transposed as it lands)
            const int e = lt + 128 * k, rr = e >> 4, part = e & 15, j = j0 + rr;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
            if (j < j_end) {
                const long long kp = klo + j;
                const size_t slot = (size_t) (ring > 0 ? kp % ring : kp);
                kv = ((const uint4*) (K + (slot * Hkv + g) * D))[part];
                vv = ((const uint4*) (V + (slot * Hkv + g) * D))[part];
            }
            *(uint4*) (ks + rr * KS + part * 8) = kv;
            const _Float16* vh = (const _Float16*) &vv;
#pragma unroll
            for (int x = 0; x < 8; ++x) vs[(part * 8 + x) * VS + rr] = vh[x];
        }
        __syncthreads();
        // S = Q K^T: 4 sub-tiles of 16 keys, the k-steps outermost (each Q operand read once a tile)
        f8 s[4];
#pragma unroll
        for (int st = 0; st < 4; ++st) s[st] = f8{};
#pragma unroll 2
        for (int kk = 0; kk < D / 16; ++kk) {
            const h8 qa = *(const h8*) (qs[wp] + m16 * KS + kk * 16 + k0);
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const h8 kb = *(const h8*) (ks + (st * 16 + m16) * KS + kk * 16 + k0);
                s[st] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa, kb, s[st]);
            }
        }
        float mx[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) mx[i] = -INFINITY;
        // (the wave's rows all see the whole tile, every key past the table's extent: no masks, no table - the
        //  global layers' tiles far from the diagonal; the flags the wave's, from its lanes' rows by ballot)
        int lo_ok = 1, far_ok = 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            lo_ok &= (j0 >= rloj[i] && j0 + KT - 1 <= rpj[i]) ? 1 : 0;
            far_ok &= (rpj[i] - (j0 + KT - 1) >= extent) ? 1 : 0;
        }
        const bool fast = __all(lo_ok && far_ok && j0 + KT <= j_end);
        if (fast) {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = s[st][i] * scale + 0.0f - rs[i];
                    s[st][i] = v;
                    mx[i] = fmaxf(mx[i], v);
                }
        } else {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int j = j0 + st * 16 + m16;
                    float v = -INFINITY;
                    if (j < j_end && j >= rloj[i] && j <= rpj[i]) {
                        const int d = rpj[i] - j;
                        v = s[st][i] * scale + (d < extent ? tbl[rto[i] + d] : 0.0f) - rs[i];
                    }
                    s[st][i] = v;
                    mx[i] = fmaxf(mx[i], v);
                }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) mx[i] = fmaxf(mx[i], __shfl_xor(mx[i], o, 32));
            const float m_new = fmaxf(m_run[i], mx[i]), ts = fl_exp(m_run[i] - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const float p = s[st][i] == -INFINITY ? 0.0f : fl_exp(s[st][i] - m_new);
                sum += p;
                ps[wp][(k0 + i) * VS + st * 16 + m16] = (_Float16) p;
            }
#pragma unroll
            for (int o = 8; o >= 1; o >>= 1) sum += __shfl_xor(sum, o, 32);
            den[i] = den[i] * ts + sum;
            m_run[i] = m_new;
            if (m16 == 0) tsc_[wp][k0 + i] = ts;
        }
        __syncthreads();
        {
            float tr[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) tr[i] = tsc_[wp][k0 + i];
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[nt][i] *= tr[i];
        }
        // O += P V: A = P (row m16, keys kk 16 + k0 ..), B = V (dimension nt 16 + m16, the same keys)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const h8 pa = *(const h8*) (ps[wp] + m16 * VS + kk * 16 + k0);
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt) {
                const h8 vb = *(const h8*) (vs + (nt * 16 + m16) * VS + kk * 16 + k0);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb, acc[nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = q0 + k0 + i;
            if (q < Q) out[((size_t) g * Q + q) * D + nt * 16 + m16] = acc[nt][i] / den[i];
        }
#endif
}

// attn_flash's second form (IK2_HIP_FLASH_V=2): S^T = K Q^T and O^T = V^T P^T, so a lane holds one query (column
// l % 16) and 8 keys / dimensions (rows 8 (l / 16) ..): its 8 probabilities a 16-key step are the second product's B
// operand as they stand (no trip through LDS), the row's max and sum one shuffle across the halves, the rescale one
// number a lane; the query's F16 values in registers; V transposed 8 x 8 in registers and stored 16 bytes at a time
template <int NW, bool Q8 = false>   // (Q8: the 8-bit cache's rows, made F16 as they are staged)
__global__ void __launch_bounds__(NW * 32) attn_flash2_kernel(const uint16_t* __restrict__ qh16, int Q, int per, int Hkv,
                                                          const uint16_t* __restrict__ K16, const uint16_t* __restrict__ V16,
                                                          int ring, long long klo, int nk, long long pf, int window,
                                                          int extent, const float* __restrict__ tbl,
                                                          const float* __restrict__ sink, float scale,
                                                          float* __restrict__ out) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    using f8 = float __attribute__((ext_vector_type(8)));
    const _Float16* qh = (const _Float16*) qh16;
    const _Float16* K = (const _Float16*) K16;
    const _Float16* V = (const _Float16*) V16;
    using fl::D;
    using fl::KT;
    using fl::KS;
    using fl::VS;
    __shared__ __align__(16) _Float16 ks[KT * KS];   // K: a key's 128 values a row
    __shared__ __align__(16) _Float16 vs[D * VS];    // V transposed: a dimension's 64 keys a row
    const int g = blockIdx.y, lt = threadIdx.x, lane = lt & 31, wp = lt >> 5, m16 = lane & 15, hi = lane >> 4;
    const int k0 = 8 * hi;
    constexpr int RB = NW * 16;   // (the block's query rows: NW waves of 16)
    const int H = Hkv * per, qb = blockIdx.x * RB, q = qb + wp * 16 + m16;
    const bool qok = q < Q;
    const int qq = qok ? q : Q - 1, t = qq / per, h = g * per + qq % per;
    h8 qa[D / 16];
    {
        const h8* qr = (const h8*) (qh + ((size_t) g * Q + qq) * D);
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) qa[kk] = qok ? qr[kk * 2 + hi] : h8{};
    }
    const int rp = qok ? (int) (pf + t - klo) : -1;   // (a row past Q: no keys)
    const long long lo = window > 0 ? pf + t - (window - 1) : 0;
    const int rlo = lo < klo ? 0 : (int) (lo - klo), rto = (t * H + h) * extent;
    const float rs = sink[(size_t) t * H + h];
    float m_run = 0.0f, den = 1.0f;
    f8 acc[D / 16];
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) acc[nt] = f8{};
    const int qlast = min(qb + RB, Q) - 1;
    const long long p_lo = pf + qb / per, p_hi = pf + qlast / per;
    long long k_lo = window > 0 ? p_lo - (window - 1) : 0;
    if (k_lo < klo) k_lo = klo;
    const int j_begin = (int) (k_lo - klo), j_end = (int) min((long long) nk, p_hi - klo + 1);
    const int vp = (lt & 127) >> 3, vk = (lt & 7) * 8;
    for (int j0 = j_begin; j0 < j_end; j0 += KT) {
        // (the tile: 64 keys' K rows, 1024 uint4s, and V's 8 x 8 blocks, 128; NW 4: each thread 8 of K's and a block
        //  of V, NW 8: threads 0 .. 127 a block of V each, 128 .. 255 8 of K's)
        if (NW == 4 || lt >= 128) {
            const int lk = NW == 4 ? lt : lt - 128;
#pragma unroll
            for (int k = 0; k < 8; ++k) {   // (K: key rows (lk / 16) + 8 k, a uint4 each)
                const int rr = (lk >> 4) + 8 * k, j = j0 + rr;
                uint4 kv = make_uint4(0u, 0u, 0u, 0u);
                if (j < j_end) {
                    const long long kp = klo + j;
                    const size_t slot = (size_t) (ring > 0 ? kp % ring : kp);
                    kv = Q8 ? kv8_get8(K16, slot, Hkv, D, g, (lk & 15) * 8) : ((const uint4*) (K + (slot * Hkv + g) * D))[lk & 15];
                }
                *(uint4*) (ks + rr * KS + (lk & 15) * 8) = kv;
            }
        }
        if (NW == 4 || lt < 128) {   // (V: keys vk .. + 7, dimensions vp 8 .. + 7, transposed in registers)
            uint32_t a[8][4];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int j = j0 + vk + k;
                uint4 vv = make_uint4(0u, 0u, 0u, 0u);
                if (j < j_end) {
                    const long long kp = klo + j;
                    const size_t slot = (size_t) (ring > 0 ? kp % ring : kp);
                    vv = Q8 ? kv8_get8(V16, slot, Hkv, D, g, vp * 8) : ((const uint4*) (V + (slot * Hkv + g) * D))[vp];
                }
                a[k][0] = vv.x;
                a[k][1] = vv.y;
                a[k][2] = vv.z;
                a[k][3] = vv.w;
            }
#pragma unroll
            for (int x = 0; x < 8; ++x) {
                const int p = x >> 1;
                uint32_t o[4];
#pragma unroll
                for (int m = 0; m < 4; ++m)
                    o[m] = (x & 1) ? ((a[2 * m][p] >> 16) | (a[2 * m + 1][p] & 0xffff0000u))
                                   : ((a[2 * m][p] & 0xffffu) | (a[2 * m + 1][p] << 16));
                *(uint4*) (vs + (vp * 8 + x) * VS + vk) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        // S^T = K Q^T: 4 sub-tiles of 16 keys; lane: query m16, keys st 16 + k0 + i
        f8 s[4];
#pragma unroll
        for (int st = 0; st < 4; ++st) s[st] = f8{};
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk)
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const h8 ka = *(const h8*) (ks + (st * 16 + m16) * KS + kk * 16 + k0);
                s[st] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qa[kk], s[st]);
            }
        float mx = -INFINITY;
        const bool fast = __all(j0 >= rlo && j0 + KT - 1 <= rp && rp - (j0 + KT - 1) >= extent && j0 + KT <= j_end);
        if (fast) {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = s[st][i] * scale + 0.0f - rs;
                    s[st][i] = v;
                    mx = fmaxf(mx, v);
                }
        } else {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int j = j0 + st * 16 + k0 + i;
                    float v = -INFINITY;
                    if (j < j_end && j >= rlo && j <= rp) {
                        const int d = rp - j;
                        v = s[st][i] * scale + (d < extent ? tbl[rto + d] : 0.0f) - rs;
                    }
                    s[st][i] = v;
                    mx = fmaxf(mx, v);
                }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float m_new = fmaxf(m_run, mx), tsc = fl_exp(m_run - m_new);
        float sum = 0.0f;
        h8 pb[4];
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = s[st][i] == -INFINITY ? 0.0f : fl_exp(s[st][i] - m_new);
                sum += p;
                pb[st][i] = (_Float16) p;
            }
        sum += __shfl_xor(sum, 16, 32);
        den = den * tsc + sum;
        m_run = m_new;
#pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) acc[nt] *= tsc;
        // O^T += V^T P^T: A = V^T (dimension nt 16 + m16, keys kk 16 + k0 ..), B = the lane's probabilities
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int nt = 0; nt < D / 16; ++nt) {
                const h8 va = *(const h8*) (vs + (nt * 16 + m16) * VS + kk * 16 + k0);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, pb[kk], acc[nt]);
            }
        __syncthreads();
    }
    if (qok) {
        float* o = out + ((size_t) g * Q + q) * D;
#pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) {
            float4* o4 = (float4*) (o + nt * 16 + k0);
            o4[0] = make_float4(acc[nt][0] / den, acc[nt][1] / den, acc[nt][2] / den, acc[nt][3] / den);
            o4[1] = make_float4(acc[nt][4] / den, acc[nt][5] / den, acc[nt][6] / den, acc[nt][7] / den);
        }
    }
#endif
}

__global__ void scatter_rows_kernel(const float* __restrict__ y, const int* __restrict__ row_of, int E,
                                    float* __restrict__ ent) {
    const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x, r = row_of[e];
    if (i < E && r >= 0) ent[(size_t) e * E + i] = y[(size_t) r * E + i];
}
// moe_sort_bits (the prompt's expert sets): hy4's moe_sort's rows - each slot's entries in token order - by a bitmap
// of the slot's tokens (a token routes to an expert once) instead of its AMD rank loop (each entry counting its slot's
// rows: the square of the counts, one block): a bit an entry (atomicOr: any order the same), each 32 tokens' word
// prefix by a wave's scan, the slots' bounds by the block's, then each entry's row from them
__global__ void sort_bits_set_kernel(const int* __restrict__ ids, int n_ent, int nu, const int* __restrict__ slot,
                                     int n_slots, int W, unsigned* __restrict__ bits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ent) return;
    const int s = slot ? slot[ids[i]] : ids[i];
    if (s < 0 || s >= n_slots) return;
    const int t = i / nu;
    atomicOr(bits + (size_t) s * W + (t >> 5), 1u << (t & 31));
}
__global__ void __launch_bounds__(1024) sort_bits_rows_kernel(const int* __restrict__ ids, int n_ent, int nu,
                                                              const int* __restrict__ slot, int n_slots, int W,
                                                              const unsigned* __restrict__ bits, int* __restrict__ wpre,
                                                              int* __restrict__ bounds, int* __restrict__ tok,
                                                              int* __restrict__ row_of, int* __restrict__ ent_of) {
    __shared__ int cnt[1024];
    __shared__ int sc[1024];
    const int lt = threadIdx.x, lane = lt & 31, nw = (int) (blockDim.x >> 5);
    for (int s = lt >> 5; s < n_slots; s += nw) {   // (a wave a slot: its words' exclusive prefix, its count)
        int carry = 0;
        for (int w0 = 0; w0 < W; w0 += 32) {
            const int w = w0 + lane;
            const int c = w < W ? __popc(bits[(size_t) s * W + w]) : 0;
            int x = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up(x, o, 32);
                if (lane >= o) x += y;
            }
            if (w < W) wpre[(size_t) s * W + w] = carry + x - c;
            carry += __shfl(x, 31, 32);
        }
        if (lane == 0) cnt[s] = carry;
    }
    __syncthreads();
    const int v = lt < n_slots ? cnt[lt] : 0;
    sc[lt] = v;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int y = lt >= o ? sc[lt - o] : 0;
        __syncthreads();
        sc[lt] += y;
        __syncthreads();
    }
    if (lt < n_slots) {
        bounds[lt] = sc[lt] - v;
        cnt[lt] = sc[lt] - v;
    }
    if (lt == 1023) bounds[n_slots] = sc[lt];
    __syncthreads();
    for (int i = lt; i < n_ent; i += (int) blockDim.x) {
        const int s = slot ? slot[ids[i]] : ids[i];
        if (s < 0 || s >= n_slots) {
            row_of[i] = -1;
            continue;
        }
        const int t = i / nu, w = t >> 5;
        const unsigned b = bits[(size_t) s * W + w];
        const int r = cnt[s] + wpre[(size_t) s * W + w] + __popc(b & ((1u << (t & 31)) - 1u));
        tok[r] = t;
        row_of[i] = r;
        ent_of[r] = i;
    }
}
// a set's rows to their entries' places (scatter_rows by the set's rows, not by all the chunk's entries)
__global__ void scatter_set_rows_kernel(const float* __restrict__ y, const int* __restrict__ ent_of, int E,
                                        float* __restrict__ ent) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E) ent[(size_t) ent_of[r] * E + i] = y[(size_t) r * E + i];
}
__global__ void combine_rows_kernel(const float* __restrict__ ent, const float* __restrict__ w, int nu, int E,
                                    float* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float v = 0.0f;
    for (int j = 0; j < nu; ++j) v = fmaf(w[(size_t) t * nu + j], ent[((size_t) t * nu + j) * E + i], v);
    out[(size_t) t * E + i] = v;
}

// ---- the window step in fewer launches (own runtime, 2026-11-24): fused forms of the step's small kernels, each part
// the same arithmetic as the kernel it replaces (the same bits) - tests/ik2_hip_fuse_test.cu checks them against those
struct MixConv {   // (y null: none) conv_add_rows_kernel's work on the block's row before its mix
    const float* y;
    const float* h0;
    float* h1;
    const float* w;
    float* blk;
};
struct MixQ8 {   // (q null: none) q8_kernel's form of each query's output - dup: dup_q8_kernel's two rows a row
    int8_t* q;
    float* d;
    int dup;
};
__device__ __forceinline__ float fz_wave_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// mix_rms2_kernel after the residual's conv (cv) and with the int8 rows of its outputs (q8); E a multiple of 32
__global__ void __launch_bounds__(1024) mix_rms3_kernel(const float* blocks, size_t ldb, int nb, float* cur, int T, int E,
                                                        const float* __restrict__ query, int nq,
                                                        const float* __restrict__ g0, const float* __restrict__ g1,
                                                        const float* __restrict__ g2, float eps, float* __restrict__ out,
                                                        MixConv cv, MixQ8 q8) {
    __shared__ float sh[33 * 16];
    __shared__ float st[kMaxStates * 4];
    const int ns = nb + 1, tid = threadIdx.x, t = blockIdx.x;
    const size_t to = (size_t) t * E;
    if (cv.y) {   // (conv_add_rows_kernel's row t: every channel of it in this block, so a barrier is enough)
        const int C = E;
        for (int c = tid; c < C; c += blockDim.x) {
            float in[4];
            for (int k = 0; k < 4; ++k) {
                const int tt = t - 3 + k;
                in[k] = tt >= 0 ? cv.y[(size_t) tt * C + c] : cv.h0[(size_t) (3 + tt) * C + c];
            }
            const float h0 = in[0], h1 = in[1], h2 = in[2], v = in[3];
            const float* w = cv.w;
            const float o = (v + w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v) * 0.44721359f;
            float* cr = cur + (size_t) t * C + c;
            if (cv.blk) {
                cv.blk[(size_t) t * C + c] = *cr;
                *cr = o;
            } else {
                *cr += o;
            }
            if (t == T - 1)
                for (int r = 0; r < 3; ++r) {
                    const int q = T + r;
                    cv.h1[(size_t) r * C + c] = q < 3 ? cv.h0[(size_t) q * C + c] : cv.y[(size_t) (q - 3) * C + c];
                }
        }
        __syncthreads();
    }
    for (int s0 = 0; s0 < ns; s0 += 4) {
        float a[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) a[i] = 0.0f;
        for (int i = tid; i < E; i += blockDim.x) {
            float qv[3];
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) qv[qi] = qi < nq ? query[(size_t) qi * E + i] : 0.0f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = s0 + u;
                if (s < ns) {
                    const float v = (s < nb ? blocks + (size_t) s * ldb : cur)[to + i];
                    a[4 * u] += v * v;
#pragma unroll
                    for (int qi = 0; qi < 3; ++qi) a[4 * u + 1 + qi] += v * qv[qi];
                }
            }
        }
        block_sum_v<16>(a, sh);
        if (tid == 0)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = s0 + u;
                if (s < ns) {
                    const float inv = rsqrtf(a[4 * u] / (float) E + 1e-6f);
                    st[4 * s] = inv;
                    for (int qi = 0; qi < nq; ++qi) st[4 * s + 1 + qi] = a[4 * u + 1 + qi] * inv;
                }
            }
    }
    __syncthreads();
    const float sc = rsqrtf((float) E);
    float vv[3][8], ss[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) continue;
        float m = -INFINITY, w[kMaxStates], den = 0.0f;
        for (int s = 0; s < ns; ++s) m = fmaxf(m, st[4 * s + 1 + qi] * sc);
        for (int s = 0; s < ns; ++s) {
            w[s] = expf(st[4 * s + 1 + qi] * sc - m);
            den += w[s];
        }
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            const int i = tid + k2 * blockDim.x;
            vv[qi][k2] = 0.0f;
            if (i >= E) continue;
            float v = 0.0f;
            for (int s = 0; s < ns; ++s) v += w[s] * st[4 * s] * (s < nb ? blocks[(size_t) s * ldb + to + i] : cur[to + i]);
            vv[qi][k2] = v / den;
            ss[qi] += vv[qi][k2] * vv[qi][k2];
        }
    }
    block_sum_v<3>(ss, sh);
    const int E32 = E / 32;
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) continue;
        const float* gain = qi == 0 ? g0 : qi == 1 ? g1 : g2;
        const float inv = rsqrtf(ss[qi] / (float) E + eps);
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            const int i = tid + k2 * blockDim.x;
            if (i < E) {   // (a whole wave in or out: E and the block a multiple of 32)
                const float o = vv[qi][k2] * inv * (gain ? gain[i] : 1.0f);
                out[((size_t) qi * T + t) * E + i] = o;
                if (q8.q) {
                    const float am = fz_wave_max(fabsf(o));
                    const float scq = am / 127.0f;
                    const int8_t qv = (int8_t) (am > 0.0f ? __float2int_rn(o / scq) : 0);
                    const int b = i / 32, lane = i & 31;
                    if (!q8.dup) {
                        const size_t r = (size_t) qi * T + t;
                        q8.q[r * E + i] = qv;
                        if (lane == 0) q8.d[r * E32 + b] = scq;
                    } else {
                        for (int h = 0; h < 2; ++h) {
                            const size_t r = 2 * ((size_t) qi * T + t) + h;
                            q8.q[r * E + i] = qv;
                            if (lane == 0) q8.d[r * E32 + b] = scq;
                        }
                    }
                }
            }
        }
    }
}
// router_bitonic_kernel (hy4) for row blockIdx.x < T, its picks and weights also into dst beside the copy of the rows x
// the other blocks make (pack3_kernel's layout: [T][E] x, [T][nu] picks, [T][nu] weights), then publish_kernel's flag
__device__ __forceinline__ float fz_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float fz_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ bool fz_before(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }
__device__ __forceinline__ void fz_step(float& v, int& i, int j, bool up, int lane) {
    const float pv = __shfl_xor_sync(0xffffffffu, v, j);
    const int pi = __shfl_xor_sync(0xffffffffu, i, j);
    const bool lower = (lane & j) == 0, pb = fz_before(pv, pi, v, i);
    if (lower == up ? pb : !pb) {
        v = pv;
        i = pi;
    }
}
struct FzGroup {   // (grp_ptr null: none) group_experts' outputs, made by the launch's last block
    unsigned long long* grp_ptr;
    int* grp_start;
    int* grp_ent;
    int* n_groups;
};
__global__ void __launch_bounds__(512) route_pub_kernel(const float* __restrict__ logits, int n_exp,
                                                        const float* __restrict__ bias, int n_used, int norm, float scale,
                                                        int* __restrict__ ids, float* __restrict__ wout, int T,
                                                        const float4* __restrict__ x, size_t n4, float* dst,
                                                        unsigned* counter, unsigned* flag, unsigned v,
                                                        const uint8_t* base, size_t stride, unsigned long long* plan,
                                                        const int* __restrict__ slot, FzGroup gr) {
    __shared__ float lv[16][16];
    __shared__ int li[16][16];
    if ((int) blockIdx.x < T) {
        const int t = blockIdx.x, e = threadIdx.x, w = e >> 5, lane = e & 31;
        const float* lr = logits + (size_t) t * n_exp;
        float val = e < n_exp ? fz_sigmoid(lr[e]) + (bias ? bias[e] : 0.0f) : -INFINITY;
        int i = e < n_exp ? e : 0x7fffffff;
        for (int k = 2; k <= 32; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1) fz_step(val, i, j, (lane & k) == 0, lane);
        if (lane < 16) {
            lv[w][lane] = val;
            li[w][lane] = i;
        }
        __syncthreads();
        for (int lists = 16; lists > 1; lists >>= 1) {
            float mv = -INFINITY;
            int mi = 0x7fffffff;
            if (w < lists / 2) {
                mv = lane < 16 ? lv[2 * w][lane] : lv[2 * w + 1][31 - lane];
                mi = lane < 16 ? li[2 * w][lane] : li[2 * w + 1][31 - lane];
                for (int j = 16; j > 0; j >>= 1) fz_step(mv, mi, j, true, lane);
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
            const float p = lane < n_used ? fz_sigmoid(lr[pick]) : 0.0f;
            const float sum = fz_warp_sum(p);
            const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
            if (lane < n_used) {
                const float wt = p / den * scale;
                const size_t j = (size_t) t * n_used + lane, TE = 4 * n4;
                ids[j] = pick;
                ((int*) dst)[TE + j] = pick;
                dst[TE + (size_t) T * n_used + j] = wt;
                if (plan) {   // (plan_kernel's work, a token: its pick's blob, or 0 and no weight - the CPU lane's)
                    const int sl = slot ? slot[pick] : pick;
                    plan[j] = sl >= 0 ? (unsigned long long) (base + (size_t) sl * stride) : 0ull;
                    wout[j] = sl < 0 ? 0.0f : wt;
                } else {
                    wout[j] = wt;
                }
            }
        }
    } else {
        const size_t nb = gridDim.x - (unsigned) T;
        for (size_t i = (size_t) (blockIdx.x - (unsigned) T) * blockDim.x + threadIdx.x; i < n4; i += nb * blockDim.x)
            ((float4*) dst)[i] = x[i];
    }
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
    if (!gr.grp_ptr || !last) return;
    // hy4's group_par_kernel (slot form) on the last block's 512 threads, a thread an expert (n_exp <= 512): the same groups
    // and entry order - the picks the other blocks wrote read past the caches
    __shared__ int ent[16 * 16];
    __shared__ int sc[512], sg[512];
    const int e = threadIdx.x, n_ent = T * n_used;
    for (int i = threadIdx.x; i < n_ent; i += blockDim.x) {
        const int x2 = *(const volatile int*) (ids + i);
        const bool out = slot && slot[x2] < 0;
        if (out) wout[i] = 0.0f;
        ent[i] = out ? -1 : x2;
    }
    __syncthreads();
    int c = 0;
    if (e < n_exp)
        for (int i = 0; i < n_ent; ++i) c += ent[i] == e;
    sc[e] = c;
    sg[e] = c > 0;
    __syncthreads();
    for (int o = 1; o < 512; o <<= 1) {
        const int a = e >= o ? sc[e - o] : 0, b = e >= o ? sg[e - o] : 0;
        __syncthreads();
        sc[e] += a;
        sg[e] += b;
        __syncthreads();
    }
    const int off = sc[e] - c, gi = sg[e] - (c > 0);
    if (e < n_exp && c > 0) {
        gr.grp_ptr[gi] = (unsigned long long) (base + (size_t) (slot ? slot[e] : e) * stride);
        gr.grp_start[gi] = off;
        int k2 = off;
        for (int i = 0; i < n_ent; ++i)
            if (ent[i] == e) gr.grp_ent[k2++] = (i / n_used) * 16 + i % n_used;
    }
    if (e == n_exp - 1) {
        gr.grp_start[sg[e]] = sc[e];
        *gr.n_groups = sg[e];
    }
}
// hy4's gemv_rows_kernel (F32, alpha 1): a block's sum over its threads in hy4's block_sum's order
__device__ float fz_block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = fz_warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.0f;
    r = fz_warp_sum(r);
    return r;
}
__device__ __forceinline__ float fz_row_dot(const float* __restrict__ wr, const float* __restrict__ xc, int k) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) acc = fmaf(wr[i], xc[i], acc);
    return fz_block_sum(acc);
}
struct KvProj {   // (the F32 GEMVs the launch makes too: the sinks, sw null: none; a global layer's conv projections, kw null: none)
    const float* sw;
    const float* sx;
    float* sy;
    int ns;
    int k;
    const float* kw;
    const float* vw;
    const float* kx;
    const float* vx;
    float* dko;
    float* dvo;
};
// kv_finish_rows_kernel's blocks (x < 2 Hkv: head x % Hkv, V when x >= Hkv), clip2_rows_kernel's (head x - 2 Hkv) and the
// sinks' GEMV (row x - 2 Hkv - H) in one launch, row blockIdx.y; with pj.kw a kv block makes its head's 4 rows of the conv
// projection first (gemv_rows_kernel's sums)
__global__ void kv_clip_rows_kernel(const float* __restrict__ in_k, const float* __restrict__ in_v,
                                    const float* __restrict__ hk0, const float* __restrict__ hv0, float* __restrict__ hk1,
                                    float* __restrict__ hv1, const float* __restrict__ k_conv,
                                    const float* __restrict__ v_conv, const float* __restrict__ dk,
                                    const float* __restrict__ dv, int T, int Hkv, int D, float thr, float eps,
                                    __half* __restrict__ K, __half* __restrict__ V, int ring, long long p0,
                                    float* __restrict__ q, int H, const float* __restrict__ qs, float qthr,
                                    float* __restrict__ rel, int n_rel, const float* __restrict__ rs, float rthr,
                                    RowMul mul, KvProj pj, int kvq) {
    __shared__ float sh[256];
    __shared__ float dloc[4];
    const int x = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    if (x >= 2 * Hkv + H) {   // (a sink: gemv_rows_kernel's block (row, column t))
        const int row = x - 2 * Hkv - H;
        const float v = fz_row_dot(pj.sw + (size_t) row * pj.k, pj.sx + (size_t) t * pj.k, pj.k);
        if (threadIdx.x == 0) pj.sy[(size_t) t * pj.ns + row] = v;
        return;
    }
    if (x < 2 * Hkv) {
        const int h = x % Hkv, isv = x / Hkv, C = Hkv * D, c = h * D + i;
        const float* y = isv ? in_v : in_k;
        const float* hist = isv ? hv0 : hk0;
        float* hnew = isv ? hv1 : hk1;
        const float* w = isv ? v_conv : k_conv;
        const float* dyn = isv ? dv : dk;
        if (pj.kw) {   // (the head's rows of the conv projection, as the GEMV writes them)
            const float* pw = isv ? pj.vw : pj.kw;
            const float* px = (isv ? pj.vx : pj.kx) + (size_t) t * pj.k;
            for (int j = 0; j < 4; ++j) {
                const float v = fz_row_dot(pw + (size_t) (h * 4 + j) * pj.k, px, pj.k);
                if (threadIdx.x == 0) {
                    dloc[j] = v;
                    (isv ? pj.dvo : pj.dko)[((size_t) t * Hkv + h) * 4 + j] = v;
                }
            }
            __syncthreads();
        }
        float o = 0.0f;
        if (i < D) {
            float in[4];
            for (int k = 0; k < 4; ++k) {
                const int tt = t - 3 + k;
                in[k] = tt >= 0 ? y[(size_t) tt * C + c] : hist[(size_t) (3 + tt) * C + c];
            }
            const float h0 = in[0], h1 = in[1], h2 = in[2], v = in[3];
            o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
            if (dyn) {
                const float* d = pj.kw ? dloc : dyn + ((size_t) t * Hkv + h) * 4;
                o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
            }
            o = (v + o) * (dyn ? 0.333333343f : 0.44721359f);
            if (t == T - 1)
                for (int r = 0; r < 3; ++r) {
                    const int qq = T + r;
                    hnew[(size_t) r * C + c] = qq < 3 ? hist[(size_t) qq * C + c] : y[(size_t) (qq - 3) * C + c];
                }
        }
        const float ss = block_sum(o * o, sh);
        const float f = isv ? rsqrtf(ss / (float) D + eps) : fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
        if (kvq) {
            const long long p = p0 + t;
            kv8_put(isv ? V : K, (size_t) (ring > 0 ? p % ring : p), Hkv, D, h, o * f, sh);
        } else if (i < D) {
            const long long p = p0 + t;
            (isv ? V : K)[(size_t) (ring > 0 ? p % ring : p) * C + c] = __float2half_rn(o * f);
        }
        return;
    }
    const int h = x - 2 * Hkv;
    float* qr = q + ((size_t) t * H + h) * D;
    float* rr = rel + ((size_t) t * H + h) * n_rel;
    const float v = i < D ? qr[i] * qs[(size_t) h * D + i] : 0.0f;
    const float r = i < n_rel ? rr[i] * rs[(size_t) h * n_rel + i] : 0.0f;
    const float fq = fminf(1.0f, qthr / (sqrtf(block_sum(v * v, sh)) + 1e-6f));
    const float fr = fminf(1.0f, rthr / (sqrtf(block_sum(r * r, sh)) + 1e-6f));
    if (i < D) qr[i] = v * fq * mul.m[t];
    if (i < n_rel) rr[i] = r * fr * mul.m[t];
}

// moe_sum_kernel (hy4; a token's: wsum_kernel - the same sums) and then lane_add_kernel's wait and add, in one launch
__global__ void moe_lane_kernel(const float* __restrict__ part, int n_used, int n, const float* __restrict__ w,
                                float* __restrict__ out, int stride16, const unsigned* flag, unsigned v,
                                const int* __restrict__ ids, const int* __restrict__ slot, int n_ids, const float* src) {
    __shared__ int need;
    if (threadIdx.x == 0) {
        int nd = 0;
        for (int j = 0; j < n_ids; ++j) nd |= slot[ids[j]] < 0;
        if (nd) {
            const unsigned long long t0 = maya::kernels::clock_ns();
            while ((int) (*(volatile const unsigned*) flag - v) < 0)
                if (maya::kernels::clock_ns() - t0 > 60ull * 1000000000ull) {
                    nd = 0;
                    break;
                }
        }
        need = nd;
    }
    __syncthreads();
    __threadfence_system();
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.0f;
    for (int j = 0; j < n_used; ++j) {
        const float wj = w[(size_t) t * n_used + j];
        if (wj != 0.0f) a = fmaf(wj, part[((size_t) t * stride16 + j) * n + i], a);
    }
    if (need) a += *(const volatile float*) (src + (size_t) t * n + i);
    out[(size_t) t * n + i] = a;
}
// hy4's gemv4s_kernel (F32 weights, a window's NC columns staged in shared memory) and gemv4_kernel (one column) with
// RPB rows a block instead of 8: each row's warp makes the same steps in the same order, so the same bits - more
// blocks for the router's 384 rows (48 blocks of 8 leave a quarter of the CUs idle)
template <int NC, int RPB>
__global__ void __launch_bounds__(RPB * 32) gemv_s_rpb_kernel(const float* __restrict__ w, size_t ld, int k, int n,
                                                              const float* __restrict__ x, size_t ldx, int ncol,
                                                              float* __restrict__ y, size_t ldy) {
    constexpr int KC = 1024;
    __shared__ float4 xs[NC][KC / 4];
    const int row = blockIdx.x * RPB + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int c0 = blockIdx.y * NC;
    const bool live = row < n;
    const float* wr = w + (size_t) (live ? row : 0) * ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += KC) {
        const int kq = min(KC, k - k0) / 4;
        __syncthreads();
        for (int idx = (int) threadIdx.x; idx < NC * kq; idx += (int) blockDim.x) {
            const int c = idx / kq, q = idx - c * kq;
            xs[c][q] = c0 + c < ncol ? *(const float4*) (x + (size_t) (c0 + c) * ldx + k0 + 4 * q)
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        if (!live) continue;
#pragma unroll 4
        for (int i = lane * 4; i < 4 * kq; i += 128) {
            const float4 wv = *(const float4*) (wr + k0 + i);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c0 + c < ncol) {
                    const float4 xv = xs[c][i / 4];
                    acc[c] = fmaf(wv.w, xv.w, fmaf(wv.z, xv.z, fmaf(wv.y, xv.y, fmaf(wv.x, xv.x, acc[c]))));
                }
        }
    }
    if (!live) return;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = fz_warp_sum(acc[c]);
        if (lane == 0 && c0 + c < ncol) y[(size_t) (c0 + c) * ldy + row] = v;
    }
}
template <int RPB>
__global__ void __launch_bounds__(RPB * 32) gemv_1_rpb_kernel(const float* __restrict__ w, size_t ld, int k, int n,
                                                              const float* __restrict__ x, float* __restrict__ y) {
    const int row = blockIdx.x * RPB + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= n) return;
    const float* wr = w + (size_t) row * ld;
    float acc = 0.0f;
    for (int i = lane * 4; i < k; i += 128) {
        const float4 wv = *(const float4*) (wr + i);
        const float4 xv = *(const float4*) (x + i);
        acc = fmaf(wv.w, xv.w, fmaf(wv.z, xv.z, fmaf(wv.y, xv.y, fmaf(wv.x, xv.x, acc))));
    }
    const float v = fz_warp_sum(acc);
    if (lane == 0) y[row] = v;
}
// ---- decode attention on the matrix cores (IK2_HIP_ATTN_MMA, own runtime 2026-11-27): attn_chunk_rows_kernel's
// partials from v_wmma_f32_16x16x16_f16, for long contexts.  Block = (a split of SK keys at positions multiple of SK,
// kv head g, 16 of the T x per query rows): the split's 64-key tiles staged in LDS as attn_flash2_kernel stages them
// (the 8-bit cache's rows made F16 there), wave w the tile's keys w 16 .. + 15: S^T = K Q^T (the queries F16, F32
// sums), the logits as the chunks make them (scale, the distance bias per key under extent, the sink), an online
// softmax a wave, O^T += V^T P^T (P in F16); the 4 waves merged at the end into the split's (max, sum, part) - the
// combine of attention_chunk_rows takes them as they are.  A row's sums do not depend on the other rows nor on T (the
// splits and their tiles at fixed positions, a row's masked tiles adding exact zeros): a window's rows give the token
// steps' bits.  Not attention_chunk_rows' bits: the queries and probabilities in F16, other orders.
// PREC: the queries and the probabilities each in two F16 halves (x = hi + lo, two products), so the products see F32
// inputs to ~2^-22 as the chunks do - only the order of the sums differs from theirs (the default;
// IK2_HIP_ATTN_MMA_PREC=0: one F16 each)
template <bool Q8, bool PREC>
__global__ void __launch_bounds__(128) attn_mma_rows_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                                           const float* __restrict__ sink, int T, int H, int Hkv,
                                                           int n_rel, const float* __restrict__ rel_bias, int extent,
                                                           const uint16_t* __restrict__ K16,
                                                           const uint16_t* __restrict__ V16, int ring, long long lo0,
                                                           long long p0, int window, int SK, float scale,
                                                           float* __restrict__ st, float* __restrict__ part) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    using f8 = float __attribute__((ext_vector_type(8)));
    using fl::D;
    using fl::KT;
    using fl::KS;
    using fl::VS;
    const _Float16* K = (const _Float16*) K16;
    const _Float16* V = (const _Float16*) V16;
    __shared__ __align__(16) _Float16 smem[KT * KS + D * VS];   // (K's tile, V's transposed; at the end the merge's floats)
    _Float16* ks = smem;
    _Float16* vs = smem + KT * KS;
    const int b = blockIdx.y, nsplit = gridDim.y, g = blockIdx.x, lt = threadIdx.x, lane = lt & 31, w = lt >> 5;
    const int m16 = lane & 15, hi = lane >> 4, k0 = 8 * hi, per = H / Hkv, TP = T * per;
    const int tr = blockIdx.z * 16 + m16;   // (the lane's query row: t per + r)
    const bool qok = tr < TP;
    const int trr = qok ? tr : TP - 1, t = trr / per, h = g * per + trr % per;
    h8 qa[D / 16], ql[PREC ? D / 16 : 1];
    {
        const float* qr = q + ((size_t) t * H + h) * D;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x = qok ? qr[kk * 16 + k0 + i] : 0.0f;
                qa[kk][i] = (_Float16) x;
                if (PREC) ql[kk][i] = (_Float16) (x - (float) qa[kk][i]);
            }
    }
    const long long pf = p0 + t;   // (the row's position: it sees keys lo .. pf)
    const long long lo = window > 0 ? (pf - (window - 1) > 0 ? pf - (window - 1) : 0) : 0;
    const float rs = sink[(size_t) t * H + h];
    const float* rr = rel + ((size_t) t * H + h) * n_rel;
    // the split's keys: positions s0 .. s1 - 1, those the block's rows may see
    const int rl = min(blockIdx.z * 16 + 15, TP - 1);
    const long long s0 = lo0 + (long long) b * SK, p_hi = p0 + rl / per;
    const long long s1 = min(s0 + SK, p_hi + 1);
    float m_run = -INFINITY, den = 0.0f;
    f8 acc[D / 16];
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) acc[nt] = f8{};
    const int vp = (lt & 127) >> 3, vk = (lt & 7) * 8;
    // (the next tile's rows fetched into registers while this one is computed: K's 8 rows (lt / 16) + 8 k, V's keys
    //  vk .. + 7 at dimensions vp 8 .. + 7; the 8-bit cache's raw bytes and scales, made F16 as they go to LDS)
    uint4 kr[8], vr[8];
    float ksc[8], vsc[8];
    auto fetch = [&](long long j0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const long long jk = j0 + (lt >> 4) + 8 * k, jv = j0 + vk + k;
            kr[k] = vr[k] = make_uint4(0u, 0u, 0u, 0u);
            ksc[k] = vsc[k] = 0.0f;
            if (jk < s1) {
                const size_t slot = (size_t) (ring > 0 ? jk % ring : jk);
                if (Q8) {
                    const uint8_t* row = (const uint8_t*) K16 + slot * kv8_row_bytes(Hkv, D);
                    const uint2 bq = *(const uint2*) (row + (size_t) g * D + (lt & 15) * 8);
                    kr[k].x = bq.x;
                    kr[k].y = bq.y;
                    ksc[k] = ((const float*) (row + (size_t) Hkv * D))[g];
                } else {
                    kr[k] = ((const uint4*) (K + (slot * Hkv + g) * D))[lt & 15];
                }
            }
            if (jv < s1) {
                const size_t slot = (size_t) (ring > 0 ? jv % ring : jv);
                if (Q8) {
                    const uint8_t* row = (const uint8_t*) V16 + slot * kv8_row_bytes(Hkv, D);
                    const uint2 bq = *(const uint2*) (row + (size_t) g * D + vp * 8);
                    vr[k].x = bq.x;
                    vr[k].y = bq.y;
                    vsc[k] = ((const float*) (row + (size_t) Hkv * D))[g];
                } else {
                    vr[k] = ((const uint4*) (V + (slot * Hkv + g) * D))[vp];
                }
            }
        }
    };
    auto f16x8 = [](const uint4& b, float sc) -> uint4 {   // (8 int8 values, as kv8_get8 makes them)
        auto qv = [](uint32_t w, int k) { return (int) (int8_t) (uint8_t) (w >> (8 * k)); };
        return make_uint4(kv8_h2(qv(b.x, 0), qv(b.x, 1), sc), kv8_h2(qv(b.x, 2), qv(b.x, 3), sc),
                          kv8_h2(qv(b.y, 0), qv(b.y, 1), sc), kv8_h2(qv(b.y, 2), qv(b.y, 3), sc));
    };
    if (s0 < s1) fetch(s0);
    for (long long j0 = s0; j0 < s1; j0 += KT) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int r = (lt >> 4) + 8 * k;
            *(uint4*) (ks + r * KS + (lt & 15) * 8) = Q8 ? f16x8(kr[k], ksc[k]) : kr[k];
        }
        {   // (V transposed in registers)
            uint32_t a[8][4];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint4 vv = Q8 ? f16x8(vr[k], vsc[k]) : vr[k];
                a[k][0] = vv.x;
                a[k][1] = vv.y;
                a[k][2] = vv.z;
                a[k][3] = vv.w;
            }
#pragma unroll
            for (int x = 0; x < 8; ++x) {
                const int p = x >> 1;
                uint32_t o[4];
#pragma unroll
                for (int m = 0; m < 4; ++m)
                    o[m] = (x & 1) ? ((a[2 * m][p] >> 16) | (a[2 * m + 1][p] & 0xffff0000u))
                                   : ((a[2 * m][p] & 0xffffu) | (a[2 * m + 1][p] << 16));
                *(uint4*) (vs + (vp * 8 + x) * VS + vk) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        if (j0 + KT < s1) fetch(j0 + KT);
        // S^T for the wave's 16 keys (rows) and the 16 queries (columns): lane (m16, hi) gets keys w 16 + k0 + i of query m16
        f8 s = f8{};
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            const h8 ka = *(const h8*) (ks + (w * 16 + m16) * KS + kk * 16 + k0);
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qa[kk], s);
            if (PREC) s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, ql[PREC ? kk : 0], s);
        }
        const long long jb = j0 + w * 16 + k0;   // (the lane's first key)
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const long long j = jb + i;
            float v = -INFINITY;
            if (qok && j < s1 && j >= lo && j <= pf) {
                const long long d = pf - j;
                v = __fmul_rn(s[i], scale);   // (the roundings spelled out)
                if (d < extent) {
                    const float* bb = rel_bias + (size_t) d * n_rel;
                    float bs = 0.0f;
                    for (int u = 0; u < n_rel; ++u) bs += rr[u] * bb[u];
                    v = __fadd_rn(v, bs);
                }
                v = __fsub_rn(v, rs);
            }
            s[i] = v;
            mx = fmaxf(mx, v);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        // (every lane runs the WMMAs: a row with no key seen yet keeps its state exactly - factor 1, probabilities 0)
        const float m_new = fmaxf(m_run, mx);
        const float tsc = m_new == -INFINITY ? 1.0f : (m_run > -INFINITY ? expf(m_run - m_new) : 0.0f);
        h8 pb, pl;
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float p = s[i] > -INFINITY ? expf(s[i] - m_new) : 0.0f;
            sum += p;
            pb[i] = (_Float16) p;
            if (PREC) pl[i] = (_Float16) (p - (float) pb[i]);
        }
        sum += __shfl_xor(sum, 16, 32);
        den = den * tsc + sum;
        m_run = m_new;
#pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) acc[nt] *= tsc;
#pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) {
            const h8 va = *(const h8*) (vs + (nt * 16 + m16) * VS + w * 16 + k0);
            acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, pb, acc[nt]);
            if (PREC) acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, pl, acc[nt]);
        }
        __syncthreads();
    }
    // the waves merged (wave order): M = max m_w, sum = sum den_w e^(m_w - M), part = sum acc_w e^(m_w - M)
    float* mw = (float*) smem;   // [4 waves][16 queries] maxima, then denominators
    float* aw = mw + 128;        // [4 waves][16 queries][D]: 8,320 floats in all, of the tiles' 8,960
    if (hi == 0) {
        mw[w * 16 + m16] = m_run;
        mw[64 + w * 16 + m16] = den;
    }
#pragma unroll
    for (int nt = 0; nt < D / 16; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) aw[((size_t) w * 16 + m16) * D + nt * 16 + k0 + i] = acc[nt][i];
    __syncthreads();
    for (int e = lt; e < 16 * D; e += 128) {   // (a thread: a query row and a dimension)
        const int qi = e / D, dd = e % D, row = blockIdx.z * 16 + qi;
        if (row >= TP) continue;
        float M = -INFINITY;
        for (int u = 0; u < 4; ++u) M = fmaxf(M, mw[u * 16 + qi]);
        float v = 0.0f, sm = 0.0f;
        if (M > -INFINITY)
            for (int u = 0; u < 4; ++u) {
                const float mu = mw[u * 16 + qi];
                const float f = mu > -INFINITY ? expf(mu - M) : 0.0f;
                v += aw[((size_t) u * 16 + qi) * D + dd] * f;
                sm += mw[64 + u * 16 + qi] * f;
            }
        const int tt = row / per, hh = g * per + row % per;
        const size_t ix = ((size_t) tt * H + hh) * nsplit + b;
        part[ix * D + dd] = v;
        if (dd == 0) {
            st[ix * 2] = M;
            st[ix * 2 + 1] = sm;
        }
    }
#endif
}
unsigned blocks_of(size_t n, int per = 256) { return (unsigned) ((n + (size_t) per - 1) / (size_t) per); }

}  // namespace

void mix(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, float eps, float* stat,
         float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3) {
        std::fprintf(stderr, "maya-ik2s mix: %d states, %d queries\n", nb + 1, nq);
        g_err.fetch_add(1);
        return;
    }
    mix_stat_kernel<<<(unsigned) (nb + 1), 256, 0, s>>>(blocks, nb, cur, E, query, nq, eps, stat);
    mix_out_kernel<<<dim3(blocks_of((size_t) E), (unsigned) nq), 256, 0, s>>>(blocks, nb, cur, E, nq, stat, out);
    check("mix");
}
void rms(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s) {
    rms_kernel<<<(unsigned) rows, 256, 0, s>>>(x, ldx, n, w, eps, y, ldy);
    check("rms");
}
void conv(float* y, float* hist, const float* w, int C, const float* dyn, int heads, cudaStream_t s) {
    conv_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, hist, w, C, dyn, heads, 1.0f / std::sqrt(dyn ? 9.0f : 5.0f));
    check("conv");
}
void head_clip(float* x, int H, int n, const float* scale, float thr, float mul, cudaStream_t s) {
    if (n > 256) {
        std::fprintf(stderr, "maya-ik2s head_clip: rows of %d\n", n);
        g_err.fetch_add(1);
        return;
    }
    head_clip_kernel<<<(unsigned) H, 256, 0, s>>>(x, n, scale, thr, mul);
    check("head_clip");
}
void attention(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
               const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p, int64_t lo,
               float scale, float* S, float* out, cudaStream_t s) {
    const int nk = (int) (p - lo + 1);
    const int per = H / Hkv;
    if (per <= 8 && (size_t) (kTile + per) * D * 4 <= 40000)
        attn_score_tile_kernel<<<dim3((unsigned) ((nk + kTile - 1) / kTile), (unsigned) Hkv), 256,
                                 (size_t) (kTile + per) * D * sizeof(float), s>>>(q, rel, sink, H, Hkv, D, n_rel, rel_bias,
                                                                                  extent, (const __half*) K, ring, (long long) p,
                                                                                  (long long) lo, nk, scale, S);
    else
        attn_score_kernel<<<blocks_of((size_t) H * nk), 256, 0, s>>>(q, rel, sink, H, Hkv, D, n_rel, rel_bias, extent,
                                                                      (const __half*) K, ring, (long long) p, (long long) lo, nk, scale, S);
    // (the softmax and the output in a launch while the keys are few: measured, 512 decoded tokens at a short context
    //  34.8 tok/s against 32.5 - and 27.3 against 28.1 after a 2.7k prompt, where the window is full: the three
    //  launches then, as before)
    if (per * 1024 <= 6144 && 1024 % D == 0 && nk <= 640) {
        attn_soft_out_kernel<<<(unsigned) Hkv, 1024, 0, s>>>(S, H, Hkv, D, (const __half*) V, ring, (long long) lo, nk, out);
        check("attention");
        return;
    }
    attn_softmax_kernel<<<(unsigned) H, 256, 0, s>>>(S, nk);
    const int nseg = (nk + kSeg - 1) / kSeg;
    float* part = S + (size_t) H * nk;
    attn_part_kernel<<<dim3((unsigned) nseg, (unsigned) Hkv), 256, 0, s>>>(S, H, Hkv, D, (const __half*) V, ring, (long long) lo, nk, nseg, part);
    attn_sum_kernel<<<(unsigned) H, 256, 0, s>>>(part, D, nseg, out);
    check("attention");
}
namespace {
// keys a tile.  Measured at a full 1024-key window (ms of attention a token over the 48 layers): 64 keys 11.2, 32
// keys 8.9, 16 keys 6.4 - the smaller the tile the faster, against what its work would say.  IK2_HIP_ATTN_TK=a,b
// (tests): a for up to 2048 keys, b beyond.
int split_tk(int nk) {
    static int t1 = 16, t2 = 16;
    static const bool init = [] {
        if (const char* e = std::getenv("IK2_HIP_ATTN_TK")) {
            int a = 0, b2 = 0;
            if (std::sscanf(e, "%d,%d", &a, &b2) == 2 && a >= 8 && a <= 64 && b2 >= 16 && b2 <= 64) t1 = a, t2 = b2;
        }
        return true;
    }();
    (void) init;
    return nk <= 2048 ? t1 : t2;
}
}  // namespace
size_t attention_split_floats(int H, int D, int64_t max_keys) {
    const int64_t nk = std::max<int64_t>(max_keys, 2048);
    // (at least 256 tiles: IK2_HIP_ATTN_TK's smallest tile, 8, over 2048 keys)
    const int64_t tiles = std::max<int64_t>((2048 + 7) / 8, (nk + 15) / 16);
    return (size_t) H * (size_t) tiles * (size_t) (D + 2);
}
void attention_split(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
                     const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p,
                     int64_t lo, float scale, float* S, const float* gate, float* out, cudaStream_t s) {
    const int nk = (int) (p - lo + 1), per = H / Hkv, TK = split_tk(nk), ntile = (nk + TK - 1) / TK;
    if (per > 8 || D > 256) {
        std::fprintf(stderr, "maya-ik2s attention_split: %d query heads a kv head, rows of %d\n", per, D);
        g_err.fetch_add(1);
        return;
    }
    float* part = S + (size_t) H * ntile * 2;
    attn_tile_kernel<<<dim3((unsigned) ntile, (unsigned) Hkv), 256,
                       ((size_t) TK * D + (size_t) per * D + (size_t) per * TK + 16) * sizeof(float), s>>>(
        q, rel, sink, H, Hkv, D, n_rel, rel_bias, extent, (const __half*) K, (const __half*) V, ring,
        (long long) (ring > 0 ? lo % ring : lo), (int) (p - lo), nk, TK, scale, S, part);
    attn_combine_kernel<<<(unsigned) H, 256, 0, s>>>(S, part, ntile, D, gate, out);
    check("attention_split");
}
bool attention_split_rows(const float* q, const float* rel, const float* sink, int T, int H, int Hkv, int D, int n_rel,
                          const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p0,
                          int64_t lo0, int window, float scale, float* S, size_t S_floats, const float* gate, float* out,
                          cudaStream_t s) {
    // (the tiles at multiples of TK positions, whatever the first key is: a row's tiles hold the same keys in a token's
    //  launch and in a window's, the keys it does not see masked - with the combine below, the same sums to the bit)
    const int TK = kATK;
    lo0 -= lo0 % TK;
    const int nk = (int) (p0 + T - lo0), per = H / Hkv, ntile = (nk + TK - 1) / TK, TP = T * per;
    if (TP > 256 || D > 256 || (size_t) T * H * ntile * (size_t) (D + 2) > S_floats) return false;
    float* part = S + (size_t) T * H * ntile * 2;
    const size_t shb = ((size_t) TK * D + (size_t) TP * TK + (size_t) TP + 16) * sizeof(float);
    const dim3 grid((unsigned) ntile, (unsigned) Hkv);
    const long long base = (long long) (ring > 0 ? lo0 % ring : lo0);
    if (T <= 4)
        attn_tile_rows_kernel<4><<<grid, 256, shb, s>>>(q, rel, sink, T, H, Hkv, D, n_rel, rel_bias, extent, (const __half*) K,
                                                        (const __half*) V, ring, base, (int) (p0 - lo0), window, nk, TK, scale,
                                                        S, part);
    else
        attn_tile_rows_kernel<8><<<grid, 256, shb, s>>>(q, rel, sink, T, H, Hkv, D, n_rel, rel_bias, extent, (const __half*) K,
                                                        (const __half*) V, ring, base, (int) (p0 - lo0), window, nk, TK, scale,
                                                        S, part);
    if (ntile <= kCombTiles)
        attn_combine_seq_kernel<<<(unsigned) (T * H), 256, (256 + (size_t) ntile) * sizeof(float), s>>>(S, part, ntile, D, gate, out);
    else   // (more factors than a block's shared memory holds: a chunk of them at a time)
        attn_combine_seq_big_kernel<<<(unsigned) (T * H), 256, 0, s>>>(S, part, ntile, D, gate, out);
    check("attention_split_rows");
    return true;
}
int attention_chunk_keys() {
    static const int ck = [] {
        const char* e = std::getenv("IK2_HIP_ATTN_CK");
        const int v = e ? std::atoi(e) : 64;
        return v == 32 || v == 64 || v == 128 ? v : 64;
    }();
    return ck;
}
bool attention_mma_rows(const float* q, const float* rel, const float* sink, int T, int H, int Hkv, int D, int n_rel,
                        const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p0,
                        int64_t lo0, int window, float scale, float* S, size_t S_floats, const float* gate, float* out,
                        cudaStream_t s, int8_t* q8, float* d8, int kvq) {
#if defined(MAYA_USE_HIP)
    if (D != fl::D || H % Hkv || (q8 && D % 32)) return false;
    // (the queries and probabilities in two F16 halves each unless IK2_HIP_ATTN_MMA_PREC=0: measured, 1-6% of the kernel
    //  for outputs 100x closer to the chunks' - max |d| / rms ~1e-5 against 1.2e-3 - so only the sums' order differs)
    static const bool prec = !(std::getenv("IK2_HIP_ATTN_MMA_PREC") && std::atoi(std::getenv("IK2_HIP_ATTN_MMA_PREC")) == 0);
    static const int SK = [] {   // (IK2_HIP_ATTN_MMA_SK: keys a split, a multiple of 64; 512 by default)
        int v = std::getenv("IK2_HIP_ATTN_MMA_SK") ? std::atoi(std::getenv("IK2_HIP_ATTN_MMA_SK")) : 512;
        return std::max(64, v / 64 * 64);
    }();
    lo0 -= lo0 % SK;
    const int nk = (int) (p0 + T - lo0), per = H / Hkv, nsplit = (nk + SK - 1) / SK, TP = T * per;
    if ((size_t) T * H * nsplit * (size_t) (D + 2) > S_floats || nsplit > 65535) return false;
    float* part = S + (size_t) T * H * nsplit * 2;
    const dim3 grid((unsigned) Hkv, (unsigned) nsplit, (unsigned) ((TP + 15) / 16));   // (the heads of a split together)
#define IK2_MMA(q8_, pr_)                                                                                               \
    attn_mma_rows_kernel<q8_, pr_><<<grid, 128, 0, s>>>(q, rel, sink, T, H, Hkv, n_rel, rel_bias, extent, K, V, ring,     \
                                                        (long long) lo0, (long long) p0, window, SK, scale, S, part)
    if (kvq && prec) IK2_MMA(true, true);
    else if (kvq) IK2_MMA(true, false);
    else if (prec) IK2_MMA(false, true);
    else IK2_MMA(false, false);
#undef IK2_MMA
    if (nsplit <= kCombTiles)
        attn_combine_seq_kernel<<<(unsigned) (T * H), 256, (256 + (size_t) nsplit) * sizeof(float), s>>>(S, part, nsplit, D,
                                                                                                     gate, out, q8, d8);
    else
        attn_combine_seq_big_kernel<<<(unsigned) (T * H), 256, 0, s>>>(S, part, nsplit, D, gate, out, q8, d8);
    check("attention_mma_rows");
    return true;
#else
    (void) q, (void) rel, (void) sink, (void) T, (void) H, (void) Hkv, (void) D, (void) n_rel, (void) rel_bias;
    (void) extent, (void) K, (void) V, (void) ring, (void) p0, (void) lo0, (void) window, (void) scale, (void) S;
    (void) S_floats, (void) gate, (void) out, (void) s, (void) q8, (void) d8, (void) kvq;
    return false;
#endif
}
bool attention_chunk_rows(const float* q, const float* rel, const float* sink, int T, int H, int Hkv, int D, int n_rel,
                          const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p0,
                          int64_t lo0, int window, float scale, float* S, size_t S_floats, const float* gate, float* out,
                          cudaStream_t s, int8_t* q8, float* d8, int kvq) {
    if (q8 && D % 32) return false;
    if (kvq && D % 4) return false;
    const int CK = attention_chunk_keys();
    lo0 -= lo0 % CK;
    const int nk = (int) (p0 + T - lo0), per = H / Hkv, ntile = (nk + CK - 1) / CK, TP = T * per;
    if (TP > 256 || D > 256 || (D & 1) || (size_t) T * H * ntile * (size_t) (D + 2) > S_floats) return false;
    float* part = S + (size_t) T * H * ntile * 2;
    // rows a block (IK2_HIP_ATTN_RPB, default 24: four tokens' rows of a kv head; measured best of 6 / 12 / 24 / all with
    // 64-key chunks, tests/ik2_hip_attn_bench), the rest in more blocks
    int rpb = 24;
    if (const char* e = std::getenv("IK2_HIP_ATTN_RPB")) rpb = std::max(1, std::atoi(e));
    rpb = std::min(rpb, TP);
    const size_t shb = ((size_t) rpb * D + (size_t) rpb * CK + (size_t) ((rpb + 3) & ~3)) * sizeof(float) +
                       (size_t) CK * (D + 2) * sizeof(__half);
    if (shb > 64 * 1024) return false;
    const dim3 grid((unsigned) ntile, (unsigned) Hkv, (unsigned) ((TP + rpb - 1) / rpb));
    const long long base = (long long) (ring > 0 ? lo0 % ring : lo0);
    // rows a thread: enough (key, row-group) items for the block's 256 threads, the most rows that leaves
    // (IK2_HIP_ATTN_RU: 1, 2, 3 or 6 forces it, when it divides the rows)
    // (a block's rows: rpb, and the last block's the rest - RU must divide both)
    const int last = TP - (TP - 1) / rpb * rpb;
    int ru = 1;
    for (int c : { 6, 3, 2 })
        if (rpb % c == 0 && last % c == 0 && (rpb / c) * CK >= 256) { ru = c; break; }
    if (const char* e = std::getenv("IK2_HIP_ATTN_RU")) {
        const int v = std::atoi(e);
        if ((v == 1 || v == 2 || v == 3 || v == 6) && rpb % v == 0 && last % v == 0) ru = v;
    }
#define IK2_ATTN_CHUNK(ck, r)                                                                                           \
    if (kvq)                                                                                                            \
        attn_chunk_rows_kernel<ck, r, true><<<grid, 256, shb, s>>>(q, rel, sink, T, H, Hkv, D, n_rel, rel_bias, extent, \
                                                                  (const __half*) K, (const __half*) V, ring, base,     \
                                                                  (int) (p0 - lo0), window, nk, scale, rpb, S, part);   \
    else                                                                                                                \
        attn_chunk_rows_kernel<ck, r><<<grid, 256, shb, s>>>(q, rel, sink, T, H, Hkv, D, n_rel, rel_bias, extent,       \
                                                            (const __half*) K, (const __half*) V, ring, base,           \
                                                            (int) (p0 - lo0), window, nk, scale, rpb, S, part)
#define IK2_ATTN_RU(ck)                                                                                                 \
    switch (ru) {                                                                                                       \
        case 6: IK2_ATTN_CHUNK(ck, 6); break;                                                                           \
        case 3: IK2_ATTN_CHUNK(ck, 3); break;                                                                           \
        case 2: IK2_ATTN_CHUNK(ck, 2); break;                                                                           \
        default: IK2_ATTN_CHUNK(ck, 1); break;                                                                          \
    }
    if (CK == 32) { IK2_ATTN_RU(32) }
    else if (CK == 128) { IK2_ATTN_RU(128) }
    else { IK2_ATTN_RU(64) }
#undef IK2_ATTN_RU
#undef IK2_ATTN_CHUNK
    if (ntile <= kCombTiles)
        attn_combine_seq_kernel<<<(unsigned) (T * H), 256, (256 + (size_t) ntile) * sizeof(float), s>>>(S, part, ntile, D, gate, out,
                                                                                                    q8, d8);
    else
        attn_combine_seq_big_kernel<<<(unsigned) (T * H), 256, 0, s>>>(S, part, ntile, D, gate, out, q8, d8);
    check("attention_chunk_rows");
    return true;
}
void sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s) {
    sigmoid_mul_kernel<<<blocks_of(n), 256, 0, s>>>(x, g, n);
    check("sigmoid_mul");
}
void add(float* y, const float* x, size_t n, cudaStream_t s) {
    add_kernel<<<blocks_of(n), 256, 0, s>>>(y, x, n);
    check("add");
}
void situ(const float* gu, int rows, int ff, float beta, float lb, float* h, cudaStream_t s) {
    if (rows <= 0) return;
    situ_kernel<<<dim3(blocks_of((size_t) ff), (unsigned) rows), 256, 0, s>>>(gu, ff, beta, lb, h);
    check("situ");
}
void wsum(const float* part, int m, int n, const float* w, float* out, cudaStream_t s) {
    wsum_kernel<<<blocks_of((size_t) n), 256, 0, s>>>(part, m, n, w, out);
    check("wsum");
}
void softcap(float* x, size_t n, float cap, cudaStream_t s) {
    softcap_kernel<<<blocks_of(n), 256, 0, s>>>(x, n, cap);
    check("softcap");
}
void argmax(const float* x, int n, float* out, cudaStream_t s) {
    argmax_kernel<<<1, 256, 0, s>>>(x, n, out);
    check("argmax");
}
void mix_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
              float eps, float* stat, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3) {
        std::fprintf(stderr, "maya-ik2s mix: %d states, %d queries\n", nb + 1, nq);
        g_err.fetch_add(1);
        return;
    }
    mix_stat_rows_kernel<<<dim3((unsigned) (nb + 1), (unsigned) T), 256, 0, s>>>(blocks, ldb, nb, cur, E, query, nq, eps, stat);
    mix_out_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) T, (unsigned) nq), 256, 0, s>>>(blocks, ldb, nb, cur, T, E,
                                                                                              nq, stat, out);
    check("mix_rows");
}
void conv_rows(const float* y, int T, float* hist, const float* w, int C, const float* dyn, int heads, float* out,
               cudaStream_t s) {
    conv_rows_kernel<<<dim3(blocks_of((size_t) C), (unsigned) T), 256, 0, s>>>(y, hist, w, C, dyn, heads,
                                                                                1.0f / std::sqrt(dyn ? 9.0f : 5.0f), out);
    conv_hist_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, T, hist, C, nullptr);
    check("conv_rows");
}
void head_clip_rows(float* x, int T, int H, int n, const float* scale, float thr, float alpha, double floor, int64_t p0,
                    cudaStream_t s) {
    if (n > 256) {
        g_err.fetch_add(1);
        return;
    }
    head_clip_rows_kernel<<<(unsigned) (T * H), 256, 0, s>>>(x, H, n, scale, thr, alpha, (float) (1.0 / floor), (long long) p0);
    check("head_clip_rows");
}
__global__ void kv8_f16_kernel(const void* cache, long long slot0, int n, int Hkv, int D, uint4* out) {
    const int per = Hkv * D / 8;   // (eight values a thread)
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * per) return;
    const int j = (int) (i / per), r = (int) (i % per), g = r / (D / 8), c = (r % (D / 8)) * 8;
    out[i] = kv8_get8(cache, (size_t) (slot0 + j), Hkv, D, g, c);
}
void kv8_f16(const uint16_t* cache, int64_t slot0, int n, int Hkv, int D, uint16_t* out, cudaStream_t s) {
    if (n <= 0) return;
    if (D % 8) {
        g_err.fetch_add(1);
        return;
    }
    const size_t items = (size_t) n * Hkv * D / 8;
    kv8_f16_kernel<<<(unsigned) ((items + 255) / 256), 256, 0, s>>>(cache, (long long) slot0, n, Hkv, D, (uint4*) out);
    check("kv8_f16");
}
void store_rows(const float* src, int T, int n, uint16_t* cache, int ring, int64_t p0, cudaStream_t s, int kvq, int Hkv) {
    if (kvq) {   // (n = Hkv D values a row, D <= 256)
        const int D = n / Hkv;
        if (D > 256 || D * Hkv != n) {
            g_err.fetch_add(1);
            return;
        }
        store_rows8_kernel<<<dim3((unsigned) Hkv, (unsigned) T), 256, 0, s>>>(src, Hkv, D, cache, ring, (long long) p0);
        check("store_rows8");
        return;
    }
    store_rows_kernel<<<dim3(blocks_of((size_t) n), (unsigned) T), 256, 0, s>>>(src, n, (__half*) cache, ring, (long long) p0);
    check("store_rows");
}
void perm_heads(const float* src, float* dst, int T, int Hkv, int per, int D, bool to_p, cudaStream_t s) {
    perm_heads_kernel<<<blocks_of((size_t) T * Hkv * per * D), 256, 0, s>>>(src, dst, T, Hkv, per, D, to_p ? 1 : 0);
    check("perm_heads");
}
void attn_softmax_rows(float* S, int T, int Hkv, int per, int nk, int64_t klo, int64_t p0, int window, int extent,
                       const float* tbl, const float* sink, float scale, cudaStream_t s) {
    attn_softmax_rows_kernel<<<(unsigned) (T * Hkv * per), 256, 0, s>>>(S, T, Hkv, per, nk, (long long) klo, (long long) p0,
                                                                         window, extent, tbl, sink, scale);
    check("attn_softmax_rows");
}
bool attn_flash(const uint16_t* qh, int Q, int per, int Hkv, int D, const uint16_t* K, const uint16_t* V, int ring,
                int64_t klo, int nk, int64_t pf, int window, int extent, const float* tbl, const float* sink, float scale,
                float* out, cudaStream_t s, int kvq) {
#if defined(MAYA_USE_HIP)
    if (D != fl::D) return false;
    static const int ver = std::getenv("IK2_HIP_FLASH_V") ? std::atoi(std::getenv("IK2_HIP_FLASH_V")) : 3;
    if (kvq)   // (the 8-bit cache: the 8-wave form, whatever IK2_HIP_FLASH_V says)
        attn_flash2_kernel<8, true><<<dim3((unsigned) ((Q + 127) / 128), (unsigned) Hkv), 256, 0, s>>>(
            qh, Q, per, Hkv, K, V, ring, (long long) klo, nk, (long long) pf, window, extent, tbl, sink, scale, out);
    else if (ver == 2)
        attn_flash2_kernel<4><<<dim3((unsigned) ((Q + 63) / 64), (unsigned) Hkv), 128, 0, s>>>(
            qh, Q, per, Hkv, K, V, ring, (long long) klo, nk, (long long) pf, window, extent, tbl, sink, scale, out);
    else if (ver == 3)
        attn_flash2_kernel<8><<<dim3((unsigned) ((Q + 127) / 128), (unsigned) Hkv), 256, 0, s>>>(
            qh, Q, per, Hkv, K, V, ring, (long long) klo, nk, (long long) pf, window, extent, tbl, sink, scale, out);
    else
        attn_flash_kernel<<<dim3((unsigned) ((Q + fl::RB - 1) / fl::RB), (unsigned) Hkv), 128, 0, s>>>(
            qh, Q, per, Hkv, K, V, ring, (long long) klo, nk, (long long) pf, window, extent, tbl, sink, scale, out);
    check("attn_flash");
    return true;
#else
    (void) qh, (void) Q, (void) per, (void) Hkv, (void) D, (void) K, (void) V, (void) ring, (void) klo, (void) nk;
    (void) pf, (void) window, (void) extent, (void) tbl, (void) sink, (void) scale, (void) out, (void) s, (void) kvq;
    return false;
#endif
}
void attn_softmax_p16(float* S, int T, int Hkv, int per, int nk, int64_t klo, int64_t p0, int window, int extent,
                      const float* tbl, const float* sink, float scale, uint16_t* P, cudaStream_t s, const float* relq,
                      const float* rel_bias, int d_rel) {
    if (relq != nullptr && extent > 1024) {
        std::fprintf(stderr, "maya-ik2s attn_softmax_p16: an extent of %d\n", extent);
        g_err.fetch_add(1);
        return;
    }
    attn_softmax_p16_kernel<<<(unsigned) (T * Hkv * per), 256, 0, s>>>(S, T, Hkv, per, nk, (long long) klo, (long long) p0,
                                                                        window, extent, tbl, sink, scale, (__half*) P,
                                                                        relq, rel_bias, d_rel);
    check("attn_softmax_p16");
}
void kv_finish(float* ky, float* vy, float* hist_k, float* hist_v, const float* k_conv, const float* v_conv,
               const float* dk, const float* dv, int Hkv, int D, float thr, float eps, uint16_t* K, uint16_t* V, int ring,
               int64_t p, cudaStream_t s, int kvq) {
    if (D > 256) {
        g_err.fetch_add(1);
        return;
    }
    kv_finish_kernel<<<dim3((unsigned) Hkv, 2), 256, 0, s>>>(ky, vy, hist_k, hist_v, k_conv, v_conv, dk, dv, Hkv, D, thr, eps,
                                                             (__half*) K, (__half*) V, (size_t) (ring > 0 ? p % ring : p), kvq);
    check("kv_finish");
}
void clip2(float* q, int H, int D, const float* qs, float qthr, float* rel, int n_rel, const float* rs, float rthr,
           float mul, cudaStream_t s) {
    if (D > 256 || n_rel > 256) {
        g_err.fetch_add(1);
        return;
    }
    clip2_kernel<<<(unsigned) H, 256, 0, s>>>(q, D, qs, qthr, rel, n_rel, rs, rthr, mul);
    check("clip2");
}
void conv_add(float* y, float* hist, const float* w, int C, float* cur, cudaStream_t s) {
    conv_add_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, hist, w, C, cur);
    check("conv_add");
}
void scatter_rows(const float* y, const int* row_of, int n, int E, float* ent, cudaStream_t s) {
    for (int e0 = 0; e0 < n; e0 += 32768) {   // (a grid's rows at most)
        const int ne = std::min(32768, n - e0);
        scatter_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) ne), 256, 0, s>>>(y, row_of + e0, E, ent + (size_t) e0 * E);
    }
    check("scatter_rows");
}
bool moe_sort_bits(const int* ids, int T, int nu, const int* slot, int n_slots, unsigned* bits, int* wpre, int* bounds,
                   int* tok, int* row_of, int* ent_of, cudaStream_t s) {
    if (n_slots < 0 || n_slots > 1024) return false;
    const int W = (T + 31) / 32, n = T * nu;
    if (n_slots > 0) {
        if (cudaMemsetAsync(bits, 0, (size_t) n_slots * W * sizeof(unsigned), s) != cudaSuccess) return false;
        if (n > 0) sort_bits_set_kernel<<<blocks_of((size_t) n), 256, 0, s>>>(ids, n, nu, slot, n_slots, W, bits);
    }
    sort_bits_rows_kernel<<<1, 1024, 0, s>>>(ids, n, nu, slot, n_slots, W, bits, wpre, bounds, tok, row_of, ent_of);
    check("moe_sort_bits");
    return true;
}
void scatter_set_rows(const float* y, const int* ent_of, int n_rows, int E, float* ent, cudaStream_t s) {
    for (int r0 = 0; r0 < n_rows; r0 += 32768) {   // (a grid's rows at most)
        const int nr = std::min(32768, n_rows - r0);
        scatter_set_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) nr), 256, 0, s>>>(y + (size_t) r0 * E, ent_of + r0,
                                                                                         E, ent);
    }
    check("scatter_set_rows");
}
void combine_rows(const float* ent, const float* w, int T, int nu, int E, float* out, cudaStream_t s) {
    combine_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) T), 256, 0, s>>>(ent, w, nu, E, out);
    check("combine_rows");
}
void mix_rms(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, const float* g0,
             const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024) {
        std::fprintf(stderr, "maya-ik2s mix_rms: %d states, %d queries, width %d\n", nb + 1, nq, E);
        g_err.fetch_add(1);
        return;
    }
    mix_rms_kernel<<<1, 1024, 0, s>>>(blocks, (size_t) E, nb, cur, 1, E, query, nq, g0, g1, g2, eps, out);
    check("mix_rms");
}
void mix_rms_rows_form(int form, const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query,
                       int nq, const float* g0, const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024) {
        g_err.fetch_add(1);
        return;
    }
    if (form == 0) mix_rms_kernel<<<(unsigned) T, 1024, 0, s>>>(blocks, ldb, nb, cur, T, E, query, nq, g0, g1, g2, eps, out);
    else mix_rms2_kernel<<<(unsigned) T, 1024, 0, s>>>(blocks, ldb, nb, cur, T, E, query, nq, g0, g1, g2, eps, out);
    check("mix_rms_rows");
}
void mix_rms_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
                  const float* g0, const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    static const bool old = std::getenv("IK2_HIP_MIX_OLD") != nullptr;   // (the barrier-per-sum form, as before)
    mix_rms_rows_form(old ? 0 : 1, blocks, ldb, nb, cur, T, E, query, nq, g0, g1, g2, eps, out, s);
}
void conv_rows_save(const float* y, int T, float* hist, const float* w, int C, const float* dyn, int heads, float* out,
                    float* save, cudaStream_t s) {
    conv_rows_kernel<<<dim3(blocks_of((size_t) C), (unsigned) T), 256, 0, s>>>(y, hist, w, C, dyn, heads,
                                                                                1.0f / std::sqrt(dyn ? 9.0f : 5.0f), out);
    conv_hist_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, T, hist, C, save);
    check("conv_rows_save");
}
void lane_add_rows(const unsigned* flag, unsigned v, const int* ids, const int* slot, int nu, const float* src, float* y,
                   int n, int T, cudaStream_t s) {
    // (the tokens' picks as one list: any outside VRAM and the wait is taken; rows of n are contiguous)
    lane_add_kernel<<<blocks_of((size_t) n * T), 256, 0, s>>>(flag, v, ids, slot, nu * T, src, y, n * T);
    check("lane_add_rows");
}
void hist_set4(float* const hist[4], const float* const h0[4], const float* const in[4], const int C[4], int keep,
               cudaStream_t s) {
    const int cm = std::max(std::max(C[0], C[1]), std::max(C[2], C[3]));
    hist_set4_kernel<<<blocks_of((size_t) cm), 256, 0, s>>>(hist[0], hist[1], hist[2], hist[3], h0[0], h0[1], h0[2], h0[3],
                                                            in[0], in[1], in[2], in[3], C[0], C[1], C[2], C[3], keep);
    check("hist_set4");
}
void lane_add(const unsigned* flag, unsigned v, const int* ids, const int* slot, int nu, const float* src, float* y, int n,
              cudaStream_t s) {
    lane_add_kernel<<<blocks_of((size_t) n), 256, 0, s>>>(flag, v, ids, slot, nu, src, y, n);
    check("lane_add");
}
int attention_rows_fit(int H, int D, int64_t keys, size_t S_floats) {
    const int64_t ntile = (keys + 15 + 15) / 16;   // (the first tile's start rounded down)
    return (int) std::min<int64_t>(256 / std::max(1, H / 8), (int64_t) (S_floats / ((size_t) H * (size_t) ntile * (size_t) (D + 2))));
}
void kv_finish_rows(const float* in_k, const float* in_v, int T, const float* hk0, const float* hv0, float* hk1, float* hv1,
                    const float* k_conv, const float* v_conv, const float* dk, const float* dv, int Hkv, int D, float thr,
                    float eps, uint16_t* K, uint16_t* V, int ring, int64_t p0, cudaStream_t s, int kvq) {
    if (D > 256) {
        g_err.fetch_add(1);
        return;
    }
    kv_finish_rows_kernel<<<dim3((unsigned) Hkv, 2, (unsigned) T), 256, 0, s>>>(in_k, in_v, hk0, hv0, hk1, hv1, k_conv, v_conv,
                                                                              dk, dv, T, Hkv, D, thr, eps, (__half*) K,
                                                                              (__half*) V, ring, (long long) p0, kvq);
    check("kv_finish_rows");
}
void clip2_rows(float* q, int T, int H, int D, const float* qs, float qthr, float* rel, int n_rel, const float* rs,
                float rthr, const float* mul, cudaStream_t s) {
    if (D > 256 || n_rel > 256 || T > 16) {
        g_err.fetch_add(1);
        return;
    }
    RowMul m;
    for (int t = 0; t < 16; ++t) m.m[t] = t < T ? mul[t] : 1.0f;
    clip2_rows_kernel<<<dim3((unsigned) H, (unsigned) T), 256, 0, s>>>(q, H, D, qs, qthr, rel, n_rel, rs, rthr, m);
    check("clip2_rows");
}
void conv_add_rows(const float* y, int T, const float* h0, float* h1, const float* w, int C, float* cur, float* blk,
                   cudaStream_t s) {
    conv_add_rows_kernel<<<dim3(blocks_of((size_t) C), (unsigned) T), 256, 0, s>>>(y, h0, h1, w, C, T, cur, blk);
    check("conv_add_rows");
}
void taps_gather(const float* const* src, unsigned vol, int n, int T, int E, float* dst, size_t ld, cudaStream_t s) {
    if (n > 8) {
        g_err.fetch_add(1);
        return;
    }
    TapSrc ts;
    for (int k = 0; k < 8; ++k) ts.p[k] = k < n ? src[k] : nullptr;
    ts.vol = vol;
    taps_gather_kernel<<<blocks_of((size_t) T * n * E), 256, 0, s>>>(ts, n, T, E, dst, ld);
    check("taps_gather");
}
void bias_rows(float* x, size_t ld, int rows, const int* ids, const float* vals, int n, cudaStream_t s) {
    if (n <= 0 || rows <= 0) return;
    bias_rows_kernel<<<dim3(blocks_of((size_t) n), (unsigned) rows), 256, 0, s>>>(x, ld, ids, vals, n);
    check("bias_rows");
}
void argmax_rows(const float* x, size_t ld, int rows, int n, float* out, cudaStream_t s) {
    argmax_rows_kernel<<<(unsigned) rows, 1024, 0, s>>>(x, ld, n, out);
    check("argmax_rows");
}
void mix_rms_rows_fused(const float* blocks, size_t ldb, int nb, float* cur, int T, int E, const float* query, int nq,
                        const float* g0, const float* g1, const float* g2, float eps, float* out, const float* cy,
                        const float* ch0, float* ch1, const float* cw, float* cblk, int8_t* q, float* d, bool dup,
                        cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024 || E % 32 || (dup && nq != 1)) {
        g_err.fetch_add(1);
        return;
    }
    const MixConv cv{cy, ch0, ch1, cw, cblk};
    const MixQ8 qq{q, d, dup ? 1 : 0};
    mix_rms3_kernel<<<(unsigned) T, 1024, 0, s>>>(blocks, ldb, nb, cur, T, E, query, nq, g0, g1, g2, eps, out, cv, qq);
    check("mix_rms_rows_fused");
}
void route_publish(const float* logits, int T, int n_exp, const float* bias, int n_used, bool norm, float scale, int* ids,
                   float* w, const float* x, size_t n, float* dst, unsigned* counter, unsigned* flag, unsigned v,
                   cudaStream_t s, const uint8_t* base, size_t stride, unsigned long long* plan, const int* slot,
                   unsigned long long* grp_ptr, int* grp_start, int* grp_ent, int* n_groups) {
    if (n_exp > 512 || n_used > 16 || n % 4 || ((n + 2 * (size_t) T * n_used) % 4) || (plan && T != 1) ||
        (grp_ptr && (plan || T * n_used > 256))) {
        g_err.fetch_add(1);
        return;
    }
    const size_t n4 = n / 4;
    const unsigned nc = (unsigned) std::max<size_t>(1, std::min<size_t>(16, (n4 + 511) / 512));
    route_pub_kernel<<<(unsigned) T + nc, 512, 0, s>>>(logits, n_exp, bias, n_used, norm ? 1 : 0, scale, ids, w, T,
                                                       (const float4*) x, n4, dst, counter, flag, v, base, stride, plan,
                                                       slot, FzGroup{grp_ptr, grp_start, grp_ent, n_groups});
    check("route_publish");
}
void kv_clip_rows(const float* in_k, const float* in_v, int T, const float* hk0, const float* hv0, float* hk1, float* hv1,
                  const float* k_conv, const float* v_conv, const float* dk, const float* dv, int Hkv, int D, float thr,
                  float eps, uint16_t* K, uint16_t* V, int ring, int64_t p0, float* q, int H, const float* qs,
                  float qthr, float* rel, int n_rel, const float* rs, float rthr, const float* mul, cudaStream_t s,
                  const float* sink_w, const float* sink_x, float* sink_y, int ns, int k, const float* kp_w,
                  const float* vp_w, const float* kp_x, const float* vp_x, float* dk_out, float* dv_out, int kvq) {
    if (D > 256 || n_rel > 256 || T > 16 || (kp_w && (dk == nullptr || dk != dk_out || dv != dv_out))) {
        g_err.fetch_add(1);
        return;
    }
    RowMul m;
    for (int t = 0; t < 16; ++t) m.m[t] = t < T ? mul[t] : 1.0f;
    const KvProj pj{sink_w, sink_x, sink_y, sink_w ? ns : 0, k, kp_w, vp_w, kp_x, vp_x, dk_out, dv_out};
    kv_clip_rows_kernel<<<dim3((unsigned) (2 * Hkv + H + pj.ns), (unsigned) T), 256, 0, s>>>(
        in_k, in_v, hk0, hv0, hk1, hv1, k_conv, v_conv, dk, dv, T, Hkv, D, thr, eps, (__half*) K, (__half*) V, ring,
        (long long) p0, q, H, qs, qthr, rel, n_rel, rs, rthr, m, pj, kvq);
    check("kv_clip_rows");
}
void moe_lane_rows(const float* part, int T, int n_used, int n, const float* w, float* out, int stride16,
                   const unsigned* flag, unsigned v, const int* ids, const int* slot, const float* src, cudaStream_t s) {
    moe_lane_kernel<<<dim3(blocks_of((size_t) n), (unsigned) T), 256, 0, s>>>(part, n_used, n, w, out, stride16, flag, v, ids,
                                                                            slot, n_used * T, src);
    check("moe_lane_rows");
}
bool gemv_f32_rows(int rpb, const float* W, size_t ld, int k, int n, const float* x, size_t ldx, int T, float* y, size_t ldy,
                   cudaStream_t s) {
    if (T < 1 || T > 16 || k % 128 || ld % 8 || ldx % 4 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    const unsigned nb = (unsigned) ((n + rpb - 1) / rpb);
#define IK2_GEMV_RPB(R)                                                                                                  \
    if (T == 1) gemv_1_rpb_kernel<R><<<nb, R * 32, 0, s>>>(W, ld, k, n, x, y);                                           \
    else if (T <= 4) gemv_s_rpb_kernel<4, R><<<dim3(nb, 1), R * 32, 0, s>>>(W, ld, k, n, x, ldx, T, y, ldy);              \
    else gemv_s_rpb_kernel<8, R><<<dim3(nb, (unsigned) ((T + 7) / 8)), R * 32, 0, s>>>(W, ld, k, n, x, ldx, T, y, ldy);
    switch (rpb) {
        case 1: IK2_GEMV_RPB(1) break;
        case 2: IK2_GEMV_RPB(2) break;
        case 4: IK2_GEMV_RPB(4) break;
        case 8: IK2_GEMV_RPB(8) break;
        default: return false;
    }
#undef IK2_GEMV_RPB
    check("gemv_f32_rows");
    return true;
}
int errors() { return g_err.exchange(0); }

// ---- (tests, IK2_HIP_KV_SIM) an 8-bit cache's rounding in place in the F16 rows ------------------------------------
__device__ inline float e4m3_round(float x) {   // (OCP E4M3 fn: round to nearest even, saturating at 448)
    const float a = fabsf(x);
    if (!(a > 0.0f)) return 0.0f;
    int e;
    frexpf(a, &e);   // (a = m 2^e, m in [0.5, 1): its binade 2^(e - 1); the subnormals' quantum 2^-9)
    const float qn = ldexpf(1.0f, max(e - 1, -6) - 3);
    return copysignf(fminf(rintf(a / qn) * qn, 448.0f), x);
}
__global__ void kv_sim_kernel(__half* K, __half* V, int ring, long long p0, int Hkv, int D, int mode) {
    __shared__ float sh[256];
    const int h = blockIdx.x, isv = blockIdx.y, t = blockIdx.z, i = threadIdx.x;
    const long long p = p0 + t;
    __half* row = (isv ? V : K) + (size_t) (ring > 0 ? p % ring : p) * Hkv * D + (size_t) h * D;
    const float x = __half2float(row[i]);
    sh[i] = fabsf(x);
    __syncthreads();
    const int span = mode == 2 ? 32 : D;
    for (int st = span / 2; st > 0; st >>= 1) {
        if (i % span < st) sh[i] = fmaxf(sh[i], sh[i + st]);
        __syncthreads();
    }
    const float am = sh[i / span * span];
    float y = x;
    if (mode == 1 || mode == 2) {
        float sc = am / 127.0f;
        if (mode == 2) sc = __half2float(__float2half_rn(sc));
        y = sc > 0.0f ? fminf(fmaxf(rintf(x / sc), -127.0f), 127.0f) * sc : 0.0f;
    } else if (mode == 3) {
        const float sc = am / 448.0f;
        y = sc > 0.0f ? e4m3_round(x / sc) * sc : 0.0f;
    } else if (mode == 4) {
        y = e4m3_round(x);
    }
    row[i] = __float2half_rn(y);
}
void kv_sim(uint16_t* K, uint16_t* V, int ring, int64_t p0, int T, int Hkv, int D, int mode, cudaStream_t s) {
    if (mode <= 0 || T <= 0) return;
    kv_sim_kernel<<<dim3((unsigned) Hkv, 2, (unsigned) T), (unsigned) D, 0, s>>>((__half*) K, (__half*) V, ring, (long long) p0,
                                                                                Hkv, D, mode);
    check("kv_sim");
}
}  // namespace ik2hip::k
