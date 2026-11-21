// src/inkling2/inkling2_kernels.cu - maya-inkling2's own kernels (inkling2_kernels.cuh).  Written for clarity first
// (the decode path's small vectors: one launch a step of the reference graph); the sums are F32 in a fixed order.
#include "inkling2_kernels.cuh"

#include <cuda_fp16.h>

#include "strata/kernels/fresh_load.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace inkling2::k {

namespace {

std::atomic<int> g_err{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "maya-inkling2 kernel %s: %s\n", what, cudaGetErrorString(e));
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
constexpr int kTile = 32, kSeg = 128;   // (a tile's shared memory against blocks: measurable with INKLING2_PHASES)
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
__global__ void attn_combine_seq_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                        const float* __restrict__ gate, float* __restrict__ out) {
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
}
// ... the same sums for more tiles than a block's shared memory holds factors for (contexts past kCombTiles * kATK
// keys: 131072): the tiles' factors a chunk of kCombTiles at a time, summed in the same order
constexpr int kCombTiles = 8192;   // (8192 floats = 32 KB of a block's shared memory)
__global__ void attn_combine_seq_big_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                            const float* __restrict__ gate, float* __restrict__ out) {
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
}
// ---- the window step's fused forms: the decode step's launches over T rows (T 1: a token) --------------------------
// block = (a kv head, K or V, a row): the row's conv from the window's inputs and the histories before it, the clip /
// the norm, the F16 cache row; the last row's blocks write the histories after the window into h1
__global__ void kv_finish_rows_kernel(const float* __restrict__ in_k, const float* __restrict__ in_v,
                                      const float* __restrict__ hk0, const float* __restrict__ hv0,
                                      float* __restrict__ hk1, float* __restrict__ hv1, const float* __restrict__ k_conv,
                                      const float* __restrict__ v_conv, const float* __restrict__ dk,
                                      const float* __restrict__ dv, int T, int Hkv, int D, float thr, float eps,
                                      __half* __restrict__ K, __half* __restrict__ V, int ring, long long p0) {
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
    if (i < D) {
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
            const unsigned long long t0 = strata::kernels::clock_ns();
            while ((int) (*(volatile const unsigned*) flag - v) < 0)
                if (strata::kernels::clock_ns() - t0 > 60ull * 1000000000ull) {   // (a minute: the lane is gone)
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
                                 __half* __restrict__ K, __half* __restrict__ V, size_t row) {
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
        (isv ? V : K)[row * C + c] = __float2half_rn(o);
    }
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

__global__ void scatter_rows_kernel(const float* __restrict__ y, const int* __restrict__ row_of, int E,
                                    float* __restrict__ ent) {
    const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x, r = row_of[e];
    if (i < E && r >= 0) ent[(size_t) e * E + i] = y[(size_t) r * E + i];
}
__global__ void combine_rows_kernel(const float* __restrict__ ent, const float* __restrict__ w, int nu, int E,
                                    float* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float v = 0.0f;
    for (int j = 0; j < nu; ++j) v = fmaf(w[(size_t) t * nu + j], ent[((size_t) t * nu + j) * E + i], v);
    out[(size_t) t * E + i] = v;
}

unsigned blocks_of(size_t n, int per = 256) { return (unsigned) ((n + (size_t) per - 1) / (size_t) per); }

}  // namespace

void mix(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, float eps, float* stat,
         float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3) {
        std::fprintf(stderr, "maya-inkling2 mix: %d states, %d queries\n", nb + 1, nq);
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
        std::fprintf(stderr, "maya-inkling2 head_clip: rows of %d\n", n);
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
// keys 8.9, 16 keys 6.4 - the smaller the tile the faster, against what its work would say.  INKLING2_ATTN_TK=a,b
// (tests): a for up to 2048 keys, b beyond.
int split_tk(int nk) {
    static int t1 = 16, t2 = 16;
    static const bool init = [] {
        if (const char* e = std::getenv("INKLING2_ATTN_TK")) {
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
    // (at least 256 tiles: INKLING2_ATTN_TK's smallest tile, 8, over 2048 keys)
    const int64_t tiles = std::max<int64_t>((2048 + 7) / 8, (nk + 15) / 16);
    return (size_t) H * (size_t) tiles * (size_t) (D + 2);
}
void attention_split(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
                     const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p,
                     int64_t lo, float scale, float* S, const float* gate, float* out, cudaStream_t s) {
    const int nk = (int) (p - lo + 1), per = H / Hkv, TK = split_tk(nk), ntile = (nk + TK - 1) / TK;
    if (per > 8 || D > 256) {
        std::fprintf(stderr, "maya-inkling2 attention_split: %d query heads a kv head, rows of %d\n", per, D);
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
        std::fprintf(stderr, "maya-inkling2 mix: %d states, %d queries\n", nb + 1, nq);
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
void store_rows(const float* src, int T, int n, uint16_t* cache, int ring, int64_t p0, cudaStream_t s) {
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
void kv_finish(float* ky, float* vy, float* hist_k, float* hist_v, const float* k_conv, const float* v_conv,
               const float* dk, const float* dv, int Hkv, int D, float thr, float eps, uint16_t* K, uint16_t* V, int ring,
               int64_t p, cudaStream_t s) {
    if (D > 256) {
        g_err.fetch_add(1);
        return;
    }
    kv_finish_kernel<<<dim3((unsigned) Hkv, 2), 256, 0, s>>>(ky, vy, hist_k, hist_v, k_conv, v_conv, dk, dv, Hkv, D, thr, eps,
                                                             (__half*) K, (__half*) V, (size_t) (ring > 0 ? p % ring : p));
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
void combine_rows(const float* ent, const float* w, int T, int nu, int E, float* out, cudaStream_t s) {
    combine_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) T), 256, 0, s>>>(ent, w, nu, E, out);
    check("combine_rows");
}
void mix_rms(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, const float* g0,
             const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024) {
        std::fprintf(stderr, "maya-inkling2 mix_rms: %d states, %d queries, width %d\n", nb + 1, nq, E);
        g_err.fetch_add(1);
        return;
    }
    mix_rms_kernel<<<1, 1024, 0, s>>>(blocks, (size_t) E, nb, cur, 1, E, query, nq, g0, g1, g2, eps, out);
    check("mix_rms");
}
void mix_rms_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
                  const float* g0, const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024) {
        g_err.fetch_add(1);
        return;
    }
    mix_rms_kernel<<<(unsigned) T, 1024, 0, s>>>(blocks, ldb, nb, cur, T, E, query, nq, g0, g1, g2, eps, out);
    check("mix_rms_rows");
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
                    float eps, uint16_t* K, uint16_t* V, int ring, int64_t p0, cudaStream_t s) {
    if (D > 256) {
        g_err.fetch_add(1);
        return;
    }
    kv_finish_rows_kernel<<<dim3((unsigned) Hkv, 2, (unsigned) T), 256, 0, s>>>(in_k, in_v, hk0, hv0, hk1, hv1, k_conv, v_conv,
                                                                              dk, dv, T, Hkv, D, thr, eps, (__half*) K,
                                                                              (__half*) V, ring, (long long) p0);
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
int errors() { return g_err.exchange(0); }

}  // namespace inkling2::k
