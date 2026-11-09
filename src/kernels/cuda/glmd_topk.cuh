// src/kernels/cuda/glmd_topk.cuh - the glm-dsa token-level top-k selector (B4.5), one 1024-thread block per row.
//
// GLM-5.3 keeps the 2048 best of up to max_ctx cells per query (Maya's dsa_topk.cuh is capped at 1024 pools and
// sorts its winners; this one has no cap on n or top and writes the winners in POSITION order, which the gather
// in mla2 and the shared-layer reuse want, so it needs no sort at all):
//   0. n <= top: the identity 0..n-1 then -1 (every visible cell is attended; scores are not read);
//   1. a radix select over the scores' order-preserving uint32 keys, 4 passes of 8 bits from the top, finds the
//      cut-off key (the top-th largest) and `need`, how many keys equal to it are taken;
//   2. each thread takes a contiguous run of cells, counts keys above the cut-off and keys equal to it; an
//      exclusive scan of the equal counts gives each thread its share of the `need` ties (the LOWER cells win, as
//      torch.topk's stable order and glmd::ref_topk choose), a second scan the output offsets;
//   3. each thread writes its winners in cell order.
// O(6n) global reads per row (L2-resident at 128K x 4 B); the multi-block version for > 32K is B10.4, the bucket
// (histogram) variant below (5 passes, B14.15, STRATA_GLMD_TOPK_HIST) writes the same cells.
// glmd::twin::topk (src/core/glm_dsa_cpu_fast_twin.cpp) is the same algorithm on the host.
#pragma once

#include "strata/kernels/glmd_fast.hpp"

#include <cuda_runtime.h>

namespace glmd::topk_detail {

constexpr int kThreads = 1024;

// a float's order-preserving key: larger float, larger key (-0 is +0; scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// inclusive Hillis-Steele scan over s[kThreads]; every thread calls it
__device__ __forceinline__ void block_scan_incl(int* s, int tid) {
    for (int off = 1; off < kThreads; off <<= 1) {
        const int a = tid >= off ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += a;
        __syncthreads();
    }
}

static __device__ void compact_top(const float* __restrict__ sc, int n, unsigned cut, int need_eq, int* __restrict__ out);

// out[0, top) = the top `top` cells of sc[0, n) in ascending cell order; 1 <= top < n; blockDim.x == kThreads.
// Every thread of the block must call it.
static __device__ void select_top_cells(const float* __restrict__ sc, int n, int top, int* __restrict__ out) {
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
    }
    __syncthreads();
    // 1. the cut-off key, a byte at a time from the top
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) s_hist[tid] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n; i += kThreads) {
            const unsigned k = score_key(sc[i]);
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
    compact_top(sc, n, s_prefix, s_need, out);
}

// 2-3. out = every cell whose key is above `cut` and the first `need_eq` cells whose key equals it, in cell order.
// Every thread of the block must call it.
static __device__ void compact_top(const float* __restrict__ sc, int n, unsigned cut, int need_eq, int* __restrict__ out) {
    __shared__ int s_eq[kThreads], s_w[kThreads];
    const int tid = threadIdx.x;
    __syncthreads();
    // 2. per-thread runs: counts, the tie shares, the output offsets
    const int chunk = (n + kThreads - 1) / kThreads;
    const int lo = min(n, tid * chunk), hi = min(n, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    s_eq[tid] = eq;
    __syncthreads();
    block_scan_incl(s_eq, tid);
    const int eq_before = s_eq[tid] - eq;
    const int take_eq = max(0, min(eq, need_eq - eq_before));
    s_w[tid] = gt + take_eq;
    __syncthreads();
    block_scan_incl(s_w, tid);
    int o = s_w[tid] - (gt + take_eq);
    // 3. the winners, in cell order
    int eq_seen = eq_before;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        if (k > cut) {
            out[o++] = i;
        } else if (k == cut) {
            if (eq_seen < need_eq) out[o++] = i;
            ++eq_seen;
        }
    }
}

// ---------------------------------------------------------------- B14.15 the bucket (histogram) selector
// The same (cut, need_eq) as the radix cut-off, found with one min / max pass, one histogram pass over kHistBuckets
// buckets spread linearly over [kmin, kmax] of the keys, and a sort of the boundary bucket only (a bitonic sort in
// shared memory of at most kHistCand keys); then the same compaction, so the output is identical to
// select_top_cells'.  A boundary bucket over kHistCand keys (heavily clustered or tied scores) falls back to the radix
// cut-off.  glmd::twin::topk_hist is this algorithm on the host (glmd_fast.hpp).
constexpr int kHistBuckets = 2048;
constexpr int kHistCand = 4096;
static_assert(kHistBuckets == glmd::kTopkHistBuckets && kHistCand == glmd::kTopkHistCand, "the twin's constants");

// the bucket of key k for the row's key range [kmin, kmin + span] (span = kmax - kmin > 0): monotone in k
__host__ __device__ __forceinline__ int hist_bucket(unsigned k, unsigned kmin, unsigned span) {
    return (int) (((unsigned long long) (k - kmin) * (unsigned long long) kHistBuckets) /
                  ((unsigned long long) span + 1ull));
}

static __device__ void select_top_cells_hist(const float* __restrict__ sc, int n, int top, int* __restrict__ out) {
    __shared__ unsigned s_hist[kHistBuckets];
    __shared__ unsigned s_cand[kHistCand];
    __shared__ unsigned s_red[2][32];
    __shared__ unsigned s_kmin, s_span;
    __shared__ int s_b, s_need_b, s_cnt, s_ncand, s_first_eq;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // A. the key range
    unsigned lo = 0xffffffffu, hi = 0u;
    for (int i = tid; i < n; i += kThreads) {
        const unsigned k = score_key(sc[i]);
        lo = min(lo, k);
        hi = max(hi, k);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    if (lane == 0) {
        s_red[0][warp] = lo;
        s_red[1][warp] = hi;
    }
    for (int i = tid; i < kHistBuckets; i += kThreads) s_hist[i] = 0u;
    __syncthreads();
    if (tid == 0) {
        unsigned a = 0xffffffffu, b = 0u;
        for (int w = 0; w < kThreads / 32; ++w) {
            a = min(a, s_red[0][w]);
            b = max(b, s_red[1][w]);
        }
        s_kmin = a;
        s_span = b - a;
        s_ncand = 0;
    }
    __syncthreads();
    const unsigned kmin = s_kmin, span = s_span;
    if (span == 0u) {   // every key equal: the first `top` cells
        compact_top(sc, n, kmin, top, out);
        return;
    }
    // B. the histogram
    for (int i = tid; i < n; i += kThreads) atomicAdd(&s_hist[hist_bucket(score_key(sc[i]), kmin, span)], 1u);
    __syncthreads();
    // the boundary bucket: warp 0, lane l owns buckets [64 l, 64 l + 64); a suffix scan over the lanes
    if (warp == 0) {
        constexpr int per = kHistBuckets / 32;
        int own = 0;
        for (int j = 0; j < per; ++j) own += (int) s_hist[lane * per + j];
        int suf = own;   // inclusive suffix sum over lanes >= lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_down_sync(0xffffffffu, suf, o);
            if (lane + o < 32) suf += v;
        }
        const int above = suf - own;   // keys in the lanes above this one
        if (above < top && suf >= top) {   // exactly one lane holds the boundary
            int cum = above;
            for (int b = lane * per + per - 1; b >= lane * per; --b) {
                const int c = (int) s_hist[b];
                if (cum + c >= top) {
                    s_b = b;
                    s_need_b = top - cum;
                    s_cnt = c;
                    break;
                }
                cum += c;
            }
        }
    }
    __syncthreads();
    const int bstar = s_b, need_b = s_need_b, cnt = s_cnt;
    if (cnt > kHistCand) {   // a heavy boundary bucket: the radix cut-off (same output)
        select_top_cells(sc, n, top, out);
        return;
    }
    // C. the boundary bucket's keys, then a descending bitonic sort of them (padded with key 0, the smallest)
    for (int i = tid; i < n; i += kThreads) {
        const unsigned k = score_key(sc[i]);
        if (hist_bucket(k, kmin, span) == bstar) s_cand[atomicAdd(&s_ncand, 1)] = k;
    }
    __syncthreads();
    int m = 1;
    while (m < cnt) m <<= 1;
    for (int i = cnt + tid; i < m; i += kThreads) s_cand[i] = 0u;
    __syncthreads();
    for (int k = 2; k <= m; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < m; i += kThreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned a = s_cand[i], b = s_cand[ixj];
                    const bool desc = (i & k) == 0;
                    if (desc ? a < b : a > b) {
                        s_cand[i] = b;
                        s_cand[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    const unsigned cut = s_cand[need_b - 1];
    if (tid == 0) s_first_eq = need_b - 1;
    __syncthreads();
    for (int i = tid; i < need_b; i += kThreads)
        if (s_cand[i] == cut && (i == 0 || s_cand[i - 1] != cut)) s_first_eq = i;
    __syncthreads();
    compact_top(sc, n, cut, need_b - s_first_eq, out);
}

// B14.15: the bucket selector, one row per block, the same contract as topk_kernel
__global__ void __launch_bounds__(kThreads) topk_hist_kernel(const float* __restrict__ score, int ld, int n_kv,
                                                             const int* __restrict__ pos, int top,
                                                             int* __restrict__ sel) {
    const int t = blockIdx.x;
    const int n = pos ? min(n_kv, pos[t] + 1) : n_kv;
    int* out = sel + (size_t) t * top;
    if (n <= top) {
        for (int i = threadIdx.x; i < top; i += blockDim.x) out[i] = i < n ? i : -1;
        return;
    }
    select_top_cells_hist(score + (size_t) t * ld, n, top, out);
}

// one row per block: sel[t * top + r]
__global__ void __launch_bounds__(kThreads) topk_kernel(const float* __restrict__ score, int ld, int n_kv,
                                                        const int* __restrict__ pos, int top, int* __restrict__ sel) {
    const int t = blockIdx.x;
    const int n = pos ? min(n_kv, pos[t] + 1) : n_kv;
    int* out = sel + (size_t) t * top;
    if (n <= top) {
        for (int i = threadIdx.x; i < top; i += blockDim.x) out[i] = i < n ? i : -1;
        return;
    }
    select_top_cells(score + (size_t) t * ld, n, top, out);
}

}  // namespace glmd::topk_detail
