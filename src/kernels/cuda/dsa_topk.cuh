// src/kernels/cuda/dsa_topk.cuh - the DSA indexer's choice of the top pools in O(n_vis) (one block, 1024 threads).
//
// The order it writes is the one the O(n_vis^2) rank loop wrote (score descending, ties by the lower pool index), so
// the selected cells - and every result built on them - are unchanged; only the cost stops growing with the square of
// the context (at 30k tokens the old loop was ~56M comparisons per DSA layer per token, in one block).
//   1. a radix select (4 passes of 8 bits) over the scores' order-preserving uint32 keys finds the cut-off key - the
//      top-th largest - and how many of the keys equal to it are taken;
//   2. one pass in pool order (each thread a contiguous run, exclusive scans for the offsets) collects the winners:
//      every key above the cut-off and the first `need` equal to it - the lower indices, as the rank loop chose;
//   3. a bitonic sort of the <= 1024 winners, packed as (key << 32) | ~index, descending.
#pragma once

#include <cuda_runtime.h>

namespace maya::kernels::dsa {

// a float's order-preserving key: larger float, larger key (-0 is +0; the scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis) in rank order; 1 <= top <= 1024,
// top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that includes
// this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    __shared__ unsigned s_hist[256];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[NT], s_eq[NT];
    __shared__ unsigned long long s_win[NT];
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
        for (int i = tid; i < n_vis; i += NT) {
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
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    // 2. the winners, in pool order
    const int chunk = (n_vis + NT - 1) / NT;
    const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    s_gt[tid] = gt;
    s_eq[tid] = eq;
    __syncthreads();
    for (int off = 1; off < NT; off <<= 1) {   // inclusive scans of both counts
        int a = 0, b = 0;
        if (tid >= off) {
            a = s_gt[tid - off];
            b = s_eq[tid - off];
        }
        __syncthreads();
        s_gt[tid] += a;
        s_eq[tid] += b;
        __syncthreads();
    }
    const int n_gt = s_gt[NT - 1];   // == top - need_eq
    int og = s_gt[tid] - gt, oe = s_eq[tid] - eq;
    for (int i = lo; i < hi; ++i) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) {
            s_win[og++] = w;
        } else if (k == cut) {
            if (oe < need_eq) s_win[n_gt + oe] = w;
            ++oe;
        }
    }
    // 3. descending bitonic sort, padded with the smallest value
    int n2 = 1;
    while (n2 < top) n2 <<= 1;
    for (int i = top + tid; i < n2; i += NT) s_win[i] = 0ull;
    __syncthreads();
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n2; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const unsigned long long a = s_win[i], b = s_win[j];
                    if ((i & size) == 0 ? a < b : a > b) {
                        s_win[i] = b;
                        s_win[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int r = tid; r < top; r += NT) {
        const int p = (int) ~(unsigned) (s_win[r] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
}

}  // namespace maya::kernels::dsa
