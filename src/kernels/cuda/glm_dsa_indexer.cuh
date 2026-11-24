// src/kernels/cuda/glm_dsa_indexer.cuh - the DSA indexer of one decode token in one call (dsa_indexer in
// glm_fast.cu): dsa_score + dsa_select's cells, in their order. Included by glm_fast.cu at file scope (CUDA only).
// Shapes: idx_heads 32, idx_key 128 (GLM-5.3-Flash).
//   o1 score_select():    FP32 scores bit-identical to dsa_score_kernel, then an exact select: a 4096-bin histogram
//                         of the scores' order keys finds the cut-off, a parallel collect + a shared-memory sort the
//                         winners (many blocks, not dsa_select's one).
//   o2 score_select_tc(): above Plan::tc_from pools, FP16 tensor-core (mma.sync, sm_80+) upper bounds for every pool,
//                         then the pools that can still be selected rescored exactly as o1 - the same cells; `score`
//                         holds bounds, not scores (no caller reads it after the select).
// One Plan per stream: its histogram and counters are left zeroed by every call.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dsa_idx {

constexpr int kKey = 128;
constexpr int kHeads = 32;
constexpr int kThreads = 256;               // score launch: 8 warps, one block per SM (the query takes ~128 registers)
constexpr int kWarps = kThreads / 32;
constexpr int kTile = 33;                   // padded row of the per-warp transpose tile
constexpr int kBins = 4096;                 // first histogram: the top 12 bits of the order key
constexpr int kBinShift = 20;
constexpr int kSortCap = 2048;              // candidates sorted at once in shared memory
constexpr int kMaxTop = 1024;
constexpr int kSmallN = 8192;               // n_vis at or below: the score launch's last block also selects
constexpr int kCollectThreads = 512;

struct Ctl {                                // every call leaves it as it found it (all zero)
    unsigned hist[kBins];
    unsigned ticket;                        // score blocks finished
    unsigned ticket2;                       // collect blocks finished
    unsigned n_cand;                        // candidates appended by the collect launch
    unsigned cut;                           // the cut-off bin, for the collect launch
    unsigned m;                             // candidates (pools in the cut-off bin or above)
    unsigned pad[3];
};

struct Plan {
    int sms = 0;
    int max_pools = 0;
    int small_n = kSmallN;                  // n_vis at or below: one launch
    int tc_from = 12288;                    // n_vis above: o2 in indexer() (crossover measured 8K-16K)
    Ctl* ctl = nullptr;
    unsigned long long* cand = nullptr;     // max_pools entries
    float* lower = nullptr;                 // max_pools entries (o2)
};

struct Args {
    const float* iq;
    const float* pooled;
    const float* iw;
    float* score;
    int* cells;
    Ctl* ctl;
    unsigned long long* cand;
    int n, top, top_pools, kpool, n_sel, pos;
    int small_n;
    float* lower;                           // o2: per-pool lower bounds (a.score holds the upper ones)
};

#ifdef GLM_DSA_INDEXER_PROF
__device__ unsigned long long g_prof[16];
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define PROF(i) do { if (threadIdx.x == 0) g_prof[i] = gtime(); } while (0)
#define PROF_MIN(i) do { if (threadIdx.x == 0) atomicMin(&g_prof[i], gtime()); } while (0)
#define PROF_MAX(i) do { if (threadIdx.x == 0) atomicMax(&g_prof[i], gtime()); } while (0)
#else
#define PROF(i) do {} while (0)
#define PROF_MIN(i) do {} while (0)
#define PROF_MAX(i) do {} while (0)
#endif

// Maya's order key (dsa_topk.cuh score_key): larger float, larger key; -0 is +0
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ unsigned long long compo(unsigned key, int i) {
    return ((unsigned long long) key << 32) | (unsigned long long) (~(unsigned) i);
}

// exclusive block scan of one int a thread; s needs NT / 32 ints
template <int NT>
__device__ __forceinline__ int block_excl_scan(int v, int* s) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = lane < NT / 32 ? s[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        if (lane < NT / 32) s[lane] = t;
    }
    __syncthreads();
    const int before = (w > 0 ? s[w - 1] : 0) + x - v;
    __syncthreads();
    return before;
}

// the bin holding the need-th largest element of h[0, nb) (bins in ascending key order): {bin, elements above it}.
// s needs NT / 32 + 2 ints.  GLOBAL: h was written by other blocks (read through L2).
template <int NT, bool GLOBAL>
__device__ __forceinline__ int2 find_cut(const unsigned* h, int nb, int need, int* s) {
    const int per = (nb + NT - 1) / NT;
    const int hi = nb - (int) threadIdx.x * per, lo = max(0, hi - per);
    int sum = 0;
    for (int b = lo; b < hi; ++b) sum += (int) (GLOBAL ? __ldcg(h + b) : h[b]);
    const int excl = block_excl_scan<NT>(sum, s);
    if (excl < need && excl + sum >= need) {
        int c = excl;
        for (int b = hi - 1; b >= lo; --b) {
            const int x = (int) (GLOBAL ? __ldcg(h + b) : h[b]);
            if (c + x >= need) {
                s[NT / 32] = b;
                s[NT / 32 + 1] = c;
                break;
            }
            c += x;
        }
    }
    __syncthreads();
    const int2 r = make_int2(s[NT / 32], s[NT / 32 + 1]);
    __syncthreads();
    return r;
}

__device__ __forceinline__ unsigned long long keep(unsigned long long mine, unsigned long long other, bool keep_max) {
    return keep_max ? (mine > other ? mine : other) : (mine < other ? mine : other);
}

// compare-exchange of in-thread pairs (r, r + S)
template <int R, int S>
__device__ __forceinline__ void cx_in(unsigned long long (&x)[R], int ebase, int size) {
#pragma unroll
    for (int r = 0; r < R; ++r)
        if ((r & S) == 0) {
            const unsigned long long a = x[r], b = x[r + S];
            const bool sw = ((ebase + r) & size) == 0 ? a < b : a > b;
            x[r] = sw ? b : a;
            x[r + S] = sw ? a : b;
        }
}

// descending bitonic sort of NT * R values in registers (element e = tid * R + r), every stage compile-time;
// shared memory only for the cross-warp strides (buf: 2 * NT * R entries, double-buffered: one barrier a stage)
template <int NT, int R, int SIZE, int S>
__device__ __forceinline__ void bstage(unsigned long long (&x)[R], unsigned long long* buf, int& ph) {
    if constexpr (S > 0) {
        const int tid = threadIdx.x, ebase = tid * R;
        if constexpr (S < R) {
            cx_in<R, S>(x, ebase, SIZE);
        } else if constexpr (S < 32 * R) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = ebase + r;
                const unsigned long long o = __shfl_xor_sync(0xffffffffu, x[r], S / R);
                x[r] = keep(x[r], o, ((e & SIZE) == 0) == ((e & S) == 0));
            }
        } else {
            unsigned long long* b = buf + ph * (NT * R);
#pragma unroll
            for (int r = 0; r < R; ++r) b[r * NT + tid] = x[r];
            __syncthreads();
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = ebase + r;
                x[r] = keep(x[r], b[r * NT + (tid ^ (S / R))], ((e & SIZE) == 0) == ((e & S) == 0));
            }
            ph ^= 1;
        }
        bstage<NT, R, SIZE, S / 2>(x, buf, ph);
    }
}
template <int NT, int R, int SIZE = 2>
__device__ __forceinline__ void reg_bitonic_desc(unsigned long long (&x)[R], unsigned long long* buf, int ph = 0) {
    if constexpr (SIZE <= NT * R) {
        bstage<NT, R, SIZE, SIZE / 2>(x, buf, ph);
        reg_bitonic_desc<NT, R, SIZE * 2>(x, buf, ph);
    }
}

// src[0, cnt) (cnt <= NT * R) sorted descending; the first a.top written as cells.  buf: 2 * NT * R entries.
template <int NT, int R>
__device__ __forceinline__ void sort_write(const Args& a, const unsigned long long* src, unsigned long long* buf, int cnt) {
    unsigned long long x[R];
    const int ebase = threadIdx.x * R;
#pragma unroll
    for (int r = 0; r < R; ++r) x[r] = ebase + r < cnt ? src[ebase + r] : 0ull;
    __syncthreads();
    reg_bitonic_desc<NT, R>(x, buf);
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (ebase + r < a.top) {
            const int p = (int) ~(unsigned) (x[r] & 0xffffffffull);
            for (int q = 0; q < a.kpool; ++q) a.cells[(ebase + r) * a.kpool + q] = p * a.kpool + q;
        }
}

template <int NT>
__device__ __forceinline__ unsigned long long block_or(unsigned long long v, unsigned long long* s) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v |= __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    unsigned long long r = 0;
    for (int i = 0; i < NT / 32; ++i) r |= s[i];
    __syncthreads();
    return r;
}

// Radix-narrow the m distinct composites of src to the ones above a cut-off: at most cap of them, and always all of
// the top `top`; compacted into dst, returns how many.  cap == top: exactly the top `top`.
// s_h: 256 unsigned; s: NT / 32 + 4 ints; s64: NT / 32 u64.  GLOBAL: src written by other blocks.
template <int NT, bool GLOBAL>
__device__ int narrow(const unsigned long long* src, int m, int top, int cap, unsigned long long* dst, unsigned* s_h,
                      int* s, unsigned long long* s64) {
    auto ld = [&](int i) { return GLOBAL ? __ldcg(src + i) : src[i]; };
    if (m <= cap) {
#pragma unroll 4
        for (int i = threadIdx.x; i < m; i += NT) dst[i] = ld(i);
        __syncthreads();
        return m;
    }
    const unsigned long long c0 = ld(0);
    unsigned long long d = 0;
#pragma unroll 4
    for (int i = threadIdx.x; i < m; i += NT) d |= ld(i) ^ c0;
    d = block_or<NT>(d, s64);
    const int hb = 63 - __clzll((long long) d);   // the highest bit where they differ (m >= 2, distinct)
    unsigned long long mask = hb >= 63 ? 0ull : (~0ull << (hb + 1));
    unsigned long long prefix = c0 & mask;
    int need = top, in_bin = m, hi = hb;
    while (hi >= 0 && (top - need) + in_bin > cap) {
        const int lo = max(0, hi - 7), nb = 1 << (hi - lo + 1);
        for (int b = threadIdx.x; b < 256; b += NT) s_h[b] = 0u;
        __syncthreads();
#pragma unroll 4
        for (int i = threadIdx.x; i < m; i += NT) {
            const unsigned long long c = ld(i);
            if ((c & mask) == prefix) atomicAdd(&s_h[(unsigned) (c >> lo) & (unsigned) (nb - 1)], 1u);
        }
        __syncthreads();
        const int2 cut = find_cut<NT, false>(s_h, nb, need, s);
        need -= cut.y;
        in_bin = (int) s_h[cut.x];
        prefix |= (unsigned long long) cut.x << lo;
        mask |= (unsigned long long) (nb - 1) << lo;
        hi = lo - 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) s[NT / 32 + 2] = 0;
    __syncthreads();
#pragma unroll 4
    for (int i = threadIdx.x; i < m; i += NT) {
        const unsigned long long c = ld(i);
        if ((c & mask) >= prefix) dst[atomicAdd(&s[NT / 32 + 2], 1)] = c;
    }
    __syncthreads();
    const int cnt = s[NT / 32 + 2];
    __syncthreads();
    return cnt;
}

// The top a.top of the m candidates (distinct composites), descending, written as cells (+ the tail, -1 elsewhere).
// in_smem: the candidates are in sbuf (m <= kSortCap); else in src (global).  sbuf: 2 * kSortCap entries.
template <int NT>
__device__ void block_final(const Args& a, const unsigned long long* src, bool in_smem, int m,
                            unsigned long long* sbuf, unsigned* s_h, int* s, unsigned long long* s64) {
    unsigned long long* fin = sbuf + kSortCap;   // exactly a.top entries end here; the sort buffers sbuf[0, 2 * top)
    int cnt;
    if (in_smem) {
        cnt = narrow<NT, false>(sbuf, m, a.top, a.top, fin, s_h, s, s64);
    } else if (m <= kSortCap) {
#pragma unroll 4
        for (int i = threadIdx.x; i < m; i += NT) sbuf[i] = __ldcg(src + i);
        __syncthreads();
        cnt = narrow<NT, false>(sbuf, m, a.top, a.top, fin, s_h, s, s64);
    } else {
        cnt = narrow<NT, true>(src, m, a.top, kSortCap, sbuf, s_h, s, s64);
        cnt = narrow<NT, false>(sbuf, cnt, a.top, a.top, fin, s_h, s, s64);
    }
    PROF(9);
#ifdef GLM_DSA_INDEXER_PROF
    if (threadIdx.x == 0) g_prof[12] = (unsigned long long) m;
#endif
    if (a.top <= NT) sort_write<NT, 1>(a, fin, sbuf, cnt);
    else if (a.top <= 2 * NT) sort_write<NT, 2>(a, fin, sbuf, cnt);
    else sort_write<NT, 4>(a, fin, sbuf, cnt);
    PROF(10);
    for (int t = a.top * a.kpool + (int) threadIdx.x; t < a.n_sel; t += NT) {
        const int q = t - a.top_pools * a.kpool, cell = a.n * a.kpool + q;
        a.cells[t] = (q >= 0 && q < a.kpool - 1 && cell <= a.pos) ? cell : -1;
    }
}

// add U groups of 32 lanes' bins (0xffffffff = none; n[u] valid) to a shared histogram: sort each group across the
// warp, then one atomic a run (same-bin shared atomics serialise); U groups interleaved for ILP
template <int U>
__device__ __forceinline__ void hist_runs(unsigned* h, unsigned (&b)[U], int lane, const int (&n)[U]) {
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const bool lo_min = ((lane & k) == 0) == ((lane & j) == 0);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const unsigned o = __shfl_xor_sync(0xffffffffu, b[u], j);
                b[u] = lo_min ? min(b[u], o) : max(b[u], o);
            }
        }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const unsigned prev = __shfl_up_sync(0xffffffffu, b[u], 1);
        const bool lead = b[u] != 0xffffffffu && (lane == 0 || prev != b[u]);
        const unsigned lm = __ballot_sync(0xffffffffu, lead);
        const unsigned rest = lane < 31 ? lm >> (lane + 1) : 0u;
        const int end = rest ? lane + __ffs(rest) : n[u];
        if (lead) atomicAdd(&h[b[u]], (unsigned) (end - lane));
    }
}

// one butterfly level: keep registers [0, O), add the partner's [O, 2O) (same head: register r holds head r ^ lane)
template <int O>
__device__ __forceinline__ void bfly(float (&v)[kHeads]) {
#pragma unroll
    for (int r = 0; r < O; ++r) v[r] += __shfl_xor_sync(0xffffffffu, v[r + O], O);
}

__host__ __device__ constexpr size_t score_smem_bytes() {
    return (size_t) (kHeads * kKey + kHeads + kWarps * 32 * kTile) * sizeof(float) + (size_t) kBins * sizeof(unsigned);
}
static_assert((kHeads * kKey + kHeads + kWarps * 32 * kTile) * 4 >= 2 * kSortCap * 8, "last block's sort buffer");
static_assert(kBins >= 2048, "o2 candidate list (2048 ints) in the histogram's space");

// lane L gets head L's dot of one pool (k0's FMA chains and butterfly tree, bit for bit); kc = elements L + 32 j
__device__ __forceinline__ float pool_dot_lane(const float (&qr)[kHeads][4], const float (&kc)[4]) {
    float v[kHeads];
#pragma unroll
    for (int r = 0; r < kHeads; ++r) {
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) dot += qr[r][j] * kc[j];
        v[r] = dot;
    }
    bfly<16>(v);
    bfly<8>(v);
    bfly<4>(v);
    bfly<2>(v);
    bfly<1>(v);
    return v[0];
}

// SELECT: also histogram + (last block) threshold / small-n selection.
// Bench probes: MATH_ONLY (no key loads); STOP 1 = return after the loop, 2 = after the histogram flush;
// HM histogram update 0 = after the loop, warp-sorted runs; in the loop: 1 = none, 2 = one atomic a lane,
// 3 = match_any, 4 = warp-sorted runs.
template <bool SELECT, bool MATH_ONLY = false, int STOP = 0, int HM = 0>
__global__ void __launch_bounds__(kThreads, 1) score_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* s_q = (float*) smem_raw;
    float* s_w = s_q + kHeads * kKey;
    float* s_tile = s_w + kHeads;
    unsigned* s_hist = (unsigned*) (s_tile + kWarps * 32 * kTile);
    __shared__ int s_scan[kWarps + 4];
    __shared__ unsigned long long s64[kWarps];
    __shared__ int s_flag;
    const int tid = threadIdx.x, L = tid & 31, warp = tid >> 5;
    if (SELECT) PROF_MIN(0);
    for (int i = tid; i < kHeads * kKey; i += kThreads) s_q[i] = a.iq[i];
    if (tid < kHeads) s_w[tid] = a.iw[tid];
    if (SELECT)
        for (int i = tid; i < kBins; i += kThreads) s_hist[i] = 0u;
    __syncthreads();
    float qr[kHeads][4];   // lane L, register r: head r ^ L, elements L + 32 j
#pragma unroll
    for (int r = 0; r < kHeads; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) qr[r][j] = s_q[(r ^ L) * kKey + L + 32 * j];
    float* tile = s_tile + warp * 32 * kTile;
    const int nw = gridDim.x * kWarps, gw = blockIdx.x * kWarps + warp;
    const int lo = (int) ((long long) gw * a.n / nw), hi = (int) ((long long) (gw + 1) * a.n / nw);
    const float* kp = a.pooled + L;
    float ka[4], kb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        ka[j] = 0.0f;
        kb[j] = 0.0f;
    }
    if (!MATH_ONLY) {
        if (lo < hi)
#pragma unroll
            for (int j = 0; j < 4; ++j) ka[j] = __ldcs(kp + (size_t) lo * kKey + 32 * j);
        if (lo + 1 < hi)
#pragma unroll
            for (int j = 0; j < 4; ++j) kb[j] = __ldcs(kp + (size_t) (lo + 1) * kKey + 32 * j);
    }
    int i = 0, base = lo;
    for (int p = lo; p < hi; ++p) {
        float kc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kc[j] = ka[j];
            ka[j] = kb[j];
        }
        if (MATH_ONLY) {
#pragma unroll
            for (int j = 0; j < 4; ++j) kc[j] = __int_as_float(0x3f800000 | ((p * 4 + j + L) & 0xffff));
        } else if (p + 2 < hi) {
#pragma unroll
            for (int j = 0; j < 4; ++j) kb[j] = __ldcs(kp + (size_t) (p + 2) * kKey + 32 * j);
        }
        tile[i * kTile + L] = pool_dot_lane(qr, kc);
        if (++i == 32 || p + 1 == hi) {
            __syncwarp();
            const bool act = L < i;
            float acc = 0.0f;
            if (act) {
#pragma unroll
                for (int h = 0; h < kHeads; ++h) acc += fmaxf(tile[L * kTile + h], 0.0f) * s_w[h];
                a.score[base + L] = acc;
            }
            if (SELECT && HM == 2 && act) atomicAdd(&s_hist[order_key(acc) >> kBinShift], 1u);
            if (SELECT && HM == 4) {
                unsigned b1[1] = {act ? order_key(acc) >> kBinShift : 0xffffffffu};
                const int n1[1] = {i};
                hist_runs<1>(s_hist, b1, L, n1);
            }
            if (SELECT && HM == 3) {
                const unsigned bin = act ? order_key(acc) >> kBinShift : 0xffffffffu;
                const unsigned peers = __match_any_sync(0xffffffffu, bin);
                if (act && L == __ffs(peers) - 1) atomicAdd(&s_hist[bin], (unsigned) __popc(peers));
            }
            __syncwarp();
            base += i;
            i = 0;
        }
    }
    if (SELECT && HM == 0) {   // the histogram as its own pass: inside the FMA loop it costs ~3 us a batch
        __syncwarp();
        constexpr int U = 4;
        for (int b0 = lo; b0 < hi; b0 += 32 * U) {
            float sc[U];
#pragma unroll
            for (int u = 0; u < U; ++u) sc[u] = b0 + 32 * u + L < hi ? a.score[b0 + 32 * u + L] : 0.0f;
            unsigned bins[U];
            int cnt[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                bins[u] = b0 + 32 * u + L < hi ? order_key(sc[u]) >> kBinShift : 0xffffffffu;
                cnt[u] = max(0, min(32, hi - b0 - 32 * u));
            }
            hist_runs<U>(s_hist, bins, L, cnt);
        }
    }
    if (!SELECT || STOP == 1) return;
    __syncthreads();
    PROF_MAX(1);
    for (int b = tid; b < kBins; b += kThreads)
        if (s_hist[b] != 0u) atomicAdd(&a.ctl->hist[b], s_hist[b]);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_flag = atomicAdd(&a.ctl->ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_flag) return;
    __threadfence();
    PROF(2);
    // the last block: the global histogram into shared memory (and zeroed for the next call), then the cut-off bin
#pragma unroll
    for (int k = 0; k < kBins / kThreads; ++k) s_hist[tid + k * kThreads] = __ldcg(&a.ctl->hist[tid + k * kThreads]);
#pragma unroll
    for (int k = 0; k < kBins / kThreads; ++k) a.ctl->hist[tid + k * kThreads] = 0u;
    if (tid == 0) a.ctl->ticket = 0u;
    if (STOP == 2) return;
    __syncthreads();
    const int2 cut = find_cut<kThreads, false>(s_hist, kBins, a.top, s_scan);
    const int m = cut.y + (int) s_hist[cut.x];
    PROF(3);
    if (a.n > a.small_n) {   // the collect launch takes over
        if (tid == 0) {
            a.ctl->cut = (unsigned) cut.x;
            a.ctl->m = (unsigned) m;
        }
        return;
    }
    unsigned long long* sbuf = (unsigned long long*) s_q;   // 32 KB over s_q / s_w / s_tile (the query is in registers)
    const bool to_smem = m <= kSortCap;
    if (tid == 0) s_scan[kWarps + 2] = 0;
    __syncthreads();
    constexpr int U = 4;   // float4 loads in flight a thread
    for (int b4 = 0; b4 * 4 < a.n; b4 += U * kThreads) {
        float f[U][4];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = (b4 + u * kThreads + tid) * 4;
            if (i + 3 < a.n) {
                const float4 x = __ldcg((const float4*) (a.score + i));
                f[u][0] = x.x;
                f[u][1] = x.y;
                f[u][2] = x.z;
                f[u][3] = x.w;
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) f[u][e] = i + e < a.n ? __ldcg(a.score + i + e) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int k = (b4 + u * kThreads + tid) * 4 + e;
                const unsigned key = order_key(f[u][e]);
                if (k < a.n && (key >> kBinShift) >= (unsigned) cut.x) {
                    const int slot = atomicAdd(&s_scan[kWarps + 2], 1);
                    if (to_smem) sbuf[slot] = compo(key, k);
                    else a.cand[slot] = compo(key, k);
                }
            }
    }
    __syncthreads();
    PROF(4);
    block_final<kThreads>(a, a.cand, to_smem, m, sbuf, s_hist, s_scan, s64);
    PROF(5);
}

// n_vis > kSmallN: every pool at or above the cut-off bin into a.cand (warp-aggregated), then the last block selects
__global__ void __launch_bounds__(kCollectThreads) collect_kernel(const Args a) {
    __shared__ unsigned long long sbuf[2 * kSortCap];
    __shared__ unsigned s_h[256];
    __shared__ int s_scan[kCollectThreads / 32 + 4];
    __shared__ unsigned long long s64[kCollectThreads / 32];
    __shared__ int s_flag;
    PROF_MIN(6);
    const unsigned cut = a.ctl->cut;
    const int m = (int) a.ctl->m;
    const int lane = threadIdx.x & 31;
    const int n4 = (a.n + 3) >> 2;
    for (int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 - (int) threadIdx.x < n4; i4 += gridDim.x * blockDim.x) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const bool in = i4 < n4;
        if (in && i4 * 4 + 3 < a.n) {
            const float4 f = __ldcg((const float4*) a.score + i4);
            v[0] = f.x;
            v[1] = f.y;
            v[2] = f.z;
            v[3] = f.w;
        } else if (in) {
            for (int e = 0; e < 4; ++e)
                if (i4 * 4 + e < a.n) v[e] = __ldcg(a.score + i4 * 4 + e);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int idx = i4 * 4 + e;
            const unsigned key = order_key(v[e]);
            const bool take = in && idx < a.n && (key >> kBinShift) >= cut;
            const unsigned bal = __ballot_sync(0xffffffffu, take);
            if (bal == 0u) continue;
            unsigned basep = 0;
            if (lane == 0) basep = atomicAdd(&a.ctl->n_cand, (unsigned) __popc(bal));
            basep = __shfl_sync(0xffffffffu, basep, 0);
            if (take) a.cand[basep + __popc(bal & ((1u << lane) - 1u))] = compo(key, idx);
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_flag = atomicAdd(&a.ctl->ticket2, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_flag) return;
    __threadfence();
    PROF(7);
    block_final<kCollectThreads>(a, a.cand, false, m, sbuf, s_h, s_scan, s64);
    PROF(8);
    if (threadIdx.x == 0) {
        a.ctl->n_cand = 0u;
        a.ctl->ticket2 = 0u;
    }
}

// n_vis == 0: Maya's select writes -1s and the tail only
__global__ void tail_only_kernel(const Args a) {
    for (int t = threadIdx.x; t < a.n_sel; t += blockDim.x) {
        const int q = t - a.top_pools * a.kpool, cell = q;
        a.cells[t] = (q >= 0 && q < a.kpool - 1 && cell <= a.pos) ? cell : -1;
    }
}

// ---------------------------------------------------------------- o2: tensor-core bounds, exact rescoring of candidates
// Each pool gets an FP16 tensor-core score a and a rigorous bound eps with |a - k0's score| <= eps (README: derivation).
// tau <= the top-th largest lower bound; only pools with a + eps >= tau can be selected.  They are rescored with k0's
// exact FP32 order and selected exactly, so the cells are k0's.  a.score receives the upper bounds, not the scores.
constexpr float kC1 = 1.0f / 256.0f;        // x sum_e v_e |k_e|: FP16 rounding of q and k (2^-10) + accumulation, 4x
constexpr float kC2 = 1.0f / 65536.0f;      // x sum_h |w_h| |d_h|: FP32 rounding of the weighted head sums, 4x
constexpr int kChunk = 2048;                // rescoring: pools scanned at once by a block

__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();   // m16n8k16 f16 mma: sm_80+
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ float relu(float x) { return fmaxf(x, 0.0f); }

// Pass 1: lower / upper bound of every pool (8 pools a warp step, m16n8k16: heads x pools, k permuted so each lane
// loads one float4 per k-step), the lower bounds' histogram, and (last block) the cut-off key tau into ctl->cut.
__global__ void __launch_bounds__(kThreads, 1) tc_bound_kernel(const Args a) {
    __shared__ __align__(16) float s_q[kHeads * kKey];
    __shared__ __align__(16) float s_v[kKey];
    __shared__ unsigned s_hist[kBins];
    __shared__ int s_scan[kWarps + 4];
    __shared__ float s_w[kHeads], s_red[kKey / 32];
    __shared__ int s_flag;
    const int tid = threadIdx.x, L = tid & 31, warp = tid >> 5, g = L >> 2, t = L & 3;
    for (int i = tid; i < kHeads * kKey; i += kThreads) s_q[i] = a.iq[i];
    for (int i = tid; i < kBins; i += kThreads) s_hist[i] = 0u;
    if (tid < kHeads) s_w[tid] = a.iw[tid];
    __syncthreads();
    float wsum = fabsf(s_w[L]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, o);
    float v = 0.0f;   // v_e = sum_h |w_h| |q_he|
    if (tid < kKey) {
#pragma unroll 8
        for (int h = 0; h < kHeads; ++h) v = fmaf(fabsf(s_w[h]), fabsf(s_q[h * kKey + tid]), v);
        float r = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
        if (L == 0) s_red[warp] = r;
    }
    __syncthreads();
    // FP16-subnormal terms, 4x: of k (c3 = 2^-23 sum_e v_e) and of q (2^-23 sum_h |w_h| on every v_e)
    const float c3 = (s_red[0] + s_red[1] + s_red[2] + s_red[3]) * (1.0f / 8388608.0f) + 1e-30f;
    if (tid < kKey) s_v[tid] = fmaf(v, kC1, wsum * (1.0f / 8388608.0f));
    unsigned afr[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            const float4 x0 = *(const float4*) (s_q + (16 * mt + g) * kKey + 16 * st + 4 * t);
            const float4 x1 = *(const float4*) (s_q + (16 * mt + g + 8) * kKey + 16 * st + 4 * t);
            afr[mt][st][0] = pack_h2(x0.x, x0.y);
            afr[mt][st][1] = pack_h2(x1.x, x1.y);
            afr[mt][st][2] = pack_h2(x0.z, x0.w);
            afr[mt][st][3] = pack_h2(x1.z, x1.w);
        }
    const float w0 = s_w[g], w1 = s_w[g + 8], w2 = s_w[16 + g], w3 = s_w[24 + g];
    __syncthreads();
    const float4* sv4 = (const float4*) s_v;
    const int nw = gridDim.x * kWarps, gw = blockIdx.x * kWarps + warp;
    const int lo = (int) ((long long) gw * a.n / nw), hi = (int) ((long long) (gw + 1) * a.n / nw);
    const float4* kp4 = (const float4*) a.pooled;
    float4 kn[8];
    auto load = [&](int G) {
        const bool ok = G + g < hi;
        const float4* src = kp4 + (size_t) (ok ? G + g : 0) * (kKey / 4) + t;
#pragma unroll
        for (int st = 0; st < 8; ++st) kn[st] = ok ? __ldcs(src + 4 * st) : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    if (lo < hi) load(lo);
    for (int G = lo; G < hi; G += 8) {
        float4 kc[8];
#pragma unroll
        for (int st = 0; st < 8; ++st) kc[st] = kn[st];
        if (G + 8 < hi) load(G + 8);
        float vs = 0.0f, d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int st = 0; st < 8; ++st) {
            const float4 vv = sv4[4 * st + t];
            vs = fmaf(vv.x, fabsf(kc[st].x), vs);
            vs = fmaf(vv.y, fabsf(kc[st].y), vs);
            vs = fmaf(vv.z, fabsf(kc[st].z), vs);
            vs = fmaf(vv.w, fabsf(kc[st].w), vs);
            const unsigned b0 = pack_h2(kc[st].x, kc[st].y), b1 = pack_h2(kc[st].z, kc[st].w);
            mma16816(d0, afr[0][st], b0, b1);
            mma16816(d1, afr[1][st], b0, b1);
        }
        // lane (g, t): heads g, g+8, 16+g, 24+g of pools 2t (d[0], d[2]) and 2t+1 (d[1], d[3])
        float sa0 = w0 * relu(d0[0]) + w1 * relu(d0[2]) + w2 * relu(d1[0]) + w3 * relu(d1[2]);
        float sa1 = w0 * relu(d0[1]) + w1 * relu(d0[3]) + w2 * relu(d1[1]) + w3 * relu(d1[3]);
        float sA0 = fabsf(w0) * fabsf(d0[0]) + fabsf(w1) * fabsf(d0[2]) + fabsf(w2) * fabsf(d1[0]) +
                    fabsf(w3) * fabsf(d1[2]);
        float sA1 = fabsf(w0) * fabsf(d0[1]) + fabsf(w1) * fabsf(d0[3]) + fabsf(w2) * fabsf(d1[1]) +
                    fabsf(w3) * fabsf(d1[3]);
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
            sa0 += __shfl_xor_sync(0xffffffffu, sa0, o);
            sa1 += __shfl_xor_sync(0xffffffffu, sa1, o);
            sA0 += __shfl_xor_sync(0xffffffffu, sA0, o);
            sA1 += __shfl_xor_sync(0xffffffffu, sA1, o);
        }
        vs += __shfl_xor_sync(0xffffffffu, vs, 1);
        vs += __shfl_xor_sync(0xffffffffu, vs, 2);
        // lane 4g + g/2 holds pool g's V (row g) and its a / A (column 2t or 2t+1 = g)
        if (t == (g >> 1) && G + g < hi) {
            const float av = (g & 1) ? sa1 : sa0, Av = (g & 1) ? sA1 : sA0;
            const float eps = __fadd_ru(__fmaf_ru(kC2, Av, vs), c3);
            float lb = __fsub_rd(av, eps), ub = __fadd_ru(av, eps);
            if (!isfinite(lb) || !isfinite(ub)) {   // FP16 overflow or NaN: always a candidate, never raises tau
                lb = -INFINITY;
                ub = INFINITY;
            }
            a.lower[G + g] = lb;
            a.score[G + g] = ub;
        }
    }
    __syncwarp();
    constexpr int U = 4;
    for (int b0 = lo; b0 < hi; b0 += 32 * U) {
        unsigned bins[U];
        int cnt[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int p = b0 + 32 * u + L;
            bins[u] = p < hi ? order_key(a.lower[p]) >> kBinShift : 0xffffffffu;
            cnt[u] = max(0, min(32, hi - b0 - 32 * u));
        }
        hist_runs<U>(s_hist, bins, L, cnt);
    }
    __syncthreads();
    for (int b = tid; b < kBins; b += kThreads)
        if (s_hist[b] != 0u) atomicAdd(&a.ctl->hist[b], s_hist[b]);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_flag = atomicAdd(&a.ctl->ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_flag) return;
    __threadfence();
#pragma unroll
    for (int k = 0; k < kBins / kThreads; ++k) s_hist[tid + k * kThreads] = __ldcg(&a.ctl->hist[tid + k * kThreads]);
#pragma unroll
    for (int k = 0; k < kBins / kThreads; ++k) a.ctl->hist[tid + k * kThreads] = 0u;
    if (tid == 0) a.ctl->ticket = 0u;
    __syncthreads();
    const int2 cut = find_cut<kThreads, false>(s_hist, kBins, a.top, s_scan);
    if (tid == 0) a.ctl->cut = (unsigned) cut.x << kBinShift;   // tau: the lowest key of the top-th lower bound's bin
}

// Pass 2: every pool whose upper bound reaches tau, rescored exactly (k0's order), appended as composites; the last
// block selects.  Dynamic smem: score_smem_bytes().
__global__ void __launch_bounds__(kThreads, 1) tc_rescore_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* s_q = (float*) smem_raw;
    float* s_w = s_q + kHeads * kKey;
    float* s_tile = s_w + kHeads;
    int* s_list = (int*) (s_tile + kWarps * 32 * kTile);   // kChunk entries (the o1 histogram's space)
    unsigned* s_h = (unsigned*) s_list;
    __shared__ int s_scan[kWarps + 4];
    __shared__ unsigned long long s64[kWarps];
    __shared__ int s_cnt, s_flag;
    const int tid = threadIdx.x, L = tid & 31, warp = tid >> 5;
    for (int i = tid; i < kHeads * kKey; i += kThreads) s_q[i] = a.iq[i];
    if (tid < kHeads) s_w[tid] = a.iw[tid];
    __syncthreads();
    float qr[kHeads][4];
#pragma unroll
    for (int r = 0; r < kHeads; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) qr[r][j] = s_q[(r ^ L) * kKey + L + 32 * j];
    float* tile = s_tile + warp * 32 * kTile;
    const unsigned tau = a.ctl->cut;
    const int blo = (int) ((long long) blockIdx.x * a.n / gridDim.x);
    const int bhi = (int) ((long long) (blockIdx.x + 1) * a.n / gridDim.x);
    for (int c0 = blo; c0 < bhi; c0 += kChunk) {
        const int cn = min(kChunk, bhi - c0);
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        for (int i0 = 0; i0 < cn; i0 += kThreads) {   // warp-uniform trip count (ballots)
            const int i = i0 + tid;
            const bool take = i < cn && order_key(a.score[c0 + i]) >= tau;
            const unsigned bal = __ballot_sync(0xffffffffu, take);
            int basep = 0;
            if (L == 0 && bal) basep = atomicAdd(&s_cnt, __popc(bal));
            basep = __shfl_sync(0xffffffffu, basep, 0);
            if (take) s_list[basep + __popc(bal & ((1u << L) - 1u))] = c0 + i;
        }
        __syncthreads();
        const int cnt = s_cnt;
        for (int b = 0; b < cnt; b += kWarps * 32) {   // warp w: entries b + w + 8 i (spread over all warps)
            const int nb = max(0, min(32, (cnt - b - warp + kWarps - 1) / kWarps));
            float ka[4], kb[4];
            auto ld = [&](int i, float (&k)[4]) {
                const float* kp = a.pooled + (size_t) s_list[b + warp + kWarps * i] * kKey + L;
#pragma unroll
                for (int j = 0; j < 4; ++j) k[j] = __ldcs(kp + 32 * j);
            };
            if (nb > 0) ld(0, ka);
            if (nb > 1) ld(1, kb);
            for (int i = 0; i < nb; ++i) {
                float kc[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    kc[j] = ka[j];
                    ka[j] = kb[j];
                }
                if (i + 2 < nb) ld(i + 2, kb);
                tile[i * kTile + L] = pool_dot_lane(qr, kc);
            }
            __syncwarp();
            const bool act = L < nb;
            unsigned long long c = 0ull;
            if (act) {
                float acc = 0.0f;
#pragma unroll
                for (int h = 0; h < kHeads; ++h) acc += fmaxf(tile[L * kTile + h], 0.0f) * s_w[h];
                c = compo(order_key(acc), s_list[b + warp + kWarps * L]);
            }
            unsigned basep = 0;
            if (L == 0) basep = atomicAdd(&a.ctl->n_cand, (unsigned) nb);
            basep = __shfl_sync(0xffffffffu, basep, 0);
            if (act) a.cand[basep + L] = c;
            __syncwarp();
        }
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_flag = atomicAdd(&a.ctl->ticket2, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_flag) return;
    __threadfence();
    const int m = (int) __ldcg(&a.ctl->n_cand);
    block_final<kThreads>(a, a.cand, false, m, (unsigned long long*) s_q, s_h, s_scan, s64);
    if (tid == 0) {
        a.ctl->pad[0] = (unsigned) m;   // candidates of the last call (bench)
        a.ctl->n_cand = 0u;
        a.ctl->ticket2 = 0u;
    }
}

// o2: the same cells as gf::dsa_score + gf::dsa_select, without the full score array (score gets upper bounds).
inline cudaError_t score_select_tc(const Plan& pl, const float* iq, const float* pooled, const float* iw, int key_dim,
                                   int idx_heads, int n_vis, int kpool, int top_pools, int n_sel, int pos, float* score,
                                   int* cells, cudaStream_t s) {
    if (key_dim != kKey || idx_heads != kHeads || n_vis < 0 || n_vis > pl.max_pools || top_pools < 1 ||
        top_pools > kMaxTop || kpool < 1 || pl.ctl == nullptr)
        return cudaErrorNotSupported;
    Args a{iq, pooled, iw, score, cells, pl.ctl, pl.cand, n_vis, top_pools < n_vis ? top_pools : n_vis, top_pools,
           kpool, n_sel, pos, pl.small_n, pl.lower};
    if (n_vis == 0) {
        tail_only_kernel<<<1, 256, 0, s>>>(a);
        return cudaGetLastError();
    }
    tc_bound_kernel<<<pl.sms, kThreads, 0, s>>>(a);
    tc_rescore_kernel<<<pl.sms, kThreads, score_smem_bytes(), s>>>(a);
    return cudaGetLastError();
}

// The engine entry: o1 up to pl.tc_from pools (also writes k0's scores), o2 above (score = upper bounds); same cells.
inline cudaError_t score_select(const Plan& pl, const float* iq, const float* pooled, const float* iw, int key_dim,
                                int idx_heads, int n_vis, int kpool, int top_pools, int n_sel, int pos, float* score,
                                int* cells, cudaStream_t s);
inline cudaError_t indexer(const Plan& pl, const float* iq, const float* pooled, const float* iw, int key_dim,
                           int idx_heads, int n_vis, int kpool, int top_pools, int n_sel, int pos, float* score,
                           int* cells, cudaStream_t s) {
    return n_vis > pl.tc_from
               ? score_select_tc(pl, iq, pooled, iw, key_dim, idx_heads, n_vis, kpool, top_pools, n_sel, pos, score, cells, s)
               : score_select(pl, iq, pooled, iw, key_dim, idx_heads, n_vis, kpool, top_pools, n_sel, pos, score, cells, s);
}

// ---------------------------------------------------------------- host side
inline cudaError_t init(Plan& pl, int max_pools) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    if ((e = cudaDeviceGetAttribute(&pl.sms, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess) return e;
    pl.max_pools = max_pools;
    if ((e = cudaMalloc(&pl.ctl, sizeof(Ctl))) != cudaSuccess) return e;
    if ((e = cudaMemset(pl.ctl, 0, sizeof(Ctl))) != cudaSuccess) return e;
    if ((e = cudaMalloc(&pl.cand, sizeof(unsigned long long) * (size_t) (max_pools > 0 ? max_pools : 1))) != cudaSuccess)
        return e;
    if ((e = cudaMalloc(&pl.lower, sizeof(float) * (size_t) (max_pools > 0 ? max_pools : 1))) != cudaSuccess) return e;
    const int smem = (int) score_smem_bytes();
    if ((e = cudaFuncSetAttribute(score_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess)
        return e;
    if ((e = cudaFuncSetAttribute(score_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess)
        return e;
    if ((e = cudaFuncSetAttribute(tc_rescore_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) != cudaSuccess)
        return e;
    return cudaFuncSetAttribute(score_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}

inline void release(Plan& pl) {
    cudaFree(pl.ctl);
    cudaFree(pl.cand);
    cudaFree(pl.lower);
    pl = Plan{};
}

// gf::dsa_score + gf::dsa_select in one call (score is written too, as Maya's).  cudaErrorNotSupported: shape not
// covered - leave it to Maya's kernels.
inline cudaError_t score_select(const Plan& pl, const float* iq, const float* pooled, const float* iw, int key_dim,
                                int idx_heads, int n_vis, int kpool, int top_pools, int n_sel, int pos, float* score,
                                int* cells, cudaStream_t s) {
    if (key_dim != kKey || idx_heads != kHeads || n_vis < 0 || n_vis > pl.max_pools || top_pools < 1 ||
        top_pools > kMaxTop || kpool < 1 || pl.ctl == nullptr)
        return cudaErrorNotSupported;
    Args a{iq, pooled, iw, score, cells, pl.ctl, pl.cand, n_vis, top_pools < n_vis ? top_pools : n_vis, top_pools,
           kpool, n_sel, pos, pl.small_n, pl.lower};
    if (n_vis == 0) {
        tail_only_kernel<<<1, 256, 0, s>>>(a);
        return cudaGetLastError();
    }
    score_kernel<true><<<pl.sms, kThreads, score_smem_bytes(), s>>>(a);
    if (n_vis > pl.small_n) {
        const int blocks = (((n_vis + 3) >> 2) + kCollectThreads - 1) / kCollectThreads;
        collect_kernel<<<blocks, kCollectThreads, 0, s>>>(a);
    }
    return cudaGetLastError();
}

// the score alone (bench probe; Maya's dsa_score bit for bit)
inline cudaError_t score_only(const Plan& pl, const float* iq, const float* pooled, const float* iw, int n_pools,
                              float* score, cudaStream_t s, bool math_only = false) {
    Args a{iq, pooled, iw, score, nullptr, nullptr, nullptr, n_pools, 0, 0, 0, 0, 0, 0, nullptr};
    if (math_only) score_kernel<false, true><<<pl.sms, kThreads, score_smem_bytes(), s>>>(a);
    else score_kernel<false><<<pl.sms, kThreads, score_smem_bytes(), s>>>(a);
    return cudaGetLastError();
}

}  // namespace dsa_idx
