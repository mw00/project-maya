// src/kernels/cuda/glm_prompt_dsa.cuh - the prompt path's DSA indexer in one pass (dsa_indexer in glm_batch.cu):
// dsa_score + dsa_select's scores bit for bit and their cells / n_sel. Shapes: idx_heads 32, idx_key 128.
//   score:  16 tokens a block (2 a warp, their queries held as mma A fragments for the whole launch), FP16 keys streamed
//           through a cp.async ring, HMMA.16816 in dsa_score's k order, its FP32 head chain; also, per token and group
//           of 16 pools, the largest order key (the select's pre-filter).
//   select: one block a token.  tau = the top-th largest group max (at least `top` pools score >= tau, so the top-th
//           score >= tau); the pools >= tau of the groups >= tau (all pools >= a per-thread-max cut while n_vis <= 16K)
//           sorted as (key, ~pool) = dsa_select's order.  Mass ties: an exact slow path, then select_top_pools.
// cp.async, __reduce_add_sync and mma.sync need sm_80: older targets compile traps, and dsa_indexer never runs there.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "dsa_topk.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define PDSA_SM80 0
#else
#define PDSA_SM80 1
#endif

namespace pdsa {

constexpr int kKey = 128, kHeads = 32;
constexpr int kWarps = 8, kThreads = 32 * kWarps;
constexpr int kTokW = 2, kTokB = kWarps * kTokW;   // tokens a warp, a block
constexpr int kStageP = 32, kStages = 4;           // pools a cp.async stage, ring depth
constexpr int kKRow = kKey + 8;                    // padded FP16 key row: 272 B, ldmatrix without bank conflicts
constexpr int kHS = 40, kTS = 32 * kHS + 16;       // dot tile [token][head][pool]: strides in floats (conflict-free)
constexpr int kGroup = 16;                         // pools a group max
constexpr int kSelT = 1024;                        // select block (Maya's select_top_pools needs 1024)
constexpr int kRV = 16;                            // select: keys a thread -> 16K pools direct, 1,024 groups
constexpr int kMaxTop = 1024;

__host__ __device__ constexpr size_t score_smem() {
    return (size_t) kStages * kStageP * kKRow * 2 + (size_t) kWarps * kTokW * kTS * 4;
}

struct ScoreArgs {
    const float* iq;          // [T][32][128] FP32 (Maya's B.iq)
    const __half* k16;        // [>= max_vis][128] FP16 = __float2half_rn(pooled)
    const float* iw;          // [T][32]
    float* score;             // [T][score_ld]
    unsigned* gmax;           // [T][gld]
    int score_ld, gld, p0, kpool, T, max_vis, chunk;
};

struct SelArgs {
    const float* score;
    const unsigned* gmax;
    int* cells;
    int* n_sel;
    unsigned* stats;          // tokens: [0] direct, [1] grouped (fast path), [2] Maya fallback, [3] slow path
    int score_ld, gld, p0, kpool, top_max, tail, n_sel_max;
};

struct Plan {
    int sms = 0, max_pools = 0, max_T = 0, gld = 0;
    __half* k16 = nullptr;    // FP16 keys scratch (max_pools rows) when the caller keeps no shadow
    unsigned* gmax = nullptr; // max_T x gld
    unsigned* stats = nullptr;
};

#ifdef GLM_PROMPT_DSA_PROF
__device__ unsigned long long g_prof[16];
__device__ int g_prof_block;
__device__ __forceinline__ void pprof(int i) {   // bench probe: phase timestamps of one select block
    if (threadIdx.x == 0 && (int) blockIdx.x == g_prof_block) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        g_prof[i] = t;
    }
}
#define PPROF(i) pprof(i)
#else
#define PPROF(i) do {} while (0)
#endif

// Maya's order key (dsa_topk.cuh score_key)
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ unsigned long long compo(unsigned key, int i) {
    return ((unsigned long long) key << 32) | (unsigned long long) (~(unsigned) i);
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, int bytes) {
#if PDSA_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_commit() {
#if PDSA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if PDSA_SM80
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
    __trap();
#endif
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
#if PDSA_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#else
    __trap();
#endif
}
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);   // Maya's rounding of q and k
    return *reinterpret_cast<const unsigned*>(&h);
}

// FP16 copy of pooled rows [p_lo, p_hi) (Maya's in-block conversion, done once)
__global__ void keys16_kernel(const float* __restrict__ pooled, __half* __restrict__ k16, long long e_lo, long long e_hi) {
    for (long long i = e_lo + 4 * ((long long) blockIdx.x * blockDim.x + threadIdx.x); i < e_hi;
         i += 4LL * gridDim.x * blockDim.x) {
        const float4 v = *(const float4*) (pooled + i);
        uint2 o;
        o.x = pack_h2(v.x, v.y);
        o.y = pack_h2(v.z, v.w);
        *(uint2*) (k16 + i) = o;
    }
}

// ---------------------------------------------------------------- score
// Block (token group of 16, chunk of a.chunk pools).  Warp w: tokens tb + 2w, +1 as A fragments (rows = heads, the
// 2 x 16-row tiles of each token); B = 8 pools a mma from the stage in shared memory; per accumulator the 8 k-steps in
// order (Maya's wmma 16x16x16 = the same HMMA.16816 chain).  Dots -> padded tile -> lane L chains token L/16's heads
// 0..31 for pools L%16 and L%16 + 16 (Maya's sum += relu(dot) * w).  MODE 3 (default): ping-pong warps; 1: chains
// after each stage; 2: chains interleaved by halves; 0: the mma loop alone (probe).  NB: n-tiles a batch.
template <int MODE = 3, int NB = 2>
__global__ void __launch_bounds__(kThreads, 1) score_kernel(const ScoreArgs a) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half* sk = (__half*) sm;
    float* sc = (float*) (sm + (size_t) kStages * kStageP * kKRow * 2);
    const int tid = threadIdx.x, L = tid & 31, warp = tid >> 5, g = L >> 2, t4 = L & 3;
    const int tb = blockIdx.x * kTokB, pc0 = blockIdx.y * a.chunk;
    const int t_last = min(a.T, tb + kTokB) - 1;
    const int vis_last = (a.p0 + t_last + 1) / a.kpool;
    if (pc0 >= vis_last) return;   // no pool of this chunk is visible to any token of the group
    const int pc1 = min(pc0 + a.chunk, vis_last);
    const int nst = (pc1 - pc0 + kStageP - 1) / kStageP;
    auto load = [&](int st) {
        const unsigned dst = smem_u32(sk + (st % kStages) * kStageP * kKRow);
#pragma unroll
        for (int i = 0; i < kStageP * 16 / kThreads; ++i) {
            const int c = tid + i * kThreads, pp = c >> 4, part = c & 15, p = pc0 + st * kStageP + pp;
            const bool ok = p < pc1;
            cp_async16(dst + (unsigned) (pp * kKRow + part * 8) * 2, a.k16 + (size_t) (ok ? p : pc0) * kKey + part * 8,
                       ok ? 16 : 0);
        }
    };
#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < nst) load(st);
        cp_commit();
    }
    const int tw = tb + warp * kTokW;
    unsigned A[kTokW][2][8][4];
#pragma unroll
    for (int tk = 0; tk < kTokW; ++tk) {
        const bool ok = tw + tk < a.T;
        const float* q = a.iq + (size_t) (ok ? tw + tk : 0) * kHeads * kKey;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const float* r0 = q + (mt * 16 + g) * kKey + kk * 16 + 2 * t4;
                const float* r1 = r0 + 8 * kKey;
                const float2 z = make_float2(0.0f, 0.0f);
                const float2 x0 = ok ? *(const float2*) r0 : z, x1 = ok ? *(const float2*) r1 : z;
                const float2 x2 = ok ? *(const float2*) (r0 + 8) : z, x3 = ok ? *(const float2*) (r1 + 8) : z;
                A[tk][mt][kk][0] = pack_h2(x0.x, x0.y);
                A[tk][mt][kk][1] = pack_h2(x1.x, x1.y);
                A[tk][mt][kk][2] = pack_h2(x2.x, x2.y);
                A[tk][mt][kk][3] = pack_h2(x3.x, x3.y);
            }
    }
    const int tk_e = L >> 4, j = L & 15, tok_e = tw + tk_e;
    const bool tok_ok = tok_e < a.T;
    const int nvis_e = tok_ok ? (a.p0 + tok_e + 1) / a.kpool : 0;
    float w[kHeads];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) w[h] = tok_ok ? a.iw[(size_t) tok_e * kHeads + h] : 0.0f;
    float* ct = sc + warp * kTokW * kTS;
    float* srow = a.score + (size_t) (tok_ok ? tok_e : 0) * a.score_ld;
    unsigned* grow = a.gmax + (size_t) (tok_ok ? tok_e : 0) * a.gld;
    const unsigned kaddr0 = smem_u32(sk + (L & 7) * kKRow + (L >> 3) * 8);
    float sink = 0.0f;
    // NB x 8 pools x 128 k for the warp's 4 (token, 16-head) accumulators each; the 8 k-steps in order, then the tile
    auto mma_nt = [&](int nt, unsigned kaddr) {
        float acc[NB][kTokW][2][4];
#pragma unroll
        for (int n = 0; n < NB; ++n)
#pragma unroll
            for (int tk = 0; tk < kTokW; ++tk)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[n][tk][mt][e] = 0.0f;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            unsigned b[NB][4];   // k-steps 2 k2 (b0, b1) and 2 k2 + 1 (b2, b3) of pools (nt + n) * 8 .. +7
#pragma unroll
            for (int n = 0; n < NB; ++n) ldsm_x4(b[n], kaddr + (unsigned) ((nt + n) * 8 * kKRow + k2 * 32) * 2);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
#pragma unroll
                for (int n = 0; n < NB; ++n)
#pragma unroll
                    for (int tk = 0; tk < kTokW; ++tk)
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt)
                            mma16816(acc[n][tk][mt], A[tk][mt][2 * k2 + kk], b[n][2 * kk], b[n][2 * kk + 1]);
        }
#pragma unroll
        for (int n = 0; n < NB; ++n)
#pragma unroll
            for (int tk = 0; tk < kTokW; ++tk)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (MODE == 0) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) sink += acc[n][tk][mt][e];
                    } else {
                        float* r0 = ct + tk * kTS + (mt * 16 + g) * kHS + (nt + n) * 8 + 2 * t4;
                        *(float2*) r0 = make_float2(acc[n][tk][mt][0], acc[n][tk][mt][1]);
                        *(float2*) (r0 + 8 * kHS) = make_float2(acc[n][tk][mt][2], acc[n][tk][mt][3]);
                    }
                }
    };
    auto chain = [&](int col) {   // Maya's epilogue: sum += relu(dot) * w over heads 0..31 in order
        const float* cr = ct + tk_e * kTS + col;
        float s = 0.0f;
#pragma unroll
        for (int h = 0; h < kHeads; ++h) s += fmaxf(cr[h * kHS], 0.0f) * w[h];
        return s;
    };
    auto emit = [&](int p, float s) {   // the score of pool p, and (lane j == 0) its 16-pool group's largest key
        unsigned k = 0u;
        if (p < nvis_e) {
            srow[p] = s;
            k = order_key(s);
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) k = max(k, __shfl_xor_sync(0xffffffffu, k, o));
        if (j == 0 && p < nvis_e) grow[p / kGroup] = k;
    };
    auto epilogue = [&](int pg) {
        const float s0 = chain(j), s1 = chain(16 + j);
        emit(pg + j, s0);
        emit(pg + 16 + j, s1);
        __syncwarp();
    };
    // MODE 3 (ping-pong): warps 4..7 share sub-cores with 0..3 (warp % 4) and run one stage behind on the chains, so
    // each sub-core's tensor pipe has one warp's mma while the other warp chains
    const bool lag = MODE == 3 && warp >= kWarps / 2;
    int pg_prev = 0;
    for (int st = 0; st <= nst; ++st) {
        if (st < nst) {
            cp_wait<kStages - 2>();
            __syncthreads();
            if (st + kStages - 1 < nst) load(st + kStages - 1);
            cp_commit();
        }
        if (lag && st > 0) epilogue(pg_prev);
        // MODE 2: the previous stage's pools 16..31 (tile columns 16..31) overlap this stage's first half
        if (MODE == 2 && st > 0) emit(pg_prev + 16 + j, chain(16 + j));
        if (st == nst) break;
        const unsigned kaddr = kaddr0 + (unsigned) ((st % kStages) * kStageP * kKRow) * 2;
        const int pg = pc0 + st * kStageP;
#pragma unroll
        for (int nt = 0; nt < 2; nt += NB) mma_nt(nt, kaddr);
        if (MODE == 2) {   // pools 0..15 overlap the second half
            __syncwarp();
            emit(pg + j, chain(j));
        }
#pragma unroll
        for (int nt = 2; nt < 4; nt += NB) mma_nt(nt, kaddr);
        __syncwarp();
        if (MODE == 1 || (MODE == 3 && !lag)) epilogue(pg);
        pg_prev = pg;
    }
    if (MODE == 0 && sink == 1.2345f) ct[L] = sink;   // keeps the probe's mma alive
}

// ---------------------------------------------------------------- select
// sum over the block (NT / 32 warps); s_red: 64 ints, double-buffered (one barrier a call)
__device__ __forceinline__ int block_sum(int v, int* s_red, int& ph) {
#if PDSA_SM80
    v = __reduce_add_sync(0xffffffffu, v);
    if ((threadIdx.x & 31) == 0) s_red[ph * 32 + (threadIdx.x >> 5)] = v;
    __syncthreads();
    int r = (int) (threadIdx.x & 31) < (int) (blockDim.x >> 5) ? s_red[ph * 32 + (threadIdx.x & 31)] : 0;
    ph ^= 1;
    return __reduce_add_sync(0xffffffffu, r);
#else
    __trap();
    return 0;
#endif
}

// exclusive scan of one int a thread in thread order; *total = the sum.  s: 34 ints.
__device__ __forceinline__ int block_scan(int v, int* s, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = lane < nw ? s[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        if (lane < nw) s[lane] = t;
    }
    __syncthreads();
    const int before = (w > 0 ? s[w - 1] : 0) + x - v;
    total = s[nw - 1];
    __syncthreads();
    return before;
}

// the k-th largest of the block's keys v[r < nr] (MSB-first bit search: the largest c with count(>= c) >= k)
template <int R>
__device__ unsigned kth_largest(const unsigned (&v)[R], int nr, int k, int* s_red, int& ph) {
    unsigned res = 0u;
    for (int b = 31; b >= 0; --b) {
        const unsigned cand = res | (1u << b);
        int c = 0;
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (r < nr) c += v[r] >= cand;
        if (block_sum(c, s_red, ph) >= k) res = cand;
    }
    return res;
}

// descending register bitonic sort of NT * R values (as glm_dsa_indexer.cuh's)
__device__ __forceinline__ unsigned long long keep(unsigned long long mine, unsigned long long other, bool keep_max) {
    return keep_max ? (mine > other ? mine : other) : (mine < other ? mine : other);
}
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

// the k-th largest of one key a thread (0 = none): the same bit search, counted by the barrier itself
__device__ __forceinline__ unsigned kth_one(unsigned v, int k) {
    unsigned res = 0u;
    for (int b = 31; b >= 0; --b) {
        const unsigned cand = res | (1u << b);
        if (__syncthreads_count(v >= cand) >= k) res = cand;
    }
    return res;
}

// src[0, cnt) (distinct composites, cnt <= NT * R) sorted descending; ranks < top written as cells.  buf: 2 NT R.
template <int NT, int R>
__device__ __forceinline__ void sort_cells(const unsigned long long* src, int cnt, unsigned long long* buf, int top,
                                           int kpool, int* cells) {
    unsigned long long x[R];
    const int ebase = threadIdx.x * R;
#pragma unroll
    for (int r = 0; r < R; ++r) x[r] = ebase + r < cnt ? src[ebase + r] : 0ull;
    __syncthreads();
    reg_bitonic_desc<NT, R>(x, buf);
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (ebase + r < top) {
            const int p = (int) ~(unsigned) (x[r] & 0xffffffffull);
            for (int q = 0; q < kpool; ++q) cells[(ebase + r) * kpool + q] = p * kpool + q;
        }
}

constexpr int kCap0 = 4096, kCapC = 2048;   // select fast path: group list, candidates (u64 each)
constexpr size_t select_smem() { return (size_t) (kCap0 + kCapC) * 8; }

// One block a token (Maya's dsa_select_kernel layout: -1s, n_sel, the selection in rank order, the tail).
// RG: group maxima a thread (n_vis up to RG * 1024 * 16 pools grouped; Maya's select beyond).
// Fast path: tau0 = the top-th largest per-thread maximum (>= top values reach it); grouped: tau = the top-th largest
// group max (from the groups >= tau0), candidates = pools >= tau of the groups >= tau; direct: candidates = pools >=
// tau0.  The candidates (<= 2,048) sorted as (key, ~pool) composites = Maya's order.  Else the slow exact path.
template <int RG>
__global__ void __launch_bounds__(kSelT, 1) select_kernel(const SelArgs a) {
    constexpr int NT = kSelT, LCAP = kRV * NT / kGroup;   // 1,024 candidate groups
    extern __shared__ __align__(16) unsigned char dsm[];
    unsigned long long* s_u = (unsigned long long*) dsm;   // kCap0: group list (fast) / sort buffers
    unsigned long long* s_c = s_u + kCap0;                 // kCapC: candidates (fast) / s_fin (slow)
    unsigned long long* s_fin = s_c;
    unsigned long long* s_buf = s_u;
    __shared__ int s_red[64];
    __shared__ int s_scan[34];
    __shared__ int s_list[LCAP];
    __shared__ int s_cnt, s_n0, s_nc, s_ncg;
    const int t = blockIdx.x, tid = threadIdx.x;
    PPROF(0);
    const int pos = a.p0 + t, n_vis = (pos + 1) / a.kpool;
    const int top = min(a.top_max, n_vis);
    const int n_sel = a.kpool * top + (a.tail ? a.kpool - 1 : 0);
    int* cells = a.cells + (size_t) t * a.n_sel_max;
    for (int i = tid; i < n_sel; i += NT) cells[i] = -1;
    if (tid == 0) a.n_sel[t] = n_sel;
    __syncthreads();
    PPROF(1);
    if (n_vis <= a.top_max) {
        for (int i = tid; i < n_vis * a.kpool; i += NT) cells[i] = i;
    } else {
        const float* sc = a.score + (size_t) t * a.score_ld;
        int ph = 0;
        const bool grouped = n_vis > kRV * NT;
        bool done = false;
        int fast_ncg = -1;   // groups at tau, when the fast path got that far
        {   // fast path
            constexpr int RX = kRV > RG ? kRV : RG;
            const int nx = grouped ? (n_vis + kGroup - 1) / kGroup : n_vis;   // stage-A values: group maxima / keys
            if (nx <= RX * NT) {
                const unsigned* gr = a.gmax + (size_t) t * a.gld;
                const int nr = (nx + NT - 1) / NT;
                unsigned x[RX], tmax = 0u;
#pragma unroll
                for (int r = 0; r < RX; ++r) {
                    const int i = r * NT + tid;
                    x[r] = r < nr && i < nx ? (grouped ? __ldcg(gr + i) : order_key(__ldcg(sc + i))) : 0u;
                    tmax = max(tmax, x[r]);
                }
                if (tid == 0) s_n0 = s_nc = s_ncg = 0;
                const unsigned tau0 = kth_one(tmax, top);   // > 0: at least top + 1 threads hold a value
                PPROF(2);
#pragma unroll
                for (int r = 0; r < RX; ++r)
                    if (r < nr && x[r] >= tau0) {
                        const int i = r * NT + tid, slot = atomicAdd(&s_n0, 1);
                        if (grouped && slot < kCap0) s_u[slot] = (unsigned long long) x[r] << 32 | (unsigned) i;
                        if (!grouped && slot < kCapC) s_c[slot] = compo(x[r], i);
                    }
                __syncthreads();
                const int c0 = s_n0;
                int m = grouped ? kCapC + 1 : c0;
                PPROF(3);
                if (grouped && c0 <= kCap0) {
                    unsigned tau;   // the top-th largest group max: it is in the list (>= tau0)
                    if (c0 <= NT) {
                        tau = kth_one(tid < c0 ? (unsigned) (s_u[tid] >> 32) : 0u, top);
                    } else {
                        unsigned y[kCap0 / NT];
#pragma unroll
                        for (int r = 0; r < kCap0 / NT; ++r) {
                            const int i = r * NT + tid;
                            y[r] = i < c0 ? (unsigned) (s_u[i] >> 32) : 0u;
                        }
                        tau = kth_largest<kCap0 / NT>(y, kCap0 / NT, top, s_red, ph);
                    }
                    PPROF(4);
                    for (int i = tid; i < c0; i += NT)
                        if ((unsigned) (s_u[i] >> 32) >= tau) {
                            const int slot = atomicAdd(&s_ncg, 1);
                            if (slot < LCAP) s_list[slot] = (int) (s_u[i] & 0xffffffffu);
                        }
                    __syncthreads();
                    const int n_cg = s_ncg;
                    fast_ncg = n_cg;
                    if (n_cg <= LCAP) {
                        constexpr int U = 8;   // loads in flight a thread
                        for (int e0 = 0; e0 < n_cg * kGroup; e0 += U * NT) {
                            float f[U];
                            int p[U];
#pragma unroll
                            for (int u = 0; u < U; ++u) {
                                const int e = e0 + u * NT + tid;
                                p[u] = e < n_cg * kGroup ? s_list[e / kGroup] * kGroup + e % kGroup : n_vis;
                                f[u] = p[u] < n_vis ? __ldcg(sc + p[u]) : 0.0f;
                            }
#pragma unroll
                            for (int u = 0; u < U; ++u) {
                                const unsigned k = order_key(f[u]);
                                if (p[u] < n_vis && k >= tau) {
                                    const int slot = atomicAdd(&s_nc, 1);
                                    if (slot < kCapC) s_c[slot] = compo(k, p[u]);
                                }
                            }
                        }
                        __syncthreads();
                        m = s_nc;
                    }
                    PPROF(5);
                }
                if (m <= kCapC) {
                    PPROF(7);
                    if (m <= NT) sort_cells<NT, 1>(s_c, m, s_u, top, a.kpool, cells);
                    else sort_cells<NT, 2>(s_c, m, s_u, top, a.kpool, cells);
                    PPROF(8);
                    if (tid == 0) atomicAdd(&a.stats[grouped ? 1 : 0], 1u);
                    done = true;
                }
            }
        }
        if (!done) {   // slow exact path (mass ties)
        __syncthreads();
        if (tid == 0) atomicAdd(&a.stats[3], 1u);
        bool fallback = false;
        int nval = n_vis;
        if (grouped) {
            const int ng = (n_vis + kGroup - 1) / kGroup;
            if (ng > RG * NT || fast_ncg > LCAP) {   // (mass ties at tau: Maya's select)
                fallback = true;
            } else {
                unsigned gm[RG];
                const unsigned* gr = a.gmax + (size_t) t * a.gld;
                const int nr = (ng + NT - 1) / NT;
#pragma unroll
                for (int r = 0; r < RG; ++r) {
                    const int i = r * NT + tid;
                    gm[r] = r < nr && i < ng ? __ldcg(gr + i) : 0u;
                }
                const unsigned tau = kth_largest<RG>(gm, nr, top, s_red, ph);
                PPROF(2);
                int base = 0;
#pragma unroll
                for (int r = 0; r < RG; ++r)
                    if (r < nr) {
                        const int f = gm[r] >= tau && r * NT + tid < ng;
                        int tot;
                        const int ex = block_scan(f, s_scan, tot);
                        if (f && base + ex < LCAP) s_list[base + ex] = r * NT + tid;
                        base += tot;
                    }
                fallback = base > LCAP;
                nval = base * kGroup;
                __syncthreads();
                PPROF(3);
            }
        }
        if (fallback) {
            if (tid == 0) atomicAdd(&a.stats[2], 1u);
            maya::kernels::dsa::select_top_pools(sc, n_vis, top, a.kpool, cells);
        } else {
            // element e = r * NT + tid, ascending in pool order (direct: pool e; grouped: pool e % 16 of group list[e / 16])
            auto pool_of = [&](int e) { return grouped ? s_list[e >> 4] * kGroup + (e & 15) : e; };
            const int nr = (nval + NT - 1) / NT;
            unsigned v[kRV];
#pragma unroll
            for (int r = 0; r < kRV; ++r) {
                const int e = r * NT + tid;
                const int p = r < nr && e < nval ? pool_of(e) : n_vis;
                v[r] = p < n_vis ? order_key(__ldcg(sc + p)) : 0u;
            }
            PPROF(4);
            const unsigned K = kth_largest<kRV>(v, nr, top, s_red, ph);
            PPROF(5);   // the top-th largest score key (> 0)
            int cgt = 0, ceq = 0;
#pragma unroll
            for (int r = 0; r < kRV; ++r)
                if (r < nr) {
                    cgt += v[r] > K;
                    ceq += v[r] == K;
                }
            const int n_gt = block_sum(cgt, s_red, ph), n_eq = block_sum(ceq, s_red, ph);
            const int need = top - n_gt;   // 1 <= need <= n_eq: the lowest pools among those tied at K
            if (tid == 0) s_cnt = 0;
            __syncthreads();
            PPROF(6);
            if (n_eq == need) {
#pragma unroll
                for (int r = 0; r < kRV; ++r)
                    if (r < nr && v[r] >= K) s_fin[atomicAdd(&s_cnt, 1)] = compo(v[r], pool_of(r * NT + tid));
            } else {
#pragma unroll
                for (int r = 0; r < kRV; ++r)
                    if (r < nr && v[r] > K) s_fin[atomicAdd(&s_cnt, 1)] = compo(v[r], pool_of(r * NT + tid));
                int base = 0;
#pragma unroll
                for (int r = 0; r < kRV; ++r)
                    if (r < nr && base < need) {   // (uniform)
                        const int f = v[r] == K;
                        int tot;
                        const int ex = block_scan(f, s_scan, tot);
                        if (f && base + ex < need) s_fin[n_gt + base + ex] = compo(v[r], pool_of(r * NT + tid));
                        base += tot;
                    }
            }
            __syncthreads();
            unsigned long long x[1] = {tid < top ? s_fin[tid] : 0ull};
            __syncthreads();
            PPROF(7);
            reg_bitonic_desc<NT, 1>(x, s_buf);
            PPROF(8);
            if (tid < top) {
                const int p = (int) ~(unsigned) (x[0] & 0xffffffffull);
                for (int q = 0; q < a.kpool; ++q) cells[tid * a.kpool + q] = p * a.kpool + q;
            }
        }
        }
    }
    __syncthreads();
    PPROF(9);
    if (threadIdx.x == 0 && a.tail) {   // Maya's tail
        for (int m = 0; m < a.kpool - 1; ++m) {
            const int cell = n_vis * a.kpool + m;
            if (cell <= pos) cells[top * a.kpool + m] = cell;
        }
    }
}

// ---------------------------------------------------------------- host side
inline cudaError_t init(Plan& pl, int max_pools, int max_T, bool keys_scratch = true) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    if ((e = cudaDeviceGetAttribute(&pl.sms, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess) return e;
    pl.max_pools = max_pools;
    pl.max_T = max_T;
    pl.gld = (max_pools + kGroup - 1) / kGroup;
    if (keys_scratch && (e = cudaMalloc(&pl.k16, (size_t) max_pools * kKey * sizeof(__half))) != cudaSuccess) return e;
    if ((e = cudaMalloc(&pl.gmax, (size_t) max_T * pl.gld * sizeof(unsigned))) != cudaSuccess) return e;
    if ((e = cudaMalloc(&pl.stats, 4 * sizeof(unsigned))) != cudaSuccess) return e;
    if ((e = cudaMemset(pl.stats, 0, 4 * sizeof(unsigned))) != cudaSuccess) return e;
    const int sm = (int) score_smem(), ss = (int) select_smem();
    const cudaFuncAttribute at = cudaFuncAttributeMaxDynamicSharedMemorySize;
    if ((e = cudaFuncSetAttribute(score_kernel<>, at, sm)) != cudaSuccess) return e;
    if ((e = cudaFuncSetAttribute(select_kernel<1>, at, ss)) != cudaSuccess) return e;
    if ((e = cudaFuncSetAttribute(select_kernel<2>, at, ss)) != cudaSuccess) return e;
    if ((e = cudaFuncSetAttribute(select_kernel<4>, at, ss)) != cudaSuccess) return e;
    if ((e = cudaFuncSetAttribute(select_kernel<8>, at, ss)) != cudaSuccess) return e;
    return cudaFuncSetAttribute(select_kernel<16>, at, ss);
}

// the shared-memory opt-in of a non-default score variant (bench probes)
template <int MODE, int NB>
inline cudaError_t allow_score() {
    return cudaFuncSetAttribute(score_kernel<MODE, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) score_smem());
}

inline void release(Plan& pl) {
    cudaFree(pl.k16);
    cudaFree(pl.gmax);
    cudaFree(pl.stats);
    pl = Plan{};
}

// FP16 shadow rows [p_lo, p_hi) of pooled (the engine can keep one per DSA layer, updated after each dsa_pool)
inline cudaError_t keys16(const float* pooled, __half* k16, int p_lo, int p_hi, cudaStream_t s) {
    if (p_hi <= p_lo) return cudaSuccess;
    const long long n4 = (long long) (p_hi - p_lo) * kKey / 4;
    const int blocks = (int) (n4 < 2048LL * 256 ? (n4 + 255) / 256 : 2048);
    keys16_kernel<<<blocks, 256, 0, s>>>(pooled, k16, (long long) p_lo * kKey, (long long) p_hi * kKey);
    return cudaGetLastError();
}

inline int pick_chunk(const Plan& pl, int T, int max_vis) {
    const long long groups = (T + kTokB - 1) / kTokB;
    long long c = (max_vis * groups + 4LL * pl.sms - 1) / (4LL * pl.sms);   // ~4 waves
    c = (c + kStageP - 1) / kStageP * kStageP;
    return (int) (c < 1024 ? 1024 : c > 4096 ? 4096 : c);
}

// gb::dsa_score's job (+ the group maxima in pl.gmax).  k16: FP16 rows [0, max_vis) of the pooled keys.
template <int MODE = 3, int NB = 2>
inline cudaError_t score(const Plan& pl, const float* iq, const __half* k16, const float* iw, int p0, int kpool, int T,
                         int max_vis, float* sc, int score_ld, cudaStream_t s, int chunk = 0) {
    if (T <= 0 || max_vis <= 0) return cudaSuccess;
    if (T > pl.max_T || max_vis > pl.max_pools) return cudaErrorNotSupported;
    if (chunk <= 0) chunk = pick_chunk(pl, T, max_vis);
    const ScoreArgs a{iq, k16, iw, sc, pl.gmax, score_ld, pl.gld, p0, kpool, T, max_vis, chunk};
    const dim3 grid((unsigned) ((T + kTokB - 1) / kTokB), (unsigned) ((max_vis + chunk - 1) / chunk));
    score_kernel<MODE, NB><<<grid, kThreads, score_smem(), s>>>(a);
    return cudaGetLastError();
}

// gb::dsa_select's job; grouped tokens read pl.gmax from score() of the same call.
inline cudaError_t select(const Plan& pl, const float* sc, int score_ld, int p0, int kpool, int top_pools_max, int tail,
                          int T, int n_sel_max, int* cells, int* n_sel, cudaStream_t s) {
    if (T <= 0) return cudaSuccess;
    if (top_pools_max > kMaxTop || T > pl.max_T) return cudaErrorNotSupported;
    const SelArgs a{sc, pl.gmax, cells, n_sel, pl.stats, score_ld, pl.gld, p0, kpool, top_pools_max, tail, n_sel_max};
    const int max_vis = (p0 + T) / kpool, ng = (max_vis + kGroup - 1) / kGroup;
    if (ng <= kSelT) select_kernel<1><<<T, kSelT, select_smem(), s>>>(a);
    else if (ng <= 2 * kSelT) select_kernel<2><<<T, kSelT, select_smem(), s>>>(a);
    else if (ng <= 4 * kSelT) select_kernel<4><<<T, kSelT, select_smem(), s>>>(a);
    else if (ng <= 8 * kSelT) select_kernel<8><<<T, kSelT, select_smem(), s>>>(a);
    else select_kernel<16><<<T, kSelT, select_smem(), s>>>(a);
    return cudaGetLastError();
}

// Drop-in for the call site's pair (glm_prefill.cu): score only when max_vis > top_pools_max, as Maya's.
// pooled16: the caller's FP16 shadow of pooled, or nullptr -> rows [0, max_vis) converted into pl.k16 first.
inline cudaError_t score_select(const Plan& pl, const float* iq, const float* pooled, const __half* pooled16,
                                const float* iw, int key_dim, int idx_heads, int p0, int kpool, int T, int max_vis,
                                float* sc, int score_ld, int top_pools_max, int tail, int n_sel_max, int* cells,
                                int* n_sel, cudaStream_t s) {
    if (key_dim != kKey || idx_heads != kHeads || top_pools_max > kMaxTop || T > pl.max_T || max_vis > pl.max_pools)
        return cudaErrorNotSupported;
    cudaError_t e = cudaSuccess;
    if (max_vis > top_pools_max) {
        if (pooled16 == nullptr) {
            if (pl.k16 == nullptr) return cudaErrorNotSupported;
            if ((e = keys16(pooled, pl.k16, 0, max_vis, s)) != cudaSuccess) return e;
            pooled16 = pl.k16;
        }
        if ((e = score(pl, iq, pooled16, iw, p0, kpool, T, max_vis, sc, score_ld, s)) != cudaSuccess) return e;
    }
    return select(pl, sc, score_ld, p0, kpool, top_pools_max, tail, T, n_sel_max, cells, n_sel, s);
}

}  // namespace pdsa
