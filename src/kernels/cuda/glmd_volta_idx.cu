// src/kernels/cuda/glmd_volta_idx.cu - glm-dsa B10.3 lightning-indexer scorers (WMMA and CUDA-core, F32 or FP16 keys:
// B10.8) and B10.4 multi-block radix top-k.  See include/strata/kernels/glmd_volta.hpp; the twins are in
// src/core/glm_dsa_cpu_volta_twin.cpp.
//
// HIP builds compile the CUDA-core scorer and the radix top-k and leave the WMMA scorer out (nvcuda::wmma: GLMD_WMMA,
// glmd_arch.cuh, is 0 there, and the scorer's kernel and launcher are not built): there idx_wmma_enabled() and
// idx_score_wmma_supported() are false and idx_score_wmma() refuses.
#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "glmd_arch.cuh"   // STRATA_GRID_CONSTANT, GLMD_WMMA (Pascal pass)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if GLMD_WMMA
#include <mma.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>

#if defined(STRATA_USE_HIP)
// The radix top-k's last block reads the other blocks' counts after their publication fences: an agent-scope acquire
// load is CUDA's __ldcg there (HIP spells __ldcg only for its vector and half types; glm_fast.cu does the same for
// float).
__device__ __forceinline__ unsigned __ldcg(const unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

namespace glmd {
namespace {

#if GLMD_WMMA
using namespace nvcuda;
#endif

std::atomic<int> g_errors{0};

void refuse(const char* what) {
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_volta %s: shape refused\n", what);
}
bool launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    g_errors.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "glmd_volta %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
// an on/off switch, default off ("auto" counts as on here: these dispatchers have no auto rule of their own)
bool env_on(const char* name) { return strata::glmenv::env_switch(name, strata::glmenv::kOff, true) != strata::glmenv::kOff; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ __half half_sat(float v) { return __float2half(fmaxf(-65504.0f, fminf(65504.0f, v))); }

// ---------------------------------------------------------------- B10.3 WMMA scorer (CUDA: nvcuda::wmma)
#if !defined(STRATA_USE_HIP)
// 8 warps x 16 cells per block, TB tokens per block (blockIdx.y), NTH = idx_heads / 16 column tiles per token.
// Shared: the queries of the block's tokens as FP16 [TB * H][D + 8] (matrix_b, col-major), the head weights, and per
// warp a 16 x (D + 8) FP16 key tile that becomes the warp's 16 x H f32 output scratch after the k-loop.
constexpr int kIdxWarps = 8;
size_t idx_wmma_smem(int TB, int H, int D) {
    return (size_t) TB * H * (D + 8) * 2 + (size_t) kIdxWarps * 16 * (D + 8) * 2 + (size_t) TB * H * 4;
}

template <int TB, int NTH, bool F16K>
__global__ void __launch_bounds__(kIdxWarps * 32) idx_wmma_kernel(const STRATA_GRID_CONSTANT IdxScoreWmma a, int ld) {
#if !GLMD_WMMA
    __trap();   // Pascal (no tensor cores): stops loudly - nothing keeps the auto scorer choice off below sm_70 yet
#else
    using FragAcc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
    constexpr int H = NTH * 16;
    const int D = a.idx_dim, LDD = D + 8;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __half* sB = (__half*) smem_raw;                       // [TB * H][LDD]
    __half* sK = sB + (size_t) TB * H * LDD;               // [warps][16][LDD]
    float* sW = (float*) (sK + (size_t) kIdxWarps * 16 * LDD);
    __shared__ int s_pos[TB];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t0 = blockIdx.y * TB, ntok = min(TB, a.T - t0);
    for (int i = tid; i < TB * H * D; i += blockDim.x) {
        const int r = i / D, e = i - r * D;   // r = tok * H + h
        const int tok = r / H;
        sB[(size_t) r * LDD + e] = tok < ntok ? half_sat(a.iq[(size_t) t0 * H * D + i]) : __float2half(0.0f);
    }
    for (int i = tid; i < TB * H; i += blockDim.x) sW[i] = i / H < ntok ? a.w[(size_t) t0 * H + i] : 0.0f;
    if (tid < TB) s_pos[tid] = tid < ntok ? (a.pos ? a.pos[t0 + tid] : a.n_kv - 1) : -1;
    __syncthreads();
    const int s0 = (blockIdx.x * kIdxWarps + warp) * 16;
    if (s0 >= a.n_kv) return;
    int maxp = -1;
#pragma unroll
    for (int t = 0; t < TB; ++t) maxp = max(maxp, s_pos[t]);
    if (s0 > maxp) {   // no token sees this tile
        if (lane < 16 && s0 + lane < a.n_kv)
            for (int t = 0; t < ntok; ++t) a.score[(size_t) (t0 + t) * ld + s0 + lane] = -INFINITY;
        return;
    }
    __half* kt = sK + (size_t) warp * 16 * LDD;
    if constexpr (F16K) {
        const int vpr = D / 8;
        for (int i = lane; i < 16 * vpr; i += 32) {
            const int r = i / vpr, c = i - r * vpr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (s0 + r < a.n_kv) v = ((const uint4*) a.keys)[(size_t) (s0 + r) * vpr + c];
            *(uint4*) (kt + r * LDD + 8 * c) = v;
        }
    } else {
        const int vpr = D / 4;
        for (int i = lane; i < 16 * vpr; i += 32) {
            const int r = i / vpr, c = i - r * vpr;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (s0 + r < a.n_kv) v = ((const float4*) a.keys)[(size_t) (s0 + r) * vpr + c];
            __half2* d = (__half2*) (kt + r * LDD + 4 * c);
            d[0] = __halves2half2(half_sat(v.x), half_sat(v.y));
            d[1] = __halves2half2(half_sat(v.z), half_sat(v.w));
        }
    }
    __syncwarp();
    FragAcc acc[TB * NTH];
#pragma unroll
    for (int j = 0; j < TB * NTH; ++j) wmma::fill_fragment(acc[j], 0.0f);
    for (int k = 0; k < D / 16; ++k) {
        FragA fa;
        wmma::load_matrix_sync(fa, kt + k * 16, LDD);
#pragma unroll
        for (int j = 0; j < TB * NTH; ++j) {
            if (j / NTH >= ntok) continue;
            FragB fb;
            wmma::load_matrix_sync(fb, sB + (size_t) j * 16 * LDD + k * 16, LDD);
            wmma::mma_sync(acc[j], fa, fb, acc[j]);
        }
    }
    __syncwarp();
    float* scr = (float*) kt;   // 16 x H f32: 64 * H <= 32 * LDD by idx_wmma_shape_rule (2 * H <= D + 8)
#pragma unroll
    for (int t = 0; t < TB; ++t) {
        if (t >= ntok) break;
#pragma unroll
        for (int n = 0; n < NTH; ++n) wmma::store_matrix_sync(scr + n * 16, acc[t * NTH + n], H, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16 && s0 + lane < a.n_kv) {
            const int s = s0 + lane;
            float v = -INFINITY;
            if (s <= s_pos[t]) {
                v = 0.0f;
                for (int h = 0; h < H; ++h) v += fmaxf(scr[lane * H + h], 0.0f) * sW[t * H + h];
            }
            a.score[(size_t) (t0 + t) * ld + s] = v;
        }
        __syncwarp();
    }
#endif
}

template <int TB, int NTH>
bool launch_idx_wmma(const IdxScoreWmma& a, int ld, cudaStream_t s) {
    const size_t smem = idx_wmma_smem(TB, NTH * 16, a.idx_dim);
    if (smem > 48 * 1024) {
        static std::atomic<unsigned> done[2]{};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev >= 32) return false;
        const int f = a.key_f16 ? 1 : 0;
        if (!(done[f].load() & (1u << dev))) {
            const cudaError_t e = a.key_f16 ? cudaFuncSetAttribute(idx_wmma_kernel<TB, NTH, true>,
                                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem)
                                            : cudaFuncSetAttribute(idx_wmma_kernel<TB, NTH, false>,
                                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            if (e != cudaSuccess) return false;
            done[f].fetch_or(1u << dev);
        }
    }
    const dim3 grid((a.n_kv + kIdxWarps * 16 - 1) / (kIdxWarps * 16), (a.T + TB - 1) / TB);
    if (a.key_f16)
        idx_wmma_kernel<TB, NTH, true><<<grid, kIdxWarps * 32, smem, s>>>(a, ld);
    else
        idx_wmma_kernel<TB, NTH, false><<<grid, kIdxWarps * 32, smem, s>>>(a, ld);
    return launch_check("idx_score_wmma");
}
#endif  // !STRATA_USE_HIP (the WMMA scorer)

// ---------------------------------------------------------------- the CUDA-core scorer (glmf::dsa_score's form)
// a warp per cell, the queries of token blockIdx.y in shared memory; F32 or FP16 keys
template <bool F16K>
__global__ void __launch_bounds__(256) idx_simple_kernel(const STRATA_GRID_CONSTANT IdxScoreWmma a, int ld) {
    extern __shared__ float s_q[];   // H * D, then H weights
    const int H = a.idx_heads, D = a.idx_dim, t = blockIdx.y;
    float* s_w = s_q + H * D;
    for (int i = threadIdx.x; i < H * D; i += blockDim.x) s_q[i] = a.iq[(size_t) t * H * D + i];
    for (int i = threadIdx.x; i < H; i += blockDim.x) s_w[i] = a.w[(size_t) t * H + i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int s = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (s >= a.n_kv) return;
    const int pt = a.pos ? a.pos[t] : a.n_kv - 1;
    float acc = -INFINITY;
    if (s <= pt) {
        acc = 0.0f;
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
            for (int e = lane; e < D; e += 32) {
                const float k = F16K ? __half2float(((const __half*) a.keys)[(size_t) D * s + e])
                                     : ((const float*) a.keys)[(size_t) D * s + e];
                dot += s_q[h * D + e] * k;
            }
            dot = warp_sum(dot);
            acc += fmaxf(dot, 0.0f) * s_w[h];
        }
    }
    if (lane == 0) a.score[(size_t) t * ld + s] = acc;
}

// ---------------------------------------------------------------- B10.4 multi-block radix top-k
constexpr int G = kTopkMbBlocks, NT = kTopkMbThreads;
// per-row scratch (u32): hist[256] | state[8] {prefix, mask, need, ctr_pass, ctr_count, ctr_write, -, -} |
// gt[G] | eq[G] | off[G] | eqb[G]
constexpr int kRowWords = 256 + 8 + 4 * G;

__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ int row_n(int n_kv, const int* pos, int t) { return pos ? min(n_kv, pos[t] + 1) : n_kv; }

__global__ void __launch_bounds__(NT) topk_mb_pass(const float* __restrict__ score, int ld, int n_kv,
                                                   const int* __restrict__ pos, int top, int shift,
                                                   unsigned* __restrict__ work) {
    __shared__ unsigned sh[256];
    __shared__ bool s_last;
    const int t = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    const int n = row_n(n_kv, pos, t);
    if (n <= top) return;
    unsigned* W = work + (size_t) t * kRowWords;
    unsigned* st = W + 256;
    const unsigned prefix = st[0], mask = st[1];
    if (tid < 256) sh[tid] = 0u;
    __syncthreads();
    const int chunk = (n + G - 1) / G, lo = min(n, b * chunk), hi = min(n, lo + chunk);
    const float* sc = score + (size_t) t * ld;
    for (int i = lo + tid; i < hi; i += NT) {
        const unsigned k = score_key(sc[i]);
        if ((k & mask) == prefix) atomicAdd(&sh[(k >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (tid < 256 && sh[tid]) atomicAdd(&W[tid], sh[tid]);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(&st[3], 1u) == (unsigned) (G - 1);
    __syncthreads();
    if (!s_last) return;
    if (tid < 256) sh[tid] = atomicExch(&W[tid], 0u);   // read and re-zero the global histogram
    __syncthreads();
    if (tid == 0) {
        int need = shift == 24 ? top : (int) st[2];
        unsigned p = prefix, m = mask;
        for (int d = 255; d >= 0; --d) {
            const int c = (int) sh[d];
            if (c >= need) {
                p |= (unsigned) d << shift;
                m |= 255u << shift;
                break;
            }
            need -= c;
        }
        st[0] = p;
        st[1] = m;
        st[2] = (unsigned) need;
        st[3] = 0u;
        __threadfence();
    }
}

__global__ void __launch_bounds__(NT) topk_mb_count(const float* __restrict__ score, int ld, int n_kv,
                                                    const int* __restrict__ pos, int top, unsigned* __restrict__ work) {
    __shared__ int s_gt[NT / 32], s_eq[NT / 32];
    __shared__ bool s_last;
    const int t = blockIdx.y, b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = row_n(n_kv, pos, t);
    if (n <= top) return;
    unsigned* W = work + (size_t) t * kRowWords;
    unsigned* st = W + 256;
    const unsigned cut = st[0];
    const int chunk = (n + G - 1) / G, lo = min(n, b * chunk), hi = min(n, lo + chunk);
    const float* sc = score + (size_t) t * ld;
    int gt = 0, eq = 0;
    for (int i = lo + tid; i < hi; i += NT) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) {
        s_gt[warp] = gt;
        s_eq[warp] = eq;
    }
    __syncthreads();
    if (tid == 0) {
        int g = 0, e = 0;
        for (int w = 0; w < NT / 32; ++w) {
            g += s_gt[w];
            e += s_eq[w];
        }
        W[264 + b] = (unsigned) g;
        W[264 + G + b] = (unsigned) e;
        __threadfence();
        s_last = atomicAdd(&st[4], 1u) == (unsigned) (G - 1);
    }
    __syncthreads();
    if (!s_last || tid != 0) return;
    // the last block: the tie shares (lower blocks first) and the output offsets
    const int need = (int) st[2];
    int run_eq = 0, run_off = 0;
    for (int bb = 0; bb < G; ++bb) {
        const int g = (int) __ldcg(&W[264 + bb]), e = (int) __ldcg(&W[264 + G + bb]);
        W[264 + 3 * G + bb] = (unsigned) run_eq;
        W[264 + 2 * G + bb] = (unsigned) run_off;
        run_off += g + max(0, min(e, need - run_eq));
        run_eq += e;
    }
    st[4] = 0u;
    __threadfence();
}

// inclusive Hillis-Steele scan over s[NT]
__device__ __forceinline__ void scan_incl(int* s, int tid) {
    for (int off = 1; off < NT; off <<= 1) {
        const int a = tid >= off ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += a;
        __syncthreads();
    }
}

__global__ void __launch_bounds__(NT) topk_mb_write(const float* __restrict__ score, int ld, int n_kv,
                                                    const int* __restrict__ pos, int top, int* __restrict__ sel,
                                                    unsigned* __restrict__ work) {
    __shared__ int s_eq[NT], s_w[NT];
    __shared__ bool s_last;
    const int t = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    const int n = row_n(n_kv, pos, t);
    int* out = sel + (size_t) t * top;
    if (n <= top) {   // identity: this block's share of [0, top)
        const int ch = (top + G - 1) / G;
        for (int i = b * ch + tid; i < min(top, (b + 1) * ch); i += NT) out[i] = i < n ? i : -1;
        return;
    }
    unsigned* W = work + (size_t) t * kRowWords;
    unsigned* st = W + 256;
    const unsigned cut = st[0];
    const int need = (int) st[2];
    const int chunk = (n + G - 1) / G, lo = min(n, b * chunk), hi = min(n, lo + chunk);
    const int tch = (hi - lo + NT - 1) / NT;
    const int tlo = min(hi, lo + tid * tch), thi = min(hi, tlo + tch);
    const float* sc = score + (size_t) t * ld;
    int gt = 0, eq = 0;
    for (int i = tlo; i < thi; ++i) {
        const unsigned k = score_key(sc[i]);
        gt += k > cut;
        eq += k == cut;
    }
    s_eq[tid] = eq;
    __syncthreads();
    scan_incl(s_eq, tid);
    const int eqb = (int) W[264 + 3 * G + b];
    const int eq_before = eqb + s_eq[tid] - eq;
    const int take = max(0, min(eq, need - eq_before));
    s_w[tid] = gt + take;
    __syncthreads();
    scan_incl(s_w, tid);
    int o = (int) W[264 + 2 * G + b] + s_w[tid] - (gt + take);
    int seen = eq_before;
    for (int i = tlo; i < thi; ++i) {
        const unsigned k = score_key(sc[i]);
        if (k > cut) {
            out[o++] = i;
        } else if (k == cut) {
            if (seen < need) out[o++] = i;
            ++seen;
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(&st[5], 1u) == (unsigned) (G - 1);
    __syncthreads();
    if (s_last && tid == 0) {   // leave the row's scratch zeroed for the next call
        st[0] = st[1] = st[2] = st[5] = 0u;
        __threadfence();
    }
}

}  // namespace

int volta_idx_launch_errors() { return g_errors.load(std::memory_order_relaxed); }

bool idx_wmma_enabled() {
#if defined(STRATA_USE_HIP)
    return false;   // the WMMA scorer is a CUDA kernel: HIP builds score with the CUDA-core one
#else
    static const bool on = env_on("MAYA_GLMD_IDX_WMMA");
    return on;
#endif
}

bool idx_score_wmma_supported(int idx_heads, int idx_dim) {
#if defined(STRATA_USE_HIP)
    (void) idx_heads;
    (void) idx_dim;
    return false;   // (no WMMA scorer in a HIP build)
#else
    return idx_wmma_shape_rule(idx_heads, idx_dim);
#endif
}

namespace {
bool idx_args_ok(const IdxScoreWmma& a) {
    return a.iq && a.w && a.keys && a.score && a.n_kv >= 0 && a.T >= 0 && a.idx_heads > 0 && a.idx_dim > 0 &&
           (a.ld == 0 || a.ld >= a.n_kv) && a.idx_dim % (a.key_f16 ? 8 : 4) == 0;
}
}  // namespace

bool idx_score_wmma(const IdxScoreWmma& a, void* stream) {
    if (!idx_args_ok(a) || !idx_score_wmma_supported(a.idx_heads, a.idx_dim)) {
        refuse("idx_score_wmma");
        return false;
    }
    if (a.n_kv == 0 || a.T == 0) return true;
#if defined(STRATA_USE_HIP)
    (void) stream;
    return false;   // (unreachable: idx_score_wmma_supported refused above)
#else
    const int ld = a.ld > 0 ? a.ld : a.n_kv;
    cudaStream_t s = (cudaStream_t) stream;
    const bool two = a.idx_heads == 32;
    bool ok;
    if (a.T == 1)
        ok = two ? launch_idx_wmma<1, 2>(a, ld, s) : launch_idx_wmma<1, 1>(a, ld, s);
    else
        ok = two ? launch_idx_wmma<kIdxTokPerBlock, 2>(a, ld, s) : launch_idx_wmma<kIdxTokPerBlock, 1>(a, ld, s);
    if (!ok) refuse("idx_score_wmma launch");
    return ok;
#endif
}

bool idx_score_simple(const IdxScoreWmma& a, void* stream) {
    if (!idx_args_ok(a) || (size_t) a.idx_heads * (a.idx_dim + 1) * 4 > 48 * 1024) {
        refuse("idx_score_simple");
        return false;
    }
    if (a.n_kv == 0 || a.T == 0) return true;
    const int ld = a.ld > 0 ? a.ld : a.n_kv;
    const dim3 grid((a.n_kv + 7) / 8, a.T);
    const size_t smem = (size_t) a.idx_heads * (a.idx_dim + 1) * 4;
    if (a.key_f16)
        idx_simple_kernel<true><<<grid, 256, smem, (cudaStream_t) stream>>>(a, ld);
    else
        idx_simple_kernel<false><<<grid, 256, smem, (cudaStream_t) stream>>>(a, ld);
    return launch_check("idx_score_simple");
}

bool idx_score(const IdxScoreWmma& a, void* stream) {
    if (idx_wmma_enabled() && (a.T > 1 || a.n_kv > kIdxWmmaMinDecode) &&
        idx_score_wmma_supported(a.idx_heads, a.idx_dim))
        return idx_score_wmma(a, stream);
    return idx_score_simple(a, stream);
}

bool topk_mb_init(void* work, int T, void* stream) {
    if (!work || T < 0) return false;
    return cudaMemsetAsync(work, 0, topk_mb_work_bytes(T), (cudaStream_t) stream) == cudaSuccess;
}

bool topk_mb(const float* score, int ld, int n_kv, int T, const int* pos, int top_k, int* sel, void* work,
             void* stream) {
    if (T <= 0) return true;
    if (top_k <= 0 || n_kv < 0 || ld < n_kv || !sel || !work || (n_kv > top_k && !score)) {
        refuse("topk_mb");
        return false;
    }
    cudaStream_t s = (cudaStream_t) stream;
    unsigned* w = (unsigned*) work;
    const dim3 grid(G, T);
    if (n_kv > top_k) {
        for (int shift = 24; shift >= 0; shift -= 8) topk_mb_pass<<<grid, NT, 0, s>>>(score, ld, n_kv, pos, top_k, shift, w);
        topk_mb_count<<<grid, NT, 0, s>>>(score, ld, n_kv, pos, top_k, w);
    }
    topk_mb_write<<<grid, NT, 0, s>>>(score, ld, n_kv, pos, top_k, sel, w);
    return launch_check("topk_mb");
}

}  // namespace glmd
