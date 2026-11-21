// src/ik2/hip/ik2_hip_dspark.cu - maya-ik2's drafter (ik2_hip_dspark.hpp; docs/INKLING2_DSPARK.md has the
// equations, read from llama.cpp PR #32218's graph_inkling2 / build_inkling2_head).
//
// One GPU.  The block's matrices stay in the GGUF's Q8_0 blocks and run through ds4_iq's GEMVs (the activations as
// int8 rows, 8 columns a launch: a matrix is read once for the block's 8 rows); the small norms, the fuse, the RoPE,
// the attention and the markov head are this file's kernels.  The head's correction goes through
// M = output @ dspark_hidden_out ([vocab][256], made at the load) instead of the target's whole output matrix a
// step, and its 7 steps run on the device one after the other: the host reads the tokens once, at the end.
#include "ik2_hip_dspark.hpp"

#include "ik2_hip_kernels.cuh"
#include "ik2_hip_pack.hpp"
#include "hy4_kernels.cuh"
#include "../ds4/ds4_iq.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

extern "C" void dequantize_row_q5_K(const void* x, float* y, int64_t k);   // (ggml-quants: the anchor's embedding row)
extern "C" void dequantize_row_q6_K(const void* x, float* y, int64_t k);

namespace ik2hip {

namespace hk = hy4::k;
namespace iq = strata::ds4::gguf::iq;

namespace {

constexpr int kF32 = 0, kQ5K = 13;
constexpr int kFeed = 16;   // positions a feed step (ds4_iq's GEMVs take 16 columns)

std::atomic<int> g_err{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "maya-ik2 dspark kernel %s: %s\n", what, cudaGetErrorString(e));
        g_err.fetch_add(1, std::memory_order_relaxed);
    }
}
unsigned blocks_of(size_t n) { return (unsigned) ((n + 255) / 256); }
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

// ---- the load's kernels ---------------------------------------------------------------------------------------------
// Q8_0 blocks (an F16 step, 32 int8) -> F16 values, n of them
__global__ void q8_to_f16_kernel(const uint8_t* __restrict__ w, size_t n, __half* __restrict__ out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* b = w + (i / 32) * 34;
    const float d = __half2float(*(const __half*) b);
    out[i] = __float2half(d * (float) ((const int8_t*) b)[2 + i % 32]);
}
// M[v][c] = y[c][v] (y rows ld apart) as int8 with a scale a row: round(y / sc[v]), sc[v] = max|y| / 127 - V is the
// padded vocabulary (the words' stride)
__global__ void pack_m_kernel(const float* __restrict__ y, size_t ld, int V, int R, int8_t* __restrict__ Mq,
                              float* __restrict__ sc) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float mx = 0.0f;
    for (int c = 0; c < R; ++c) mx = fmaxf(mx, fabsf(y[(size_t) c * ld + v]));
    const float d = mx / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    // (column c of row v at byte c % 4 of word [c / 4][v]: the head's threads, a row each, then read side by side)
    for (int c = 0; c < R; ++c) Mq[((size_t) (c / 4) * V + v) * 4 + (size_t) (c % 4)] = (int8_t) lrintf(y[(size_t) c * ld + v] * id);
    sc[v] = d;
}
__global__ void float_to_half_kernel(const float* __restrict__ x, size_t n, __half* __restrict__ y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// ---- the context's kernels --------------------------------------------------------------------------------------------
// block (r, l): out[l][r] = rms(sum_k w[l][k] taps[r][k]) gain[l]
__global__ void fuse_kernel(const float* __restrict__ taps, size_t ld, int E, int NT, const float* __restrict__ w,
                            const float* __restrict__ gain, float eps, int n, float* __restrict__ out) {
    __shared__ float sh[256];
    const int r = blockIdx.x, l = blockIdx.y;
    const float* x = taps + (size_t) r * ld;
    float* y = out + ((size_t) l * n + r) * E;
    float wk[16];
    for (int kk = 0; kk < NT; ++kk) wk[kk] = w[l * NT + kk];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float v = 0.0f;
        for (int kk = 0; kk < NT; ++kk) v = fmaf(wk[kk], x[(size_t) kk * E + i], v);
        y[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) y[i] = y[i] * inv * gain[(size_t) l * E + i];
}
// RoPE over all D values of every head, pairs (i, i + D / 2) (ggml's NEOX), theta_i = pos base^(-2 i / D):
// x [rows][H][D] in place, row t at position pos0 + t
__global__ void rope_neox_kernel(float* __restrict__ x, int rows, int H, int D, long long pos0, double log_base) {
    const int half = D / 2;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * H * half) return;
    const int j = (int) (i % half);
    const size_t rh = i / half;
    const int t = (int) (rh / H);
    const double th = (double) (pos0 + t) * exp(-(double) j / (double) half * log_base);
    const float c = (float) cos(th), s = (float) sin(th);
    float* v = x + rh * D;
    const float a = v[j], b = v[j + half];
    v[j] = a * c - b * s;
    v[j + half] = a * s + b * c;
}
__global__ void half_to_float_kernel(const __half* __restrict__ x, int n, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

// ---- the block's attention --------------------------------------------------------------------------------------------
// Every row of the block reads the same keys: the nctx cached contexts at positions lo .. lo + nctx - 1 (row slot
// p % ring, ring 0: p; F16 [slots][Hkv][D]), then the block's own T rows (kb / vb F32 [T][Hkv][D]); nk = nctx + T.
// Split over the keys, as the engine's k::attention_split: block (tile, kv head) = TK keys of one kv head for its
// T * per queries (per = H / Hkv query heads a kv head, all T rows) - the keys read once into shared memory, each
// query's scores, their softmax within the tile (st: its maximum and sum) and the tile's part of the output - then a
// combine a query.  A plain softmax of scale q . k: no sink, no bias.
// (kMaxTiles: the combine's shared factors; kPartAcc = the 48 queries - 8 rows x 6 heads a kv head - over the 2 groups
//  of 128 threads, which the load checks)
constexpr int kMaxTiles = 4096, kPartAcc = 24;
__global__ void attn_tile_kernel(const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
                                 const __half* __restrict__ K, const __half* __restrict__ V, int ring, long long base,
                                 int nctx, int T, int H, int Hkv, int D, int TK, float scale, float* __restrict__ st,
                                 float* __restrict__ part) {
    extern __shared__ float sm[];   // [TK][D] keys, [NQ][TK] scores / weights, [NQ] maxima
    const int g = blockIdx.y, tile = blockIdx.x, ntile = gridDim.x, j0 = tile * TK, per = H / Hkv, tid = threadIdx.x;
    const int NQ = T * per, nk = nctx + T, n = min(TK, nk - j0);
    float* sk = sm;
    float* sw = sk + (size_t) TK * D;
    float* smb = sw + (size_t) NQ * TK;
    for (int i = tid; i < n * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nctx) {
            long long sl = base + j;   // (wrapped once: no 64-bit modulo a value)
            if (ring > 0 && sl >= ring) sl -= ring;
            sk[i] = __half2float(K[((size_t) sl * Hkv + g) * D + i % D]);
        } else
            sk[i] = kb[((size_t) (j - nctx) * Hkv + g) * D + i % D];
    }
    __syncthreads();
    for (int e = tid; e < NQ * TK; e += blockDim.x) {
        const int r = e / TK, jj = e % TK;
        float a = -INFINITY;
        if (jj < n) {
            const float* qr = q + ((size_t) (r / per) * H + (size_t) g * per + r % per) * D;
            const float* kr = sk + jj * D;
            a = 0.0f;
            for (int i = 0; i < D; ++i) a = fmaf(qr[i], kr[i], a);
            a *= scale;
        }
        sw[e] = a;
    }
    __syncthreads();
    if (tid < NQ) {
        float m = -INFINITY;
        for (int jj = 0; jj < n; ++jj) m = fmaxf(m, sw[tid * TK + jj]);
        smb[tid] = m;
    }
    __syncthreads();
    for (int e = tid; e < NQ * TK; e += blockDim.x) sw[e] = e % TK < n ? expf(sw[e] - smb[e / TK]) : 0.0f;
    __syncthreads();
    if (tid < NQ) {
        float sum = 0.0f;
        for (int jj = 0; jj < n; ++jj) sum += sw[tid * TK + jj];
        float* o = st + (((size_t) (tid / per) * H + (size_t) g * per + tid % per) * ntile + tile) * 2;
        o[0] = smb[tid];
        o[1] = sum;
    }
    // the tile's part: thread (group, i) takes value i of its group's queries (the queries split over the groups)
    const int ng = blockDim.x / D, grp = tid / D, i = tid % D;
    if (grp < ng) {
        const int qn = (NQ + ng - 1) / ng, r0 = grp * qn, r1 = min(NQ, r0 + qn);
        float v[kPartAcc];
#pragma unroll
        for (int r = 0; r < kPartAcc; ++r) v[r] = 0.0f;
        for (int jj = 0; jj < n; ++jj) {
            const int j = j0 + jj;
            float x;
            if (j < nctx) {
                long long sl = base + j;
                if (ring > 0 && sl >= ring) sl -= ring;
                x = __half2float(V[((size_t) sl * Hkv + g) * D + i]);
            } else
                x = vb[((size_t) (j - nctx) * Hkv + g) * D + i];
#pragma unroll
            for (int r = 0; r < kPartAcc; ++r)
                if (r0 + r < r1) v[r] = fmaf(sw[(r0 + r) * TK + jj], x, v[r]);
        }
        for (int r = r0; r < r1; ++r)
            part[(((size_t) (r / per) * H + (size_t) g * per + r % per) * ntile + tile) * D + i] = v[r - r0];
    }
}
// block = a query (row t, head h): the tiles' parts under one maximum
__global__ void attn_combine_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                    float* __restrict__ out) {
    __shared__ float sh[256];
    __shared__ float w[kMaxTiles];
    const int qi = blockIdx.x, tid = threadIdx.x;
    const float* sq = st + (size_t) qi * ntile * 2;
    float m = -INFINITY;
    for (int b = tid; b < ntile; b += blockDim.x) m = fmaxf(m, sq[2 * b]);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) {
        const float e = expf(sq[2 * b] - m);
        w[b] = e;
        sum += sq[2 * b + 1] * e;
    }
    sum = block_sum(sum, sh);
    if (tid >= D) return;
    const float* pq = part + (size_t) qi * ntile * D + tid;
    float v = 0.0f;
    for (int b = 0; b < ntile; ++b) v = fmaf(pq[(size_t) b * D], w[b], v);
    out[(size_t) qi * D + tid] = v / sum;
}

// The GEMM form's middle step: block = a query's row of raw dots over the nk keys -> its softmax of scale s, as F16
__global__ void soft_f16_kernel(const float* __restrict__ S, int nk, float scale, __half* __restrict__ P) {
    __shared__ float sh[256];
    const float* s = S + (size_t) blockIdx.x * nk;
    __half* p = P + (size_t) blockIdx.x * nk;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) mx = fmaxf(mx, s[j]);
    mx = block_max(mx, sh);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) sum += expf((s[j] - mx) * scale);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) p[j] = __float2half(expf((s[j] - mx) * scale) * inv);
}

// ---- a Q8_0 GEMV for exactly NC columns (ds4_iq's mv_kernel - a warp a row, a lane a 32-value block of it, each block
// read once and dotted with every column's int8 row - without its 16 accumulators and its column test a block)
template <int NC>
__global__ void q8mv_kernel(const uint8_t* __restrict__ w, size_t ld, int k, int n, const int8_t* __restrict__ xq,
                            const float* __restrict__ xd, float* __restrict__ y, size_t ycs) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= n) return;
    const uint8_t* wr = w + (size_t) row * ld;
    const int nsb = k / 32;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int kb = lane; kb < nsb; kb += 32) {
        const uint16_t* b = (const uint16_t*) (wr + (size_t) kb * 34);   // (an F16 step, 32 int8: 2-byte aligned)
        const float d = __half2float(*(const __half*) b);
        int q[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) q[t] = (int) ((uint32_t) b[1 + 2 * t] | ((uint32_t) b[2 + 2 * t] << 16));
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int4* xp = (const int4*) (xq + (size_t) c * k + (size_t) kb * 32);
            const int4 xa = xp[0], xb = xp[1];
            int dot = __dp4a(q[0], xa.x, 0);
            dot = __dp4a(q[1], xa.y, dot);
            dot = __dp4a(q[2], xa.z, dot);
            dot = __dp4a(q[3], xa.w, dot);
            dot = __dp4a(q[4], xb.x, dot);
            dot = __dp4a(q[5], xb.y, dot);
            dot = __dp4a(q[6], xb.z, dot);
            dot = __dp4a(q[7], xb.w, dot);
            acc[c] = fmaf(xd[(size_t) c * nsb + kb] * d, (float) dot, acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[(size_t) c * ycs + row] = v;
    }
}

// ---- the markov head ----------------------------------------------------------------------------------------------------
// block r: z[r] = W[r] . [state | w1[*tok] | hid | ctx] + b[r]   (W F16 [3 R][2 R + 2 E], w1 F16 [vocab][R])
__global__ void joint_kernel(const __half* __restrict__ W, const float* __restrict__ b, const float* __restrict__ state,
                             const __half* __restrict__ w1, const int* __restrict__ tok, const float* __restrict__ hid,
                             const float* __restrict__ ctx, int R, int E, float* __restrict__ z,
                             const float* __restrict__ cum, float cmin) {
    __shared__ float sh[256];
    if (*cum < cmin) return;   // (the drafts are cut before this one: its step is not needed)
    const int r = blockIdx.x, n = 2 * R + 2 * E;
    const __half* wr = W + (size_t) r * n;
    const __half* e = w1 + (size_t) (*tok) * R;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = i < R ? state[i] : i < 2 * R ? __half2float(e[i - R]) : i < 2 * R + E ? hid[i - 2 * R]
                                                                                             : ctx[i - 2 * R - E];
        acc = fmaf(__half2float(wr[i]), x, acc);
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) z[r] = acc + b[r];
}
// lg[v] = cap tanh((base[v] + M[v] . u + m0[v]) / cap), v < V, with u = tanh(z[2 R ..]) and M[v] = sc[v] Mq[v]; and
// each block's part of the pick: bmax[b] its largest value, bidx[b] where (the lowest index), bsum[b] = sum exp(lg -
// bmax[b])
__global__ void logits_kernel(const float* __restrict__ base, const int8_t* __restrict__ Mq, const float* __restrict__ sc,
                              const float* __restrict__ m0, const float* __restrict__ z, int R, int V, int Vs, float cap,
                              float* __restrict__ lg, float* __restrict__ bmax, int* __restrict__ bidx,
                              float* __restrict__ bsum, const float* __restrict__ cum, float cmin) {
    if (*cum < cmin) return;
    // u = tanh(.) lies in (-1, 1): as 16-bit integers the products with M's int8 sum exactly in 32 bits (R <= 256)
    __shared__ int us[256];
    __shared__ float sv[256];
    __shared__ int si[256];
    for (int i = threadIdx.x; i < R; i += blockDim.x) us[i] = (int) lrintf(tanhf(z[2 * R + i]) * 32767.0f);
    __syncthreads();
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    float val = -INFINITY;
    if (v < V) {
        const uint32_t* mw = (const uint32_t*) Mq;   // (words [R / 4][Vs]: pack_m_kernel's layout)
        int acc = 0;
        for (int c = 0; c < R / 4; ++c) {
            const uint32_t w = mw[(size_t) c * Vs + v];
            acc += (int) (int8_t) (w & 255) * us[4 * c] + (int) (int8_t) ((w >> 8) & 255) * us[4 * c + 1] +
                   (int) (int8_t) ((w >> 16) & 255) * us[4 * c + 2] + (int) (int8_t) (w >> 24) * us[4 * c + 3];
        }
        val = cap * tanhf((base[v] + (float) acc * (sc[v] * (1.0f / 32767.0f)) + m0[v]) / cap);
        lg[v] = val;
    }
    sv[threadIdx.x] = val;
    si[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            const float v2 = sv[threadIdx.x + o];
            const int i2 = si[threadIdx.x + o];
            if (v2 > sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x])) {
                sv[threadIdx.x] = v2;
                si[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    const float mx = sv[0];
    const int at = si[0];
    __syncthreads();
    sv[threadIdx.x] = v < V ? expf(val - mx) : 0.0f;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sv[threadIdx.x] += sv[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bmax[blockIdx.x] = mx;
        bidx[blockIdx.x] = at;
        bsum[blockIdx.x] = sv[0];
    }
}
// the blocks' parts together: *tok = the argmax (ties: the lowest index), *prob = its softmax probability; and the
// step's state update, after every reader of the old state: state = cand + gate (state - cand)
__global__ void pick_kernel(const float* __restrict__ bmax, const int* __restrict__ bidx, const float* __restrict__ bsum,
                            int nb, int* __restrict__ tok, float* __restrict__ prob, const float* __restrict__ z, int R,
                            float* __restrict__ state, float* __restrict__ cum, float cmin) {
    if (*cum < cmin) {
        if (threadIdx.x == 0) {
            *tok = -1;
            *prob = 0.0f;
        }
        return;
    }
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        const float gt = 1.0f / (1.0f + expf(-z[i])), c = tanhf(z[R + i]);
        state[i] = c + gt * (state[i] - c);
    }
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float v = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < nb; i += blockDim.x)
        if (bmax[i] > v || (bmax[i] == v && bidx[i] < at)) {
            v = bmax[i];
            at = bidx[i];
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
    const float mx = bv[0];
    const int best = bi[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) sum += bsum[i] * expf(bmax[i] - mx);
    bv[threadIdx.x] = sum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) bv[threadIdx.x] += bv[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *tok = best;
        *prob = 1.0f / bv[0];
        *cum *= 1.0f / bv[0];
    }
}

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, f = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (f == 0) u = s;
        else {
            int sh = 0;
            uint32_t ff = f;
            while (!(ff & 1024)) {
                ff <<= 1;
                ++sh;
            }
            u = s | ((uint32_t) (113 - sh) << 23) | ((ff & 1023) << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (f << 13);
    else u = s | ((e + 112) << 23) | (f << 13);
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

// the drafter's index (tools/ik2/convert_dspark.py): dspark.txt, tensors.txt - the format of the model's pack
struct Index {
    std::string dir;
    std::map<std::string, std::string> kv;
    std::map<std::string, Tensor> t;
    std::vector<const uint8_t*> map;
    std::vector<size_t> map_bytes;

    bool load(const std::string& d, std::string& err) {
        dir = d;
        std::ifstream gf(dir + "/dspark.txt");
        if (!gf) {
            err = dir + "/dspark.txt: cannot open (tools/ik2/convert_dspark.py writes it)";
            return false;
        }
        for (std::string line; std::getline(gf, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) kv[line.substr(0, sp)] = line.substr(sp + 1);
        }
        if (str("format").rfind("maya-ik2-dspark", 0) != 0) {
            err = dir + "/dspark.txt: not a maya-ik2-dspark index";
            return false;
        }
        std::ifstream f(dir + "/tensors.txt");
        if (!f) {
            err = dir + "/tensors.txt: cannot open";
            return false;
        }
        std::vector<std::string> files;
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream is(line);
            std::string tag;
            is >> tag;
            if (tag == "file") {
                int i;
                std::string p;
                is >> i;
                std::getline(is >> std::ws, p);
                files.push_back(p);
            } else if (tag == "t") {
                Tensor x;
                is >> x.name >> x.type >> x.file >> x.off >> x.bytes;
                for (int64_t v; is >> v;) x.ne.push_back(v);
                t[x.name] = std::move(x);
            }
        }
        const std::string md = str("model_dir", "..");
        const std::string base = !md.empty() && md[0] == '/' ? md : dir + "/" + md;
        for (const std::string& p : files) {
            const std::string full = !p.empty() && p[0] == '/' ? p : base + "/" + p;
            size_t sz = 0;
            const uint8_t* mp = mm3::os::map_file(full, sz);
            if (mp == nullptr) {
                err = full + ": cannot open or map";
                return false;
            }
            map.push_back(mp);
            map_bytes.push_back(sz);
        }
        for (const auto& [n, x] : t)
            if (x.file < 0 || x.file >= (int) map.size() || x.off + x.bytes > map_bytes[(size_t) x.file]) {
                err = n + ": outside its file";
                return false;
            }
        return true;
    }
    std::string str(const std::string& key, const std::string& def = "") const {
        auto it = kv.find(key);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& key, double def = 0) const {
        auto it = kv.find(key);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> ints(const std::string& key) const {
        std::vector<int> v;
        std::istringstream is(str(key));
        for (std::string x; std::getline(is, x, ',');) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    const uint8_t* data(const Tensor& x) const { return map[(size_t) x.file] + x.off; }
};

struct Mat {
    const uint8_t* w = nullptr;
    size_t ld = 0;
    int type = 0, k = 0, n = 0;
};

struct DLayer {
    bool swa = false;
    Mat q, kk, v, o, kc, vc, gate, up, down;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *qn = nullptr, *kn = nullptr, *kcn = nullptr;
    uint16_t *K = nullptr, *V = nullptr;   // the contexts' rows, F16 [slots][Hkv][D]
    int ring = 0;                          // 0: a row a position
};

}  // namespace

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            err = std::string("dspark: ") + #x + ": " + cudaGetErrorString(e_);                  \
            return false;                                                                        \
        }                                                                                        \
    } while (0)

struct Dspark::Impl {
    DsparkOptions o;
    DsparkStats st;
    Index ix, jx;
    Pack tp;
    int dev = 0;
    cudaStream_t s = nullptr;
    std::vector<void*> allocs;
    size_t used = 0;

    int NL = 0, E = 0, FF = 0, H = 0, Hkv = 0, D = 0, R = 0, NT = 0, B = 0, Vp = 0, Vu = 0, n_swa = 0;
    float eps = 1e-6f, cap = 30.0f;
    double log_base = 0;
    std::vector<int> targets;
    std::vector<DLayer> L;
    Mat out;   // the target's output matrix
    float *gain = nullptr, *fw = nullptr, *mask_x = nullptr, *out_norm = nullptr, *tok_norm = nullptr;
    float *bj = nullptr, *m0 = nullptr, *msc = nullptr;
    uint16_t *w1 = nullptr, *Wj = nullptr, *fring = nullptr;
    int8_t* Mq = nullptr;
    int FR = 0;
    // scratch
    float *x = nullptr, *a = nullptr, *emb = nullptr, *qv = nullptr, *kv = nullptr, *vv = nullptr, *ao = nullptr,
          *ob = nullptr, *gu = nullptr, *hh = nullptr, *ast = nullptr, *apart = nullptr, *hN = nullptr, *base = nullptr, *lg = nullptr,
          *z = nullptr, *state = nullptr, *ctx = nullptr, *prob = nullptr, *taps = nullptr,
          *fz = nullptr, *kc = nullptr, *vc = nullptr, *xd = nullptr;
    int8_t* xq = nullptr;
    int *tok = nullptr, *bidx = nullptr;
    float *bmax = nullptr, *bsum = nullptr, *x0 = nullptr, *cumd = nullptr;
    int max_tiles = 0, tile = 16;
    cublasHandle_t blas = nullptr;
    float *Sg = nullptr, *qp = nullptr, *op = nullptr;
    uint16_t *P16 = nullptr, *xh16 = nullptr, *kb16 = nullptr, *vb16 = nullptr;
    // the anchor's embedding row
    const uint8_t* tok_embd = nullptr;
    size_t tok_row = 0;
    int tok_type = 0;
    std::vector<float> h_x;
    int64_t end = 0, hi = -1;   // the context's end; the furthest position ever fed (what the rings have overwritten)
    float cum_min = 0.5f;
    // (--parallel) a conversation's state each: every layer's K / V rows, the context ring, end and hi - the bound
    // one's in the fields above, the others' here (state 0's too while another is bound)
    struct State {
        std::vector<uint16_t*> K, V;
        uint16_t* fring = nullptr;
        int64_t end = 0, hi = -1;
    };
    std::vector<State> states;
    int bound = 0;
    void save_state(State& x) {
        x.K.resize(L.size());
        x.V.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            x.K[i] = L[i].K;
            x.V[i] = L[i].V;
        }
        x.fring = fring;
        x.end = end;
        x.hi = hi;
    }
    void load_state(const State& x) {
        for (size_t i = 0; i < L.size(); ++i) {
            L[i].K = x.K[i];
            L[i].V = x.V[i];
        }
        fring = x.fring;
        end = x.end;
        hi = x.hi;
    }
    bool add_states(int n, std::string& err) {
        if (states.empty()) {
            states.resize(1);
            save_state(states[0]);
        }
        CK(cudaSetDevice(dev));
        while ((int) states.size() < n) {
            State x;
            for (const DLayer& l : L) {
                const size_t rows = l.swa ? (size_t) l.ring : (size_t) o.max_ctx, kd = (size_t) Hkv * D;
                uint16_t *k = nullptr, *v = nullptr;
                if (!alloc(k, rows * kd, err) || !alloc(v, rows * kd, err)) return false;
                x.K.push_back(k);
                x.V.push_back(v);
            }
            if (!alloc(x.fring, (size_t) FR * E, err)) return false;
            states.push_back(x);
        }
        st.vram_gb = (double) used / 1e9;
        return true;
    }
    void bind(int i) {
        if (i == bound || i < 0 || i >= (int) states.size()) return;
        save_state(states[(size_t) bound]);
        load_state(states[(size_t) i]);
        bound = i;
    }

    ~Impl() {
        if (s != nullptr || !allocs.empty()) {
            (void) cudaSetDevice(dev);
            for (void* p : allocs) (void) cudaFree(p);
            if (blas != nullptr) (void) cublasDestroy(blas);
            if (s != nullptr) (void) cudaStreamDestroy(s);
        }
    }

    template <typename T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        const size_t bytes = std::max<size_t>(n * sizeof(T), 16);
        const cudaError_t e = cudaMalloc(&v, bytes);
        if (e != cudaSuccess) {
            err = "dspark: " + std::to_string(bytes >> 20) + " MiB on GPU " + std::to_string(dev) + ": " +
                  cudaGetErrorString(e);
            return false;
        }
        allocs.push_back(v);
        used += bytes;
        p = (T*) v;
        return true;
    }
    void release(void* p) {
        for (size_t i = 0; i < allocs.size(); ++i)
            if (allocs[i] == p) {
                allocs.erase(allocs.begin() + (long) i);
                break;
            }
        (void) cudaFree(p);
    }
    bool h2d(void* d, const void* h, size_t bytes, std::string& err) {
        CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
        return true;
    }
    const Tensor* need(const Index& x, const std::string& n, std::string& err, std::initializer_list<int64_t> ne) {
        const Tensor* t = x.get(n);
        if (!t) {
            err = x.dir + ": no " + n;
            return nullptr;
        }
        size_t i = 0;
        bool ok = t->ne.size() == ne.size();
        for (int64_t v : ne) ok = ok && i < t->ne.size() && t->ne[i++] == v;
        if (!ok) {
            err = x.dir + ": " + n + " does not have the shape mtp_config gives it";
            return nullptr;
        }
        return t;
    }
    // an F32 vector on the device
    bool vec(const std::string& n, float*& p, int want, std::string& err) {
        const Tensor* t = need(ix, n, err, {want});
        if (!t) return false;
        if (t->type != "F32") {
            err = n + ": " + t->type + ", F32 expected";
            return false;
        }
        return alloc(p, (size_t) want, err) && h2d(p, ix.data(*t), t->bytes, err);
    }
    // a matrix (ggml [k, n]) on the host as F32 rows: F32, BF16, F16 or Q8_0
    bool host_f32(const Index& x, const std::string& n, int kk, int nn, std::vector<float>& f, std::string& err) {
        const Tensor* t = need(x, n, err, {kk, nn});
        if (!t) return false;
        const size_t nv = (size_t) kk * nn;
        const uint8_t* p = x.data(*t);
        f.resize(nv);
        if (t->type == "F32" && t->bytes == nv * 4) std::memcpy(f.data(), p, nv * 4);
        else if (t->type == "BF16" && t->bytes == nv * 2)
            for (size_t i = 0; i < nv; ++i) {
                uint16_t hv;
                std::memcpy(&hv, p + 2 * i, 2);
                const uint32_t u = (uint32_t) hv << 16;
                std::memcpy(&f[i], &u, 4);
            }
        else if (t->type == "F16" && t->bytes == nv * 2)
            for (size_t i = 0; i < nv; ++i) {
                uint16_t hv;
                std::memcpy(&hv, p + 2 * i, 2);
                f[i] = half_to_float(hv);
            }
        else if (t->type == "Q8_0" && kk % 32 == 0 && t->bytes == nv / 32 * 34)
            for (size_t b = 0; b < nv / 32; ++b) {
                uint16_t hv;
                std::memcpy(&hv, p + b * 34, 2);
                const float d = half_to_float(hv);
                for (int i = 0; i < 32; ++i) f[b * 32 + (size_t) i] = d * (float) (int8_t) p[b * 34 + 2 + (size_t) i];
            }
        else {
            err = n + ": " + t->type + ", not a type the drafter reads";
            return false;
        }
        return true;
    }
    // a block-type matrix on the device as it is in the file
    bool mat(const std::string& n, Mat& m2, int kk, int nn, std::string& err) {
        const Tensor* t = need(ix, n, err, {kk, nn});
        if (!t) return false;
        if (t->type != "Q8_0" || kk % 32 != 0 || t->bytes != iq::row_bytes(iq::kQ8_0, kk) * (size_t) nn) {
            err = n + ": " + t->type + " - the drafter's block matrices are Q8_0 (the Q8_0 index: "
                  "tools/ik2/convert_dspark.py --type q8_0)";
            return false;
        }
        uint8_t* d = nullptr;
        if (!alloc(d, t->bytes, err) || !h2d(d, ix.data(*t), t->bytes, err)) return false;
        m2.w = d;
        m2.type = iq::kQ8_0;
        m2.k = kk;
        m2.n = nn;
        m2.ld = iq::row_bytes(iq::kQ8_0, kk);
        return true;
    }
    // y[c] = M x[c] for nc <= 16 rows of x (their int8 rows in xq / xd already)
    static iq::MvJob job(const Mat& m2, const int8_t* q, const float* d, int nc, float* y, size_t ycs) {
        iq::MvJob j;
        j.w = m2.w;
        j.ld = m2.ld;
        j.type = m2.type;
        j.k = m2.k;
        j.n = m2.n;
        j.ncol = nc;
        j.xq = q;
        j.xd = d;
        j.y = y;
        j.ycs = ycs;
        return j;
    }
    bool mv(std::initializer_list<iq::MvJob> jobs, std::string& err) {
        if (!iq::mv(jobs.begin(), (int) jobs.size(), s)) {
            err = "dspark: a GEMV ds4_iq does not take";
            return false;
        }
        return true;
    }
    // the block's 8 rows through a Q8_0 matrix: this file's 8-column kernel, or ds4_iq's
    bool mv8(std::initializer_list<iq::MvJob> jobs, std::string& err) {
        if (!o.own_gemv || B != 8) return mv(jobs, err);
        for (const iq::MvJob& j : jobs) {
            if (j.type != iq::kQ8_0 || j.ncol != 8) return mv(jobs, err);
            q8mv_kernel<8><<<(unsigned) ((j.n + 7) / 8), 256, 0, s>>>(j.w, j.ld, j.k, j.n, j.xq, j.xd, j.y, j.ycs);
        }
        check("q8mv");
        return true;
    }
    // The attention as GEMMs over each kv head's keys (the engine's attention_rows): S = K^T Q, the softmax, O = V P -
    // the caches' F16 rows as they lie (a key a column, kv heads D apart), the block's own rows a third product
    bool attention_gemm(DLayer& l, int64_t lo, int nctx, float scale, std::string& err) {
        const int per = H / Hkv, Q = B * per, KD = Hkv * D, nk = nctx + B;
        const float one = 1.0f, zero = 0.0f;
        ik2hip::k::perm_heads(qv, qp, B, Hkv, per, D, true, s);
        hk::to_f16(qp, (size_t) D, Hkv * Q, D, xh16, (size_t) D, s);
        hk::to_f16(kv, (size_t) KD, B, KD, kb16, (size_t) KD, s);
        hk::to_f16(vv, (size_t) KD, B, KD, vb16, (size_t) KD, s);
        const size_t slot0 = (size_t) (l.ring > 0 ? lo % l.ring : lo);
        const int n1 = l.ring > 0 ? (int) std::min<int64_t>(nctx, (int64_t) l.ring - (int64_t) slot0) : nctx, n2 = nctx - n1;
        auto bg = [&](decltype(CUBLAS_OP_T) ta, int mm, int kk, const uint16_t* A, const uint16_t* Bm, int ldb, long long sb,
                      const float* beta, float* C, int ldc, long long sc) {
            return cublasGemmStridedBatchedEx(blas, ta, CUBLAS_OP_N, mm, Q, kk, &one, A, CUDA_R_16F, KD, D, Bm, CUDA_R_16F,
                                              ldb, sb, beta, C, CUDA_R_32F, ldc, sc, Hkv, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
        };
        const long long sq = (long long) Q * D, ss = (long long) Q * nk;
        bool ok = true;
        if (n1 > 0) ok = ok && bg(CUBLAS_OP_T, n1, D, l.K + slot0 * KD, xh16, D, sq, &zero, Sg, nk, ss);
        if (n2 > 0) ok = ok && bg(CUBLAS_OP_T, n2, D, l.K, xh16, D, sq, &zero, Sg + n1, nk, ss);
        ok = ok && bg(CUBLAS_OP_T, B, D, kb16, xh16, D, sq, &zero, Sg + nctx, nk, ss);
        soft_f16_kernel<<<(unsigned) (Hkv * Q), 256, 0, s>>>(Sg, nk, scale, (__half*) P16);
        check("soft_f16");
        ok = ok && bg(CUBLAS_OP_N, D, B, vb16, P16 + nctx, nk, ss, &zero, op, D, sq);
        if (n1 > 0) ok = ok && bg(CUBLAS_OP_N, D, n1, l.V + slot0 * KD, P16, nk, ss, &one, op, D, sq);
        if (n2 > 0) ok = ok && bg(CUBLAS_OP_N, D, n2, l.V, P16 + n1, nk, ss, &one, op, D, sq);
        if (!ok) {
            err = "dspark: an attention GEMM failed";
            return false;
        }
        ik2hip::k::perm_heads(op, ao, B, Hkv, per, D, false, s);
        return true;
    }
    bool sync(std::string& err) {
        CK(cudaStreamSynchronize(s));
        if (g_err.exchange(0) != 0) {   // (the shared kernels count their own failures: the engine's check)
            err = "dspark: a kernel launch failed";
            return false;
        }
        return true;
    }

    bool load(const DsparkOptions& opt, std::string& err) {
        const double t0 = now_ms();
        o = opt;
        cum_min = o.cum_min;
        dev = o.gpu;
        if (!ix.load(o.index, err)) return false;
        if (!o.joint_index.empty() && !jx.load(o.joint_index, err)) return false;
        if (!tp.load(o.target_pack, err)) return false;
        NL = (int) ix.num("n_layer");
        E = (int) ix.num("n_embd");
        FF = (int) ix.num("n_ff");
        H = (int) ix.num("n_head");
        Hkv = (int) ix.num("n_head_kv");
        D = (int) ix.num("head_dim");
        R = (int) ix.num("markov_rank");
        B = (int) ix.num("block_size");
        Vp = (int) ix.num("n_vocab");
        Vu = (int) ix.num("n_vocab_unpadded");
        n_swa = (int) ix.num("n_swa");
        eps = (float) ix.num("eps", 1e-6);
        cap = (float) ix.num("softcap", 30.0);
        log_base = std::log(ix.num("rope_theta", 1e6));
        targets = ix.ints("target_layers");
        NT = (int) targets.size();
        const std::vector<int> swa = ix.ints("swa");
        const Geo& tg = tp.g;
        if (NL < 1 || NL > 15 || NT != Dspark::kTaps || B != Dspark::kBlock || (int) swa.size() != NL || E != tg.n_embd ||
            Vp != tg.n_vocab || Vu != tg.n_vocab_unpadded || E % 32 || FF % 32 || (H * D) % 32 || (Hkv * D) % 32 ||
            D > 128 || D % 2 || H % Hkv || (B * (H / Hkv) + 256 / D - 1) / (256 / D) > kPartAcc || R < 4 || R > 256 || R % 4 || n_swa < 1 || Vu > Vp || cap <= 0 ||
            targets.back() != tg.n_layer || (int) ix.num("target_n_layer") != tg.n_layer) {
            err = o.index + ": a drafter maya-ik2 does not run with this target";
            return false;
        }
        CK(cudaSetDevice(dev));
        CK(cudaStreamCreate(&s));

        // ---- the target's side: the embedding table (in place), its norm, the output matrix
        {
            const Tensor* t = tp.get("token_embd.weight");
            const Tensor* tn = tp.get("token_embd_norm.weight");
            if (!t || !tn || tn->type != "F32" || (int64_t) tn->bytes != (int64_t) E * 4 || t->n(0) != E || t->n(1) != Vp) {
                err = o.target_pack + ": token_embd / token_embd_norm are not the target's";
                return false;
            }
            tok_type = t->type == "Q5_K" ? kQ5K : t->type == "Q6_K" ? iq::kQ6_K : t->type == "F32" ? kF32 : -1;
            if (tok_type < 0) {
                err = "token_embd.weight: " + t->type + ", Q5_K / Q6_K / F32 expected";
                return false;
            }
            tok_embd = tp.data(*t);
            tok_row = (size_t) t->bytes / (size_t) Vp;
            h_x.resize((size_t) E);
            if (!alloc(tok_norm, (size_t) E, err) || !h2d(tok_norm, tp.data(*tn), tn->bytes, err)) return false;
            if (o.output_dev != nullptr) {
                out.w = o.output_dev;
                out.type = o.output_type;
                out.ld = o.output_ld;
            } else {
                const Tensor* to = tp.get("output.weight");
                if (!to || to->n(0) != E || to->n(1) != Vp) {
                    err = o.target_pack + ": no output.weight of the target's shape";
                    return false;
                }
                out.type = to->type == "Q6_K" ? iq::kQ6_K : to->type == "Q8_0" ? iq::kQ8_0 : -1;
                if (out.type < 0 || E % iq::block_values(out.type) || iq::row_bytes(out.type, E) * (size_t) Vp != to->bytes) {
                    err = "output.weight: " + to->type + ", Q6_K / Q8_0 expected";
                    return false;
                }
                uint8_t* d = nullptr;
                if (!alloc(d, to->bytes, err) || !h2d(d, tp.data(*to), to->bytes, err)) return false;
                out.w = d;
                out.ld = iq::row_bytes(out.type, E);
            }
            out.k = E;
            out.n = Vp;
            if (!iq::mv_supported(out.type) || out.ld != iq::row_bytes(out.type, E)) {
                err = "dspark: the target's output matrix is not rows ds4_iq's GEMVs read";
                return false;
            }
        }

        // ---- the drafter's layers
        const int HD = H * D, KD = Hkv * D;
        L.resize((size_t) NL);
        for (int il = 0; il < NL; ++il) {
            DLayer& l = L[(size_t) il];
            const std::string p = "blk." + std::to_string(il) + ".";
            l.swa = swa[(size_t) il] != 0;
            if (!vec(p + "attn_norm.weight", l.attn_norm, E, err) || !vec(p + "ffn_norm.weight", l.ffn_norm, E, err) ||
                !vec(p + "attn_q_norm.weight", l.qn, D, err) || !vec(p + "attn_k_norm.weight", l.kn, D, err) ||
                !vec(p + "attn_k_ctx_norm.weight", l.kcn, D, err) || !mat(p + "attn_q.weight", l.q, E, HD, err) ||
                !mat(p + "attn_k.weight", l.kk, E, KD, err) || !mat(p + "attn_v.weight", l.v, E, KD, err) ||
                !mat(p + "attn_output.weight", l.o, HD, E, err) || !mat(p + "attn_k_ctx.weight", l.kc, E, KD, err) ||
                !mat(p + "attn_v_ctx.weight", l.vc, E, KD, err) || !mat(p + "ffn_gate.weight", l.gate, E, FF, err) ||
                !mat(p + "ffn_up.weight", l.up, E, FF, err) || !mat(p + "ffn_down.weight", l.down, FF, E, err))
                return false;
            l.ring = l.swa ? n_swa + std::max(0, o.rewind) + kFeed : 0;
            const size_t rows = l.swa ? (size_t) l.ring : (size_t) o.max_ctx;
            if (!alloc(l.K, rows * (size_t) KD, err) || !alloc(l.V, rows * (size_t) KD, err)) return false;
        }
        // the fuse (softmax over the taps a row) and the contexts' gains: the layers', then the markov head's
        {
            const Tensor* t = need(ix, "dspark_fuse.weight", err, {NT, NL + 1});
            if (!t) return false;
            if (t->type != "F32") {
                err = "dspark_fuse.weight: F32 expected";
                return false;
            }
            std::vector<float> w((size_t) (NL + 1) * NT);
            std::memcpy(w.data(), ix.data(*t), w.size() * 4);
            for (int r = 0; r <= NL; ++r) {
                float* row = w.data() + (size_t) r * NT;
                const float mx = *std::max_element(row, row + NT);
                double sum = 0;
                for (int i = 0; i < NT; ++i) sum += std::exp((double) row[i] - mx);
                for (int i = 0; i < NT; ++i) row[i] = (float) (std::exp((double) row[i] - mx) / sum);
            }
            if (!alloc(fw, w.size(), err) || !h2d(fw, w.data(), w.size() * 4, err)) return false;
            if (!alloc(gain, (size_t) (NL + 1) * E, err)) return false;
            for (int r = 0; r <= NL; ++r) {
                const std::string n = r < NL ? "blk." + std::to_string(r) + ".dflare_norm.weight" : "dspark_norm.weight";
                const Tensor* g = need(ix, n, err, {E});
                if (!g) return false;
                if (g->type != "F32") {
                    err = n + ": F32 expected";
                    return false;
                }
                if (!h2d(gain + (size_t) r * E, ix.data(*g), g->bytes, err)) return false;
            }
        }
        if (!vec("output_norm.weight", out_norm, E, err) || !vec("mask_embd.weight", mask_x, E, err) ||
            !vec("joint_proj.bias", bj, 3 * R, err))
            return false;

        // ---- scratch (before the head's tables: M's making borrows some)
        // (64: the largest tile whose shared memory fits, 44 KB; 8 the measured default)
        tile = o.attn_tile >= 4 && o.attn_tile <= 64 ? o.attn_tile : 8;
        max_tiles = (int) std::min<int64_t>(kMaxTiles, (o.max_ctx + B + tile - 1) / tile);
        const int qk = std::max(std::max(E, FF), HD);
        if (!alloc(x, (size_t) B * E, err) || !alloc(a, (size_t) B * E, err) || !alloc(emb, (size_t) E, err) ||
            !alloc(qv, (size_t) B * HD, err) || !alloc(kv, (size_t) B * KD, err) || !alloc(vv, (size_t) B * KD, err) ||
            !alloc(ao, (size_t) B * HD, err) || !alloc(ob, (size_t) B * E, err) || !alloc(gu, (size_t) 2 * B * FF, err) ||
            !alloc(hh, (size_t) B * FF, err) || !alloc(ast, (size_t) B * H * max_tiles * 2, err) ||
            !alloc(apart, (size_t) B * H * max_tiles * D, err) || !alloc(hN, (size_t) B * E, err) ||
            !alloc(base, (size_t) (B - 1) * Vp, err) || !alloc(lg, (size_t) Vp, err) || !alloc(z, (size_t) 3 * R, err) ||
            !alloc(state, (size_t) R, err) || !alloc(ctx, (size_t) E, err) ||
            !alloc(prob, (size_t) B, err) || !alloc(tok, (size_t) B, err) ||
            !alloc(bmax, (size_t) blocks_of((size_t) Vp), err) || !alloc(bidx, (size_t) blocks_of((size_t) Vp), err) ||
            !alloc(bsum, (size_t) blocks_of((size_t) Vp), err) || !alloc(x0, (size_t) B * E, err) || !alloc(cumd, 4, err) ||
            !alloc(taps, (size_t) kFeed * NT * E, err) || !alloc(fz, (size_t) (NL + 1) * kFeed * E, err) ||
            !alloc(kc, (size_t) kFeed * KD, err) || !alloc(vc, (size_t) kFeed * KD, err) ||
            !alloc(xq, (size_t) kFeed * qk, err) || !alloc(xd, (size_t) kFeed * qk / 32, err))
            return false;
        for (int i = 0; i < B; ++i)   // (a block's rows before the anchor's: the mask embedding)
            CK(cudaMemcpy(x0 + (size_t) i * E, mask_x, (size_t) E * 4, cudaMemcpyDeviceToDevice));
        if (o.attn_gemm_min >= 0) {   // (the GEMM form of the attention: the long contexts')
            const size_t nkm = (size_t) o.max_ctx + (size_t) B;
            if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, s) != CUBLAS_STATUS_SUCCESS) {
                err = "dspark: no BLAS handle on GPU " + std::to_string(dev);
                return false;
            }
            if (!alloc(Sg, (size_t) B * H * nkm, err) || !alloc(P16, (size_t) B * H * nkm, err) ||
                !alloc(qp, (size_t) B * HD, err) || !alloc(op, (size_t) B * HD, err) || !alloc(xh16, (size_t) B * HD, err) ||
                !alloc(kb16, (size_t) B * KD, err) || !alloc(vb16, (size_t) B * KD, err))
                return false;
            // (the BLAS sets itself up at its first product - a quarter of a second here: not in a draft)
            {
                const float one = 1.0f, zero = 0.0f;
                CK(cudaMemset(xh16, 0, (size_t) B * HD * 2));
                CK(cudaMemset(kb16, 0, (size_t) B * KD * 2));
                const int Q = B * (H / Hkv);
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, B, Q, D, &one, kb16, CUDA_R_16F, KD, D, xh16,
                                               CUDA_R_16F, D, (long long) Q * D, &zero, Sg, CUDA_R_32F, (int) nkm, (long long) Q * nkm, Hkv,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    err = "dspark: the BLAS does not take the attention's F16 products";
                    return false;
                }
                CK(cudaMemset(P16, 0, (size_t) B * H * nkm * 2));   // (and the other product's form)
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, Q, B, &one, kb16, CUDA_R_16F, KD, D, P16,
                                               CUDA_R_16F, (int) nkm, (long long) Q * nkm, &zero, op, CUDA_R_32F, D,
                                               (long long) Q * D, Hkv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) {
                    err = "dspark: the BLAS does not take the attention's F16 products";
                    return false;
                }
                CK(cudaStreamSynchronize(s));
            }
        }
        {   // the tiled attention past kMaxTiles tiles takes larger tiles (draft()'s TK doubles: the same sums, slower);
            // its shared memory caps them - with the GEMM form (the default: from attn_gemm_min keys on, sized from
            // max_ctx) it sees fewer keys than that, without it a context past what the largest tile covers is refused
            // here, not left to a launch that cannot run
            const int64_t keys = blas != nullptr ? std::min<int64_t>(o.max_ctx + B, std::max(0, o.attn_gemm_min - 1))
                                                 : o.max_ctx + B;
            int TK = tile;
            while ((keys + TK - 1) / TK > max_tiles) TK *= 2;
            const size_t shm = ((size_t) TK * D + (size_t) B * (H / Hkv) * TK + (size_t) B * (H / Hkv)) * sizeof(float);
            cudaDeviceProp prop{};
            CK(cudaGetDeviceProperties(&prop, dev));
            const size_t lim = prop.sharedMemPerBlock;
            if (shm > lim) {
                err = "dspark: " + std::to_string(keys) + " keys need tiles of " + std::to_string(TK) + " (" +
                      std::to_string(shm >> 10) + " KB of shared memory, " + std::to_string(lim >> 10) +
                      " KB on GPU " + std::to_string(dev) + "): the GEMM form of the attention (attn_gemm_min >= 0) "
                      "takes a context this long";
                return false;
            }
        }
        FR = (int) std::min<int64_t>(o.max_ctx, (int64_t) std::max(0, o.rewind) + 2 * kFeed);   // (a rewind's reach, two feeds)
        if (!alloc(fring, (size_t) FR * E, err)) return false;

        // ---- the markov head: joint_proj as F32, markov_w1 as F16 rows, M = output @ hidden_out and m0 = output @ its bias
        {
            std::vector<float> f;
            const int jn = 2 * R + 2 * E;
            if (!host_f32(o.joint_index.empty() ? ix : jx, "joint_proj.weight", jn, 3 * R, f, err)) return false;
            float* tmpj = nullptr;
            if (!alloc(Wj, f.size(), err) || !alloc(tmpj, f.size(), err) || !h2d(tmpj, f.data(), f.size() * 4, err)) return false;
            float_to_half_kernel<<<blocks_of(f.size()), 256, 0, s>>>(tmpj, f.size(), (__half*) Wj);
            check("joint f16");
            if (!sync(err)) return false;
            release(tmpj);
            used -= f.size() * 4;
            const Tensor* t1 = need(ix, "markov_w1.weight", err, {R, Vp});
            if (!t1) return false;
            if (!alloc(w1, (size_t) Vp * R, err)) return false;
            if (t1->type == "Q8_0" && R % 32 == 0 && t1->bytes == (size_t) Vp * R / 32 * 34) {
                uint8_t* tmp = nullptr;
                if (!alloc(tmp, t1->bytes, err) || !h2d(tmp, ix.data(*t1), t1->bytes, err)) return false;
                q8_to_f16_kernel<<<blocks_of((size_t) Vp * R), 256, 0, s>>>(tmp, (size_t) Vp * R, (__half*) w1);
                check("q8_to_f16");
                if (!sync(err)) return false;
                release(tmp);
                used -= t1->bytes;
            } else {
                err = "markov_w1.weight: " + t1->type + ", Q8_0 expected";
                return false;
            }
            // X [R + 1][E]: hidden_out's columns, then its bias
            std::vector<float> ho, hb;
            if (!host_f32(ix, "dspark_hidden_out.weight", R, E, ho, err)) return false;
            const Tensor* tb = need(ix, "dspark_hidden_out.bias", err, {E});
            if (!tb) return false;
            std::vector<float> X((size_t) (R + 1) * E);
            for (int e = 0; e < E; ++e)
                for (int c = 0; c < R; ++c) X[(size_t) c * E + e] = ho[(size_t) e * R + c];
            std::memcpy(X.data() + (size_t) R * E, ix.data(*tb), (size_t) E * 4);
            float *dX = nullptr, *dY = nullptr, *dxd = nullptr;
            int8_t* dxq = nullptr;
            const size_t ybytes = (size_t) (R + 1) * Vp * 4;
            if (!alloc(dX, X.size(), err) || !h2d(dX, X.data(), X.size() * 4, err) || !alloc(dY, (size_t) (R + 1) * Vp, err) ||
                !alloc(dxq, (size_t) (R + 1) * E, err) || !alloc(dxd, (size_t) (R + 1) * E / 32, err))
                return false;
            hk::q8_rows(dX, (size_t) E, R + 1, E, dxq, dxd, s);
            for (int c0 = 0; c0 <= R; c0 += 16) {
                const int nc = std::min(16, R + 1 - c0);
                if (!mv({job(out, dxq + (size_t) c0 * E, dxd + (size_t) c0 * E / 32, nc, dY + (size_t) c0 * Vp, (size_t) Vp)}, err))
                    return false;
            }
            if (!alloc(Mq, (size_t) Vp * R, err) || !alloc(msc, (size_t) Vp, err) || !alloc(m0, (size_t) Vp, err)) return false;
            pack_m_kernel<<<blocks_of((size_t) Vp), 256, 0, s>>>(dY, (size_t) Vp, Vp, R, Mq, msc);
            check("pack_m");
            CK(cudaMemcpyAsync(m0, dY + (size_t) R * Vp, (size_t) Vp * 4, cudaMemcpyDeviceToDevice, s));
            if (!sync(err)) return false;
            release(dX);
            release(dY);
            release(dxq);
            release(dxd);
            used -= X.size() * 4 + ybytes + (size_t) (R + 1) * E + (size_t) (R + 1) * E / 32 * 4;
        }
        st.vram_gb = (double) used / 1e9;
        st.load_s = (now_ms() - t0) / 1e3;
        if (o.verbose)
            std::fprintf(stderr, "dspark: %d layers on GPU %d, %.2f GB (%s the target's output matrix), %.1f s\n", NL, dev,
                         st.vram_gb, o.output_dev ? "without" : "with a copy of", st.load_s);
        return true;
    }

    // n <= kFeed positions from pos0, their taps on the device (rows ld apart)
    bool feed_step(const float* d_taps, size_t ld, int n, int64_t pos0, std::string& err) {
        const int KD = Hkv * D;
        fuse_kernel<<<dim3((unsigned) n, (unsigned) (NL + 1)), 256, 0, s>>>(d_taps, ld, E, NT, fw, gain, eps, n, fz);
        check("fuse");
        for (int il = 0; il < NL; ++il) {
            DLayer& l = L[(size_t) il];
            hk::q8_rows(fz + (size_t) il * n * E, (size_t) E, n, E, xq, xd, s);
            if (!mv({job(l.kc, xq, xd, n, kc, (size_t) KD), job(l.vc, xq, xd, n, vc, (size_t) KD)}, err)) return false;
            ik2hip::k::rms(kc, (size_t) D, n * Hkv, D, l.kcn, eps, kc, (size_t) D, s);
            rope_neox_kernel<<<blocks_of((size_t) n * Hkv * D / 2), 256, 0, s>>>(kc, n, Hkv, D, (long long) pos0, log_base);
            check("rope");
            ik2hip::k::store_rows(kc, n, KD, l.K, l.ring, pos0, s);
            ik2hip::k::store_rows(vc, n, KD, l.V, l.ring, pos0, s);
        }
        ik2hip::k::store_rows(fz + (size_t) NL * n * E, n, E, fring, FR, pos0, s);
        return true;
    }
    bool feed(const float* taps_in, size_t ld, bool on_dev, int n, int64_t pos0, std::string& err) {
        if (n <= 0) return true;
        if (pos0 != end || pos0 + n > o.max_ctx) {
            err = "dspark: feed at " + std::to_string(pos0) + " + " + std::to_string(n) + ", the context ends at " +
                  std::to_string(end) + " (max " + std::to_string(o.max_ctx) + ")";
            return false;
        }
        const double t0 = now_ms();
        CK(cudaSetDevice(dev));
        for (int off = 0; off < n; off += kFeed) {
            const int nc = std::min(kFeed, n - off);
            const float* src = taps_in + (size_t) off * ld;
            if (!on_dev) {
                CK(cudaMemcpy(taps, src, (size_t) nc * NT * E * 4, cudaMemcpyHostToDevice));   // (the caller's buffer)
                if (!feed_step(taps, (size_t) NT * E, nc, pos0 + off, err)) return false;
            } else if (!feed_step(src, ld, nc, pos0 + off, err))
                return false;
        }
        end = pos0 + n;
        hi = std::max(hi, end - 1);
        st.fed += n;
        st.feed_ms += now_ms() - t0;
        return true;
    }

    bool draft(int anchor, int max_n, std::vector<int>& toks, std::vector<float>& probs, std::string& err) {
        toks.clear();
        probs.clear();
        const int64_t n = end;
        max_n = std::min(max_n, B - 1);
        if (max_n < 1 || n < 1 || n + B > o.max_ctx || anchor < 0 || anchor >= Vu) return true;
        // the rows a block reads must still be in the rings: the context row of n - 1, the window layers' contexts
        const int64_t lo_swa = std::max<int64_t>(0, n - n_swa);
        if (n - 1 <= hi - FR) return true;
        for (const DLayer& l : L)
            if (l.swa && lo_swa <= hi - l.ring) return true;
        const double t0 = now_ms();
        double tp0 = t0;
        auto phase = [&](double& acc) -> bool {
            if (!o.profile) return true;
            if (!sync(err)) return false;
            const double t = now_ms();
            acc += t - tp0;
            tp0 = t;
            return true;
        };
        CK(cudaSetDevice(dev));
        const int HD = H * D, KD = Hkv * D;
        const float one = 1.0f;
        // ---- the rows: the anchor's normed embedding, then the mask embedding
        {
            const uint8_t* row = tok_embd + (size_t) anchor * tok_row;
            if (tok_type == kQ5K) dequantize_row_q5_K(row, h_x.data(), E);
            else if (tok_type == iq::kQ6_K) dequantize_row_q6_K(row, h_x.data(), E);
            else std::memcpy(h_x.data(), row, (size_t) E * 4);
            CK(cudaMemcpyAsync(emb, h_x.data(), (size_t) E * 4, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(x, x0, (size_t) B * E * 4, cudaMemcpyDeviceToDevice, s));
            ik2hip::k::rms(emb, (size_t) E, 1, E, tok_norm, eps, x, (size_t) E, s);
            CK(cudaMemcpyAsync(tok, &anchor, sizeof(int), cudaMemcpyHostToDevice, s));
            CK(cudaMemsetAsync(state, 0, (size_t) R * 4, s));
            CK(cudaMemcpyAsync(cumd, &one, 4, cudaMemcpyHostToDevice, s));
            half_to_float_kernel<<<blocks_of((size_t) E), 256, 0, s>>>((const __half*) fring + (size_t) ((n - 1) % FR) * E, E, ctx);
            check("ctx");
        }
        if (!phase(st.embed_ms)) return false;
        // ---- the layers
        const float scale = 1.0f / std::sqrt((float) D);
        for (int il = 0; il < NL; ++il) {
            DLayer& l = L[(size_t) il];
            ik2hip::k::rms(x, (size_t) E, B, E, l.attn_norm, eps, a, (size_t) E, s);
            hk::q8_rows(a, (size_t) E, B, E, xq, xd, s);
            if (!mv8({job(l.q, xq, xd, B, qv, (size_t) HD), job(l.kk, xq, xd, B, kv, (size_t) KD),
                     job(l.v, xq, xd, B, vv, (size_t) KD)}, err))
                return false;
            ik2hip::k::rms(qv, (size_t) D, B * H, D, l.qn, eps, qv, (size_t) D, s);
            ik2hip::k::rms(kv, (size_t) D, B * Hkv, D, l.kn, eps, kv, (size_t) D, s);
            rope_neox_kernel<<<blocks_of((size_t) B * H * D / 2), 256, 0, s>>>(qv, B, H, D, (long long) n, log_base);
            rope_neox_kernel<<<blocks_of((size_t) B * Hkv * D / 2), 256, 0, s>>>(kv, B, Hkv, D, (long long) n, log_base);
            if (!phase(st.qkv_ms)) return false;
            const int64_t lo = l.swa ? lo_swa : 0;
            const int nctx = (int) (n - lo);
            if (blas != nullptr && nctx + B >= o.attn_gemm_min) {
                if (!attention_gemm(l, lo, nctx, scale, err)) return false;
            } else {
                // (attn_tile keys a tile; more only where a context needs more than kMaxTiles tiles)
                const int nk = nctx + B;
                int TK = tile;
                while ((nk + TK - 1) / TK > max_tiles) TK *= 2;
                const int ntile = (nk + TK - 1) / TK;
                // (256 threads a tile: 512 / 1024 - a thread a (query, key) pair - measured slower on an R9700)
                attn_tile_kernel<<<dim3((unsigned) ntile, (unsigned) Hkv), 256,
                                   ((size_t) TK * D + (size_t) B * (H / Hkv) * TK + (size_t) B * (H / Hkv)) * sizeof(float), s>>>(
                    qv, kv, vv, (const __half*) l.K, (const __half*) l.V, l.ring, (long long) (l.ring > 0 ? lo % l.ring : lo), nctx, B,
                    H, Hkv, D, TK, scale, ast, apart);
                attn_combine_kernel<<<(unsigned) (B * H), 256, 0, s>>>(ast, apart, ntile, D, ao);
            }
            check("attention");
            if (!phase(st.attn_ms)) return false;
            hk::q8_rows(ao, (size_t) HD, B, HD, xq, xd, s);
            if (!mv8({job(l.o, xq, xd, B, ob, (size_t) E)}, err)) return false;
            ik2hip::k::add(x, ob, (size_t) B * E, s);
            if (!phase(st.o_ms)) return false;
            ik2hip::k::rms(x, (size_t) E, B, E, l.ffn_norm, eps, a, (size_t) E, s);
            hk::q8_rows(a, (size_t) E, B, E, xq, xd, s);
            if (!mv8({job(l.gate, xq, xd, B, gu, (size_t) FF), job(l.up, xq, xd, B, gu + (size_t) B * FF, (size_t) FF)}, err))
                return false;
            hk::swiglu(gu, gu + (size_t) B * FF, (size_t) FF, B, FF, hh, (size_t) FF, s);
            hk::q8_rows(hh, (size_t) FF, B, FF, xq, xd, s);
            if (!mv8({job(l.down, xq, xd, B, ob, (size_t) E)}, err)) return false;
            ik2hip::k::add(x, ob, (size_t) B * E, s);
            if (!phase(st.ffn_ms)) return false;
        }
        ik2hip::k::rms(x, (size_t) E, B, E, out_norm, eps, hN, (size_t) E, s);
        if (!phase(st.layers_ms)) return false;
        if (o.profile) st.layers_ms = st.qkv_ms + st.attn_ms + st.o_ms + st.ffn_ms;
        // ---- the base logits of the draft rows (the target's output matrix, read once)
        hk::q8_rows(hN + (size_t) E, (size_t) E, max_n, E, xq, xd, s);
        if (!mv({job(out, xq, xd, max_n, base, (size_t) Vp)}, err)) return false;
        if (!phase(st.base_ms)) return false;
        // ---- the markov head, a step a draft: all on the device
        for (int i = 1; i <= max_n; ++i) {
            joint_kernel<<<(unsigned) (3 * R), 256, 0, s>>>((const __half*) Wj, bj, state, (const __half*) w1, tok + (i - 1),
                                                        hN + (size_t) i * E, ctx, R, E, z, cumd, cum_min);
            logits_kernel<<<blocks_of((size_t) Vu), 256, 0, s>>>(base + (size_t) (i - 1) * Vp, Mq, msc, m0, z, R, Vu, Vp, cap, lg,
                                                              bmax, bidx, bsum, cumd, cum_min);
            pick_kernel<<<1, 1024, 0, s>>>(bmax, bidx, bsum, (int) blocks_of((size_t) Vu), tok + i, prob + i, z, R, state, cumd,
                                           cum_min);
        }
        check("head");
        int h_tok[Dspark::kBlock];
        float h_prob[Dspark::kBlock];
        CK(cudaMemcpyAsync(h_tok, tok, sizeof(int) * (size_t) (max_n + 1), cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(h_prob, prob, sizeof(float) * (size_t) (max_n + 1), cudaMemcpyDeviceToHost, s));
        if (!sync(err)) return false;
        if (o.profile) st.head_ms += now_ms() - tp0;
        float cum = 1.0f;
        for (int i = 1; i <= max_n; ++i) {
            cum *= h_prob[i];
            if (h_tok[i] < 0 || cum < cum_min) break;
            toks.push_back(h_tok[i]);
            probs.push_back(h_prob[i]);
        }
        ++st.drafts;
        st.draft_ms += now_ms() - t0;
        return true;
    }
};

Dspark::Dspark() : m(new Impl) {}
Dspark::~Dspark() = default;
bool Dspark::load(const DsparkOptions& o, std::string& err) { return m->load(o, err); }
bool Dspark::feed(const float* taps, int n, int64_t pos0, std::string& err) {
    return m->feed(taps, (size_t) kTaps * (size_t) m->E, false, n, pos0, err);
}
bool Dspark::feed_dev(const float* d_taps, size_t ld, int n, int64_t pos0, std::string& err, void* read_done) {
    if (!m->feed(d_taps, ld, true, n, pos0, err)) return false;
    if (read_done != nullptr && cudaEventRecord((cudaEvent_t) read_done, m->s) != cudaSuccess) {
        err = "dspark: the feed's event: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}
bool Dspark::draft(int anchor, int max_n, std::vector<int>& toks, std::vector<float>& probs, std::string& err) {
    return m->draft(anchor, max_n, toks, probs, err);
}
void Dspark::rollback(int64_t pos) { m->end = std::max<int64_t>(0, std::min(m->end, pos)); }
bool Dspark::add_states(int n, std::string& err) { return m->add_states(n, err); }
void Dspark::bind(int i) { m->bind(i); }
void Dspark::reset() {
    m->end = 0;
    m->hi = -1;
}
int64_t Dspark::end() const { return m->end; }
void Dspark::set_cum_min(float c) { m->cum_min = c; }
float Dspark::cum_min() const { return m->cum_min; }
const std::vector<int>& Dspark::target_layers() const { return m->targets; }
int Dspark::n_embd() const { return m->E; }
int Dspark::vocab() const { return m->Vu; }
const DsparkStats& Dspark::stats() const { return m->st; }

}  // namespace ik2hip
