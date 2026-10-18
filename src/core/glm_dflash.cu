// src/core/glm_dflash.cu - the DFlash2 drafter (include/strata/core/glm_dflash.hpp) and its decode on the GLM trunk.
//
// The drafter's forward follows beellama's llama_model_dflash (src/models/dflash.cpp), DFlash2 path:
//   context (per committed position p, from the trunk): g = rms(fc . [f_k0 | f_k1 | ...]) * enc.output_norm, then
//     per layer K = rope_p(rms_head(Wk g) * k_norm), V = Wv g into the layer's ring slot p % ring;
//   block (rows t = 0..B-1 at positions n_past + t, row 0 the anchor token, the rest <mask>): h = emb(tokens), then
//     per layer:  x = rms(h) * attn_norm;  c = conv_proj . x;  x = conv(x, c, side 0);
//                 q, k, v = Wq x, Wk x, Wv x (per-head rms * q_norm / k_norm, NeoX rope at the row's position);
//                 a = softmax(q . [ring keys in the window | the block's keys] / sqrt(d)) . values (non-causal);
//                 h += conv(Wo a, c, side 1);
//                 x = rms(h) * ffn_norm;  c = ffn_conv_proj . x;  x = conv(x, c, side 0);
//                 h += conv(Wdown (silu(Wgate x) * Wup x), c, side 1)
//   then y = rms(h) * output_norm; the TARGET's head gives each mask row its top-k candidates and their logits, and
//   selector_hidden . y the row's gate; the score of candidate j after predecessor token q is
//   logit_j + sum_r prev[q][r] * gate[r] * next[cand_j][r], and the greedy walk from the anchor picks the drafts.
// conv(x, c, side): out[t][e] = sum_tap (c[t][side][tap][e / group] + base[side][tap][e]) * x[t - tap][e], within the
// block (x[t - tap] = 0 before row 0).  The context window is the GGUF's sliding window: a row at position P reads
// the ring's positions above P - window.
#include "strata/core/glm_dflash.hpp"

#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace strata::core {

namespace {

// the ggml blocks the selector tables may come in (ggml-common.h's layouts)
struct BlkQ8_0 {
    __half d;
    int8_t qs[32];
};
struct BlkQ4K {
    __half2 dm;   // d, dmin
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlkQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};
static_assert(sizeof(BlkQ8_0) == 34 && sizeof(BlkQ4K) == 144 && sizeof(BlkQ6K) == 210, "ggml block layouts");

constexpr int kTopKMax = 32;    // selector candidates a row (the GGUF's selector_top_k)
constexpr int kInjRows = 64;    // context positions an injection batch carries
constexpr int kHeadDim = 128;   // the attention / rope kernels' head size

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block (blockDim.x a multiple of 32, <= 1024); every thread gets it
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

// ---- the trunk side: a pass's rows -> the mean of their hc streams (the drafter's feature of that layer)
__global__ void hc_mean_kernel(const float* __restrict__ R, int hc, int E, float* __restrict__ out) {
    const int t = blockIdx.x;
    const float* r = R + (size_t) t * hc * E;
    const float inv = 1.0f / (float) hc;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += r[(size_t) s * E + e];
        out[(size_t) t * E + e] = a * inv;
    }
}

// y = x * rsqrt(mean(x^2) + eps) * w, a row per block
__global__ void rms_rows_kernel(const float* __restrict__ x, int ldx, const float* __restrict__ w, float eps, int n,
                                float* __restrict__ y, int ldy) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, red) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

// per head (blockIdx.x) of row blockIdx.y: rms over the head * w, then NeoX rope at position pos0 + row
__global__ void head_norm_rope_kernel(float* __restrict__ v, int n_heads, const float* __restrict__ w, float eps,
                                      int64_t pos0, float theta_base) {
    __shared__ float red[32];
    const int h = blockIdx.x, row = blockIdx.y, j = threadIdx.x;   // j < kHeadDim / 2
    float* p = v + ((size_t) row * n_heads + h) * kHeadDim;
    const float x0 = p[j], x1 = p[j + kHeadDim / 2];
    const float r = rsqrtf(block_sum(x0 * x0 + x1 * x1, red) / (float) kHeadDim + eps);
    const float a = x0 * r * w[j], b = x1 * r * w[j + kHeadDim / 2];
    const float theta = (float) (pos0 + row) * powf(theta_base, -2.0f * (float) j / (float) kHeadDim);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    p[j] = a * cs - b * sn;
    p[j + kHeadDim / 2] = a * sn + b * cs;
}

// conv: see the file's header.  c rows hold [side][tap][group] floats (ldc apart), base [side][tap][E]
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ c, int ldc,
                            const float* __restrict__ base, int side, int E, int taps, int group,
                            float* __restrict__ out) {
    const int t = blockIdx.x, ng = E / group;
    const float* ct = c + (size_t) t * ldc + (size_t) side * taps * ng;
    const float* bs = base + (size_t) side * taps * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float acc = 0.0f;
        for (int tap = 0; tap < taps && tap <= t; ++tap)
            acc += (ct[tap * ng + e / group] + bs[(size_t) tap * E + e]) * x[(size_t) (t - tap) * E + e];
        out[(size_t) t * E + e] = acc;
    }
}

__global__ void add_kernel(float* __restrict__ h, const float* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += a[i];
}

__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ o) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        o[i] = x / (1.0f + __expf(-x)) * u[i];
    }
}

// the context's K / V rows (positions pos0 + r) into a layer's ring (FP16), kvd values a row
__global__ void kv_store_kernel(const float* __restrict__ K, const float* __restrict__ V, int kvd, int64_t pos0,
                                int ring, __half* __restrict__ Kr, __half* __restrict__ Vr) {
    const int r = blockIdx.x;
    const size_t slot = (size_t) ((pos0 + r) % ring);
    for (int i = threadIdx.x; i < kvd; i += blockDim.x) {
        Kr[slot * kvd + i] = __float2half(K[(size_t) r * kvd + i]);
        Vr[slot * kvd + i] = __float2half(V[(size_t) r * kvd + i]);
    }
}

__global__ void kv_pos_kernel(int64_t pos0, int n, int ring, long long* __restrict__ kv_pos) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n) kv_pos[(pos0 + r) % ring] = (long long) (pos0 + r);
}

// one (head, row) per block, kHeadDim threads: the row at position n_past + row reads the ring's positions in
// [max(0, P - window + 1), n_past) whose slot holds that position, and every row of the block (non-causal)
__global__ void attn_kernel(const float* __restrict__ Q, const __half* __restrict__ Kr, const __half* __restrict__ Vr,
                            const long long* __restrict__ kv_pos, const float* __restrict__ Kb,
                            const float* __restrict__ Vb, int rows, int n_head, int n_kv, int64_t n_past, int ring,
                            int window, float scale, float* __restrict__ out) {
    extern __shared__ float sc[];
    __shared__ float qs[kHeadDim];
    __shared__ float red[32];
    const int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x, lane = d & 31, wid = d >> 5, nw = kHeadDim / 32;
    const int kvh = h / (n_head / n_kv), kvd = n_kv * kHeadDim;
    const int64_t P = n_past + i;
    const int64_t lo = P - window + 1 > 0 ? P - window + 1 : 0;
    const int nc = n_past > lo ? (int) (n_past - lo) : 0, nk = nc + rows;
    qs[d] = Q[((size_t) i * n_head + h) * kHeadDim + d];
    __syncthreads();
    // scores: a warp per key, 4 dims a lane
    for (int k = wid; k < nk; k += nw) {
        float acc = 0.0f;
        bool ok = true;
        if (k < nc) {
            const int64_t p = lo + k;
            const size_t slot = (size_t) (p % ring);
            ok = kv_pos[slot] == (long long) p;
            const __half* kr = Kr + slot * kvd + (size_t) kvh * kHeadDim;
            for (int e = lane; e < kHeadDim; e += 32) acc += qs[e] * __half2float(kr[e]);
        } else {
            const float* kb = Kb + (size_t) (k - nc) * kvd + (size_t) kvh * kHeadDim;
            for (int e = lane; e < kHeadDim; e += 32) acc += qs[e] * kb[e];
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[k] = ok ? acc * scale : -INFINITY;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int k = d; k < nk; k += kHeadDim) m = fmaxf(m, sc[k]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < nw; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int k = d; k < nk; k += kHeadDim) {
        const float e = sc[k] == -INFINITY ? 0.0f : __expf(sc[k] - m);
        sc[k] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const float pk = sc[k];
        if (pk == 0.0f) continue;
        if (k < nc) {
            const size_t slot = (size_t) ((lo + k) % ring);
            acc += pk * __half2float(Vr[slot * kvd + (size_t) kvh * kHeadDim + d]);
        } else {
            acc += pk * Vb[(size_t) (k - nc) * kvd + (size_t) kvh * kHeadDim + d];
        }
    }
    out[((size_t) i * n_head + h) * kHeadDim + d] = sum > 0.0f ? acc / sum : 0.0f;
}

// The split attention (the same keys and masks as attn_kernel, which reads each K / V row once a (head, row): over
// the 2048-key window of a long context that was most of a draft).  Block (kv head, split) holds every query that
// reads that kv head - its q heads x the block's rows, kAttnQ at most - and runs an online softmax over its split's
// keys, kAttnKeys at a time, each K / V row read once for all of them; it leaves per query the running max, the sum
// and the weighted V sum (kHeadDim + 2 floats), which attn_merge_kernel folds over the splits.  256 threads.
constexpr int kAttnQ = 32, kAttnKeys = 16, kAttnSplits = 32, kAttnPart = kHeadDim + 2;

__device__ __forceinline__ float half_warp_max(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float half_warp_sum(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(256) attn_split_kernel(const float* __restrict__ Q, const __half* __restrict__ Kr,
                                                         const __half* __restrict__ Vr,
                                                         const long long* __restrict__ kv_pos,
                                                         const float* __restrict__ Kb, const float* __restrict__ Vb,
                                                         int rows, int n_head, int n_kv, int64_t n_past, int ring,
                                                         int window, float scale, int per_split,
                                                         float* __restrict__ part) {
    __shared__ float qs[kAttnQ * (kHeadDim + 1)];
    __shared__ float ks[kAttnKeys * (kHeadDim + 1)];
    __shared__ float vs[kAttnKeys * kHeadDim];
    __shared__ float sc[kAttnQ * kAttnKeys];
    __shared__ float mrun[kAttnQ], lrun[kAttnQ], corr[kAttnQ];
    __shared__ long long kp[kAttnKeys];   // the key's position, -1 a block key, -2 none
    const int t = threadIdx.x, kvh = blockIdx.x, G = n_head / n_kv, nq = G * rows, kvd = n_kv * kHeadDim;
    const int64_t lo0 = n_past - window + 1 > 0 ? n_past - window + 1 : 0;
    const int nc = n_past > lo0 ? (int) (n_past - lo0) : 0, nk = nc + rows;
    const int k0 = blockIdx.y * per_split, k1 = min(nk, k0 + per_split);
    // query q = g x rows + i: head kvh x G + g at row i
    for (int e = t; e < kAttnQ * kHeadDim; e += blockDim.x) {
        const int q = e / kHeadDim, d = e % kHeadDim;
        qs[q * (kHeadDim + 1) + d] =
            q < nq ? Q[((size_t) (q % rows) * n_head + kvh * G + q / rows) * kHeadDim + d] : 0.0f;
    }
    if (t < kAttnQ) {
        mrun[t] = -INFINITY;
        lrun[t] = 0.0f;
    }
    const int d = t & (kHeadDim - 1), qh = t >> 7;   // the weighted sum: dim d of queries qh x 16 ..
    const int j = t & (kAttnKeys - 1);              // the scores: key j of query (t >> 4) + 16 r
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (int c = k0; c < k1; c += kAttnKeys) {
        __syncthreads();   // the previous chunk read
        if (t < kAttnKeys) {
            const int k = c + t;
            long long pos = -2;
            if (k < k1 && k >= nc) {
                pos = -1;
            } else if (k < k1) {
                const int64_t p = lo0 + k;
                if (kv_pos[p % ring] == (long long) p) pos = (long long) p;
            }
            kp[t] = pos;
        }
        __syncthreads();
        for (int e = t; e < kAttnKeys * kHeadDim; e += blockDim.x) {
            const int jj = e / kHeadDim, dd = e % kHeadDim;
            const long long pos = kp[jj];
            float kv = 0.0f, vv = 0.0f;
            if (pos >= 0) {
                const size_t off = (size_t) (pos % ring) * kvd + (size_t) kvh * kHeadDim + dd;
                kv = __half2float(Kr[off]);
                vv = __half2float(Vr[off]);
            } else if (pos == -1) {
                const size_t off = (size_t) (c + jj - nc) * kvd + (size_t) kvh * kHeadDim + dd;
                kv = Kb[off];
                vv = Vb[off];
            }
            ks[jj * (kHeadDim + 1) + dd] = kv;
            vs[jj * kHeadDim + dd] = vv;
        }
        __syncthreads();
        const long long pos = kp[j];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int q = (t >> 4) + 16 * r;
            float sv = -INFINITY;
            // row i reads the context at or past its own window's start
            if (q < nq && pos != -2 && (pos == -1 || pos >= n_past + q % rows - window + 1)) {
                float a = 0.0f;
#pragma unroll 8
                for (int e = 0; e < kHeadDim; ++e) a += qs[q * (kHeadDim + 1) + e] * ks[j * (kHeadDim + 1) + e];
                sv = a * scale;
            }
            const float mo = mrun[q];
            const float mn = fmaxf(mo, half_warp_max(sv));
            const float pw = sv == -INFINITY ? 0.0f : __expf(sv - mn);
            sc[q * kAttnKeys + j] = pw;
            const float ps = half_warp_sum(pw);
            __syncwarp();
            if (j == 0) {
                const float cf = mo == -INFINITY ? 0.0f : __expf(mo - mn);
                corr[q] = cf;
                lrun[q] = lrun[q] * cf + ps;
                mrun[q] = mn;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int q = qh * 16 + r;
            float a = acc[r] * corr[q];
#pragma unroll
            for (int k = 0; k < kAttnKeys; ++k) a += sc[q * kAttnKeys + k] * vs[k * kHeadDim + d];
            acc[r] = a;
        }
    }
    __syncthreads();
    float* P = part + ((size_t) blockIdx.y * n_kv + kvh) * kAttnQ * kAttnPart;
#pragma unroll
    for (int r = 0; r < 16; ++r) P[(size_t) (qh * 16 + r) * kAttnPart + 2 + d] = acc[r];
    if (t < kAttnQ) {
        P[(size_t) t * kAttnPart] = mrun[t];
        P[(size_t) t * kAttnPart + 1] = lrun[t];
    }
}

// block (head, row), kHeadDim threads: the splits' partials of that query into its attention output
__global__ void attn_merge_kernel(const float* __restrict__ part, int splits, int rows, int n_head, int n_kv,
                                  float* __restrict__ out) {
    const int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x, G = n_head / n_kv;
    const int kvh = h / G, q = (h % G) * rows + i;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part[(((size_t) s * n_kv + kvh) * kAttnQ + q) * kAttnPart]);
    float L = 0.0f, a = 0.0f;
    if (M != -INFINITY)
        for (int s = 0; s < splits; ++s) {
            const float* P = part + (((size_t) s * n_kv + kvh) * kAttnQ + q) * kAttnPart;
            if (P[0] == -INFINITY) continue;
            const float f = __expf(P[0] - M);
            L += f * P[1];
            a += f * P[2 + d];
        }
    out[((size_t) i * n_head + h) * kHeadDim + d] = L > 0.0f ? a / L : 0.0f;
}

// the top k of each row (descending; ties by the lower id) in two steps: each of kTopKSlices blocks takes a slice
// of the row (in shared memory) and finds the slice's top k - k rounds of a block-wide arg-max under the last pick -
// then one warp a row takes the top k of the slices' candidates the same way.  The same picks as one block over the
// whole row, which took ~0.5-0.65 ms for the vocabulary (16 rounds over 154,880 logits on one SM); ~0.05 now
constexpr int kTopKSlices = 64;
constexpr int kHeadSamples = 6;   // the head's timed drafts a variant (draft_start)

__device__ __forceinline__ bool tk_before(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }
__device__ __forceinline__ bool tk_after(float v, int i, float lv, int li) { return v < lv || (v == lv && i > li); }

// (row_slices: the slices a row's candidates hold, this launch's at the first of them; id_off: added to the ids - the
//  head split's helper takes the row's later slices of its own columns)
__global__ void __launch_bounds__(256) topk_part_kernel(const float* __restrict__ x, int n, int k, int slice,
                                                        int* __restrict__ part_i, float* __restrict__ part_v,
                                                        int row_slices, int id_off) {
    extern __shared__ float xs[];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ float last_v;
    __shared__ int last_i;
    const int lo = blockIdx.x * slice, len = max(0, min(n - lo, slice));
    const float* xr = x + (size_t) blockIdx.y * n + lo;
    for (int i = threadIdx.x; i < len; i += blockDim.x) xs[i] = xr[i];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    if (threadIdx.x == 0) {
        last_v = INFINITY;
        last_i = -1;
    }
    __syncthreads();
    const size_t o = ((size_t) blockIdx.y * row_slices + blockIdx.x) * k;
    const int id0 = lo + id_off;
    for (int r = 0; r < k; ++r) {
        const float lv = last_v;
        const int li = last_i;
        float v = -INFINITY;
        int vi = 0x7fffffff;
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            const float y = xs[i];
            if (tk_after(y, id0 + i, lv, li) && tk_before(y, id0 + i, v, vi)) {
                v = y;
                vi = id0 + i;
            }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, d);
            const int oi = __shfl_xor_sync(0xffffffffu, vi, d);
            if (tk_before(ov, oi, v, vi)) {
                v = ov;
                vi = oi;
            }
        }
        if (lane == 0) {
            bv[w] = v;
            bi[w] = vi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int bj = bi[0];
            for (int j = 1; j < nw; ++j)
                if (tk_before(bv[j], bi[j], b, bj)) {
                    b = bv[j];
                    bj = bi[j];
                }
            part_i[o + r] = bj;
            part_v[o + r] = b;
            last_v = b;
            last_i = bj;
        }
        __syncthreads();
    }
}

// one warp a row: the top k of the row's kTopKSlices x k candidates
__global__ void __launch_bounds__(32) topk_merge_kernel(const int* __restrict__ part_i, const float* __restrict__ part_v,
                                                        int k, int* __restrict__ ids, float* __restrict__ vals) {
    extern __shared__ float cv[];
    const int c = kTopKSlices * k;
    int* ci = (int*) (cv + c);
    const size_t o = (size_t) blockIdx.x * c;
    for (int j = threadIdx.x; j < c; j += 32) {
        cv[j] = part_v[o + j];
        ci[j] = part_i[o + j];
    }
    __syncwarp();
    float lv = INFINITY;
    int li = -1;
    for (int r = 0; r < k; ++r) {
        float v = -INFINITY;
        int vi = 0x7fffffff;
        for (int j = threadIdx.x; j < c; j += 32) {
            const float y = cv[j];
            const int yi = ci[j];
            if (tk_after(y, yi, lv, li) && tk_before(y, yi, v, vi)) {
                v = y;
                vi = yi;
            }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, d);
            const int oi = __shfl_xor_sync(0xffffffffu, vi, d);
            if (tk_before(ov, oi, v, vi)) {
                v = ov;
                vi = oi;
            }
        }
        if (threadIdx.x == 0) {
            ids[(size_t) blockIdx.x * k + r] = vi;
            vals[(size_t) blockIdx.x * k + r] = v;
        }
        lv = v;
        li = vi;
    }
}

// element e of row `row` of a selector table (rows of `rank` values in ggml type `type`)
__device__ float tab_at(const uint8_t* tab, int type, int rank, int row, int e) {
    switch (type) {
        case 0: return ((const float*) tab)[(size_t) row * rank + e];
        case 1: return __half2float(((const __half*) tab)[(size_t) row * rank + e]);
        case 30: {
            const uint16_t b = ((const uint16_t*) tab)[(size_t) row * rank + e];
            return __uint_as_float((unsigned int) b << 16);
        }
        case 8: {   // Q8_0
            const BlkQ8_0* q = (const BlkQ8_0*) tab + (size_t) row * (rank / 32) + e / 32;
            return __half2float(q->d) * (float) q->qs[e % 32];
        }
        case 12: {   // Q4_K
            const BlkQ4K* q = (const BlkQ4K*) tab + (size_t) row * (rank / 256) + e / 256;
            const int r = e % 256, j = r / 64, l = r % 32, hi = (r % 64) >= 32, is = 2 * j + hi;
            uint8_t sc, mn;
            if (is < 4) {
                sc = q->scales[is] & 63;
                mn = q->scales[is + 4] & 63;
            } else {
                sc = (q->scales[is + 4] & 0xF) | ((q->scales[is - 4] >> 6) << 4);
                mn = (q->scales[is + 4] >> 4) | ((q->scales[is] >> 6) << 4);
            }
            const uint8_t b = q->qs[32 * j + l];
            const int v = hi ? b >> 4 : b & 0xF;
            return __low2float(q->dm) * (float) sc * (float) v - __high2float(q->dm) * (float) mn;
        }
        case 14: {   // Q6_K
            const BlkQ6K* q = (const BlkQ6K*) tab + (size_t) row * (rank / 256) + e / 256;
            const int r = e % 256, n = r / 128, rr = r % 128, qt = rr / 32, l = rr % 32, is = l / 16;
            const uint8_t* ql = q->ql + 64 * n;
            const uint8_t* qh = q->qh + 32 * n;
            const int8_t* sc = q->scales + 8 * n;
            int v;
            switch (qt) {
                case 0: v = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
                case 1: v = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
                case 2: v = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
                default: v = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
            }
            return __half2float(q->d) * (float) sc[is + 2 * qt] * (float) (v - 32);
        }
        default: return 0.0f;
    }
}

// the selector's greedy walk over the rows 0..n-1 (the block's mask rows 1..n): `rank` threads.  Row i's candidates
// cand[i][0..k), their logits lg[i][..], its gate[i][0..rank); the predecessor of row 0 is the anchor, of row i the
// pick of row i - 1.  tok[i] = the pick, conf[i] = its softmax probability among the row's k scores.
__global__ void walk_kernel(const int* __restrict__ cand, const float* __restrict__ lg, const float* __restrict__ gate,
                            int k, int rank, const uint8_t* __restrict__ prev_tab, const uint8_t* __restrict__ next_tab,
                            int tab_type, int anchor, int n, int* __restrict__ tok, float* __restrict__ conf) {
    __shared__ float as[1024];
    __shared__ float s[kTopKMax];
    __shared__ int pred;
    const int r = threadIdx.x, lane = r & 31, w = r >> 5, nw = (int) (blockDim.x >> 5);
    if (r == 0) pred = anchor;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        as[r] = tab_at(prev_tab, tab_type, rank, pred, r) * gate[(size_t) i * rank + r];
        __syncthreads();
        // each warp a share of the candidates, a candidate's dot over the rank in one warp
        for (int j = w; j < k; j += nw) {
            const int c = cand[(size_t) i * k + j];
            float t = 0.0f;
            for (int e = lane; e < rank; e += 32) t += as[e] * tab_at(next_tab, tab_type, rank, c, e);
            t = warp_sum(t);
            if (lane == 0) s[j] = lg[(size_t) i * k + j] + t;
        }
        __syncthreads();
        if (r == 0) {
            int b = 0;
            for (int j = 1; j < k; ++j)
                if (s[j] > s[b]) b = j;
            float z = 0.0f;
            for (int j = 0; j < k; ++j) z += __expf(s[j] - s[b]);
            tok[i] = cand[(size_t) i * k + b];
            conf[i] = z > 0.0f ? 1.0f / z : 0.0f;
            pred = tok[i];
        }
        __syncthreads();
    }
}

// STRATA_GLM_DFLASH_DUMP=<dir> (debug): a draft's first layer, stage by stage, as <dir>/<name>.f32 (synchronous)
void dump(const char* name, const float* dev, size_t n, cudaStream_t s) {
    static const char* dir = getenv("STRATA_GLM_DFLASH_DUMP");
    if (dir == nullptr || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> b(n);
    cudaMemcpy(b.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
        std::fwrite(b.data(), sizeof(float), n, f);
        std::fclose(f);
    }
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("glm dflash: ") + what + ": " + cudaGetErrorString(e);
    cudaGetLastError();
    return false;
}

}  // namespace

// ---------------------------------------------------------------- the drafter
struct GlmDflash::Impl {
    int dev = 0;
    std::string path;
    // geometry (the GGUF's dflash.* keys)
    int E = 0, n_layer = 0, n_head = 0, n_kv = 0, n_ff = 0, block = 0, taps = 0, group = 0, rank = 0, top_k = 0;
    int window = 0, ring = 0, K = 0;   // K: target layers
    float eps = 1e-6f, rope_base = 10000.0f;
    int mask_tok = -1;
    std::vector<int> tl;               // target layers
    // weights
    struct Mat {
        const void* w = nullptr;
        int type = 0, n_in = 0, n_out = 0;
    };
    struct Layer {
        const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        const float *attn_base = nullptr, *ffn_base = nullptr;
        Mat q, k, v, o, attn_conv, ffn_conv, gate, up, down;
    };
    std::vector<Layer> L;
    Mat fc, sel_hidden;
    const float *enc_norm = nullptr, *out_norm = nullptr;
    const uint8_t *sel_prev = nullptr, *sel_next = nullptr;
    int sel_type = 0;
    void* arena = nullptr;
    size_t arena_bytes = 0;
    // the K/V ring: per layer ring x (n_kv x 128) FP16 keys and values; kv_pos[slot]: the position held (-1 none)
    __half *Kr = nullptr, *Vr = nullptr;
    long long* kv_pos = nullptr;
    std::vector<int64_t> kv_pos_h;
    // the trunk's staging (pinned): ring x K x E floats; per slot the position staged and the layers that arrived
    float* stage = nullptr;
    std::mutex mu;
    std::vector<int64_t> st_pos;
    std::vector<uint32_t> st_mask;
    // buffers
    void* buf = nullptr;
    size_t buf_bytes = 0;
    float *feat = nullptr, *g = nullptr, *gn = nullptr, *kc = nullptr, *vc = nullptr;
    void *fq = nullptr, *gq = nullptr;
    float *h = nullptr, *x = nullptr, *xc = nullptr, *cbuf = nullptr, *q = nullptr, *kb = nullptr, *vb = nullptr;
    float *a = nullptr, *o = nullptr, *gt = nullptr, *up = nullptr, *act = nullptr, *logits = nullptr, *sg = nullptr;
    void *xq = nullptr, *aq = nullptr, *actq = nullptr;
    float* attn_part = nullptr;        // the split attention's partials: kAttnSplits x n_kv x kAttnQ x kAttnPart
    cudaStream_t ds = nullptr;         // its own stream (stream()): it drafts while the target's parts run
    cudaStream_t is = nullptr;         // the context injection's stream (inject: beside the next draft's first layer)
    cudaEvent_t ev_s = nullptr;        // ... ordered after the drafter's stream
    std::vector<cudaEvent_t> ev_kv;    // ... each layer's context rows in the ring (the draft's attention waits)
    bool inj_pending = false;          // an injection the next draft waits for, layer by layer
    cudaStream_t pend_s = nullptr;     // draft_start's stream, its drafts (0: none pending) and cut
    int pend_n = 0;
    float pend_p_min = 0.0f;
    float* sel_lg = nullptr;
    int* sel_id = nullptr;
    float* part_lg = nullptr;          // the top k's slice candidates: block x kTopKSlices x top_k
    int* part_id = nullptr;
    int* tok = nullptr;
    float* conf = nullptr;
    int* tok_h = nullptr;
    float* conf_h = nullptr;
    float* emb_h = nullptr;            // pinned: the block's embedding rows
    cudaEvent_t emb_ev = nullptr;      // ... their last copy to the device
    std::vector<float> mask_row;
    int head_mma = -1;                 // the head's several columns on the tensor cores (-1: not decided yet)
    const void* head_mma_w = nullptr;  // ... for this head's weights
    int head_type = 0;
    cudaEvent_t head_ev[2] = {nullptr, nullptr};
    std::vector<float> head_t[2];      // the head's ms: dp4a, tensor cores
    int head_try = 0;                  // the variant the pending draft timed
    bool head_timed = false;
    int cdyn = 0;                      // the conv projection's width: 2 x taps x E / group
    int V = 0;                         // the target's vocabulary (bind)
    struct HeadSplit {                 // split_head: the head's columns [c0, V) on the helper (lo == nullptr: none)
        const WSlot *lo = nullptr, *hi = nullptr;
        int c0 = 0, dev = -1, rows = 0;
        cudaStream_t s = nullptr;
        float *x = nullptr, *lg = nullptr;   // on the helper: the rows in, their logits
        void* cand = nullptr;                // ... their slices' candidates: ids, then values
        size_t cand_bytes = 0;
        float* x_h = nullptr;                // pinned: the rows out, the candidates back
        void* cand_h = nullptr;
        cudaEvent_t ev_x = nullptr, ev_c = nullptr;
    } hs;

    size_t q8(int n) const { return (size_t) n / 32 * 36; }
};

GlmDflash::GlmDflash() : d_(new Impl) {}

GlmDflash::~GlmDflash() {
    if (!d_) return;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(d_->dev);
    if (d_->arena) cudaFree(d_->arena);
    if (d_->buf) cudaFree(d_->buf);
    if (d_->Kr) cudaFree(d_->Kr);
    if (d_->Vr) cudaFree(d_->Vr);
    if (d_->kv_pos) cudaFree(d_->kv_pos);
    if (d_->stage) cudaFreeHost(d_->stage);
    if (d_->tok_h) cudaFreeHost(d_->tok_h);
    if (d_->conf_h) cudaFreeHost(d_->conf_h);
    if (d_->emb_h) cudaFreeHost(d_->emb_h);
    if (d_->emb_ev) cudaEventDestroy(d_->emb_ev);
    for (cudaEvent_t e : d_->head_ev)
        if (e) cudaEventDestroy(e);
    if (d_->ds) cudaStreamDestroy(d_->ds);
    cudaSetDevice(cur);
}

int GlmDflash::dev() const { return d_->dev; }

// The drafter's stream at the device's highest priority: a round's drafts gate every draft token's entry into the
// pipelined window, and at the round's start the head's card is still running the layer a cancelled token was in
// (the skip takes effect at the next layer) - ~1.7 ms of such work shared the GPU with each draft and stretched it
// from ~3.4 to 4-6 ms (EXL3-4.05bpw over 9 GPUs, nsys).  STRATA_GLM_DFLASH_PRIO=0: an ordinary stream (A/B)
cudaStream_t GlmDflash::stream() {
    if (d_->ds == nullptr) {
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(d_->dev);
        static const bool prio = [] {
            const char* v = getenv("STRATA_GLM_DFLASH_PRIO");
            return v == nullptr || std::atoi(v) != 0;
        }();
        int least = 0, greatest = 0;
        if (!prio || cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
            cudaGetLastError();
            greatest = least = 0;
        }
        if (cudaStreamCreateWithPriority(&d_->ds, cudaStreamNonBlocking, greatest) != cudaSuccess) {
            cudaGetLastError();
            d_->ds = nullptr;
        }
        cudaSetDevice(cur);
    }
    return d_->ds;
}
int GlmDflash::block() const { return d_->block; }
int GlmDflash::ctx() const { return d_->window; }
const std::vector<int>& GlmDflash::target_layers() const { return d_->tl; }
size_t GlmDflash::bytes() const {
    const Impl& m = *d_;
    return m.arena_bytes + m.buf_bytes + (size_t) 2 * m.n_layer * m.ring * m.n_kv * kHeadDim * sizeof(__half) +
           (size_t) m.ring * sizeof(long long);
}

std::string GlmDflash::describe() const {
    const Impl& m = *d_;
    std::string t;
    for (size_t i = 0; i < m.tl.size(); ++i) t += (i ? "," : "") + std::to_string(m.tl[i]);
    char b[320];
    std::snprintf(b, sizeof b, "%d layers, hidden %d, %d heads (%d kv), block %d, window %d, selector top-%d rank %d, "
                               "reads trunk layers %s; %.0f MB on CUDA%d",
                  m.n_layer, m.E, m.n_head, m.n_kv, m.block, m.window, m.top_k, m.rank, t.c_str(),
                  (double) bytes() / 1048576.0, m.dev);
    return b;
}

bool GlmDflash::load(const std::string& path, int dev, std::string& err) {
    Impl& m = *d_;
    m.dev = dev;
    m.path = path;
    std::unique_ptr<strata::GgufFile> gf_;
    try {
        gf_.reset(new strata::GgufFile(path));
    } catch (const std::exception& e) {
        err = "glm dflash: " + path + ": " + e.what();
        return false;
    }
    const strata::GgufFile& G = *gf_;
    const auto* arch = G.get("general.architecture");
    if (arch == nullptr || arch->s != "dflash") {
        err = "glm dflash: " + path + " is not a DFlash draft (general.architecture " +
              (arch ? "'" + arch->s + "'" : std::string("missing")) + ")";
        return false;
    }
    const auto num = [&](const char* k, double def) {
        const auto* v = G.get(std::string("dflash.") + k);
        return v != nullptr && v->is_num() ? v->num() : def;
    };
    m.E = (int) num("embedding_length", 0);
    m.n_layer = (int) num("block_count", 0);
    m.n_head = (int) num("attention.head_count", 0);
    m.n_kv = (int) num("attention.head_count_kv", m.n_head);
    m.n_ff = (int) num("feed_forward_length", 0);
    m.block = (int) num("block_size", 16);
    m.taps = (int) num("conv_kernel_size", 0);
    m.group = (int) num("conv_group_size", 0);
    m.rank = (int) num("selector_rank", 0);
    m.top_k = (int) num("selector_top_k", 0);
    m.window = (int) num("attention.sliding_window", 0);
    m.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_base = (float) num("rope.freq_base", 10000.0);
    const int head_dim = (int) num("attention.key_length", m.n_head > 0 ? m.E / m.n_head : 0);
    const int vdim = (int) num("attention.value_length", head_dim);
    if (const auto* c = G.get("dflash.attention.causal"); c != nullptr && c->is_num() && c->u != 0) {
        err = "glm dflash: " + path + " asks for causal block attention (this drafter attends non-causally)";
        return false;
    }
    if (const auto* tlv = G.get("dflash.target_layers"); tlv != nullptr && tlv->type == strata::MetaType::ARRAY)
        for (const auto& it : tlv->items) m.tl.push_back((int) it.num());
    if (const auto* mt = G.get("tokenizer.ggml.mask_token_id"); mt != nullptr && mt->is_num()) m.mask_tok = (int) mt->u;
    m.K = (int) m.tl.size();
    if (m.E <= 0 || m.n_layer <= 0 || m.n_head <= 0 || m.n_kv <= 0 || m.n_ff <= 0 || m.K <= 0 || m.K > 32 ||
        m.mask_tok < 0) {
        err = "glm dflash: " + path + ": incomplete dflash.* metadata (hidden size, layers, heads, target_layers, "
              "the mask token)";
        return false;
    }
    if (head_dim != kHeadDim || vdim != kHeadDim || m.n_head % m.n_kv != 0 || m.E % 32 != 0 || m.n_ff % 32 != 0) {
        err = "glm dflash: " + path + ": head size " + std::to_string(head_dim) + " (this drafter runs " +
              std::to_string(kHeadDim) + "), " + std::to_string(m.n_head) + " / " + std::to_string(m.n_kv) + " heads";
        return false;
    }
    if (m.rank <= 0 || m.top_k <= 0 || m.top_k > kTopKMax || m.taps <= 0 || m.group <= 0 || m.E % m.group != 0 ||
        G.find("selector_hidden.weight") == nullptr) {
        err = "glm dflash: " + path + " is not a DFlash2 draft (no candidate selector / grouped convolution) - only "
              "DFlash2 drafts are supported";
        return false;
    }
    if (m.rank > 1024 || m.rank % 32 != 0) {
        err = "glm dflash: selector rank " + std::to_string(m.rank) + " (1..1024, a multiple of 32)";
        return false;
    }
    if (m.block < 2 || m.block > gf::kMaxMvCols) {
        err = "glm dflash: block size " + std::to_string(m.block) + " (2.." + std::to_string(gf::kMaxMvCols) + ")";
        return false;
    }
    // the context a draft reads: the sliding window (every layer of these drafts slides); without one, the most
    // recent STRATA_GLM_DFLASH_CTX positions (4096)
    if (m.window <= 0) {
        m.window = 4096;
        if (const char* c = getenv("STRATA_GLM_DFLASH_CTX")) m.window = std::max(64, std::atoi(c));
    }
    m.ring = m.window + 2 * gf::kMaxMvCols;
    m.cdyn = 2 * m.taps * (m.E / m.group);
    // ---- the weights: every tensor raw into one arena
    std::vector<const strata::TensorInfo*> ts;
    size_t total = 0;
    const auto al = [](size_t b) { return (b + 255u) & ~(size_t) 255u; };
    const auto tbytes = [](const strata::TensorInfo& t) -> size_t {
        int el = 0, by = 0;
        if (!strata::block_geometry(t.type, el, by)) return 0;
        return (size_t) (t.elements() / (uint64_t) el) * (size_t) by;
    };
    for (const auto& t : G.tensors()) {
        ts.push_back(&t);
        total += al(tbytes(t));
    }
    if (!ck(cudaSetDevice(dev), "set device", err)) return false;
    if (!ck(cudaMalloc(&m.arena, total), "weights", err)) return false;
    m.arena_bytes = total;
    std::map<std::string, std::pair<const void*, const strata::TensorInfo*>> at;
    {
        uint8_t* p = (uint8_t*) m.arena;
        for (const auto* t : ts) {
            const size_t b = tbytes(*t);
            if (b == 0) {
                err = "glm dflash: tensor " + t->name + " has type " + t->type_name() + " (unsupported)";
                return false;
            }
            if (!ck(cudaMemcpy(p, G.tensor_data(*t), b, cudaMemcpyHostToDevice), "weights upload", err)) return false;
            at[t->name] = {p, t};
            p += al(b);
        }
    }
    const auto f32 = [&](const std::string& n, int64_t want, const float*& out) -> bool {
        const auto it = at.find(n);
        if (it == at.end() || it->second.second->type != 0 || (want > 0 && (int64_t) it->second.second->elements() != want)) {
            err = "glm dflash: " + path + ": tensor " + n + " missing or not F32 of " + std::to_string(want);
            return false;
        }
        out = (const float*) it->second.first;
        return true;
    };
    const auto mat = [&](const std::string& n, int n_in, int n_out, Impl::Mat& out) -> bool {
        const auto it = at.find(n);
        if (it == at.end()) {
            err = "glm dflash: " + path + ": tensor " + n + " missing";
            return false;
        }
        const auto& t = *it->second.second;
        if (t.shape.size() != 2 || (int) t.shape[0] != n_in || (int) t.shape[1] != n_out) {
            err = "glm dflash: tensor " + n + " is not " + std::to_string(n_in) + " x " + std::to_string(n_out);
            return false;
        }
        if (!gf::mv_supported((int) t.type)) {
            err = "glm dflash: tensor " + n + " has type " + t.type_name() + " (the GEMV does not serve it)";
            return false;
        }
        out.w = it->second.first;
        out.type = (int) t.type;
        out.n_in = n_in;
        out.n_out = n_out;
        return true;
    };
    const int qd = m.n_head * kHeadDim, kvd = m.n_kv * kHeadDim, cdyn = m.cdyn;
    if (!mat("fc.weight", m.K * m.E, m.E, m.fc) || !f32("enc.output_norm.weight", m.E, m.enc_norm) ||
        !f32("output_norm.weight", m.E, m.out_norm) || !mat("selector_hidden.weight", m.E, m.rank, m.sel_hidden))
        return false;
    {
        const auto ip = at.find("selector_predecessor.weight"), in = at.find("selector_successor.weight");
        if (ip == at.end() || in == at.end() || ip->second.second->type != in->second.second->type ||
            ip->second.second->shape.size() != 2 || (int) ip->second.second->shape[0] != m.rank) {
            err = "glm dflash: " + path + ": selector_predecessor / selector_successor missing or mismatched";
            return false;
        }
        m.sel_type = (int) ip->second.second->type;
        const int st = m.sel_type;
        const bool ok = st == 0 || st == 1 || st == 30 || (st == 8 && m.rank % 32 == 0) ||
                        ((st == 12 || st == 14) && m.rank % 256 == 0);
        if (!ok) {
            err = std::string("glm dflash: selector tables in ") + ip->second.second->type_name() +
                  " (F32 / F16 / BF16 / Q8_0 / Q4_K / Q6_K are read)";
            return false;
        }
        m.sel_prev = (const uint8_t*) ip->second.first;
        m.sel_next = (const uint8_t*) in->second.first;
        m.V = (int) ip->second.second->shape[1];
    }
    m.L.resize((size_t) m.n_layer);
    for (int il = 0; il < m.n_layer; ++il) {
        auto& Ly = m.L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        if (!f32(P + "attn_norm.weight", m.E, Ly.attn_norm) || !f32(P + "ffn_norm.weight", m.E, Ly.ffn_norm) ||
            !f32(P + "attn_q_norm.weight", kHeadDim, Ly.q_norm) || !f32(P + "attn_k_norm.weight", kHeadDim, Ly.k_norm) ||
            !f32(P + "attn_conv_base", (int64_t) 2 * m.taps * m.E, Ly.attn_base) ||
            !f32(P + "ffn_conv_base", (int64_t) 2 * m.taps * m.E, Ly.ffn_base) ||
            !mat(P + "attn_q.weight", m.E, qd, Ly.q) || !mat(P + "attn_k.weight", m.E, kvd, Ly.k) ||
            !mat(P + "attn_v.weight", m.E, kvd, Ly.v) || !mat(P + "attn_output.weight", qd, m.E, Ly.o) ||
            !mat(P + "attn_conv_proj.weight", m.E, cdyn, Ly.attn_conv) ||
            !mat(P + "ffn_conv_proj.weight", m.E, cdyn, Ly.ffn_conv) || !mat(P + "ffn_gate.weight", m.E, m.n_ff, Ly.gate) ||
            !mat(P + "ffn_up.weight", m.E, m.n_ff, Ly.up) || !mat(P + "ffn_down.weight", m.n_ff, m.E, Ly.down))
            return false;
    }
    // ---- the ring
    const size_t ring_b = (size_t) m.n_layer * m.ring * kvd * sizeof(__half);
    if (!ck(cudaMalloc((void**) &m.Kr, ring_b), "the K ring", err) || !ck(cudaMalloc((void**) &m.Vr, ring_b), "the V ring", err) ||
        !ck(cudaMalloc((void**) &m.kv_pos, (size_t) m.ring * sizeof(long long)), "the ring's positions", err))
        return false;
    // ---- the buffers (the block's rows: up to block; the injection's: kInjRows)
    const int B = m.block, R = kInjRows, KE = m.K * m.E;
    const size_t f4 = sizeof(float);
    const size_t need = al((size_t) R * KE * f4) + al((size_t) R * m.q8(KE)) + 2 * al((size_t) R * m.E * f4) +
                        al((size_t) R * m.q8(m.E)) + 2 * al((size_t) R * kvd * f4) +
                        4 * al((size_t) B * m.E * f4) + al((size_t) B * cdyn * f4) + al((size_t) B * qd * f4) +
                        2 * al((size_t) B * kvd * f4) + al((size_t) B * qd * f4) +
                        3 * al((size_t) B * m.n_ff * f4) + al((size_t) B * (size_t) m.V * f4) +
                        al((size_t) B * m.rank * f4) + al((size_t) B * m.q8(std::max(m.E, qd))) +
                        al((size_t) B * m.q8(qd)) + al((size_t) B * m.q8(m.n_ff)) + al((size_t) B * m.top_k * f4) +
                        al((size_t) B * m.top_k * sizeof(int)) + al((size_t) B * kTopKSlices * m.top_k * f4) +
                        al((size_t) B * kTopKSlices * m.top_k * sizeof(int)) + al((size_t) B * sizeof(int)) + al((size_t) B * f4) +
                        al((size_t) kAttnSplits * m.n_kv * kAttnQ * kAttnPart * f4);
    if (!ck(cudaMalloc(&m.buf, need), "buffers", err)) return false;
    m.buf_bytes = need;
    uint8_t* p = (uint8_t*) m.buf;
    const auto tf = [&](size_t n) {
        float* r = (float*) p;
        p += al(n * f4);
        return r;
    };
    const auto tb = [&](size_t n) {
        void* r = p;
        p += al(n);
        return r;
    };
    m.feat = tf((size_t) R * KE);
    m.fq = tb((size_t) R * m.q8(KE));
    m.g = tf((size_t) R * m.E);
    m.gn = tf((size_t) R * m.E);
    m.gq = tb((size_t) R * m.q8(m.E));
    m.kc = tf((size_t) R * kvd);
    m.vc = tf((size_t) R * kvd);
    m.h = tf((size_t) B * m.E);
    m.x = tf((size_t) B * m.E);
    m.xc = tf((size_t) B * m.E);
    m.o = tf((size_t) B * m.E);
    m.cbuf = tf((size_t) B * cdyn);
    m.q = tf((size_t) B * qd);
    m.kb = tf((size_t) B * kvd);
    m.vb = tf((size_t) B * kvd);
    m.a = tf((size_t) B * qd);
    m.gt = tf((size_t) B * m.n_ff);
    m.up = tf((size_t) B * m.n_ff);
    m.act = tf((size_t) B * m.n_ff);
    m.logits = tf((size_t) B * (size_t) m.V);
    m.sg = tf((size_t) B * m.rank);
    m.xq = tb((size_t) B * m.q8(std::max(m.E, qd)));
    m.aq = tb((size_t) B * m.q8(qd));
    m.actq = tb((size_t) B * m.q8(m.n_ff));
    m.sel_lg = tf((size_t) B * m.top_k);
    m.sel_id = (int*) tb((size_t) B * m.top_k * sizeof(int));
    m.part_lg = tf((size_t) B * kTopKSlices * m.top_k);
    m.part_id = (int*) tb((size_t) B * kTopKSlices * m.top_k * sizeof(int));
    m.tok = (int*) tb((size_t) B * sizeof(int));
    m.conf = tf((size_t) B);
    m.attn_part = tf((size_t) kAttnSplits * m.n_kv * kAttnQ * kAttnPart);
    if ((size_t) (p - (uint8_t*) m.buf) > need) {
        err = "glm dflash: the buffers overflowed their estimate";
        return false;
    }
    if (!ck(cudaHostAlloc((void**) &m.stage, (size_t) m.ring * KE * f4, cudaHostAllocPortable), "the staging", err) ||
        !ck(cudaHostAlloc((void**) &m.tok_h, (size_t) B * sizeof(int), cudaHostAllocPortable), "pinned", err) ||
        !ck(cudaHostAlloc((void**) &m.conf_h, (size_t) B * f4, cudaHostAllocPortable), "pinned", err) ||
        !ck(cudaHostAlloc((void**) &m.emb_h, (size_t) B * m.E * f4, cudaHostAllocPortable), "pinned", err))
        return false;
    m.kv_pos_h.assign((size_t) m.ring, -1);
    m.st_pos.assign((size_t) m.ring, -1);
    m.st_mask.assign((size_t) m.ring, 0u);
    if (!ck(cudaMemset(m.kv_pos, 0xff, (size_t) m.ring * sizeof(long long)), "ring reset", err)) return false;
    return true;
}

bool GlmDflash::bind(int target_embd, int target_vocab, int target_layers, std::string& err) {
    Impl& m = *d_;
    if (target_embd != m.E) {
        err = "glm dflash: the draft's hidden size " + std::to_string(m.E) + " is not the model's " +
              std::to_string(target_embd);
        return false;
    }
    if (target_vocab != m.V || m.mask_tok >= target_vocab) {
        err = "glm dflash: the draft's vocabulary (" + std::to_string(m.V) + ") is not the model's (" +
              std::to_string(target_vocab) + ")";
        return false;
    }
    for (int l : m.tl)
        if (l < 0 || l >= target_layers) {
            err = "glm dflash: the draft reads layer " + std::to_string(l) + " of a " + std::to_string(target_layers) +
                  "-layer model";
            return false;
        }
    return true;
}

int GlmDflash::ring_size() const { return d_->ring; }
size_t GlmDflash::stage_pitch() const { return (size_t) d_->K * d_->E * sizeof(float); }

float* GlmDflash::stage_row(int64_t p, int k) const {
    const Impl& m = *d_;
    return m.stage + ((size_t) (p % m.ring) * m.K + (size_t) k) * m.E;
}

void GlmDflash::staged(int64_t p, int k) {
    Impl& m = *d_;
    std::lock_guard<std::mutex> lk(m.mu);
    const size_t s = (size_t) (p % m.ring);
    if (m.st_pos[s] != p) {
        m.st_pos[s] = p;
        m.st_mask[s] = 0u;
    }
    m.st_mask[s] |= 1u << k;
}

void GlmDflash::reset(int64_t from) {
    Impl& m = *d_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        for (size_t s = 0; s < m.st_pos.size(); ++s)
            if (m.st_pos[s] >= from) {
                m.st_pos[s] = -1;
                m.st_mask[s] = 0u;
            }
    }
    // (inject() skips a position whose slot already holds it: a stale one would stay for good)
    for (int64_t& p : m.kv_pos_h)
        if (p >= from) p = -1;
    static_assert(sizeof(int64_t) == sizeof(long long), "kv_pos_h mirrors kv_pos");
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(m.dev);
    cudaDeviceSynchronize();
    cudaMemcpy(m.kv_pos, m.kv_pos_h.data(), (size_t) m.ring * sizeof(long long), cudaMemcpyHostToDevice);
    // waited for: a copy from pageable memory can return before it landed, and the drafter's own stream is not
    // ordered after the legacy one
    cudaDeviceSynchronize();
    cudaSetDevice(cur);
}

// The float rows a GEMV of these weights reads besides the q8_1 ones: F32 / BF16 weights read the floats (gf::MvJob's
// xf), the quantized types only the q8_1 (xf stays nullptr there).  The drafter's GEMVs gave none, so a BF16 drafter
// (the published GLM-5.3-Flash-DFlash2-bf16.gguf) read from address 0 at its first draft (issue #173).
static const float* xf_for(int type, const float* x) {
    return type == gf::kTypeF32 || type == gf::kTypeBF16 ? x : nullptr;
}

// mv over `cols` columns in pieces of kMaxMvCols
static bool mv_cols(gf::MvJob j, int cols, size_t xq_s, size_t xf_s, size_t y_s, cudaStream_t s) {
    for (int c0 = 0; c0 < cols; c0 += gf::kMaxMvCols) {
        gf::MvJob jj = j;
        jj.approx = true;   // the drafter's ring: float rounding may differ (tensor-core GEMVs)
        jj.ncol = std::min(gf::kMaxMvCols, cols - c0);
        jj.xq_cstride = (int64_t) xq_s;
        jj.xf_cstride = j.xf ? (int64_t) xf_s : 0;
        jj.y_cstride = (int64_t) y_s;
        if (jj.xq) jj.xq = (const uint8_t*) j.xq + (size_t) c0 * xq_s;
        if (jj.xf) jj.xf = j.xf + (size_t) c0 * xf_s;
        jj.y = j.y + (size_t) c0 * y_s;
        if (!gf::mv(&jj, 1, s)) return false;
    }
    return true;
}

bool GlmDflash::inject(int64_t lo, int64_t hi, cudaStream_t s, std::string& err) {
    Impl& m = *d_;
    if (hi <= lo) return true;
    lo = std::max<int64_t>(lo, hi - m.window);
    const uint32_t full = m.K >= 32 ? 0xffffffffu : (1u << m.K) - 1u;
    const int KE = m.K * m.E, kvd = m.n_kv * kHeadDim;
    cudaSetDevice(m.dev);
    // on a stream of its own, after the drafter's: the next draft's first layer runs beside it, each layer's attention
    // waiting for that layer's context rows (ev_kv).  STRATA_GLM_DFLASH_INJ_OVERLAP=0: on `s` (A/B)
    static const bool ovl = [] {
        const char* v = getenv("STRATA_GLM_DFLASH_INJ_OVERLAP");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (ovl && m.is == nullptr) {
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) greatest = 0;
        bool ok = cudaStreamCreateWithPriority(&m.is, cudaStreamNonBlocking, greatest) == cudaSuccess &&
                  cudaEventCreateWithFlags(&m.ev_s, cudaEventDisableTiming) == cudaSuccess;
        m.ev_kv.assign((size_t) m.n_layer, nullptr);
        for (auto& e : m.ev_kv) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            m.is = nullptr;
        }
    }
    const cudaStream_t ws = s;
    if (ovl && m.is != nullptr) {
        cudaEventRecord(m.ev_s, ws);
        cudaStreamWaitEvent(m.is, m.ev_s, 0);
        s = m.is;
    }
    // the runs of consecutive positions to take: staged in full, not in the ring yet
    std::vector<std::pair<int64_t, int64_t>> runs;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        for (int64_t p = lo; p < hi; ++p) {
            const size_t sl = (size_t) (p % m.ring);
            const bool take = m.st_pos[sl] == p && m.st_mask[sl] == full && m.kv_pos_h[sl] != p;
            if (!take) continue;
            if (!runs.empty() && runs.back().second == p && runs.back().second - runs.back().first < kInjRows)
                ++runs.back().second;
            else
                runs.push_back({p, p + 1});
        }
    }
    for (const auto& r : runs) {
        const int64_t p0 = r.first;
        const int n = (int) (r.second - r.first);
        // the staged rows (each position's K layers contiguous) -> feat; a run may wrap around the ring
        for (int i = 0; i < n;) {
            const size_t sl = (size_t) ((p0 + i) % m.ring);
            const int len = (int) std::min<int64_t>(n - i, (int64_t) m.ring - (int64_t) sl);
            if (!ck(cudaMemcpyAsync(m.feat + (size_t) i * KE, m.stage + sl * KE, (size_t) len * KE * sizeof(float),
                                    cudaMemcpyHostToDevice, s),
                    "staging upload", err))
                return false;
            i += len;
        }
        // (the positions first: a draft's attention of a layer reads them once that layer's rows are in)
        kv_pos_kernel<<<(n + 127) / 128, 128, 0, s>>>(p0, n, m.ring, m.kv_pos);
        gf::quantize_q8_1(m.feat, m.fq, n * KE, s);
        if (!mv_cols({m.fc.w, m.fq, xf_for(m.fc.type, m.feat), m.g, nullptr, 1.0f, m.fc.type, KE, m.E}, n, m.q8(KE),
                     KE, m.E, s)) {
            err = "glm dflash: fc";
            return false;
        }
        rms_rows_kernel<<<n, 256, 0, s>>>(m.g, m.E, m.enc_norm, m.eps, m.E, m.gn, m.E);
        gf::quantize_q8_1(m.gn, m.gq, n * m.E, s);
        for (int il = 0; il < m.n_layer; ++il) {
            const auto& Ly = m.L[(size_t) il];
            if (!mv_cols({Ly.k.w, m.gq, xf_for(Ly.k.type, m.gn), m.kc, nullptr, 1.0f, Ly.k.type, m.E, kvd}, n,
                         m.q8(m.E), m.E, kvd, s) ||
                !mv_cols({Ly.v.w, m.gq, xf_for(Ly.v.type, m.gn), m.vc, nullptr, 1.0f, Ly.v.type, m.E, kvd}, n,
                         m.q8(m.E), m.E, kvd, s)) {
                err = "glm dflash: context K / V";
                return false;
            }
            head_norm_rope_kernel<<<dim3(m.n_kv, n), kHeadDim / 2, 0, s>>>(m.kc, m.n_kv, Ly.k_norm, m.eps, p0,
                                                                           m.rope_base);
            const size_t lay = (size_t) il * m.ring * kvd;
            kv_store_kernel<<<n, 256, 0, s>>>(m.kc, m.vc, kvd, p0, m.ring, m.Kr + lay, m.Vr + lay);
            if (s != ws) cudaEventRecord(m.ev_kv[(size_t) il], s);
        }
        for (int i = 0; i < n; ++i) m.kv_pos_h[(size_t) ((p0 + i) % m.ring)] = p0 + i;
    }
    if (s != ws && !runs.empty()) m.inj_pending = true;
    if (!ck(cudaGetLastError(), "inject", err)) return false;
    return true;
}

bool GlmDflash::draft_start(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head,
                            const uint8_t* emb, int emb_type, cudaStream_t s, std::string& err) {
    Impl& m = *d_;
    m.pend_n = 0;
    n_max = std::min(n_max, m.block - 1);
    if (n_max < 1) return true;
    const int B = n_max + 1, E = m.E, qd = m.n_head * kHeadDim, kvd = m.n_kv * kHeadDim, cd = m.cdyn;
    cudaSetDevice(m.dev);
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) emb_type);
    if (tt == nullptr || tt->to_float == nullptr || emb == nullptr) {
        err = "glm dflash: no embedding dequantizer";
        return false;
    }
    const size_t row_b = ggml_row_size((ggml_type) emb_type, E);
    if (m.mask_row.empty()) {
        m.mask_row.resize((size_t) E);
        tt->to_float(emb + (size_t) m.mask_tok * row_b, m.mask_row.data(), E);
    }
    // emb_h may still feed the previous draft's copy: wait for that copy alone (the whole stream would wait for the
    // injection queued just before - the host then enqueued the draft ~0.3 ms late each round)
    if (m.emb_ev == nullptr) cudaEventCreateWithFlags(&m.emb_ev, cudaEventDisableTiming);
    cudaEventSynchronize(m.emb_ev);
    tt->to_float(emb + (size_t) anchor * row_b, m.emb_h, E);
    for (int t = 1; t < B; ++t) std::memcpy(m.emb_h + (size_t) t * E, m.mask_row.data(), (size_t) E * sizeof(float));
    cudaMemcpyAsync(m.h, m.emb_h, (size_t) B * E * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaEventRecord(m.emb_ev, s);
    // (x: the float rows xq was quantized from, w.n_in apart - what F32 / BF16 weights read: xf_for)
    const auto job = [](const Impl::Mat& w, const void* xq, const float* x, float* y) {
        gf::MvJob j{w.w, xq, xf_for(w.type, x), y, nullptr, 1.0f, w.type, w.n_in, w.n_out};
        j.approx = true;   // drafts are proposals: float rounding may differ (tensor-core GEMVs)
        return j;
    };
    const auto cols = [&](gf::MvJob j, int n, size_t xs, size_t ys) {
        j.ncol = n;
        j.xq_cstride = (int64_t) xs;
        if (j.xf) j.xf_cstride = j.n_in;
        j.y_cstride = (int64_t) ys;
        return j;
    };
    const float scale = 1.0f / std::sqrt((float) kHeadDim);
    const int64_t lo0 = std::max<int64_t>(0, n_past - m.window + 1);
    const int nk_max = (int) (n_past - lo0) + B;
    const size_t smem = (size_t) nk_max * sizeof(float);
    // the split attention when a kv head's queries fit one block (STRATA_GLM_DFLASH_ATTN=0: attn_kernel, for A/B);
    // the keys in splits of whole chunks, about kAttnSplits of them
    static const bool split_opt = [] {
        const char* v = getenv("STRATA_GLM_DFLASH_ATTN");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool split = split_opt && (m.n_head / m.n_kv) * B <= kAttnQ;
    const int chunks = (nk_max + kAttnKeys - 1) / kAttnKeys;
    const int per_split = (chunks + kAttnSplits - 1) / kAttnSplits * kAttnKeys;
    const int splits = (nk_max + per_split - 1) / per_split;
    if (!split && smem > 48 * 1024) {
        static bool once = false;
        if (!once) {
            once = true;
            cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        }
        if (smem > 96 * 1024) {
            err = "glm dflash: a window of " + std::to_string(nk_max) + " keys";
            return false;
        }
    }
    for (int il = 0; il < m.n_layer; ++il) {
        const auto& Ly = m.L[(size_t) il];
        // attention
        rms_rows_kernel<<<B, 256, 0, s>>>(m.h, E, Ly.attn_norm, m.eps, E, m.x, E);
        if (il == 0) dump("h0", m.h, (size_t) B * E, s), dump("x0", m.x, (size_t) B * E, s);
        gf::quantize_q8_1(m.x, m.xq, B * E, s);
        gf::MvJob jc = cols(job(Ly.attn_conv, m.xq, m.x, m.cbuf), B, m.q8(E), cd);
        if (!gf::mv(&jc, 1, s)) { err = "glm dflash: attn conv proj"; return false; }
        conv_kernel<<<B, 256, 0, s>>>(m.x, m.cbuf, cd, Ly.attn_base, 0, E, m.taps, m.group, m.xc);
        if (il == 0) dump("c0", m.cbuf, (size_t) B * cd, s), dump("xc0", m.xc, (size_t) B * E, s);
        gf::quantize_q8_1(m.xc, m.xq, B * E, s);
        gf::MvJob j3[3] = {cols(job(Ly.q, m.xq, m.xc, m.q), B, m.q8(E), qd),
                           cols(job(Ly.k, m.xq, m.xc, m.kb), B, m.q8(E), kvd),
                           cols(job(Ly.v, m.xq, m.xc, m.vb), B, m.q8(E), kvd)};
        if (!gf::mv(j3, 3, s)) { err = "glm dflash: q / k / v"; return false; }
        head_norm_rope_kernel<<<dim3(m.n_head, B), kHeadDim / 2, 0, s>>>(m.q, m.n_head, Ly.q_norm, m.eps, n_past,
                                                                          m.rope_base);
        head_norm_rope_kernel<<<dim3(m.n_kv, B), kHeadDim / 2, 0, s>>>(m.kb, m.n_kv, Ly.k_norm, m.eps, n_past,
                                                                        m.rope_base);
        if (il == 0) dump("q0", m.q, (size_t) B * qd, s), dump("k0", m.kb, (size_t) B * kvd, s), dump("v0", m.vb, (size_t) B * kvd, s);
        const size_t lay = (size_t) il * m.ring * kvd;
        if (m.inj_pending) cudaStreamWaitEvent(s, m.ev_kv[(size_t) il], 0);   // (this layer's context rows: inject)
        if (split) {
            attn_split_kernel<<<dim3(m.n_kv, splits), 256, 0, s>>>(m.q, m.Kr + lay, m.Vr + lay, m.kv_pos, m.kb, m.vb,
                                                                  B, m.n_head, m.n_kv, n_past, m.ring, m.window, scale,
                                                                  per_split, m.attn_part);
            attn_merge_kernel<<<dim3(m.n_head, B), kHeadDim, 0, s>>>(m.attn_part, splits, B, m.n_head, m.n_kv, m.a);
        } else {
            attn_kernel<<<dim3(m.n_head, B), kHeadDim, smem, s>>>(m.q, m.Kr + lay, m.Vr + lay, m.kv_pos, m.kb, m.vb,
                                                                  B, m.n_head, m.n_kv, n_past, m.ring, m.window, scale,
                                                                  m.a);
        }
        if (il == 0) dump("a0", m.a, (size_t) B * qd, s);
        gf::quantize_q8_1(m.a, m.aq, B * qd, s);
        gf::MvJob jo = cols(job(Ly.o, m.aq, m.a, m.o), B, m.q8(qd), E);
        if (!gf::mv(&jo, 1, s)) { err = "glm dflash: attn output"; return false; }
        conv_kernel<<<B, 256, 0, s>>>(m.o, m.cbuf, cd, Ly.attn_base, 1, E, m.taps, m.group, m.xc);
        if (il == 0) dump("o0", m.o, (size_t) B * E, s), dump("oc0", m.xc, (size_t) B * E, s);
        add_kernel<<<(B * E + 255) / 256, 256, 0, s>>>(m.h, m.xc, B * E);
        // feed-forward
        rms_rows_kernel<<<B, 256, 0, s>>>(m.h, E, Ly.ffn_norm, m.eps, E, m.x, E);
        gf::quantize_q8_1(m.x, m.xq, B * E, s);
        gf::MvJob jf = cols(job(Ly.ffn_conv, m.xq, m.x, m.cbuf), B, m.q8(E), cd);
        if (!gf::mv(&jf, 1, s)) { err = "glm dflash: ffn conv proj"; return false; }
        conv_kernel<<<B, 256, 0, s>>>(m.x, m.cbuf, cd, Ly.ffn_base, 0, E, m.taps, m.group, m.xc);
        gf::quantize_q8_1(m.xc, m.xq, B * E, s);
        gf::MvJob j2[2] = {cols(job(Ly.gate, m.xq, m.xc, m.gt), B, m.q8(E), m.n_ff),
                           cols(job(Ly.up, m.xq, m.xc, m.up), B, m.q8(E), m.n_ff)};
        if (!gf::mv(j2, 2, s)) { err = "glm dflash: ffn gate / up"; return false; }
        silu_mul_kernel<<<(B * m.n_ff + 255) / 256, 256, 0, s>>>(m.gt, m.up, B * m.n_ff, m.act);
        gf::quantize_q8_1(m.act, m.actq, B * m.n_ff, s);
        gf::MvJob jd = cols(job(Ly.down, m.actq, m.act, m.o), B, m.q8(m.n_ff), E);
        if (!gf::mv(&jd, 1, s)) { err = "glm dflash: ffn down"; return false; }
        conv_kernel<<<B, 256, 0, s>>>(m.o, m.cbuf, cd, Ly.ffn_base, 1, E, m.taps, m.group, m.xc);
        add_kernel<<<(B * E + 255) / 256, 256, 0, s>>>(m.h, m.xc, B * E);
        if (il == 0) dump("l0", m.h, (size_t) B * E, s);
    }
    m.inj_pending = false;
    // the mask rows' final norm, the target's head, the selector's gate, the candidates and the walk
    const int n = B - 1;
    rms_rows_kernel<<<n, 256, 0, s>>>(m.h + E, E, m.out_norm, m.eps, E, m.x, E);
    gf::quantize_q8_1(m.x, m.xq, n * E, s);
    // the head split (split_head): the helper's columns there while this half runs here, each side its slices' top k
    Impl::HeadSplit& hs = m.hs;
    const int slice = (m.V + kTopKSlices - 1) / kTopKSlices;
    const int s_lo = hs.lo != nullptr ? hs.c0 / slice : 0, s_hi = kTopKSlices - s_lo;
    const size_t cand_n = (size_t) n * s_hi * m.top_k;
    const bool hsplit = hs.lo != nullptr && n <= hs.rows && hs.c0 % slice == 0 && s_lo > 0 && s_lo < kTopKSlices &&
                       cand_n * (sizeof(int) + sizeof(float)) <= hs.cand_bytes;
    gf::MvJob jh{hsplit ? hs.lo->q : head.q, m.xq, m.x, m.logits, nullptr, 1.0f, head.type, E, hsplit ? hs.c0 : m.V};
    if (head.type == 0) jh.w = head.f32;
    jh.ncol = n;
    m.head_type = head.type;
    jh.xq_cstride = (int64_t) m.q8(E);
    jh.xf_cstride = E;
    jh.y_cstride = m.V;
    // the head's several columns on the tensor cores (MvJob::approx) or the dp4a kernel, whichever this device runs
    // faster: the first drafts of a full block alternate the two and time the head with events (read after the
    // draft's own sync), the median of kHeadSamples each decides (a Q8_0 head at 7 columns: RTX 3090 2.3 against
    // 1.1 ms; RTX 5060 Ti 1.6 against 1.7, where dp4a already reads at the memory's speed).
    // STRATA_GLM_DFLASH_HEAD_MMA=0/1 forces either
    bool timed = false;
    if (n >= 5) {
        if (m.head_mma_w != jh.w) {   // a new head: decide again
            m.head_mma_w = jh.w;
            m.head_mma = -1;
            m.head_t[0].clear();
            m.head_t[1].clear();
            const char* v = getenv("STRATA_GLM_DFLASH_HEAD_MMA");
            if (v != nullptr && v[0]) m.head_mma = std::atoi(v) != 0 ? 1 : 0;
        }
        if (m.head_mma < 0 && n == m.block - 1) {
            if (m.head_ev[0] == nullptr) {
                cudaEventCreate(&m.head_ev[0]);
                cudaEventCreate(&m.head_ev[1]);
            }
            m.head_try = (int) ((m.head_t[0].size() + m.head_t[1].size()) % 2);
            jh.approx = m.head_try == 1;
            timed = true;
        } else if (m.head_mma >= 0) {
            jh.approx = m.head_mma == 1;
        }
    }
    bool hi_ok = true;
    if (hsplit) {
        const int nc = m.V - hs.c0;
        cudaMemcpyAsync(hs.x_h, m.x, (size_t) n * E * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaEventRecord(hs.ev_x, s);
        cudaSetDevice(hs.dev);
        cudaStreamWaitEvent(hs.s, hs.ev_x, 0);
        cudaMemcpyAsync(hs.x, hs.x_h, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice, hs.s);
        gf::MvJob jx{hs.hi->q, nullptr, hs.x, hs.lg, nullptr, 1.0f, hs.hi->type, E, nc};
        jx.ncol = n;
        jx.approx = jh.approx;
        jx.xf_cstride = E;
        jx.y_cstride = nc;
        hi_ok = gf::mv(&jx, 1, hs.s);
        int* cid = (int*) hs.cand;
        float* cv = (float*) (cid + cand_n);
        topk_part_kernel<<<dim3(s_hi, n), 256, (size_t) slice * sizeof(float), hs.s>>>(hs.lg, nc, m.top_k, slice, cid, cv,
                                                                                       s_hi, hs.c0);
        cudaMemcpyAsync(hs.cand_h, hs.cand, cand_n * (sizeof(int) + sizeof(float)), cudaMemcpyDeviceToHost, hs.s);
        cudaEventRecord(hs.ev_c, hs.s);
        cudaSetDevice(m.dev);
    }
    if (timed) cudaEventRecord(m.head_ev[0], s);
    const bool head_ok = gf::mv(&jh, 1, s) && hi_ok;
    if (timed) cudaEventRecord(m.head_ev[1], s);
    m.head_timed = timed;
    gf::MvJob jg = cols(job(m.sel_hidden, m.xq, m.x, m.sg), n, m.q8(E), m.rank);
    if (!head_ok || !gf::mv(&jg, 1, s)) {
        err = "glm dflash: the head / the selector gate (head type " + std::to_string(head.type) + ")";
        return false;
    }
    topk_part_kernel<<<dim3(hsplit ? s_lo : kTopKSlices, n), 256, (size_t) slice * sizeof(float), s>>>(
        m.logits, m.V, m.top_k, slice, m.part_id, m.part_lg, kTopKSlices, 0);
    if (hsplit) {   // the helper's slices' candidates beside this half's (rows kTopKSlices x top_k apart)
        const int* cid = (const int*) hs.cand_h;
        const float* cv = (const float*) (cid + cand_n);
        const size_t w = (size_t) s_hi * m.top_k, pitch = (size_t) kTopKSlices * m.top_k, off = (size_t) s_lo * m.top_k;
        cudaStreamWaitEvent(s, hs.ev_c, 0);
        cudaMemcpy2DAsync(m.part_id + off, pitch * sizeof(int), cid, w * sizeof(int), w * sizeof(int), (size_t) n,
                          cudaMemcpyHostToDevice, s);
        cudaMemcpy2DAsync(m.part_lg + off, pitch * sizeof(float), cv, w * sizeof(float), w * sizeof(float), (size_t) n,
                          cudaMemcpyHostToDevice, s);
    }
    topk_merge_kernel<<<n, 32, (size_t) kTopKSlices * m.top_k * (sizeof(float) + sizeof(int)), s>>>(
        m.part_id, m.part_lg, m.top_k, m.sel_id, m.sel_lg);
    walk_kernel<<<1, m.rank, 0, s>>>(m.sel_id, m.sel_lg, m.sg, m.top_k, m.rank, m.sel_prev, m.sel_next, m.sel_type,
                                     anchor, n, m.tok, m.conf);
    cudaMemcpyAsync(m.tok_h, m.tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(m.conf_h, m.conf, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost, s);
    m.pend_s = s;
    m.pend_n = n;
    m.pend_p_min = p_min;
    return true;
}

bool GlmDflash::draft_finish(std::vector<int32_t>& out, std::string& err, std::vector<float>* conf, int n_min,
                             float p_cum) {
    Impl& m = *d_;
    out.clear();
    const int n = m.pend_n;
    m.pend_n = 0;
    if (n <= 0) return true;
    if (!ck(cudaStreamSynchronize(m.pend_s), "draft", err)) return false;
    if (m.head_timed) {
        m.head_timed = false;
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, m.head_ev[0], m.head_ev[1]) == cudaSuccess) m.head_t[m.head_try].push_back(ms);
        if ((int) m.head_t[0].size() >= kHeadSamples && (int) m.head_t[1].size() >= kHeadSamples) {
            const auto med = [](std::vector<float> v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            const float t0 = med(m.head_t[0]), t1 = med(m.head_t[1]);
            m.head_mma = t1 < 0.95f * t0 ? 1 : 0;
            std::fprintf(stderr, "glm dflash: the head (type %d, %d columns) %.2f ms on the dp4a kernel, %.2f ms on the "
                                 "tensor cores (medians of %d) -> %s\n", m.head_type, n, t0, t1, kHeadSamples,
                         m.head_mma ? "tensor cores" : "dp4a");
        }
    }
    if (conf != nullptr) conf->assign(m.conf_h, m.conf_h + n);
    float cum = 1.0f;
    for (int i = 0; i < n; ++i) {
        cum *= m.conf_h[i];
        if (i >= n_min && ((m.pend_p_min > 0.0f && m.conf_h[i] < m.pend_p_min) || (p_cum > 0.0f && cum < p_cum))) break;
        out.push_back(m.tok_h[i]);
    }
    return true;
}

bool GlmDflash::split_head(const WSlot* lo, const WSlot* hi, int c0, int hdev, cudaStream_t hs, float* hx, float* hlg,
                           void* hcand, size_t cand_bytes, int rows, std::string& err) {
    Impl& m = *d_;
    const char* v = getenv("STRATA_GLM_DFLASH_HEAD_SPLIT");
    if (lo == nullptr || hi == nullptr || hs == nullptr || hx == nullptr || hlg == nullptr || hcand == nullptr ||
        (v != nullptr && std::atoi(v) == 0))
        return true;
    Impl::HeadSplit& h = m.hs;
    cudaSetDevice(m.dev);
    if (cudaHostAlloc((void**) &h.x_h, (size_t) rows * m.E * sizeof(float), cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc(&h.cand_h, cand_bytes, cudaHostAllocPortable) != cudaSuccess ||
        cudaEventCreateWithFlags(&h.ev_x, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        err = "glm dflash: the head split's pinned rows did not allocate";
        return false;
    }
    cudaSetDevice(hdev);
    const bool ok = cudaEventCreateWithFlags(&h.ev_c, cudaEventDisableTiming) == cudaSuccess;
    cudaSetDevice(m.dev);
    if (!ok) {
        cudaGetLastError();
        err = "glm dflash: the head split's event";
        return false;
    }
    h.lo = lo;
    h.hi = hi;
    h.c0 = c0;
    h.dev = hdev;
    h.s = hs;
    h.x = hx;
    h.lg = hlg;
    h.cand = hcand;
    h.cand_bytes = cand_bytes;
    h.rows = rows;
    std::fprintf(stderr, "glm dflash: a draft's head reads columns [0, %d) on CUDA%d and the rest on CUDA%d at once\n", c0,
                 m.dev, hdev);
    return true;
}

bool GlmDflash::draft(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head, const uint8_t* emb,
                      int emb_type, cudaStream_t s, std::vector<int32_t>& out, std::string& err,
                      std::vector<float>* conf) {
    return draft_start(anchor, n_past, n_max, p_min, head, emb, emb_type, s, err) && draft_finish(out, err, conf);
}

int GlmDflash::top_k() const { return d_->top_k; }

bool GlmDflash::last_rows(int n, std::vector<float>& hidden, std::vector<int>& cand, std::vector<float>& logit,
                          std::vector<int>& pick, std::vector<float>& conf) const {
    const Impl& m = *d_;
    if (n < 1 || n >= m.block) return false;
    cudaSetDevice(m.dev);
    hidden.resize((size_t) n * m.E);
    cand.resize((size_t) n * m.top_k);
    logit.resize((size_t) n * m.top_k);
    pick.resize((size_t) n);
    conf.resize((size_t) n);
    return cudaMemcpy(hidden.data(), m.x, hidden.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(cand.data(), m.sel_id, cand.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(logit.data(), m.sel_lg, logit.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(pick.data(), m.tok, pick.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(conf.data(), m.conf, conf.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

namespace {
// decode_dflash's next-round draft queued from a thread of its own (kick), waited for at the top of the round (wait)
struct DraftAhead {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<bool(std::string&)> job;
    bool has = false, quit = false, done = true, ok = true;
    std::string err;
    void kick(std::function<bool(std::string&)> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!th.joinable()) th = std::thread([this] { loop(); });
            job = std::move(f);
            has = true;
            done = false;
        }
        cv.notify_all();
    }
    bool wait(std::string& e) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return done; });
        if (!ok) e = err;
        return ok;
    }
    void loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return quit || has; });
            if (!has) return;
            auto f = std::move(job);
            has = false;
            lk.unlock();
            std::string e;
            const bool r = f(e);
            lk.lock();
            ok = r;
            err = e;
            done = true;
            cv.notify_all();
        }
    }
    ~DraftAhead() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};
}  // namespace

// ---------------------------------------------------------------- the trunk's side (Glm5Model members)
bool Glm5Model::dflash_load(const std::string& path, int dev, std::string& err) {
    dflash_.reset(new GlmDflash());
    if (!dflash_->load(path, dev, err)) {
        dflash_.reset();
        return false;
    }
    return true;
}

bool Glm5Model::dflash_bind(std::string& err) {
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    GlmDflash* D = tail->dflash_.get();
    if (D == nullptr) {
        // asked for and not loaded (the slow path already said why it drafts nothing): an error, not a quiet decode
        // without drafting
        if (!dflash_path_.empty() && fast_ != nullptr) {
            err = "glm dflash: " + dflash_path_ + " was not loaded on the part with the head";
            return false;
        }
        return true;
    }
    if (fast_ == nullptr || tail->fast_ == nullptr) {
        err = "glm dflash: needs the fast path";
        return false;
    }
    // one drafter at a time: the NextN block's decode and its states would run beside the DFlash windows
    // (load_pack_env sets STRATA_GLM_NO_MTP, so the block is never loaded with a DFlash drafter)
    if (has_mtp()) {
        err = "glm dflash: the NextN (MTP) draft block is loaded too - DFlash replaces it (MAYA_GLM_NO_MTP=1)";
        return false;
    }
    if (tail->dev_ != D->dev()) {
        err = "glm dflash: the draft sits on CUDA" + std::to_string(D->dev()) + ", the head on CUDA" +
              std::to_string(tail->dev_);
        return false;
    }
    if (!D->bind(g_.n_embd, g_.n_vocab, g_.n_layers, err)) return false;
    const auto& tl = D->target_layers();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        m->cap_k_.assign((size_t) g_.n_layers + 1, -1);
        for (size_t k = 0; k < tl.size(); ++k)
            if (tl[k] >= m->l0_ && tl[k] < m->l1_) m->cap_k_[(size_t) tl[k]] = (int) k;
        m->cap_d_ = D;
        cudaSetDevice(m->dev_);
        if (m->fast_->cap == nullptr) {
            m->fast_->cap_rows = kWinMax;
            if (cudaMalloc((void**) &m->fast_->cap, (size_t) kWinMax * g_.n_embd * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                err = "glm dflash: the capture rows did not allocate";
                return false;
            }
        }
    }
    // the drafts' head on the head split too: its helper idles while the drafter runs (the round's anchor is on the
    // first parts then)
    if (tail->head_split_ready()) {
        Glm5Model* h = tail->head_helper_;
        const auto lo = tail->ws_map_.find("output.weight.lo");
        const auto hi = h->ws_map_.find("output.weight.hi");
        if (lo != tail->ws_map_.end() && hi != h->ws_map_.end() &&
            !D->split_head(&lo->second, &hi->second, tail->head_c0_, h->dev_, h->hd_s_, h->hd_x_, h->hd_lg_,
                           h->hd_cand_, kHdCandBytes, kWinMax, err))
            return false;
        cudaSetDevice(dev_);
    }
    // a two-GPU split's batched window (dflash_pipelined): each part's window layer on both parts' CPU pools - the
    // other part's pool idles while the window is on this one (STRATA_GLM_WIN_LANES=0: its own only)
    const bool win_pipe = dflash_pipelined(2);
    static const bool win_lanes = [] {
        const char* v = getenv("STRATA_GLM_WIN_LANES");
        return v == nullptr || std::atoi(v) != 0;
    }();
    Glm5Model* pb = split_next_.get();
    if (!win_pipe && win_lanes && pb != nullptr && pb->split_next_ == nullptr && fast_->cpu_pool && pb->fast_->cpu_pool &&
        fast_->cpu_pool != pb->fast_->cpu_pool && fast_->win_lane[0] == nullptr && pb->fast_->win_lane[0] == nullptr) {
        win_lanes_make(fast_->cpu_pool, pb->fast_->cpu_pool, fast_->cpu_pin, pb->fast_->cpu_pin);
        pb->win_lanes_make(pb->fast_->cpu_pool, fast_->cpu_pool, pb->fast_->cpu_pin, fast_->cpu_pin);
        std::fprintf(stderr, "glm dflash: a batched window's layer on both parts' CPU pools (CUDA%d + CUDA%d)\n", dev_,
                     pb->dev_);
    }
    cudaSetDevice(dev_);
    std::fprintf(stderr, "glm dflash: %s\n", D->describe().c_str());
    return true;
}

// rows [0, n) of R (hc x n_embd floats each, positions p0..): the input of layer il.  The rows the drafter reads -
// at or past cap_floor_ - leave as their stream mean through `scratch` (at least min(n, rows) rows) to the staging
void Glm5Model::dflash_capture(int il, const float* R, int n, int64_t p0, float* scratch, int rows, cudaStream_t s) {
    if (cap_d_ == nullptr || il >= (int) cap_k_.size() || cap_k_[(size_t) il] < 0 || scratch == nullptr || rows <= 0)
        return;
    const int k = cap_k_[(size_t) il];
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    const int t0 = (int) std::max<int64_t>(0, cap_floor_ - p0);
    for (int a = t0; a < n; a += rows) {
        const int m = std::min(rows, n - a);
        hc_mean_kernel<<<m, 256, 0, s>>>(R + (size_t) a * g.hc * E, g.hc, E, scratch);
        // a run of positions is contiguous in the staging until the ring wraps
        const int ring = cap_d_->ring_size();
        for (int i = 0; i < m;) {
            const int64_t p = p0 + a + i;
            const int len = (int) std::min<int64_t>(m - i, (int64_t) ring - p % ring);
            cudaMemcpy2DAsync(cap_d_->stage_row(p, k), (size_t) cap_d_->stage_pitch(), scratch + (size_t) i * E,
                              (size_t) E * sizeof(float), (size_t) E * sizeof(float), (size_t) len,
                              cudaMemcpyDeviceToHost, s);
            for (int j = 0; j < len; ++j) cap_d_->staged(p + j, k);
            i += len;
        }
    }
}

bool Glm5Model::dflash_ready() const { return fast_ != nullptr && cap_d_ != nullptr; }

void Glm5Model::dflash_reset() {
    if (cap_d_) cap_d_->reset();
}

// The DFlash decode: each round the drafter's block for the last accepted token (up to n_max drafts, or fewer when the
// selector's confidence falls below p_min), a verify window of that token and the drafts, its longest right prefix and
// the pick after it kept (window_commit) - the same tokens the token-at-a-time loop produces for the same samples -
// and the kept positions' trunk features into the drafter's ring.  After the prompt's last forward().
bool Glm5Model::decode_dflash(strata::kernels::SamplerParams& sp, int64_t max_new, int n_max, float p_min,
                              const std::function<bool(int)>& emit, int64_t& produced, std::string& err, int n_min,
                              float p_cum) {
    produced = 0;
    if (!dflash_ready()) {
        err = "glm dflash: no draft loaded";
        return false;
    }
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    GlmDflash& D = *cap_d_;
    // the drafter's own stream on the head's device: a draft runs beside the pipelined window's first token
    cudaStream_t ts = D.stream() != nullptr ? D.stream() : tail->fast_->cs;
    const int n_max_req = n_max;
    if (n_max < 0) n_max = D.block() - 1;   // the default: the drafter's whole block (see below for a split)
    n_max = std::max(1, std::min({n_max, D.block() - 1, kWinMax - 1}));
    const WSlot& head = tail->ws_map_.at("output.weight");
    static const bool prof = getenv("STRATA_GLM_DFLASH_PROF") != nullptr;
    // STRATA_GLM_DFLASH_TL=<file>: a line a pipelined round - ms from its anchor's start: drafts ready, drafts pushed,
    // each pick (the tokens out of the last part), the commit done
    static FILE* tl = [] {
        const char* f = getenv("STRATA_GLM_DFLASH_TL");
        return f != nullptr && f[0] ? std::fopen(f, "a") : nullptr;
    }();
    auto t_anchor = std::chrono::steady_clock::now();
    std::vector<double> tl_pick;
    double tl_begun = 0, tl_inj = 0, tl_ds = 0;   // the anchor's begin returned, inject queued, draft queued
    const auto ms_since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    double t_draft = 0, t_win = 0, t_rest = 0;
    double ph_begin = 0, ph_dwait = 0, ph_push = 0, ph_end = 0;   // the pipelined rounds' host phases
    bool begun = false;              // the pipelined window's anchor already queued (at the end of the last round)
    int64_t inj_lo = 0, inj_hi = 0;  // the last round's kept positions, not yet in the drafter's ring
    int64_t ph_pushed = 0;
    int64_t rounds = 0, drafted = 0, accepted = 0, plain = 0;
    auto tnow = std::chrono::steady_clock::now();
    const auto lap = [&](double& acc) {
        if (!prof) return;
        const auto t = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t - tnow).count();
        tnow = t;
    };
    // the first token: the prompt's last forward left its logits on the tail
    int y = -1;
    if (sp.greedy && last_tok_ >= 0) {
        y = last_tok_;
    } else {
        y = sample_token(sp, err);
        if (y < 0) return false;
    }
    y = forced(y);
    sp.counter += 1;
    ++produced;
    if (!emit(y) || produced >= max_new || pos_ + 1 >= max_ctx_) return true;
    // the prompt's positions (their passes are done) into the ring
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
    }
    cudaSetDevice(tail->dev_);
    if (!D.inject(0, pos_, ts, err)) return false;
    std::vector<int32_t> d;
    std::vector<int32_t> tok, out;
    std::vector<float> conf;
    // STRATA_GLM_DFLASH_STATS=<file>: a line a round - how many drafts the window checked and kept, the draft's and
    // the window's ms, the round's CPU-lane experts and VRAM misses, the drafts' selector confidences (for choosing
    // how many to draft).  STRATA_GLM_DFLASH_EXPLORE=1: every
    // round drafts the whole block but checks a random number of its drafts (1..n_max), so the stats cover every
    // window size; the tokens are the same either way
    static FILE* stats = [] {
        const char* f = getenv("STRATA_GLM_DFLASH_STATS");
        return f != nullptr && f[0] ? std::fopen(f, "a") : nullptr;
    }();
    static const bool explore = getenv("STRATA_GLM_DFLASH_EXPLORE") != nullptr;
    // the pipelined window (its parts on consecutive tokens at once) or the batched one: dflash_pipelined
    const bool pipe = split_next_ != nullptr && dflash_pipelined(std::max(2, n_parts_));
    // the defaults (n_max / p_min < 0): on the pipelined window a draft after a wrong one costs little (its work is
    // skipped) - Maya-L over 9 GPUs, 5 prompts: 0.3 / 0.5 / 0.7 / none within ~2% of each other, 0.5 the steadiest on
    // a long context (53.8 against 43-52 tok/s at 0.3).  The batched window on two GPUs: a draft a round whatever its
    // confidence, then more while the product of the confidences stays at 0.9, up to kSplitDrafts - a window of two
    // costs 1.24 lone tokens there (81 against 65 ms), each more token ~28 ms; per-round stats with every window size
    // (STRATA_GLM_DFLASH_EXPLORE, 3090 + 5060 Ti) put this at 23.4 tok/s against 21.05 for one draft a round.  One GPU:
    // the 0.95 cut, no draft below it (a window of two costs 1.56 lone tokens: 97 against 62.5 ms; the same stats put
    // it within 1% of the best cut or product rule)
    const bool batched_split = split_next_ != nullptr && !pipe;
    constexpr int kSplitDrafts = 4;
    if (n_max_req < 0 && batched_split) n_max = std::min(n_max, kSplitDrafts);
    if (p_min < 0.0f) p_min = pipe ? 0.5f : batched_split ? 0.0f : 0.95f;
    // at least n_min drafts whatever their confidence, and later drafts only while the product of the confidences
    // stays at p_cum (STRATA_GLM_DFLASH_MIN / _CUM; the request's spec_n_min / spec_cum)
    if (n_min < 0)
        n_min = getenv("STRATA_GLM_DFLASH_MIN") ? std::max(0, std::atoi(getenv("STRATA_GLM_DFLASH_MIN")))
                                                : batched_split ? 1 : 0;
    if (p_cum < 0.0f)
        p_cum = getenv("STRATA_GLM_DFLASH_CUM") ? (float) std::atof(getenv("STRATA_GLM_DFLASH_CUM"))
                                                : batched_split ? 0.9f : 0.0f;
    if (pipe) n_max = std::min(n_max, kPipeMax - 1);   // the pipelined window's slots
    // the last window's late tokens and the commit behind them done before the caller reads the state - on every
    // way out, an error's too
    const auto drain = [&] {
        if (!pipe) return;
        win_commit_wait();
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            cudaSetDevice(m->dev_);
            cudaStreamSynchronize(m->fast_->cs);
        }
        cudaSetDevice(tail->dev_);
        cudaStreamSynchronize(ts);
        cudaSetDevice(dev_);
    };
    struct Drain {
        const std::function<void()> f;
        ~Drain() { f(); }
    } drain_on_exit{drain};
    // the pipelined window: the next round's injection and draft queued from a thread of their own as soon as the
    // round's last pick is in, beside the commit and the next anchor's begin here (their ~1.3 ms of launches were
    // behind those, the drafter's GPU idle meanwhile).  STRATA_GLM_DFLASH_AHEAD=0: at the top of the round (A/B).
    // (Declared after drain_on_exit: its job is done before the streams are drained.)
    static const bool ahead_on = [] {
        const char* v = getenv("STRATA_GLM_DFLASH_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    DraftAhead ahead;
    bool ahead_queued = false;
    uint64_t rng = 0x9e3779b97f4a7c15ull ^ (uint64_t) pos_;
    for (;;) {
        const int64_t N = pos_;
        tnow = std::chrono::steady_clock::now();
        const auto t_round = tnow;
        const FastStats st0 = stats != nullptr ? fast_stats() : FastStats{};
        const auto lane_of = [&] {
            const FastStats st1 = fast_stats();
            return std::make_pair(st1.cpu_experts - st0.cpu_experts, st1.misses - st0.misses);
        };
        cudaSetDevice(tail->dev_);
        const int room = (int) std::min<int64_t>(n_max, max_ctx_ - N - 1);
        d.clear();
        // the pipelined window's anchor pass (enqueued through every part; after the first round it was already
        // begun at the end of the last one, right behind its commit), then the drafter takes the last round's kept
        // positions and drafts on its own stream while the anchor runs.  After the first round a window follows the
        // previous one's late tokens without waiting for the devices; every 8th waits, so the tiers' boundary runs
        // even when no window finds the devices idle
        const auto hp0 = std::chrono::steady_clock::now();
        if (pipe && !begun) {
            t_anchor = std::chrono::steady_clock::now();
            if (!win_pipe_begin(y, err, &sp, rounds % 8 == 0)) return false;
        }
        begun = false;
        if (ahead_queued) {   // (queued at the end of the last round)
            ahead_queued = false;
            if (!ahead.wait(err)) return false;
            if (tl != nullptr) tl_inj = tl_ds = ms_since(t_anchor);
        } else {
            if (inj_hi > inj_lo) {
                cudaSetDevice(tail->dev_);
                if (!D.inject(inj_lo, inj_hi, ts, err)) return false;
                inj_lo = inj_hi = 0;
            }
            if (tl != nullptr) tl_inj = ms_since(t_anchor);
            if (room > 0 && !D.draft_start(y, N, explore ? D.block() - 1 : room, explore ? 0.0f : p_min, head,
                                           pack_emb_src_, pack_emb_type_, ts, err))
                return false;
            if (tl != nullptr) tl_ds = ms_since(t_anchor);
        }
        const auto hp1 = std::chrono::steady_clock::now();
        if (!D.draft_finish(d, err, stats != nullptr ? &conf : nullptr, explore ? 0 : n_min, explore ? 0.0f : p_cum))
            return false;
        if (prof) {
            ph_begin += std::chrono::duration<double, std::milli>(hp1 - hp0).count();
            ph_dwait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hp1).count();
        }
        if (explore && !d.empty()) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            const int nk = 1 + (int) ((rng >> 33) % (uint64_t) std::min<int>((int) d.size(), std::max(1, room)));
            d.resize((size_t) nk);
        }
        const auto t_drafted = std::chrono::steady_clock::now();
        lap(t_draft);
        if (d.empty() && !pipe) {
            // no draft passed the confidence cut: the token path for y (cheaper than a window of one; its features
            // are captured for the drafter there), the next token picked as the token loop picks it
            std::vector<float> lg;
            if (!forward({y}, lg, err)) return false;
            const int y2 = forced(sample_token(sp, err));
            if (y2 < 0) return false;
            if (stats != nullptr) {
                const auto t_tok = std::chrono::steady_clock::now();
                const auto lm = lane_of();
                std::fprintf(stats, "0 0 %.3f %.3f %llu %llu\n",
                             std::chrono::duration<double, std::milli>(t_drafted - t_round).count(),
                             std::chrono::duration<double, std::milli>(t_tok - t_drafted).count(),
                             (unsigned long long) lm.first, (unsigned long long) lm.second);
            }
            lap(t_win);
            ++rounds;
            ++plain;
            cudaSetDevice(tail->dev_);
            if (!D.inject(N, N + 1, ts, err)) return false;
            lap(t_rest);
            sp.counter += 1;
            ++produced;
            last_tok_ = y2;
            if (!emit(y2) || produced >= max_new || pos_ + 1 >= max_ctx_) break;
            y = y2;
            continue;
        }
        const int T = 1 + (int) d.size();
        tok.assign((size_t) T, 0);
        out.assign((size_t) T, 0);
        tok[0] = y;
        for (int i = 1; i < T; ++i) tok[(size_t) i] = d[(size_t) i - 1];
        // row t's token as it will go out: the window's pick, or --spec-follow's continuation there (forced() takes
        // one a token as they are emitted below; a draft stands while it is the token that goes out)
        const auto goes_out = [&](int t) {
            const int32_t f = follow_peek((size_t) t);   // (forced() returns it whatever else is forced)
            return f >= 0 ? f : out[(size_t) t];
        };
        if (pipe) {
            const auto hq0 = std::chrono::steady_clock::now();
            for (int i = 1; i < T; ++i)
                if (!win_pipe_push(tok[(size_t) i], err)) return false;
            const auto hq1 = std::chrono::steady_clock::now();
            const double tl_ready = std::chrono::duration<double, std::milli>(hq0 - t_anchor).count();
            const double tl_pushed = std::chrono::duration<double, std::milli>(hq1 - t_anchor).count();
            tl_pick.clear();
            // the picks as their tokens come out of the last part, up to the first that is not the next draft: the
            // tokens after it are still in flight, and the commit and the next window queue behind them
            for (int t = 0; t < T; ++t) {
                if (!win_pipe_pick(t, out[(size_t) t], err)) return false;
                if (tl != nullptr) tl_pick.push_back(ms_since(t_anchor));
                if (t + 1 >= T) break;
                if (tok[(size_t) t + 1] != goes_out(t)) {
                    win_pipe_cancel_from(t + 1);   // the drafts after it: their work still queued is skipped
                    break;
                }
            }
            if (!win_pipe_settle(err)) return false;
            if (tl != nullptr) {
                const auto b0 = win_pipe_begun_at(0), bl = win_pipe_begun_at(n_parts_ - 1);
                std::fprintf(tl, "%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f |", T - 1, tl_ready, tl_pushed, tl_begun, tl_inj,
                             tl_ds, std::chrono::duration<double, std::milli>(b0 - t_anchor).count(),
                             std::chrono::duration<double, std::milli>(bl - t_anchor).count());
                for (const double v : tl_pick) std::fprintf(tl, " %.3f", v);
                std::fprintf(tl, "\n");
            }
            if (prof) {
                ph_push += std::chrono::duration<double, std::milli>(hq1 - hq0).count();
                ph_pushed += T - 1;
                ph_end += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hq1).count();
            }
        } else if (!window(tok.data(), T, out.data(), err, &sp)) {
            return false;
        }
        lap(t_win);
        // the longest right prefix of the drafts; each kept token goes out in turn (a forced next token - the
        // reasoning closed by the caller - ends the run there, like a wrong draft)
        int acc = 0;
        while (acc < T - 1 && tok[(size_t) acc + 1] == goes_out(acc)) ++acc;
        if (stats != nullptr) {
            const auto t_win = std::chrono::steady_clock::now();
            const auto lm = lane_of();
            std::fprintf(stats, "%d %d %.3f %.3f %llu %llu", T - 1, acc,
                         std::chrono::duration<double, std::milli>(t_drafted - t_round).count(),
                         std::chrono::duration<double, std::milli>(t_win - t_drafted).count(),
                         (unsigned long long) lm.first, (unsigned long long) lm.second);
            for (const float c : conf) std::fprintf(stats, " %.4f", c);
            std::fprintf(stats, "\n");
        }
        spec_steps_ += (uint64_t) (T - 1);
        spec_hits_ += (uint64_t) acc;
        ++rounds;
        drafted += T - 1;
        accepted += acc;
        bool more = true;
        int kept = 0;   // drafts emitted
        for (; kept < acc; ++kept) {
            sp.counter += 1;
            ++produced;
            // (the kept draft, through forced() as every emitted token: --spec-follow's continuation moves on and
            //  counts the model's own pick here)
            more = emit(forced(out[(size_t) kept])) && produced < max_new;
            if (!more || force_tok_ >= 0) {
                ++kept;
                break;
            }
        }
        const int keep = 1 + kept;
        const int y2 = forced(out[(size_t) keep - 1]);
        if (pipe && more && ahead_on) {   // the next round's injection and draft (see ahead)
            const int64_t N2 = N + keep;
            const int room2 = (int) std::min<int64_t>(n_max, max_ctx_ - N2 - 1);
            const int nd2 = explore ? D.block() - 1 : room2;
            const float pm2 = explore ? 0.0f : p_min;
            const int dev_t = tail->dev_;
            const uint8_t* emb = pack_emb_src_;
            const int emb_type = pack_emb_type_;
            ahead.kick([&D, &head, ts, dev_t, N, N2, room2, nd2, pm2, y2, emb, emb_type](std::string& e) {
                cudaSetDevice(dev_t);
                if (!D.inject(N, N2, ts, e)) return false;
                return room2 <= 0 || D.draft_start(y2, N2, nd2, pm2, head, emb, emb_type, ts, e);
            });
            ahead_queued = true;
        }
        // (the pipelined window: the parts' commit enqueues not waited for - the next anchor's begin queues behind each
        //  on its crew thread; STRATA_GLM_WIN_COMMIT_WAIT=1: waited for, A/B)
        static const bool commit_wait = [] {
            const char* v = getenv("STRATA_GLM_WIN_COMMIT_WAIT");
            return v != nullptr && std::atoi(v) != 0;
        }();
        if (!window_commit(keep, err, !pipe || commit_wait || !more)) return false;
        if (!pipe) {
            cudaSetDevice(tail->dev_);
            if (!D.inject(N, N + keep, ts, err)) return false;
        }
        lap(t_rest);
        if (!more) break;
        sp.counter += 1;
        ++produced;
        last_tok_ = y2;
        if (!emit(y2) || produced >= max_new || pos_ + 1 >= max_ctx_) break;
        y = y2;
        if (pipe) {
            // the next window's anchor now, behind the commit; the drafter's catch-up at the top of the round
            if (tl != nullptr) std::fprintf(tl, "c %.3f\n", ms_since(t_anchor));
            t_anchor = std::chrono::steady_clock::now();
            if (!win_pipe_begin(y, err, &sp, (rounds % 8) == 0)) return false;
            if (tl != nullptr) tl_begun = ms_since(t_anchor);
            begun = true;
            inj_lo = ahead_queued ? 0 : N;   // (the thread took them)
            inj_hi = ahead_queued ? 0 : N + keep;
        }
    }
    cudaSetDevice(dev_);
    if (stats != nullptr) std::fflush(stats);
    if (tl != nullptr) std::fflush(tl);
    win_prof_report();
    if (prof && rounds > 0)
        std::fprintf(stderr, "glm dflash prof (%lld rounds, %lld of them no draft: %.2f drafts, %.2f accepted a round): "
                             "draft %.2f ms | verify window (or the token) %.2f ms | commit + inject %.2f ms\n",
                     (long long) rounds, (long long) plain, (double) drafted / rounds, (double) accepted / rounds,
                     t_draft / rounds, t_win / rounds, t_rest / rounds);
    if (prof && rounds > 0 && pipe)
        std::fprintf(stderr, "glm dflash pipe (ms a round): anchor enqueue %.2f | draft wait %.2f | drafts enqueue %.2f "
                             "(%.2f a token) | picks %.2f\n",
                     ph_begin / rounds, ph_dwait / rounds, ph_push / rounds, ph_push / std::max<int64_t>(1, ph_pushed),
                     ph_end / rounds);
    return true;
}

}  // namespace strata::core
