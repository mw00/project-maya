// src/kernels/cuda/glmd_ref.cu - glm-dsa F32 reference kernels (B3.4).
//
// Contracts, layouts and the GLM-5.3 attention scale derivation: include/strata/kernels/glmd_ref.hpp.  The CPU
// twin is src/core/glm_dsa_cpu_ref.cpp.  Everything here is plain F32 with simple launch shapes (a warp per output
// row, a block per (head, token)) and no fast math: these kernels are the yardstick for the fast path, so they
// favour being easy to read over being quick.
#include "strata/kernels/glmd_ref.hpp"

#include "glmd_rope.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace glmd {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum or max for blockDim.x == kThreads; every thread gets the result
template <bool kMax>
__device__ float block_reduce(float v) {
    __shared__ float s_part[kWarps];
    __shared__ float s_out;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = kMax ? warp_max(v) : warp_sum(v);
    if (lane == 0) s_part[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float r = lane < kWarps ? s_part[lane] : (kMax ? -INFINITY : 0.0f);
        r = kMax ? warp_max(r) : warp_sum(r);
        if (lane == 0) s_out = r;
    }
    __syncthreads();
    const float r = s_out;
    __syncthreads();
    return r;
}

// one thread per (pair, row, token)
__global__ void rope_norm_kernel(float* __restrict__ x, int row_len, int rows_per_token, int T, int rot_off,
                                 int n_rot, const int* __restrict__ pos, const float* __restrict__ inv_freq) {
    const int half = n_rot / 2;
    const long long n = (long long) half * rows_per_token * T;
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    const int i = (int) (g % half);
    const long long row = g / half;
    const int t = (int) (row / rows_per_token);
    float* xr = x + row * row_len + rot_off;
    float c, s;
    rope_cs(pos[t], inv_freq[i], c, s);
    rope_pair(xr[2 * i], xr[2 * i + 1], c, s);
}

// a warp per output row; grid (ceil(out / kWarps), T)
__global__ void gemv_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int in,
                            int out, float alpha) {
    const int o = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (o >= out) return;
    const float* wr = W + (size_t) o * in;
    const float* xt = x + (size_t) t * in;
    float acc = 0.0f;
    for (int i = lane; i < in; i += 32) acc += wr[i] * xt[i];
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * out + o] = alpha * acc;
}

// a block per token: LayerNorm(w, b, eps) in place, then NORM rope on dims 0..n_rot-1 at pos[t]
__global__ void layer_norm_rope_kernel(float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ b, float eps, int n, int n_rot,
                                       const int* __restrict__ pos, const float* __restrict__ inv_freq) {
    float* xt = x + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) s += xt[i];
    const float mean = block_reduce<false>(s) / (float) n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) v += (xt[i] - mean) * (xt[i] - mean);
    const float var = block_reduce<false>(v) / (float) n;
    const float inv = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += kThreads) xt[i] = (xt[i] - mean) * inv * w[i] + b[i];
    __syncthreads();
    for (int i = threadIdx.x; i < n_rot / 2; i += kThreads) {
        float c, sn;
        rope_cs(pos[blockIdx.x], inv_freq[i], c, sn);
        rope_pair(xt[2 * i], xt[2 * i + 1], c, sn);
    }
}

// one thread per (cell, token); grid (ceil(n_kv / kThreads), T)
__global__ void indexer_score_kernel(const float* __restrict__ iq, const float* __restrict__ ik,
                                     const float* __restrict__ w, int idx_heads, int idx_dim, int n_kv,
                                     const int* __restrict__ pos, float* __restrict__ score) {
    const int s = blockIdx.x * kThreads + threadIdx.x, t = blockIdx.y;
    if (s >= n_kv) return;
    float* out = score + (size_t) t * n_kv + s;
    if (s > pos[t]) {
        *out = -INFINITY;
        return;
    }
    const float* k = ik + (size_t) s * idx_dim;
    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        const float* q = iq + ((size_t) t * idx_heads + h) * idx_dim;
        float d = 0.0f;
        for (int e = 0; e < idx_dim; ++e) d += q[e] * k[e];
        acc += w[(size_t) t * idx_heads + h] * fmaxf(d, 0.0f);
    }
    *out = acc;
}

// a block per token: each finite cell's rank = number of cells that beat it (higher score, or equal score and a
// lower index); ranks below top_k are written.  -inf cells are not candidates and never beat a finite one.
__global__ void topk_kernel(const float* __restrict__ score, int n_kv, int top_k, int* __restrict__ sel) {
    const float* sc = score + (size_t) blockIdx.x * n_kv;
    int* row = sel + (size_t) blockIdx.x * top_k;
    for (int r = threadIdx.x; r < top_k; r += kThreads) row[r] = -1;
    __syncthreads();
    for (int s = threadIdx.x; s < n_kv; s += kThreads) {
        const float v = sc[s];
        if (v == -INFINITY) continue;
        int rank = 0;
        for (int j = 0; j < n_kv && rank < top_k; ++j) {
            const float u = sc[j];
            rank += (u > v) || (u == v && j < s);
        }
        if (rank < top_k) row[rank] = s;
    }
}

// a block per (head, token)
__global__ void absorb_q_kernel(const float* __restrict__ q, const float* __restrict__ wk_b, int n_head,
                                int qk_nope, int qk_rope, int kv_lora, float* __restrict__ Q) {
    extern __shared__ float s_qn[];
    const int h = blockIdx.x, t = blockIdx.y;
    const float* qh = q + ((size_t) t * n_head + h) * (qk_nope + qk_rope);
    float* Qh = Q + ((size_t) t * n_head + h) * (kv_lora + qk_rope);
    for (int c = threadIdx.x; c < qk_nope; c += kThreads) s_qn[c] = qh[c];
    __syncthreads();
    for (int r = threadIdx.x; r < kv_lora; r += kThreads) {
        const float* wr = wk_b + ((size_t) h * kv_lora + r) * qk_nope;
        float acc = 0.0f;
        for (int c = 0; c < qk_nope; ++c) acc += wr[c] * s_qn[c];
        Qh[r] = acc;
    }
    for (int i = threadIdx.x; i < qk_rope; i += kThreads) Qh[kv_lora + i] = qh[qk_nope + i];
}

// a block per (head, token); shared: Q row (q_dim), ctx (kv_lora), logits (n_sel)
__global__ void mla_attention_kernel(const float* __restrict__ Q, const float* __restrict__ kv,
                                     const int* __restrict__ cells, const float* __restrict__ wv_b, int n_head,
                                     int kv_lora, int qk_rope, int v_head, int n_sel, float scale,
                                     float* __restrict__ out) {
    extern __shared__ float s_mem[];
    const int q_dim = kv_lora + qk_rope;
    float* s_q = s_mem;
    float* s_ctx = s_q + q_dim;
    float* s_l = s_ctx + kv_lora;
    const int h = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int* ct = cells + (size_t) t * n_sel;
    const float* Qh = Q + ((size_t) t * n_head + h) * q_dim;
    for (int i = threadIdx.x; i < q_dim; i += kThreads) s_q[i] = Qh[i];
    __syncthreads();

    // logits: a warp per cell, one dot over [c_kv | k_pe]
    for (int s = wid; s < n_sel; s += kWarps) {
        const int cell = ct[s];
        float d = 0.0f;
        if (cell >= 0) {
            const float* k = kv + (size_t) cell * q_dim;
            for (int i = lane; i < q_dim; i += 32) d += s_q[i] * k[i];
            d = warp_sum(d);
        }
        if (lane == 0) s_l[s] = cell >= 0 ? d * scale : -INFINITY;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int s = threadIdx.x; s < n_sel; s += kThreads) m = fmaxf(m, s_l[s]);
    m = block_reduce<true>(m);
    float sum = 0.0f;
    for (int s = threadIdx.x; s < n_sel; s += kThreads) {
        const float p = ct[s] >= 0 ? expf(s_l[s] - m) : 0.0f;
        s_l[s] = p;
        sum += p;
    }
    sum = block_reduce<false>(sum);   // its barriers also publish the probabilities in s_l
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int c = threadIdx.x; c < kv_lora; c += kThreads) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = ct[s];
            if (cell >= 0) acc += s_l[s] * kv[(size_t) cell * q_dim + c];
        }
        s_ctx[c] = acc * inv;
    }
    __syncthreads();

    float* oh = out + ((size_t) t * n_head + h) * v_head;
    for (int v = threadIdx.x; v < v_head; v += kThreads) {
        const float* wr = wv_b + ((size_t) h * v_head + v) * kv_lora;
        float acc = 0.0f;
        for (int c = 0; c < kv_lora; ++c) acc += wr[c] * s_ctx[c];
        oh[v] = acc;
    }
}

cudaStream_t as_stream(void* stream) { return static_cast<cudaStream_t>(stream); }

}  // namespace

void ref_rope_norm(float* x, int row_len, int rows_per_token, int T, int rot_off, int n_rot, const int* pos,
                   const float* inv_freq, void* stream) {
    if (T <= 0 || rows_per_token <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || rot_off + n_rot > row_len) {
        std::fprintf(stderr, "glmd::ref_rope_norm: n_rot %d at %d does not fit an even span of row %d\n", n_rot,
                     rot_off, row_len);
        return;
    }
    const long long n = (long long) (n_rot / 2) * rows_per_token * T;
    rope_norm_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, as_stream(stream)>>>(
        x, row_len, rows_per_token, T, rot_off, n_rot, pos, inv_freq);
}

void ref_gemv(const float* W, const float* x, float* y, int in, int out, int T, float alpha, void* stream) {
    if (T <= 0 || out <= 0) return;
    const dim3 grid((unsigned) ((out + kWarps - 1) / kWarps), (unsigned) T);
    gemv_kernel<<<grid, kThreads, 0, as_stream(stream)>>>(W, x, y, in, out, alpha);
}

void ref_indexer_key(const float* x, const float* w_idx_k, const float* ln_w, const float* ln_b, float eps,
                     int n_embd, int idx_dim, int T, int n_rot, const int* pos, const float* inv_freq, float* ik,
                     void* stream) {
    if (T <= 0) return;
    ref_gemv(w_idx_k, x, ik, n_embd, idx_dim, T, 1.0f, stream);
    layer_norm_rope_kernel<<<(unsigned) T, kThreads, 0, as_stream(stream)>>>(ik, ln_w, ln_b, eps, idx_dim, n_rot,
                                                                            pos, inv_freq);
}

void ref_indexer_query(const float* q_lat, const float* w_idx_q_b, int q_lora, int idx_heads, int idx_dim, int T,
                       int n_rot, const int* pos, const float* inv_freq, float* iq, void* stream) {
    ref_gemv(w_idx_q_b, q_lat, iq, q_lora, idx_heads * idx_dim, T, 1.0f, stream);
    ref_rope_norm(iq, idx_dim, idx_heads, T, 0, n_rot, pos, inv_freq, stream);
}

void ref_indexer_weights(const float* x, const float* w_idx_proj, int n_embd, int idx_heads, int idx_dim, int T,
                         float* w, void* stream) {
    // llama.cpp: ggml_scale(weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head))), 1/64 for GLM-5.3
    ref_gemv(w_idx_proj, x, w, n_embd, idx_heads, T, 1.0f / std::sqrt((float) (idx_dim * idx_heads)), stream);
}

void ref_indexer_score(const float* iq, const float* ik_cache, const float* w, int idx_heads, int idx_dim, int T,
                       int n_kv, const int* pos, float* score, void* stream) {
    if (T <= 0 || n_kv <= 0) return;
    const dim3 grid((unsigned) ((n_kv + kThreads - 1) / kThreads), (unsigned) T);
    indexer_score_kernel<<<grid, kThreads, 0, as_stream(stream)>>>(iq, ik_cache, w, idx_heads, idx_dim, n_kv, pos,
                                                                  score);
}

void ref_topk(const float* score, int n_kv, int T, int top_k, int* sel, void* stream) {
    if (T <= 0 || top_k <= 0) return;
    topk_kernel<<<(unsigned) T, kThreads, 0, as_stream(stream)>>>(score, n_kv, top_k, sel);
}

void ref_mla_absorb_q(const float* q, const float* wk_b, int n_head, int qk_nope, int qk_rope, int kv_lora, int T,
                      float* Q, void* stream) {
    if (T <= 0 || n_head <= 0) return;
    const dim3 grid((unsigned) n_head, (unsigned) T);
    absorb_q_kernel<<<grid, kThreads, (size_t) qk_nope * sizeof(float), as_stream(stream)>>>(q, wk_b, n_head, qk_nope,
                                                                                              qk_rope, kv_lora, Q);
}

void ref_mla_attention(const float* Q, const float* kv, const int* cells, const float* wv_b, int n_head,
                       int kv_lora, int qk_rope, int v_head, int T, int n_sel, float scale, float* out,
                       void* stream) {
    if (T <= 0 || n_head <= 0) return;
    if (n_sel <= 0 || n_sel > kRefMaxSel) {
        std::fprintf(stderr, "glmd::ref_mla_attention: n_sel %d outside [1, %d]\n", n_sel, kRefMaxSel);
        return;
    }
    // GLM-5.3: at most (2 * 512 + 64 + 8192) floats = 37 KB, under the 48 KB default
    const size_t shmem = (size_t) (2 * kv_lora + qk_rope + n_sel) * sizeof(float);
    if (shmem > 48 * 1024) {
        std::fprintf(stderr, "glmd::ref_mla_attention: %zu B of shared memory for kv_lora %d, n_sel %d\n", shmem,
                     kv_lora, n_sel);
        return;
    }
    const dim3 grid((unsigned) n_head, (unsigned) T);
    mla_attention_kernel<<<grid, kThreads, shmem, as_stream(stream)>>>(Q, kv, cells, wv_b, n_head, kv_lora, qk_rope,
                                                                       v_head, n_sel, scale, out);
}

}  // namespace glmd
