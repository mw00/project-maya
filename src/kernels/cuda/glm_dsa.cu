// src/kernels/cuda/glm_dsa.cu - the glm5-next DSA indexer and absorbed-MLA attention.
// See glm_dsa.hpp for the math and layouts.
//
// Four kernels:
//
//   pool      one thread per (key-dim element, pool): kpool exponentials, then the weighted sum.
//             The softmax is PER KEY-DIM ELEMENT - the quirk the parity harness pins.
//   score     one thread per (query, pool): a relu'd dot per indexer head, weighted and summed.
//   select    one thread per query: the router rank formula (stable descending, ties by lower
//             index) over the VISIBLE pools, then cell expansion and the tail append, padded -1.
//             O(n_vis^2) comparisons per query; decode's n_vis is small, and prefill's is the one
//             place a radix top-k would pay - later, behind the same parity.
//   attention one block per (query, head): scores over the valid cells (shared), softmax on one
//             thread, ctx in shared (kv_lora floats), then the v-absorbed output.  Padding cells
//             are EXCLUDED from the softmax and from the ctx gather, which is numerically the
//             graph's -inf additive mask (they would contribute exp(-inf) = 0).
//
// All arithmetic f32 with precise expf (the glm_ffn accuracy contract: __expf's argument-scaled ulp
// error is visible from |x| ~ 10); the parity harness holds the kernels to ref/glm.py's f64 at f32
// tolerance.
#include "maya/kernels/glm_dsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace maya::kernels {
namespace {

constexpr int DS_THREADS = 256;
constexpr int DS_MAX_KPOOL = 16;

__global__ void glm_dsa_pool_kernel(const float* __restrict__ ik, const float* __restrict__ ig,
                                    const float* __restrict__ ape, int key_dim, int kpool, int n_pools,
                                    float* __restrict__ pooled) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = key_dim * n_pools;
    if (idx >= total) return;
    const int p = idx / key_dim;
    const int e = idx % key_dim;

    // per-key-dim softmax over the pool's members: logits_m = ig[e, member_m] + ape[e, m]
    float logits[DS_MAX_KPOOL];
    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) {
        logits[m] = ig[(size_t) e + key_dim * (p * kpool + m)] + ape[(size_t) e + key_dim * m];
        mx = fmaxf(mx, logits[m]);
    }
    float denom = 0.0f;
    for (int m = 0; m < kpool; ++m) {
        logits[m] = expf(logits[m] - mx);
        denom += logits[m];
    }
    float acc = 0.0f;
    for (int m = 0; m < kpool; ++m) acc += (logits[m] / denom) * ik[(size_t) e + key_dim * (p * kpool + m)];
    pooled[(size_t) e + key_dim * p] = acc;
}

__global__ void glm_dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                     const float* __restrict__ weights, int key_dim, int idx_heads,
                                     int n_tokens, int n_pools, float* __restrict__ score) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_pools;
    if (idx >= total) return;
    const int p = idx % n_pools;
    const int t = idx / n_pools;

    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        float dot = 0.0f;
        for (int e = 0; e < key_dim; ++e)
            dot += iq[(size_t) e + key_dim * h + (size_t) key_dim * idx_heads * t] * pooled[(size_t) e + key_dim * p];
        acc += fmaxf(dot, 0.0f) * weights[(size_t) h + idx_heads * t];
    }
    score[(size_t) p + (size_t) n_pools * t] = acc;
}

__global__ void glm_dsa_select_kernel(const float* __restrict__ score, int n_pools, int kpool, int top_pools,
                                      int select_tail, int T, int n_sel, const int* __restrict__ pos,
                                      int* __restrict__ cells) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    // pos[t] is the query's GLOBAL position: a T=1 streaming call passes its cache slot, a
    // batched call passes the identity.  Visibility follows the global position either way.
    const int n_vis = (pos[t] + 1) / kpool;   // pools whose last member is <= pos[t]
    int* row = cells + (size_t) n_sel * t;
    for (int s = 0; s < n_sel; ++s) row[s] = -1;

    for (int p = 0; p < n_vis; ++p) {
        const float v = score[(size_t) p + (size_t) n_pools * t];
        int rank = 0;
        for (int q = 0; q < n_vis; ++q) {
            const float u = score[(size_t) q + (size_t) n_pools * t];
            rank += u > v ? 1 : 0;
            rank += (q < p && u == v) ? 1 : 0;
        }
        if (rank < top_pools)
            for (int m = 0; m < kpool; ++m) row[rank * kpool + m] = p * kpool + m;
    }
    if (select_tail) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos[t]) row[top_pools * kpool + m] = cell;
        }
    }
}

__global__ void glm_mla_attention_kernel(const float* __restrict__ q_abs, const float* __restrict__ latents,
                                         const int* __restrict__ cells, const float* __restrict__ wv_b,
                                         int kv_lora, int n_head, int v_head, int qk_nope, int n_sel,
                                         float* __restrict__ out) {
    const int block = blockIdx.x;
    const int h = block % n_head;
    const int t = block / n_head;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int* row = cells + (size_t) n_sel * t;
    const float scale = rsqrtf((float) qk_nope);

    extern __shared__ float smem[];   // probs (n_sel) then ctx (kv_lora)
    float* s_prob = smem;
    float* s_ctx = smem + n_sel;

    // scores over the valid cells: s_s = (q_abs_h . latent_s) * scale
    for (int s = tid; s < n_sel; s += nt) {
        const int cell = row[s];
        if (cell < 0) {
            s_prob[s] = -INFINITY;
            continue;
        }
        const float* lat = latents + (size_t) kv_lora * cell;
        const float* qa = q_abs + (size_t) kv_lora * h + (size_t) kv_lora * n_head * t;
        float dot = 0.0f;
        for (int e = 0; e < kv_lora; ++e) dot += qa[e] * lat[e];
        s_prob[s] = dot * scale;
    }
    __syncthreads();

    // softmax over the VALID cells, on one thread (n_sel <= ~520 values)
    if (tid == 0) {
        float mx = -INFINITY;
        for (int s = 0; s < n_sel; ++s) mx = fmaxf(mx, s_prob[s]);
        float denom = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const float e = (s_prob[s] == -INFINITY) ? 0.0f : expf(s_prob[s] - mx);
            s_prob[s] = e;
            denom += e;
        }
        const float inv = 1.0f / denom;
        for (int s = 0; s < n_sel; ++s) s_prob[s] *= inv;
    }
    __syncthreads();

    // ctx[e] = sum_s prob_s * latent_s, over the VALID cells only
    for (int e = tid; e < kv_lora; e += nt) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s)
            if (row[s] >= 0) acc += s_prob[s] * latents[(size_t) e + (size_t) kv_lora * row[s]];
        s_ctx[e] = acc;
    }
    __syncthreads();

    // out[v, h, t] = sum_c wv_b[c, v, h] * ctx[c]
    for (int v = tid; v < v_head; v += nt) {
        float acc = 0.0f;
        for (int c = 0; c < kv_lora; ++c)
            acc += wv_b[(size_t) c + (size_t) kv_lora * v + (size_t) kv_lora * v_head * h] * s_ctx[c];
        out[(size_t) v + v_head * h + (size_t) v_head * n_head * t] = acc;
    }
}

}  // namespace

void glm_dsa_pool(const float* ik, const float* ig, const float* ape, int key_dim, int kpool, int n_pools,
                  float* pooled, void* stream) {
    if (key_dim <= 0 || kpool <= 0 || n_pools <= 0) return;
    if (kpool > DS_MAX_KPOOL) {
        std::fprintf(stderr, "glm_dsa_pool: kpool %d exceeds the kernel's %d\n", kpool, DS_MAX_KPOOL);
        std::exit(1);
    }
    const int total = key_dim * n_pools;
    glm_dsa_pool_kernel<<<(total + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        ik, ig, ape, key_dim, kpool, n_pools, pooled);
}

void glm_dsa_score(const float* iq, const float* pooled, const float* weights, int key_dim, int idx_heads,
                   int n_tokens, int n_pools, float* score, void* stream) {
    if (key_dim <= 0 || idx_heads <= 0 || n_tokens <= 0 || n_pools <= 0) return;
    const int total = n_tokens * n_pools;
    glm_dsa_score_kernel<<<(total + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        iq, pooled, weights, key_dim, idx_heads, n_tokens, n_pools, score);
}

void glm_dsa_select(const float* score, int n_pools, int kpool, int top_pools, int select_tail, int T,
                    int n_sel, const int* pos, int* cells, void* stream) {
    // top_pools == 0 is LEGITIMATE at the sequence's start (no pool completed yet): with
    // select_tail the current token is still visible through the tail cells, so the kernel
    // must run and write them - returning here would leave the previous call's cells stale
    // and the attention reading zeroed latents (the bug that made position 0 diverge).
    if (n_pools < 0 || kpool <= 0 || top_pools < 0 || T <= 0) return;
    glm_dsa_select_kernel<<<(T + DS_THREADS - 1) / DS_THREADS, DS_THREADS, 0, (cudaStream_t) stream>>>(
        score, n_pools, kpool, top_pools, select_tail, T, n_sel, pos, cells);
}

void glm_mla_attention(const float* q_abs, const float* latents, const int* cells, const float* wv_b,
                       int kv_lora, int n_head, int v_head, int qk_nope, int T, int n_sel, float* out,
                       void* stream) {
    if (kv_lora <= 0 || n_head <= 0 || v_head <= 0 || T <= 0 || n_sel <= 0) return;
    const size_t smem = ((size_t) n_sel + kv_lora) * sizeof(float);
    glm_mla_attention_kernel<<<T * n_head, DS_THREADS, smem, (cudaStream_t) stream>>>(
        q_abs, latents, cells, wv_b, kv_lora, n_head, v_head, qk_nope, n_sel, out);
}

}  // namespace maya::kernels
