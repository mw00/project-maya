// src/core/glm_dsa_cpu_ref.cpp - the CPU twin of the glm-dsa F32 reference kernels (B3.4).
//
// Same contracts as glmd::ref_* (include/strata/kernels/glmd_ref.hpp), plain loops, every dot product, norm
// statistic and softmax sum accumulated in double and rounded to float once.  The rope is the shared header's
// float rotation, so the GPU kernels and this file rotate with the same arithmetic.
#include "strata/kernels/glmd_ref.hpp"

#include "../kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace glmd::cpu {

void rope_norm(float* x, int row_len, int rows_per_token, int T, int rot_off, int n_rot, const int* pos,
               const float* inv_freq) {
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < rows_per_token; ++r)
            rope_row(x + ((size_t) t * rows_per_token + r) * row_len + rot_off, n_rot, pos[t], inv_freq);
}

void gemv(const float* W, const float* x, float* y, int in, int out, int T, float alpha) {
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < out; ++o) {
            const float* wr = W + (size_t) o * in;
            const float* xt = x + (size_t) t * in;
            double acc = 0.0;
            for (int i = 0; i < in; ++i) acc += (double) wr[i] * (double) xt[i];
            y[(size_t) t * out + o] = (float) ((double) alpha * acc);
        }
}

// LayerNorm with weight and bias, biased variance (torch nn.LayerNorm, ggml LLM_NORM)
void layer_norm(float* x, const float* w, const float* b, float eps, int n, int T) {
    for (int t = 0; t < T; ++t) {
        float* xt = x + (size_t) t * n;
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += xt[i];
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) var += ((double) xt[i] - mean) * ((double) xt[i] - mean);
        var /= n;
        const double inv = 1.0 / std::sqrt(var + (double) eps);
        for (int i = 0; i < n; ++i) xt[i] = (float) (((double) xt[i] - mean) * inv * (double) w[i] + (double) b[i]);
    }
}

void indexer_key(const float* x, const float* w_idx_k, const float* ln_w, const float* ln_b, float eps, int n_embd,
                 int idx_dim, int T, int n_rot, const int* pos, const float* inv_freq, float* ik) {
    gemv(w_idx_k, x, ik, n_embd, idx_dim, T, 1.0f);
    layer_norm(ik, ln_w, ln_b, eps, idx_dim, T);
    rope_norm(ik, idx_dim, 1, T, 0, n_rot, pos, inv_freq);
}

void indexer_query(const float* q_lat, const float* w_idx_q_b, int q_lora, int idx_heads, int idx_dim, int T,
                   int n_rot, const int* pos, const float* inv_freq, float* iq) {
    gemv(w_idx_q_b, q_lat, iq, q_lora, idx_heads * idx_dim, T, 1.0f);
    rope_norm(iq, idx_dim, idx_heads, T, 0, n_rot, pos, inv_freq);
}

void indexer_weights(const float* x, const float* w_idx_proj, int n_embd, int idx_heads, int idx_dim, int T,
                     float* w) {
    // llama.cpp: ggml_scale(weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head))), 1/64 for GLM-5.3
    gemv(w_idx_proj, x, w, n_embd, idx_heads, T, 1.0f / std::sqrt((float) (idx_dim * idx_heads)));
}

void indexer_score(const float* iq, const float* ik_cache, const float* w, int idx_heads, int idx_dim, int T,
                   int n_kv, const int* pos, float* score) {
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < n_kv; ++s) {
            float& out = score[(size_t) t * n_kv + s];
            if (s > pos[t]) {
                out = -std::numeric_limits<float>::infinity();
                continue;
            }
            const float* k = ik_cache + (size_t) s * idx_dim;
            double acc = 0.0;
            for (int h = 0; h < idx_heads; ++h) {
                const float* q = iq + ((size_t) t * idx_heads + h) * idx_dim;
                double d = 0.0;
                for (int e = 0; e < idx_dim; ++e) d += (double) q[e] * (double) k[e];
                acc += (double) w[(size_t) t * idx_heads + h] * std::max(d, 0.0);
            }
            out = (float) acc;
        }
}

void topk(const float* score, int n_kv, int T, int top_k, int* sel) {
    std::vector<int> idx;
    for (int t = 0; t < T; ++t) {
        const float* sc = score + (size_t) t * n_kv;
        idx.clear();
        for (int s = 0; s < n_kv; ++s)
            if (sc[s] != -std::numeric_limits<float>::infinity()) idx.push_back(s);
        // descending score, ties by the lower cell: a stable sort over ascending indices keeps that order
        std::stable_sort(idx.begin(), idx.end(), [sc](int a, int b) { return sc[a] > sc[b]; });
        int* row = sel + (size_t) t * top_k;
        for (int r = 0; r < top_k; ++r) row[r] = r < (int) idx.size() ? idx[r] : -1;
    }
}

void mla_absorb_q(const float* q, const float* wk_b, int n_head, int qk_nope, int qk_rope, int kv_lora, int T,
                  float* Q) {
    const int qk_head = qk_nope + qk_rope, q_dim = kv_lora + qk_rope;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < n_head; ++h) {
            const float* qh = q + ((size_t) t * n_head + h) * qk_head;
            float* Qh = Q + ((size_t) t * n_head + h) * q_dim;
            for (int r = 0; r < kv_lora; ++r) {
                const float* wr = wk_b + ((size_t) h * kv_lora + r) * qk_nope;
                double acc = 0.0;
                for (int c = 0; c < qk_nope; ++c) acc += (double) wr[c] * (double) qh[c];
                Qh[r] = (float) acc;
            }
            for (int i = 0; i < qk_rope; ++i) Qh[kv_lora + i] = qh[qk_nope + i];
        }
}

void mla_attention(const float* Q, const float* kv, const int* cells, const float* wv_b, int n_head, int kv_lora,
                   int qk_rope, int v_head, int T, int n_sel, float scale, float* out) {
    const int q_dim = kv_lora + qk_rope;
    std::vector<double> logit(n_sel), ctx(kv_lora);
    for (int t = 0; t < T; ++t) {
        const int* ct = cells + (size_t) t * n_sel;
        for (int h = 0; h < n_head; ++h) {
            const float* Qh = Q + ((size_t) t * n_head + h) * q_dim;
            double mx = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < n_sel; ++s) {
                if (ct[s] < 0) continue;
                const float* k = kv + (size_t) ct[s] * q_dim;
                double d = 0.0;
                for (int i = 0; i < q_dim; ++i) d += (double) Qh[i] * (double) k[i];
                logit[s] = d * (double) scale;
                mx = std::max(mx, logit[s]);
            }
            std::fill(ctx.begin(), ctx.end(), 0.0);
            double sum = 0.0;
            for (int s = 0; s < n_sel; ++s) {
                if (ct[s] < 0) continue;
                const double p = std::exp(logit[s] - mx);
                sum += p;
                const float* c_kv = kv + (size_t) ct[s] * q_dim;
                for (int c = 0; c < kv_lora; ++c) ctx[c] += p * (double) c_kv[c];
            }
            float* oh = out + ((size_t) t * n_head + h) * v_head;
            for (int v = 0; v < v_head; ++v) {
                if (sum == 0.0) {
                    oh[v] = 0.0f;
                    continue;
                }
                const float* wr = wv_b + ((size_t) h * v_head + v) * kv_lora;
                double acc = 0.0;
                for (int c = 0; c < kv_lora; ++c) acc += (double) wr[c] * (ctx[c] / sum);
                oh[v] = (float) acc;
            }
        }
    }
}

}  // namespace glmd::cpu
