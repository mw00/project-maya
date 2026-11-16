// include/strata/kernels/glmd_ref.hpp - glm-dsa F32 reference kernels and their CPU twins (B3.4).
//
// Correctness first: these are the slow, obviously-right versions the fast path (B4) and the reference forward
// (B3.5) are checked against.  Every function exists twice with the same contract: glmd::ref_* runs on the GPU in
// F32 (device pointers, async on `stream`), glmd::cpu::* runs on the host with double accumulation and float
// storage.  tests/glm_dsa/glmd_ref_parity.cu compares the two; tests/glm_dsa/glmd_ref_cpu_test.cpp checks the CPU
// side against naive loops and the non-absorbed attention.  The rotation is src/kernels/cuda/glmd_rope.cuh.
//
// Per query token t at position pos[t] (GLM-5.3 sizes in brackets; GLM-5.3-SPEC.md dataflow steps 5-7):
//   indexer key     ik = rope_0..63( LayerNorm_{w,b,eps}( W_idx_k x ) )            [6144 -> 128, eps 1e-6]
//   indexer query   iq_h = rope_0..63( (W_idx_q_b q_lat)_h )                        [2048 -> 32 x 128]
//   head weights    w_h = (W_idx_proj x)_h / sqrt(idx_heads * idx_dim)               [6144 -> 32, / 64]
//   score           I(t, s) = sum_h w_h * relu(iq_h . ik_s) for s <= pos[t], -inf otherwise
//   top-k           the min(k, n_valid) best s, score descending, ties by the lower s, -1 padding  [k 2048]
//   absorb          Q_h = [ W_kb_h^T q_nope_h | q_pe_h ]                             [192 -> 512, + 64]
//   attention       l_s = (Q_h . [c_kv(s) | k_pe(s)]) * scale over the selected cells, p = softmax(l),
//                   ctx_h = sum_s p_s c_kv(s), out_h = W_vb_h ctx_h                   [512 -> 256]
// The index-key cache holds one key per position: cell s of ik_cache is position s.  llama.cpp also multiplies
// iq and ik by a normalized 128x128 Hadamard before the score; it is orthonormal, so the score is the same in exact
// arithmetic, and the reference leaves it out (it matters only to a quantized index cache).
//
// THE ATTENTION SCALE FOR GLM-5.3 IS 1/16 (glmd::kGlm53KqScale).  llama.cpp glm-dsa.cpp:
//     attn_factor_org = attn_factor * (1 + 0.1 * log(1 / freq_scale))
//     mscale          = attn_factor_org * (1 + 0.1 * rope_yarn_log_mul * log(1 / freq_scale))
//     kq_scale        = mscale * mscale / sqrt(n_embd_head_k),   n_embd_head_k = n_embd_head_k_mla() = key_length_mla
// GLM-5.3 has rope_type default (no scaling): freq_scale = 1, so both log terms are 0, and with no YaRN
// (ext_factor 0) and no rope.scaling.attn_factor key, attn_factor = yarn_attn_factor = rope_attn_factor = 1
// (llama-context.cpp), hence mscale = 1.  key_length_mla = qk_nope + qk_rope = 192 + 64 = 256, so
// kq_scale = 1 / sqrt(256) = 1/16 exactly.  HF agrees: scaling = qk_head_dim^-0.5 = 256^-0.5, with
// yarn mscale a no-op without a factor (GLM-5.3-SPEC.md, "attention softmax scale").  It is NOT 1/sqrt(qk_nope)
// (Maya's Flash rule, 1/sqrt(192)) and NOT 1/sqrt(kv_lora + qk_rope).  The scale is still passed explicitly so a
// mutation test can feed a wrong one.
//
// LAYOUT CONTRACTS (f32, ggml order - the first listed dimension is contiguous):
//   x       : (n_embd, T)                     residual-stream input of the layer, after attn_norm
//   q_lat   : (q_lora, T)                     RMS-normed query latent
//   W (any) : GGUF {in, out} -> W[o * in + i] (one output row contiguous)
//   ik      : (idx_dim, T) / ik_cache (idx_dim, n_kv)
//   iq      : (idx_dim, idx_heads, T)
//   w       : (idx_heads, T)
//   score   : (n_kv, T)
//   sel     : (top_k, T)                      cell indices, -1 padding
//   q       : (qk_nope + qk_rope, n_head, T)  [q_nope | q_pe], q_pe already rotated
//   wk_b    : GGUF attn_k_b {qk_nope, kv_lora, n_head} -> wk_b[(h * kv_lora + r) * qk_nope + c]
//   Q       : (kv_lora + qk_rope, n_head, T)  [q_abs | q_pe]
//   kv      : (kv_lora + qk_rope, n_cache)    cache rows [c_kv | k_pe], k_pe already rotated
//   cells   : (n_sel, T)                      -1 = padding (excluded from the softmax)
//   wv_b    : GGUF attn_v_b {kv_lora, v_head, n_head} -> wv_b[(h * v_head + v) * kv_lora + c]
//   out     : (v_head, n_head, T)
// Positions are a DEVICE array for the ref_* functions (graph-capture safe), a host array for cpu::*.
#pragma once

#include <cstdint>

namespace glmd {

/// GLM-5.3 absorbed-MLA softmax scale: 1 / sqrt(key_length_mla) = 1 / sqrt(192 + 64) (derivation above).
constexpr float kGlm53KqScale = 0.0625f;

// ---- GPU F32 reference kernels ------------------------------------------------------------------------------

/// NORM rope in place on dims [rot_off, rot_off + n_rot) of every row: x is (row_len, rows_per_token, T) and all
/// rows of token t use pos[t].  inv_freq is a device array of n_rot/2 (glmd::rope_inv_freq on the host).
void ref_rope_norm(float* x, int row_len, int rows_per_token, int T, int rot_off, int n_rot, const int* pos,
                   const float* inv_freq, void* stream);

/// y[o, t] = alpha * sum_i W[o * in + i] x[i, t].
void ref_gemv(const float* W, const float* x, float* y, int in, int out, int T, float alpha, void* stream);

/// ik = rope( LayerNorm(W_idx_k x) ) on dims 0..n_rot-1.  ln_w, ln_b: (idx_dim).
void ref_indexer_key(const float* x, const float* w_idx_k, const float* ln_w, const float* ln_b, float eps,
                     int n_embd, int idx_dim, int T, int n_rot, const int* pos, const float* inv_freq, float* ik,
                     void* stream);

/// iq = rope( W_idx_q_b q_lat ) per head on dims 0..n_rot-1.
void ref_indexer_query(const float* q_lat, const float* w_idx_q_b, int q_lora, int idx_heads, int idx_dim, int T,
                       int n_rot, const int* pos, const float* inv_freq, float* iq, void* stream);

/// w = (W_idx_proj x) / sqrt(idx_heads * idx_dim).
void ref_indexer_weights(const float* x, const float* w_idx_proj, int n_embd, int idx_heads, int idx_dim, int T,
                         float* w, void* stream);

/// score[s, t] = sum_h w[h, t] * relu(iq_h(t) . ik_cache[s]) for s <= pos[t] (and s < n_kv), -inf otherwise.
void ref_indexer_score(const float* iq, const float* ik_cache, const float* w, int idx_heads, int idx_dim, int T,
                       int n_kv, const int* pos, float* score, void* stream);

/// sel[:, t] = the min(top_k, finite count) best cells of score[:, t], score descending, ties by the lower cell,
/// padded with -1.  O(n_kv^2) per token: a reference, not a fast path.
void ref_topk(const float* score, int n_kv, int T, int top_k, int* sel, void* stream);

/// Q = [W_kb_h^T q_nope_h | q_pe_h] per head.
void ref_mla_absorb_q(const float* q, const float* wk_b, int n_head, int qk_nope, int qk_rope, int kv_lora, int T,
                      float* Q, void* stream);

/// The absorbed attention over cells (see above).  n_sel <= kRefMaxSel.  A row with no valid cell writes zeros.
constexpr int kRefMaxSel = 8192;
void ref_mla_attention(const float* Q, const float* kv, const int* cells, const float* wv_b, int n_head,
                       int kv_lora, int qk_rope, int v_head, int T, int n_sel, float scale, float* out,
                       void* stream);

// ---- CPU reference (double accumulation) --------------------------------------------------------------------

namespace cpu {

void rope_norm(float* x, int row_len, int rows_per_token, int T, int rot_off, int n_rot, const int* pos,
               const float* inv_freq);
void gemv(const float* W, const float* x, float* y, int in, int out, int T, float alpha);
void layer_norm(float* x, const float* w, const float* b, float eps, int n, int T);
void indexer_key(const float* x, const float* w_idx_k, const float* ln_w, const float* ln_b, float eps, int n_embd,
                 int idx_dim, int T, int n_rot, const int* pos, const float* inv_freq, float* ik);
void indexer_query(const float* q_lat, const float* w_idx_q_b, int q_lora, int idx_heads, int idx_dim, int T,
                   int n_rot, const int* pos, const float* inv_freq, float* iq);
void indexer_weights(const float* x, const float* w_idx_proj, int n_embd, int idx_heads, int idx_dim, int T,
                     float* w);
void indexer_score(const float* iq, const float* ik_cache, const float* w, int idx_heads, int idx_dim, int T,
                   int n_kv, const int* pos, float* score);
void topk(const float* score, int n_kv, int T, int top_k, int* sel);
void mla_absorb_q(const float* q, const float* wk_b, int n_head, int qk_nope, int qk_rope, int kv_lora, int T,
                  float* Q);
void mla_attention(const float* Q, const float* kv, const int* cells, const float* wv_b, int n_head, int kv_lora,
                   int qk_rope, int v_head, int T, int n_sel, float scale, float* out);

}  // namespace cpu

}  // namespace glmd
