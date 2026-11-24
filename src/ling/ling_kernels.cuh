// src/ling/ling_kernels.cuh - Ling-3.2-air (bailingmoe3) CUDA kernels (ling_kernels.cu).
// Activations are F32 in ggml order (features fastest) unless noted otherwise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ling::k {

/// Fused KDA decode prep (1 token): causal conv1d + SiLU + history slide on q/k/v (conv_state [3][d_inner][d_conv-1],
/// matching glmb::kda_conv), ggml L2 norm on q/k per head, and full-width decay gate
/// g1 = lb * sigmoid(f * ssm_a[h]) where ssm_a holds positive exp(A_log) and f already includes dt_bias.
void kda_prep_full(const float* q_in, const float* k_in, const float* v_in, const float* f_in, const float* conv_q,
                   const float* conv_k, const float* conv_v, float* conv_state, const float* ssm_a, float lb, float eps,
                   int n_head, int head_dim, int d_conv, float* q_out, float* k_out, float* v_out, float* g1_out,
                   cudaStream_t s);

/// Fused KDA decode recurrence (1 token, head_dim == 128) + head RMSNorm * norm_w * sigmoid(g2) + Q8 quantization
/// matching hy4::k::q8_rows (xq_out [d_inner], xd_out [d_inner/32]).  beta_raw is pre-sigmoid.
/// raw_o_opt (optional): pre-norm recurrence output [d_inner] for parity seam dumps.
void kda_rec_q8(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
                const float* g2, const float* norm_w, float eps, int n_head, int head_dim, int8_t* xq_out,
                float* xd_out, float* raw_o_opt, cudaStream_t s);

/// Grouped sigmoid MoE router for T tokens (n_exp <= 512, n_group <= 16, n_used <= 16):
/// p = sigmoid(logits), sel = p + bias, top-2 sum per group, keep top n_group_used groups, top-n_used by sel,
/// weights = p / max(sum(p), 2^-14) * w_scale.  Deterministic lower-index tie-breaking matching ref/ling.py.
void router_grouped(const float* logits, const float* bias, int T, int n_exp, int n_used, int n_group, int n_group_used,
                    bool w_norm, float w_scale, int* ids_out, float* w_out, cudaStream_t s);

/// Write T tokens of kv_a_mqa output ([T][kv_lora + n_rot]) into split (lat, k_pe) buffers matching glmd::mla_dec
/// and glmd::mla_prompt_wmma: RMSNorm(kv_lora) -> lat (FP16 or 544-byte INT8 records), double-precision NORM RoPE
/// -> k_pe (FP16).  pos_dev non-null: per-token device positions; else p0 + t.
void kv_store_split(const float* kv_in, int T, int kv_lora, int n_rot, const float* kv_a_norm, float eps,
                    const int* pos_dev, int p0, float base, void* lat, bool lat_q8, uint16_t* k_pe, cudaStream_t s);

/// MLA query prep on qm [T][n_head][qk_head]:
///   - Applies double-precision NORM RoPE in place to qm[t][h][qk_nope .. qk_head).
///   - If q8n != nullptr: quantizes qm[t][h][0 .. qk_nope) into q8_heads layout (row h*T + t) for iq::mv_heads.
///   - If qm16 != nullptr: converts qm[t][h][0 .. qk_nope) to FP16 at stride qk_head for strided-batched cuBLAS.
void mla_q_prep(float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, const int* pos_dev, int p0,
                float base, int8_t* q8n, float* d8n, uint16_t* qm16, cudaStream_t s);

/// MLA per-head output gate: o[t][h][e] *= sigmoid(gate[t][h]), with optional fused Q8 or FP16 output.
void head_gate_out(float* o, const float* gate, int T, int n_head, int head_dim, int8_t* xq_out, float* xd_out,
                   uint16_t* out16, cudaStream_t s);

/// Post-SiLU clamped SwiGLU: h = min(silu(g), L) * clamp(u, -L, L) (L <= 1e-6: no clamp).
/// Writes any non-null outputs: h_opt (F32), (xq_out, xd_out) (Q8), out16 (FP16).
void swiglu_clamp_out(const float* g, const float* u, int rows, int n, float limit, float* h_opt, int8_t* xq_out,
                      float* xd_out, uint16_t* out16, cudaStream_t s);

/// Batch embedding lookup for T device token IDs (F32=0, F16=1, Q8_0=8, BF16=30).
bool emb_rows(const uint8_t* w, int type, int64_t k, const int* ids_dev, int T, float* out, cudaStream_t s);

/// MTP prologue: concat [rms(emb) * enorm ; rms(h) * hnorm] (2 * E values) -> optional F32 + Q8.
void concat_norm_eh(const float* emb, const float* enorm, const float* h, const float* hnorm, float eps, int E,
                    float* out_f32, int8_t* xq_out, float* xd_out, cudaStream_t s);

/// x[i] += a[i] + b[i] for i < n.
void add2(float* x, const float* a, const float* b, int64_t n, cudaStream_t s);

/// dst[i] = i for i < n.
void fill_iota(int* dst, int n, cudaStream_t s);

/// Tiled dense MLA prefill helpers:
/// Apply double-precision NORM RoPE on q_pe and write head-major FP16 q_nope16 [H][T][qk_nope], q_pe16 [H][T][n_rot].
void mla_q_split_rope_f16(const float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, int pos0, float base,
                          uint16_t* q_nope16, uint16_t* q_pe16, cudaStream_t s);

/// Dequantize BK rows of INT8 MLA latent cache starting at k0 -> FP16 [BK][kv_lora].
void lat_q8_to_f16(const void* lat, int k0, int BK, int kv_lora, uint16_t* out16, cudaStream_t s);

/// Initialize online softmax state m [H][T] = -inf, l [H][T] = 0, o_acc [H][T][v_head] = 0.
void mla_init_state(float* m, float* l, float* o_acc, int H, int T, int v_head, cudaStream_t s);

/// Online causal softmax over pre-scaled scores S [H][BQ][BK] -> P16 [H][BQ][BK], rescaling o_acc [H][T][v_head].
void mla_online_softmax(const float* S, int H, int T, int q0, int BQ, int k0, int BK, int pos0, float* m, float* l,
                        float* o_acc, int v_head, uint16_t* P16, cudaStream_t s);

/// Finalize tiled MLA output: out16 [T][H][v_head] = FP16((o_acc[h][t][d] / l[h][t]) * sigmoid(gate[t][h])).
void mla_o_finalize_gate(const float* o_acc, const float* l, const float* gate, int T, int H, int v_head,
                         uint16_t* out16, cudaStream_t s);

/// Compute max_e (bounds[e+1] - bounds[e]) into *out_dev (1 int).
void max_rows_from_bounds(const int* bounds, int n_slots, int* out_dev, cudaStream_t s);

}  // namespace ling::k

