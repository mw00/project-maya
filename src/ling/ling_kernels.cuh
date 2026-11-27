// src/ling/ling_kernels.cuh - Ling-3.2-air (bailingmoe3) CUDA kernels (ling_kernels.cu).
// Activations are F32 in ggml order (features fastest) unless noted otherwise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "hy4_kernels.cuh"

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
                    bool w_norm, float w_scale, int* ids_out, float* w_out, cudaStream_t s, int* err_flag = nullptr);

/// Write T tokens of kv_a_mqa output ([T][kv_lora + n_rot]) into split (lat, k_pe) buffers matching glmd::mla_dec
/// and glmd::mla_prompt_wmma: RMSNorm(kv_lora) -> lat (FP16 or 544-byte INT8 records), double-precision NORM RoPE
/// -> k_pe (FP16).  pos_dev non-null: per-token device positions; else p0 + t.
void kv_store_split(const float* kv_in, int T, int kv_lora, int n_rot, const float* kv_a_norm, float eps,
                    const int* pos_dev, int p0, float base, void* lat, bool lat_q8, uint16_t* k_pe, cudaStream_t s,
                    const int* rope_delta_dev = nullptr, const int* mrope_pos = nullptr, int mrope_hw = 24,
                    int mrope_stride = 0);

/// MLA query prep on qm [T][n_head][qk_head]:
///   - Applies double-precision NORM RoPE in place to qm[t][h][qk_nope .. qk_head).
///   - If q8n != nullptr: quantizes qm[t][h][0 .. qk_nope) into q8_heads layout (row h*T + t) for iq::mv_heads.
///   - If qm16 != nullptr: converts qm[t][h][0 .. qk_nope) to FP16 at stride qk_head for strided-batched cuBLAS.
void mla_q_prep(float* qm, int T, int n_head, int qk_head, int qk_nope, int n_rot, const int* pos_dev, int p0,
                float base, int8_t* q8n, float* d8n, uint16_t* qm16, cudaStream_t s,
                const int* rope_delta_dev = nullptr, const int* mrope_pos = nullptr, int mrope_hw = 24,
                int mrope_stride = 0);

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
                          uint16_t* q_nope16, uint16_t* q_pe16, cudaStream_t s, const int* mrope_pos = nullptr,
                          int mrope_hw = 24, int mrope_stride = 0);

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

/// Fast O(N) adaptive expert cache resolver (drop-in replacement for hy4::k::fetch_resolve):
/// Immediate O(1) fast-exit when n_miss == 0; single-pass warp-parallel victim selection when n_miss > 0.
void fetch_resolve_fast(const int* ids, int T, int n_used, const hy4::k::FetchLayer& fl, const hy4::k::FetchScratch& fs,
                        int k_gpu, bool cache, float decay, cudaStream_t s);

/// Fast 2-stage parallel argmax over n floats (vocab = 157184) -> *out_id (1 int on device).
/// scratch_val / scratch_idx must hold at least 128 floats / 128 ints.
void argmax_f32(const float* logits, int n, float* scratch_val, int* scratch_idx, int* out_id, cudaStream_t s);

/// Fused residual add + RMSNorm + Q8_0 row quantization (T >= 1, E <= 4096, E % 32 == 0).
/// Updates x[t] += a[t] (+ b[t]) in-place, writes normalized F32 to cur_out (optional), and Q8_0 to (xq, xd).
void add_rmsnorm_q8(float* x, const float* a, const float* w, int T, int E, float eps, float* cur_out, int8_t* xq,
                    float* xd, cudaStream_t s);
void add2_rmsnorm_q8(float* x, const float* a, const float* b, const float* w, int T, int E, float eps, float* cur_out,
                     int8_t* xq, float* xd, cudaStream_t s);

/// Pack routing outputs (ids, w, who) into one contiguous 96-byte device struct for single D2H transfer in kCpu mode.
struct RoutePack {
    int ids[8];
    float w[8];
    int who[8];
};
void pack_route_cpu(const int* ids, const float* w, const int* who, int n_used, RoutePack* dst, cudaStream_t s);

/// Portable WMMA split-K MLA decode (H = 32, kv_lora = 512, qk_rope = 64; supports T >= 1 and FP16 / INT8 KV).
/// Vectorized coalesced KV loads + register-accumulated WMMA (TILE=32 on sm_80+, TILE=16 on sm_70/75).
size_t mla_dec_wmma_part_bytes(int n_head, int max_T = 1);
void mla_dec_wmma_init();
bool mla_dec_wmma(const float* q_abs, const float* q, int q_stride, int q_pe_off, const void* lat, int lat_q8,
                  const uint16_t* k_pe, int n_sel_cap, const int* n_sel_dev, int T, int n_head, int kv_lora,
                  int qk_rope, float scale, float* part, float* ctx, cudaStream_t s);


/// Windowed KDA prep (W <= 4): snapshots conv_state into conv_bak (if non-null) and processes W sequential tokens.
void kda_prep_window(const float* q_in, const float* k_in, const float* v_in, const float* f_in, const float* conv_q,
                     const float* conv_k, const float* conv_v, float* conv_state, float* conv_bak, const float* ssm_a,
                     float lb, float eps, int W, int n_head, int head_dim, int d_conv, float* q_out, float* k_out,
                     float* v_out, float* g1_out, cudaStream_t s);

/// Roll back KDA conv_state from conv_bak to exact post-step-a state (1 <= a < W).
void kda_conv_rollback(const float* conv_bak, const float* q_in, const float* k_in, const float* v_in, int a,
                       int n_head, int head_dim, int d_conv, float* conv_state, cudaStream_t s);

/// Windowed register-resident KDA recurrence (W <= 4): reads state once, writes final W-step state to state_tmp
/// (leaving state untouched for exact rollback), writes W rows of quantized gated outputs to (xq_out, xd_out).
void kda_rec_window_q8(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw,
                       const float* state, float* state_tmp, const float* g2, const float* norm_w, float eps, int W,
                       int n_head, int head_dim, int8_t* xq_out, float* xd_out, cudaStream_t s);

/// Commit KDA recurrent state for `a` accepted tokens (1 <= a <= W): D2D copy state_tmp->state if a==W,
/// else exact in-register replay of steps 0..a-1 from untouched state.
void kda_commit_window(const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
                       const float* state_tmp, int a, int W, int n_head, int head_dim, cudaStream_t s);

/// Pack routing outputs for a window W <= 4 into one contiguous device struct for single D2H transfer in kCpu mode.
struct RoutePackWin {
    int ids[32];
    float w[32];
    int who[32];
};
void pack_route_win_cpu(const int* ids, const float* w, const int* who, int n_ent, RoutePackWin* dst, cudaStream_t s);

/// Fast 2-stage on-device top-1 + softmax confidence probability for a single logit row (writes 2 floats: {bits(id), prob}).
void top1_prob_f32(const float* logits, int n, float* scratch_val, int* scratch_idx, float* out_pair, cudaStream_t s);

/// Fast batched on-device argmax for W logit rows [W][n] -> out_ids[W].
void argmax_rows_f32(const float* logits, int W, int n, float* scratch_val, int* scratch_idx, int* out_ids,
                     cudaStream_t s);

/// Copy W contiguous rows from src [W][E] into ring buffer ring [R][E] at positions (pos0 + t) % R.
void store_hring(const float* src, int W, int E, float* ring, int R, int pos0, cudaStream_t s);
void set_step(int* d_step, int tok, int pos, int n_kv, int rope_delta, cudaStream_t s);
/// d_step: [tok, pos, n_kv, rope_delta]; on AMD (no tensor-core window kernel) set_step_win also writes each window
/// row's own cache length at d_step[kStepWinCounts + i] (kStepInts in all), for the row-by-row MLA decode
constexpr int kStepWinCounts = 4;
#if defined(MAYA_USE_HIP)
constexpr int kStepInts = 8;
#else
constexpr int kStepInts = 4;
#endif
void set_step_win(int* d_step, int* d_tok, int* d_pos, int t0, int t1, int t2, int t3, int W, int pos0, int rope_delta,
                  cudaStream_t s);

/// Per-256 Q8_K activation quantization matching maya::ds4::par::quant bit-for-bit (for --lane-parity / --repeatable).
void q8k_rows(const float* x, int rows, int k, int8_t* xq, float* xd, cudaStream_t s);

/// Startup bit-for-bit self-check of fused kernels (add_rmsnorm_q8, add2_rmsnorm_q8, swiglu_clamp_out, kda_prep_full).
/// Returns true on exact match; writes mismatch detail into `err` and returns false otherwise.
bool verify_fused_kernels(cudaStream_t s, std::string& err);

}  // namespace ling::k

