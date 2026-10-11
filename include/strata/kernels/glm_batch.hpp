// include/strata/kernels/glm_batch.hpp - the glm5-next BATCHED kernels: a chunk of T prompt tokens through one
// layer at a time (src/core/glm_prefill.cu).  Same arithmetic as the one-token kernels of glm_fast.hpp, laid out
// per token ([T][...] row-major).  The projections themselves are not here: they are cuBLAS GEMMs (quantized
// weights dequantized to FP16, BF16 weights widened to F32) and llama.cpp's MMQ for the routed experts.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::glmb {

/// Launches of this family that failed so far (each is also printed): the prompt path errors out on growth.
int launch_errors();

/// dst[i] = float(src[i]) for BF16 bits.
void bf16_to_f32(const uint16_t* src, float* dst, int64_t n, cudaStream_t s);
/// dst[i] = half(src[i]) (FP16 bits).
void f32_to_f16(const float* src, void* dst, int64_t n, cudaStream_t s);
/// dst[i] = half(src[i]) for BF16 bits (FP16 bits out).
void bf16_to_f16(const uint16_t* src, void* dst, int64_t n, cudaStream_t s);
/// R[t][s][e] = emb[t][e] for the 4 streams.
void embed_rows(const float* emb, float* R, int T, int n_embd, cudaStream_t s);

/// The mHC write half (in place) and the read half's sum of squares: when block_out != nullptr,
///   R[t][d] = post[t][d] * block_out[t] + sum_s comb[t][d][s] * R[t][s];  then ss[t] = sum R[t]^2 (ss may be null).
void hc_update(const float* block_out, float* R, const float* post, const float* comb, float* ss, int T, int n_embd,
               cudaStream_t s);
/// The mHC read half's tail per token, from mix[t][24] (the hc_fn dots of R[t]) and ss[t]: pre/post/comb (the
/// Sinkhorn), x[t] = rms_norm(sum_s pre_s R[t][s]) * norm_w as F32 and FP16.  n_embd <= 4096.
struct HcFinishArgs {
    const float* mix = nullptr;
    const float* ss = nullptr;
    const float* R = nullptr;
    const float* w_scale = nullptr;
    const float* w_base = nullptr;
    const float* norm_w = nullptr;
    float norm_eps = 1e-5f, hc_eps = 1e-6f;
    int iters = 20, n_embd = 4096, T = 0;
    float* pre = nullptr;    // T x 4
    float* post = nullptr;   // T x 4
    float* comb = nullptr;   // T x 16
    float* x = nullptr;      // T x n_embd
    void* x16 = nullptr;     // T x n_embd FP16
};
void hc_finish(const HcFinishArgs& a, cudaStream_t s);

/// KDA's three causal convs over the chunk (+SiLU): out_r[t][c] from proj_r[t - hist .. t][c], the history from
/// conv_state ([3][d_inner][hist], oldest first) for t < hist.  conv_state is NOT advanced (kda_conv_state).
void kda_conv(const float* const proj[3], const float* const conv_w[3], const float* conv_state, float* const out[3],
              int T, int d_inner, int d_conv, cudaStream_t s);
/// The history slides to the chunk's last d_conv - 1 inputs.
void kda_conv_state(const float* const proj[3], float* conv_state, int T, int d_inner, int d_conv, cudaStream_t s);
/// The gated delta recurrence over the chunk, per head with its state in registers: q/k l2-normalised,
/// g1 = lb * sigmoid(-(g1_raw + dt_bias) * ssm_a[h]), beta = sigmoid(beta_raw), the output gate
/// rms_norm(o) * norm_w * sigmoid(g2_raw) -> out16 (FP16, T x d_inner).  head_dim 128.
void kda_rec(const float* q, const float* k, const float* v, const float* g1_raw, const float* dt_bias,
             const float* ssm_a, float lower_bound, const float* beta_raw, float* state, const float* g2_raw,
             const float* norm_w, float eps, int n_head, int T, void* out16, cudaStream_t s);
/// The same in three parts: q/k's l2 norms and the decay over every (token, head) at once (in place: q, k and g1_raw
/// become the normalised q, k and the decay), the recurrence with each head's value columns over kv_split blocks (a
/// column's state, delta and output need only its column; the key dims are shared) - its raw output over v - and the
/// output gate.  kv_split 4 or 8 (0: 4).  Equal to kda_rec up to float summation order.
void kda_rec_split(float* q, float* k, float* v, float* g1_raw, const float* dt_bias, const float* ssm_a,
                   float lower_bound, const float* beta_raw, float* state, const float* g2_raw, const float* norm_w,
                   float eps, int n_head, int T, void* out16, cudaStream_t s, int kv_split = 0);

/// DSA after the first projections, per token t at position p0 + t: q_a norm (-> qr F32 + FP16), kv_a norm into the
/// latent cache, the indexer key's layer norm and the compressor gate into their caches.
struct DsaPrepArgs {
    const float* qr_raw = nullptr; const float* q_a_norm = nullptr; float* qr = nullptr; void* qr16 = nullptr;
    int q_lora = 1536;
    const float* kv_raw = nullptr; const float* kv_norm = nullptr; uint16_t* lat = nullptr; int kv_lora = 512;
    int lat_q8 = 0;   // INT8 latent records (kv_lora codes + kv_lora / 32 FP16 scales) instead of FP16 rows
    // KV streaming (glm_kv_stream.hpp): the rows also go to the host copy, and to their block's VRAM slot when lat_table
    // names one (lat: the identity-layout cache - the prompt path's staging copy for a streamed layer - or null)
    uint16_t* lat_host = nullptr; uint16_t* lat_slots = nullptr; const int* lat_table = nullptr; int lat_page = 4;
    const float* ik_raw = nullptr; const float* k_norm_w = nullptr; const float* k_norm_b = nullptr;
    float* ik_cache = nullptr;
    const float* ig_raw = nullptr; float* ig_cache = nullptr;
    int idx_key = 128;
    int ring = 1 << 30;   // the ik / ig caches hold positions modulo ring (a multiple of kpool)
    int p0 = 0, T = 0;
    float eps = 1e-5f;
};
void dsa_prep(const DsaPrepArgs& a, cudaStream_t s);
/// The pooled keys of pools [pool0, pool0 + n) (their cells all in the caches, positions modulo ring).
void dsa_pool(const float* ik_cache, const float* ig_cache, const float* ape, float* pooled, int idx_key, int kpool,
              int pool0, int n, cudaStream_t s, int ring = 1 << 30);
/// score[t][p] = sum_h relu(iq[t]_h . pooled_p) * iw[t][h] for the pools visible at position p0 + t.
void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int p0, int kpool,
               int T, int max_vis, float* score, int score_ld, cudaStream_t s);
/// Per token: the top pools (all of them while they fit; else the stable top-k of the scores) as cells, then the
/// tail cells - cells[t][0 .. n_sel[t]), -1 = masked; rows of n_sel_max.
void dsa_select(const float* score, int score_ld, int p0, int kpool, int top_pools_max, int tail, int T,
                int n_sel_max, int* cells, int* n_sel, cudaStream_t s);
#if defined(STRATA_USE_HIP)
/// The same attention with Q in FP16 ([T][n_head][512]) on the WMMA units (rocWMMA, RDNA3/RDNA4): FP16 operands, F32
/// accumulation and softmax.  n_head % 16 == 0, kv_lora 512.
void mla_attn_f16q(const uint16_t* q16, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max,
                   int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8 = false);
/// The same attention with Q in F32 on wave32 WMMA intrinsics (gfx11/gfx12): the FP32 query and softmax probability
/// are each split into FP16 high + residual, both products through WMMA with F32 accumulation and online softmax.
/// n_head % 16 == 0, kv_lora 512. lat_q8 expands INT8 codes with one FP16 scale per 32 values as they are loaded.
void mla_attn_wmma2(const float* q_abs, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max,
                    int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8 = false);
/// True on the current device where mla_attn_wmma2 runs (gfx11/gfx12); mla_wmma2_default() is true on gfx12 (RDNA4),
/// where the prompt path prefers it over mla_attn_f16q.
bool mla_wmma2_supported();
bool mla_wmma2_default();
#endif
/// Absorbed MLA attention per token and head over the token's cells: ctx[t][h] = softmax(q_abs[t][h] . lat_c *
/// scale) . lat_c.  kv_lora 512, n_head % 16 == 0.
void mla_attn(const float* q_abs, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head,
              int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8 = false);

/// h[t] = rms(mean of the 4 streams of R[t]) * w - the final hidden state per row (the NextN block's input).
void head_rows(const float* R, const float* w, float eps, int T, int n_embd, float* h, cudaStream_t s);
/// out16[t] = [rms(emb[t]) * enorm, rms(h[t]) * hnorm] in FP16 (eh_proj's input, 2 n_embd per row).
void mtp_in_rows(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int T, int n_embd,
                 void* out16, cudaStream_t s);
/// x[t] = rms(in[t]) * w as F32 and FP16.  n_embd <= 4096, a multiple of 256.
void rms_rows(const float* in, const float* w, float eps, int T, int n_embd, float* x, void* x16, cudaStream_t s);

/// The router per token (router_sigmoid's arithmetic: sigmoid, bias selection, stable top-k, normalised weights).
void route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int T, int* ids,
           float* w, cudaStream_t s);
/// Per expert: counts[e] and the stable rank (in (t, j) order) of every routed entry (rank[t*k + j]).
void expert_count(const int* ids, int n, int n_expert, int* counts, int* rank, cudaStream_t s);
/// dst[i] = src[i] by the SMs: dst may be pinned host memory, written over the bus without a copy engine (a small
/// readback that must not queue behind a large device-to-host copy on another stream).
void copy_i32(const int* src, int* dst, int n, cudaStream_t s);
/// row = base[ids[i]] + rank[i]: row_tok[row] = i / k, pos[i] = row.
void expert_scatter(const int* ids, const int* rank, const int* base, int n, int k, int* row_tok, int* pos,
                    cudaStream_t s);
/// h[r][c] = silu(min(gu[r][c], limit)) * clamp(gu[r][n_ff + c], +-limit), rows of [gate | up].
void swiglu_rows(const float* gu, float* h, int rows, int n_ff, float limit, cudaStream_t s, int ld_h = 0);
// (ld_h: h's row stride, n_ff when 0; h = gu with ld_h = 2 n_ff writes each row's result over its gate half)
/// The same from separate gate/up arrays, FP16 out.
void swiglu_f16(const float* gate, const float* up, void* h16, int64_t n, float limit, cudaStream_t s);
/// ffn[t] = sum_j w[t][j] * out[pos[t*k + j]] + sh[t] (plan order, like the one-token combine).
/// ffn[t] += sum_j w[t][j] * out[pos[t][j] - base] over the routes j of t whose sorted row pos[t][j] is in [lo, hi):
/// a window of expert rows added as it fills (out holds sorted rows base, base + 1, ...).
void moe_combine_add(const float* out, int base, int lo, int hi, const int* pos, const float* w, int T, int k, int n_embd,
                     float* ffn, cudaStream_t s);
void moe_combine(const float* out, const int* pos, const float* w, const float* sh, int T, int k, int n_embd,
                 float* ffn, cudaStream_t s);

}  // namespace strata::kernels::glmb
