// include/strata/kernels/dense_kernels.hpp - the kernels a DENSE qwen35 model (Qwen3.8-27B) needs and the MoE
// engine does not have.
//
// The GDN recurrence, the quantized GEMVs (native_mmvq), RoPE, the per-head RMSNorm and the attention output gate
// already exist and are reused by src/core/dense_model.cpp.  What was missing, because the MoE model routes its
// attention through the QSA indexer and has no plain FFN:
//
//   * an RMSNorm that writes to a second buffer          (dense_rms_norm)
//   * SwiGLU                                             (dense_swiglu)
//   * an F32 GEMV for the two tiny GDN projections       (dense_gemv_f32)   ssm_alpha / ssm_beta, n_out = 48
//   * a K/V cache and full causal attention over it      (dense_kv_append, dense_attn_decode)
//
// The attention is a split-K flash decode: one block per (head, chunk of 256 cells) keeps an online softmax
// (m, l, acc) per warp, the warps are merged inside the block, and a second kernel merges the chunks.  Every
// query head reads the K/V head `h / (n_head / n_kv)` (GQA: contiguous groups, the same map qsa.hpp pins).
//
// KV layout: [n_kv][max_ctx][head_dim] IEEE half, so one head's cells are contiguous.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w        (w NOT shifted: the GGUF already holds 1 + w)
void dense_rms_norm(const float* x, const float* w, float* out, int rows, int cols, float eps, void* stream);

/// y[h, :] = rms(o[h, :]) * w * silu(z[h, :]) per 128-wide head (Qwen3.5 gated delta net closing norm; SiLU, not sigmoid)
void dense_gdn_out_norm_silu(const float* o, const float* z, const float* w, float* y, int heads, int head_dim, float eps,
                             void* stream);

/// out[i] = silu(gate[i]) * up[i]
void dense_swiglu(const float* gate, const float* up, float* out, int64_t n, void* stream);

/// y[r] = sum_i W[r * n_in + i] * x[i]           W is a GGUF matrix stored as F32: ne0 = n_in is the contiguous axis
void dense_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);

/// dst[0..n) = value   (the RoPE position of every row)
void dense_fill_i32(int32_t* dst, int n, int32_t value, void* stream);

/// Writes this token's K and V (n_kv, head_dim) f32 into cell `pos` of every head's cache as half.
void dense_kv_append(uint16_t* k_cache, uint16_t* v_cache, const float* k, const float* v, int pos, int n_kv,
                     int head_dim, int max_ctx, void* stream);

/// Bytes of scratch dense_attn_decode needs for a cache of `max_ctx` cells.
uint64_t dense_attn_scratch_bytes(int n_head, int head_dim, int max_ctx);

/// out (n_head, head_dim) = softmax(q . K^T * scale) V over cells [0, n_ctx), n_ctx >= 1 (the current token's
/// cell is already appended).  q is (n_head, head_dim) f32, already normalised and rotated.
void dense_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, float* scratch,
                       int n_head, int n_kv, int head_dim, int n_ctx, int max_ctx, float scale, void* stream);

}  // namespace strata::kernels
