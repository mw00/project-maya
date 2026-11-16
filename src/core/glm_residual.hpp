// src/core/glm_residual.hpp - the residual seams of the fast decode path (glm-dsa B2.3).
//
// glm5-next (Flash) carries a 4-stream mHC residual; glm-dsa (hc = 1, Glm5Geometry::mhc() false) one plain pre-norm
// residual vector.  The fast path moves the residual through four seams, and these helpers are the only place that
// knows which kind it is:
//   res_embed  the embedding row -> R            mHC: gf::embed_streams (Maya's call, unchanged)   plain: R = emb
//   res_read   R (+= block_out) -> x, xq          plain only: gf::rms_q8 with the add (as fast_mtp already does)
//   res_close  R += block_out (before a hop)      plain only: the last layer's FFN output, glmres::add
//   res_head   R -> x, xq for the output head     mHC: gf::head_prep (unchanged)   plain: gf::rms_q8(R, null, norm)
// The hop between parts and the speculative hop copy res_hop_bytes(g) = residual_floats() * 4 bytes (Flash 4 x E,
// glm-dsa E = 24 KB).  The mHC per-layer read/write (gf::hc / gf::hc_post) stays inside Maya's fast_layers; glm-dsa
// runs Glm5Model::fast_layers_plain (glm_residual.cu) instead, so it never calls head_prep, embed_streams or hc.
#pragma once

#include "strata/core/glm_model.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace strata::core::glmres {

/// y[i] += x[i] for i < n (the new plain-residual add; any n > 0).
void add(float* y, const float* x, int n, cudaStream_t s);

/// The embedding row emb (n_embd floats, device) into the residual R at the start of a token.
void res_embed(const Glm5Geometry& g, const float* emb, float* R, cudaStream_t s);
/// Plain residual read: R += block_out (when not null), then x = rms(R) * norm_w and xq = q8_1(x).
void res_read(float* R, const float* block_out, const float* norm_w, float eps, int n_embd, float* x, void* xq,
              cudaStream_t s);
/// Plain residual close: R += block_out (the part's last FFN output), so a hop or the head reads a complete R.
void res_close(float* R, const float* block_out, int n_embd, cudaStream_t s);
/// The head's prologue: x = rms(R) * output_norm (mHC: of the stream mean), xq = q8_1(x).
void res_head(const Glm5Geometry& g, const float* R, const float* norm_w, float* x, void* xq, cudaStream_t s);
/// Bytes of the residual a part hands to the next one (and the speculative hop slots hold).
inline size_t res_hop_bytes(const Glm5Geometry& g) { return (size_t) g.residual_floats() * sizeof(float); }

}  // namespace strata::core::glmres
