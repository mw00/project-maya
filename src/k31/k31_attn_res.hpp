// src/k31/k31_attn_res.hpp - Kimi K3.1's Block Attention Residuals mix (docs/KIMI-K31.md section 2.2).
//
// At every site (before each layer's attention and FFN, and once before the final norm) the residual stream is a
// softmax mix of the banked block states and the current block's partial sum, per token:
//
//     s_i = sum_d w_d * v_{i,d} / sqrt(mean_d(v_i^2) + eps)     w = the site's fused res_score vector
//     p   = softmax_i(s)
//     mix = sum_i p_i * v_i                                      the RAW candidates; the norm only scores them
//
// all in float64 with one rounding to float32 (ref/kimi_k31.py res_mix).  The candidates are passed in the order
// the oracle stacks them: the banked blocks oldest first, then the partial sum.  Which states those are is the
// runner's bookkeeping (blocks pushed at layers 0, B, 2B, ... before that layer's attention updates the partial; a
// push restarts the partial), checked in tests/k31_attn_res_test.cu.  The state is float32 by contract (bf16 state
// measured KLD 0.014).  K3.1 mixes at most 9 candidates (8 blocks + the partial); the kernel takes up to 16.
#pragma once

#include <cstdint>

namespace strata::k31 {

inline constexpr int kK31MaxCand = 16;

struct AttnResArgs {
    const float* cand[kK31MaxCand] = {};   // candidate i: [n_tokens][n_embd] float32 (blocks oldest first, partial last)
    int n_cand = 0, n_tokens = 0, n_embd = 0;
    const float* score_w = nullptr;        // [n_embd]: the site's res_score (res_norm.weight * res_proj.weight)
    float eps = 1e-5f;                     // attention.layer_norm_rms_epsilon, as the GGUF holds it
    float* out = nullptr;                  // [n_tokens][n_embd]; may alias the partial (the last candidate)
    float* probs = nullptr;                // optional [n_tokens][n_cand] (seam dumps)
};

/// Returns false, launching nothing, outside the contract: n_cand outside 1..kK31MaxCand, n_embd < 1, n_tokens < 0,
/// missing buffers.  With one candidate the result is that candidate exactly.
bool attn_res_launch(const AttnResArgs& a, void* stream);

}  // namespace strata::k31
