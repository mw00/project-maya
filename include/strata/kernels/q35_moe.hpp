// include/strata/kernels/q35_moe.hpp - the MoE FFN pieces of a qwen35moe model (Qwen3.6-35B-A3B): routing, the
// expert grouping of the (token, expert) pairs, the weighted combine.  The expert products are llama.cpp's MMQ
// (prefill/moe_mmq.hpp) over the GGUF's stacked expert tensors.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Per row of logits (T rows of n_exp): softmax, the top k (ties to the lower id), their weights renormalized to
/// sum 1 then times `scale` -> ids[T][k], w[T][k].  (llama.cpp build_moe_ffn: SOFTMAX gating, norm_w)
void q35_moe_route(const float* logits, int T, int n_exp, int k, float scale, int32_t* ids, float* w, void* stream);

/// The same with the logit rows `ld` apart (the router's matrix carries an extra row: the shared expert's gate).
void q35_moe_route_ld(const float* logits, int ld, int T, int n_exp, int k, float scale, int32_t* ids, float* w,
                      void* stream);
/// q35_moe_combine with the shared gate's logits `sh_ld` apart (sh_logit[t * sh_ld]).
void q35_moe_combine_ld(const float* y, const int32_t* pos_of_pair, const float* w, int T, int k, int E,
                        const float* sh_logit, int sh_ld, const float* sh, float* out, bool add, void* stream);

/// The pairs (t, j) grouped by expert: bounds[e] .. bounds[e + 1] are expert e's positions; tok_of_pos[pos] = t,
/// pos_of_pair[t * k + j] = pos; max_rows[0] = the most pairs one expert has.  scratch: n_exp + 1 ints.
void q35_moe_group(const int32_t* ids, int T, int k, int n_exp, int32_t* bounds, int32_t* tok_of_pos,
                   int32_t* pos_of_pair, int32_t* max_rows, int32_t* scratch, void* stream);

/// out[t][e] (+)= sum_j w[t][j] * y[pos_of_pair[t * k + j]][e] + gate_sig[t] * sh[t][e] (gate_sig: sigmoid of the
/// shared expert's gate logit, sh the shared expert's output; both may be null).  add: accumulate into out.
void q35_moe_combine(const float* y, const int32_t* pos_of_pair, const float* w, int T, int k, int E,
                     const float* sh_logit, const float* sh, float* out, bool add, void* stream);

}  // namespace strata::kernels
