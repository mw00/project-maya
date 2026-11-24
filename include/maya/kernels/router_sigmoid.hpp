// include/maya/kernels/router_sigmoid.hpp - the glm5-next MoE router (sigmoid + selection bias).
//
// Phase 2 of docs/GLM5-FLASH.md.  The qwen4exp router (`router_top10.hpp`) is softmax over all
// experts with renormalised weights; glm5-next is DeepSeek noaux_tc, whose semantics
// (`ref/glm.py::moe_ffn`, transcribed from llama-graph.cpp's build_moe_ffn) are easy to blur in
// exactly one place, and the parity harness exists because of it:
//
//     p        = sigmoid(logits)              UNBIASED - this is what the weights are made of
//     sel      = p + bias                     the bias enters the SELECTION ONLY (ffn_exp_probs_b)
//     ids      = stable argsort(-sel)[:k]     ties break toward the LOWER index
//     w        = gather(p, ids)
//     w       /= max(sum(w), 6.103515625e-5)  the 2**-14 clamp, on the renormalisation
//     w       *= w_scale                      routed_scaling_factor (2.5 on the real model)
//
// THE TRAP: applying the bias to the weights too.  DeepSeek's own reference adds the bias for the
// top-k and then uses the UNBIASED sigmoid values as weights; llama.cpp keeps them separate
// ("leave probs unbiased as it's later used to get expert weights"), and so does the oracle.
#pragma once

#include <cstdint>

namespace maya::kernels {

/// logits (n_tokens, n_expert) f32; `bias` may be null (then sel == p); `norm_w` is the config's
/// norm_topk_prob.  ids (n_tokens, k) and weights (n_tokens, k) are device buffers, the same
/// contract the doorbell and the expert pool already consume from router_top10.
void router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k,
                    float w_scale, bool norm_w, int* ids, float* weights, void* stream);

}  // namespace maya::kernels
