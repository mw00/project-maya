// src/k31/k31_kda.hpp - Kimi K3.1's KDA layer (the 69 linear-attention layers), a thin adapter over the shared
// library's GLM KDA kernels (include/strata/kernels/glm_kda.hpp) and the GDN l2 norm (gdn.hpp).
//
// Per KDA layer, with x the layer input (attn_norm) and the five GEMVs done by the caller:
//
//     q, k, v = silu(causal_conv4(attn_{q,k,v} x))          glm_kda_conv, history carried in conv_state
//     q, k    = q / sqrt(|q|^2 + 1e-6) per head             gdn_l2_norm (the l2 eps, never the rms eps)
//     g1      = -5 * sigmoid(exp(A_log) * (f_b(f_a x) + dt_bias))
//                                                           k31 prep (the dt_bias add) + glm_kda_gate, ssm_a =
//                                                           -exp(A_log) sliced by the converter to the 96 KDA heads
//                                                           (the checkpoint stores 128, the tail +0.0)
//     beta    = sigmoid(ssm_beta x)                         k31 prep, the sigmoid in float64 as the oracle's
//     o       = delta rule over T tokens, per head          glm_kda_recurrence, state carried in ssm_state;
//               (S = diag(exp g1) S; S += k (beta (v - S^T k))^T; o = S^T q / sqrt(128))
//     gated   = rms_norm(o, ssm_norm) * sigmoid(ssm_g x)    glm_kda_out_gate (K3.1's g2 is one full-rank GEMV)
//
// then attn_output (12288 -> 7168) is an ordinary GEMM on `gated`.  Every buffer is GLM's layout: features fastest,
// head h at columns [h*head_dim, (h+1)*head_dim), so the GGUF tensors and the GEMV outputs go in with no transposes
// and no copies.  The one pass the GLM signature adds is the dt_bias add (glm_kda_gate takes the biased gate).
//
// Precision (measured by tests/k31_kda_test.cu): glm_kda.cu computes in float32 with __expf (the conv's SiLU, the
// gate's sigmoid, the decay, the output gate) and rsqrtf, and keeps the state in float32; the oracle keeps it in
// float64.  The test holds the composed layer to the oracle at a measured relative bound.
#pragma once

#include <cstdint>

namespace strata::k31 {

inline constexpr int kKdaHeads = 96;            // linear_attn_config.num_heads
inline constexpr int kKdaHeadDim = 128;         // head_dim (dk = dv)
inline constexpr int kKdaConv = 4;              // short_conv_kernel_size
inline constexpr float kKdaGateLowerBound = -5.0f;
inline constexpr float kKdaL2Eps = 1e-6f;       // fla's l2norm eps (q, k); the output norm uses the model's rms eps
inline constexpr int kKdaMaxHeadDim = 256;      // glm_kda_recurrence: one thread per state column, 256 threads
inline constexpr int kKdaMaxConv = 17;          // glm_kda_conv: 16 history taps staged in registers

struct KdaLayerArgs {
    int n_tokens = 0, n_head = 0, head_dim = 0, d_conv = 0;
    float lower_bound = kKdaGateLowerBound;
    float eps_l2 = kKdaL2Eps;
    float eps_rms = 0.0f;                       // the model's rms epsilon (the output norm)
    // the GEMV outputs, [n_tokens][d_inner] (beta_pre [n_tokens][n_head]); read only
    const float* xq = nullptr;
    const float* xk = nullptr;
    const float* xv = nullptr;
    const float* f_pre = nullptr;               // ssm_f_b (ssm_f_a x), before dt_bias
    const float* beta_pre = nullptr;            // ssm_beta x
    const float* g2 = nullptr;                  // ssm_g x
    // weights
    const float* conv_w[3] = {nullptr, nullptr, nullptr};   // ssm_conv1d_{q,k,v} as stored: [d_inner][d_conv],
                                                            // taps oldest first (the last multiplies token t)
    const float* dt_bias = nullptr;             // [d_inner]
    const float* ssm_a = nullptr;               // [n_head], -exp(A_log[:n_head])
    const float* norm_w = nullptr;              // ssm_norm [head_dim]
    // the sequence's state, updated in place (zero both for a fresh sequence)
    float* conv_state[3] = {nullptr, nullptr, nullptr};     // [d_inner][d_conv-1]: the last inputs, oldest first
    float* ssm_state = nullptr;                 // [n_head][head_dim][head_dim], S[h] row-major, row = KEY axis
    // outputs (caller-owned, [n_tokens][d_inner] unless noted); each is also a seam the parity test reads
    float* q = nullptr;                         // silu(conv) then l2-normalised
    float* k = nullptr;                         // silu(conv) then l2-normalised
    float* v = nullptr;                         // silu(conv)
    float* g1 = nullptr;                        // the decay gate (log space, in (lower_bound, 0))
    float* beta = nullptr;                      // [n_tokens][n_head]
    float* o = nullptr;                         // the recurrence's output (1/sqrt(head_dim) included)
    float* gated = nullptr;                     // attn_output's input; holds the biased gate until the last step
};

/// Runs the layer on `stream`.  Returns false, launching nothing, outside the contract: n_tokens < 0, n_head < 1,
/// head_dim outside 1..kKdaMaxHeadDim, d_conv outside 2..kKdaMaxConv, lower_bound >= 0, eps_l2 or eps_rms not > 0,
/// a missing buffer, or a CUDA error already pending.  n_tokens == 0 is a no-op (state untouched).  The output
/// buffers must not overlap each other or the inputs.  Also false if a launch reports an error.
bool kda_layer_launch(const KdaLayerArgs& a, void* stream);

}  // namespace strata::k31
