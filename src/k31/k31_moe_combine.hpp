// src/k31/k31_moe_combine.hpp - Kimi K3.1's latent MoE combine (layers 1-92).
//
// After the 16 routed experts ran in the 3584-wide latent (z = routed_expert_down_proj x), ref/kimi_k31.py moe_ffn
// does, per token:
//
//     moe    = float32( sum_i w_i * part_i )                 float64 accumulation, pick by pick       (ffn_moe_out)
//     normed = float32( float32(moe / sqrt(mean(moe^2) + eps)) * routed_norm )  the mean in float64  (ffn_routed_normed)
//     r      = routed_expert_up_proj @ normed                3584 -> 7168, a GEMV                     (ffn_routed_out)
//     out    = float32( r + shared )                         shared experts on the full-width x       (ffn_out)
//
// The norm sits AFTER the weighted sum, and it couples all 16 picks: the combine cannot start before every pick's
// part (including the CPU lane's) is in.  This file holds the sum + norm kernel and the residual add; the up GEMV is
// the engine's ordinary dense GEMV (iq::mv / gguf::mv / cuBLAS with CUBLAS_DEFAULT_MATH).  eps is the file's
// attention.layer_norm_rms_epsilon (1e-5 released) as the float the GGUF holds.
// Gated by tests/k31_moe_combine_test.cu against tools/kimi_k31_moe_dump.py (the oracle's seams on the fixture) and a
// float64 reference at K3.1's shape; the "picks truncated to 8" mutation must fail it.
#pragma once

#include "k31_route.hpp"

#include <cstdint>

namespace strata::k31 {

inline constexpr int kK31MaxLatent = 8192;   // the combine's shared row (latent floats)

struct CombineArgs {
    const float* parts = nullptr;    // [n_tokens][k][latent]: pick i's down output, in the router's rank order
    const float* w = nullptr;        // [n_tokens][k]: the router's weights (k31_route), same order
    const float* norm_w = nullptr;   // [latent]: ffn_routed_norm (routed_expert_norm)
    int n_tokens = 0, k = kPicks31, latent = 0;
    float eps = 1e-5f;
    float* moe_out = nullptr;        // optional [n_tokens][latent]: the routed sum before the norm (seam dumps)
    float* normed = nullptr;         // [n_tokens][latent]
};

/// Sum + norm.  Returns false, launching nothing, outside the contract: k outside 1..kK31MaxPicks, latent outside
/// 1..kK31MaxLatent, n_tokens < 0, missing buffers.
bool moe_combine_launch(const CombineArgs& a, void* stream);

/// out[i] = a[i] + b[i] (float32), the routed + shared residual; out may alias a or b.
bool add_launch(const float* a, const float* b, float* out, int64_t n, void* stream);

}  // namespace strata::k31
