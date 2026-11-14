// src/k31/k31_route.hpp - Kimi K3.1's MoE router: 16 of 896 experts, sigmoid scores, noaux_tc selection bias.
//
//     p      = sigmoid(logits)               float64, rounded to float32 (ref/kimi_k31.py's own sigmoid, so the
//                                            scores - and every tie - are the oracle's)
//     sel    = float32(p + bias)             the bias (exp_probs_b, e_score_correction_bias) PICKS; it never weights
//     ids    = the k largest sel, descending; ties go to the LOWER expert id (ggml_top_k / ref top_k_desc)
//     w      = p[ids] / (sum(p[ids]) + eps) * scale     summed in float64 in rank order, rounded once to float32
//
// eps is Moonshot's 1e-20 (kK31RenormEps), not GLM's 2^-14 clamp (router_sigmoid.cu): the two differ only when
// every selected sigmoid is tiny, and the golden vectors carry such a case.  scale = routed_scaling_factor (1.0
// released).  A NaN score ranks after every -inf, ties by lower id, as the oracle's stable argsort on -score puts
// NaN last (a picked NaN's weight is NaN, which the caller's checks see).  One block of up to 1024 threads a token,
// one thread an expert, so E <= 1024; k <= kK31MaxPicks.  The grid is n_tokens blocks (CUDA allows 2^31-1; a HIP
// build should chunk calls at 65535 tokens until its grid limit is checked).
// Not router_sigmoid (it exits above 512 experts and clamps at 2^-14) and not DeepSeek's route_kernel
// (sqrt-softplus scores, rows kPicks = 8 apart).  Gated by tests/k31_route_test.cu against tools/kimi_k31_golden.py.
#pragma once

#include <cstdint>

namespace strata::k31 {

inline constexpr int kPicks31 = 16;          // num_experts_per_token (released)
inline constexpr int kK31MaxExperts = 1024;  // one thread an expert
inline constexpr int kK31MaxPicks = 32;      // the selection loop's bound (16 used)
inline constexpr double kK31RenormEps = 1e-20;   // a double, as the oracle adds it (1e-20f is 9.99999968e-21)
static_assert(kPicks31 >= 16, "K3.1 routes 16 experts a token; no structure here may assume 8");
static_assert(kPicks31 <= kK31MaxPicks, "kPicks31 within the kernel's bound");

struct RouteArgs {
    const float* logits = nullptr;   // [n_tokens][n_expert], one token's logits contiguous
    const float* bias = nullptr;     // [n_expert] or null (then sel = p)
    int n_tokens = 0, n_expert = 0, k = kPicks31;
    float scale = 1.0f;              // routed_scaling_factor
    bool norm = true;                // expert_weights_norm (norm_topk_prob)
    double eps = kK31RenormEps;      // the renormalisation's + eps
    int32_t* ids = nullptr;          // [n_tokens][k], rank order (largest sel first)
    float* weights = nullptr;        // [n_tokens][k], same order
    float* probs = nullptr;          // optional [n_tokens][n_expert]: p (for seam dumps), may be null
};

/// Launch on `stream`.  Returns false, launching nothing, when the arguments are outside the kernel's contract:
/// n_expert outside 1..1024, k outside 1..min(n_expert, kK31MaxPicks), n_tokens < 0, missing buffers, or a CUDA
/// error already pending (refused, not blamed on this launch).  n_tokens == 0 is a no-op.  Also false if the
/// launch itself reports an error.
bool route_launch(const RouteArgs& a, void* stream);

}  // namespace strata::k31
