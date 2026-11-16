// include/strata/kernels/glmd_route.hpp - the glm-dsa port B5.1: the decode router's selection and weight arithmetic
// on the host, the CPU twin of glm_fast.cu moe_route_kernel (sigmoid + exp_probs_b for selection only, stable top-k,
// normalised weights x routed_scaling_factor).  Header-only, no CUDA: tests/glm_dsa/glm_dsa_router_*_test.cpp check it
// against an F64 HF-style oracle on the CPU and the device kernel against both.
//
// glm-dsa (GLM-5.3 744B) routing, as HF modeling_glm_moe_dsa / DeepSeek-V3:
//   p_e   = sigmoid(logit_e)                      logit = ffn_gate_inp (F32) . x, 256 experts
//   sel_e = p_e + exp_probs_b_e                    the bias picks the experts, it never weights them
//   top-8 of sel (ties: the lower expert id)
//   w_i   = p_i / (sum_j p_j + 1e-20) * 2.5        expert_weights_norm, expert_weights_scale (from the geometry)
// The shared expert is added unscaled; there is no SwiGLU clamp (the geometry's +INF limit makes the kernels'
// fminf/fmaxf no-ops for every finite and infinite value).
//
// The normalisation differs from llama.cpp's (and Maya's): llama.cpp divides by max(sum, 6.103515625e-5) (ggml_clamp,
// 2^-14), HF by sum + 1e-20.  The two agree whenever sum >= 6.1e-5 (the 1e-20 moves the F64
// denominator by at most an ulp, far below the F32 weights' precision); a smaller sum needs every selected
// sigmoid below 6.1e-5 (every selected logit below about -9.7, 8 of 8): never seen on a real router.  Below that HF still
// normalises the weights to 2.5 while llama.cpp scales them down.  glm-dsa follows HF for glm-dsa (kRouteNormHf) and
// keeps Maya's clamp for Flash (kRouteNormClamp, bit-identical).
#pragma once

#include <cmath>
#include <cstdint>

namespace glmd {

/// The values of glm_fast.hpp's gf::kRouteNormClamp / gf::kRouteNormHf (the GPU test static_asserts they match).
constexpr int kRouteNormClamp = 0, kRouteNormHf = 1;

namespace cpu {

/// The device route's selection and weights for one router row, in its arithmetic: F32 sigmoid (expf), F32 selection
/// score p + bias, k rounds of argmax (ties: the lower index; a non-finite score ends the route, ids -1), the sum of the
/// k selected p in F64, inv = (float) (1 / D), w_i = (float) ((double) p_i * inv * w_scale).  D per `norm_mode` (see
/// above); norm_w false: inv = 1.  ids/w hold k entries.  false when a logit or a score is not finite (the kernel then
/// publishes an invalid route).
inline bool route_select(const float* logits, const float* bias, int E, int k, float w_scale, bool norm_w,
                         int norm_mode, int* ids, float* w) {
    if (E < 1 || E > 512 || k < 1 || k > 8) return false;
    float p[512], sel[512];
    bool ok = true;
    for (int e = 0; e < E; ++e) {
        p[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        sel[e] = bias ? p[e] + bias[e] : p[e];
        ok = ok && std::isfinite(logits[e]) && std::isfinite(sel[e]);
    }
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int e = 0; e < E; ++e)
            if (sel[e] > bv) {   // ascending e: a tie keeps the lower index
                bv = sel[e];
                bi = e;
            }
        ids[r] = bi >= 0 && std::isfinite(bv) ? bi : -1;
        if (ids[r] < 0) ok = false;
        else sel[bi] = -INFINITY;
    }
    if (!ok) return false;
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += (double) p[ids[i]];
    const float inv = (float) (!norm_w                       ? 1.0
                               : norm_mode == kRouteNormHf ? 1.0 / (sum + 1e-20)
                                                           : 1.0 / std::fmax(sum, 6.103515625e-5));
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) p[ids[i]] * inv * w_scale);
    return true;
}

/// The SwiGLU of the decode kernels with GLM's clamp (gate above, up both sides): with limit = +INF every fminf/fmaxf
/// returns its other operand, so h = silu(g) * u exactly (glm-dsa); Flash's limit is 10.
inline float swiglu_clamped(float g, float u, float limit) {
    const float gc = std::fmin(g, limit);
    const float uc = std::fmin(std::fmax(u, -limit), limit);
    return (gc / (1.0f + std::exp(-gc))) * uc;
}

}  // namespace cpu
}  // namespace glmd
