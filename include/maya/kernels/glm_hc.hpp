// include/maya/kernels/glm_hc.hpp - the glm5-next mHC (manifold-constrained hyper-connections).
//
// Phase 2 of docs/GLM5-FLASH.md.  The qwen4exp gated residual (`gr.hpp`) collapses the stream stack through
// a low-rank bottleneck; glm5-next's mHC replaces that with a direct projection to (2+hc)*hc mixing values,
// three affine gates, and a Sinkhorn normalization that runs PER TOKEN - the reference is
// `ref/glm.py::hc_pre` / `hc_sinkhorn` / `hc_post`, transcribed from
// `ref/upstream-llama.cpp/glm5-next.cpp` L371-548.  The math, for one token, R the (hc, n_embd) stack:
//
//     flat      = R flattened, stream-major (element (e, s) at s*n_embd + e)
//     inv       = rsqrt(mean(flat^2) + norm_eps)      ONE rms over the WHOLE stack - the four streams are
//                                                     normalized JOINTLY, not per stream (gr's rival reading)
//     mixes[m]  = sum_k w_fn[m*hc_dim + k] * flat[k] * inv     (hc_mix_dim = (2+hc)*hc values)
//     pre[s]    = sigmoid(mixes[s]*scale[0] + base[s]) + hc_eps
//     post[s]   = 2 * sigmoid(mixes[hc+s]*scale[1] + base[hc+s])
//     comb_raw[d, s] = mixes[2hc + d + hc*s]*scale[2] + base[2hc + d + hc*s]     SRC-MAJOR rows
//     comb      = sinkhorn(comb_raw): softmax over dst, +hc_eps, normalize each dst row, then
//                 (iters-1) x [normalize each src column, normalize each dst row]
//     mixed[e]  = sum_s pre[s] * R[s*n_embd + e]
//
// and the write-back after the mixer ran:
//
//     out[d*n_embd + e] = block_out[e] * post[d] + sum_s comb[d, s] * residual[s*n_embd + e]
//
// WHERE THIS DIFFERS FROM `gr.hpp`, because the two are easy to blur:
//   * the rms norm covers the whole stack JOINTLY (gr does one PER STREAM) - and the norm is applied to
//     the projection's INPUT only; `mixed` sums the RAW streams scaled by the gates;
//   * there is no bottleneck rank and no inject vector: the write-back mixes the OLD streams by the
//     Sinkhorn matrix and adds the gated block output to every stream;
//   * the Sinkhorn is data-dependent per token, so it is a device kernel here, not pack-time preprocessing;
//   * the pre-gate gets a `+hc_eps` FLOOR (ggml_scale_bias(pre, 1, eps)); the post-gate is `2*sigmoid`
//     with no floor.
//
// PRECISION CONTRACT: F32 weights and F32 activations throughout, matching `ref/glm.py` on the synthetic
// model.  The real model's hc_* weights are BF16 in the conversion; whether they ride the pack's
// BF16-promoted path (like gr's) is a capture-time variant to settle in Phase 3 wiring, and the parity
// harness pins the F32 contract first so the refinement has a fixed reference.
#pragma once

#include <cstdint>
#include <cstddef>

namespace maya::kernels {

struct GlmHcShapes {
    int64_t n_embd = 0;
    int64_t hc = 0;            // the architecture fixes 4; the kernels stay generic in it
    int64_t sinkhorn_iters = 0;

    int64_t hc_dim() const { return hc * n_embd; }
    int64_t hc_mix_dim() const { return (2 + hc) * hc; }
};

/// One float: the joint-stack inverse rms, computed by its own kernel because the mixes GEMV
/// consumes it.  Carved by the caller once, like every other kernel workspace (P2.T10).
struct GlmHcWorkspace {
    float* inv_rms = nullptr;
    size_t bytes = 0;
};

size_t glm_hc_workspace_bytes(const GlmHcShapes& s);
void glm_hc_workspace_init(const GlmHcShapes& s, void* base, GlmHcWorkspace& ws);

/// The read half.  R is (hc*n_embd) f32, stream-major; w_fn is (hc_mix_dim * hc_dim) with
/// `w_fn[m*hc_dim + k]` contiguous in k (the GGUF {hc_dim, hc_mix_dim} manifest layout, so the
/// GEMV is warp-per-output-row over a contiguous reduction - gr.cu's mapping, no transposes);
/// w_scale is (3,), w_base is (hc_mix_dim,).  Outputs: mixed (n_embd), pre (hc), post (hc),
/// comb (hc*hc) with comb[d*hc + s].
///
/// `norm_eps` is the model's rms epsilon (the norm), `hc_eps` the hyper-connection epsilon (the
/// gate floor and every Sinkhorn divisor's guard) - two different constants in the config.
void glm_hc_pre(const float* R, const float* w_fn, const float* w_scale, const float* w_base,
                float norm_eps, float hc_eps, const GlmHcShapes& s, GlmHcWorkspace& ws,
                float* mixed, float* pre, float* post, float* comb, void* stream);

/// The write half.  block_out (n_embd), residual (hc*n_embd), post (hc), comb (hc*hc, d*hc + s).
/// out must not alias residual (the engine writes into a fresh stack, as hc_post's concat does).
void glm_hc_post(const float* block_out, const float* residual, const float* post, const float* comb,
                 const GlmHcShapes& s, float* out, void* stream);

}  // namespace maya::kernels
