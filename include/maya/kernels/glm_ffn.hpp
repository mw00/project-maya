// include/maya/kernels/glm_ffn.hpp - the glm5-next FFN activation: the clamped SwiGLU.
//
// Phase 2 of docs/GLM5-FLASH.md.  glm5-next's FFNs (the 3 dense lead layers, the routed experts,
// and the shared expert) all use the GLM-family clamped SwiGLU, whose kernel semantics
// (`ref/glm.py::swiglu_clamp`, transcribed from ggml's ggml_swiglu_clamp, cpu-ops.cpp L3449) are
// easy to read past:
//
//     gate = min(gate, limit)            clamped ABOVE ONLY - there is no lower clamp on the gate
//     up   = clamp(up, -limit, +limit)   clamped on BOTH sides
//     out  = silu(gate) * up
//
// The rival reading - clamping the gate on both sides, or clamping silu(gate) instead of the raw
// gate - produces plausible numbers and different ones, which is what the parity harness is for.
// The limit comes from the config's swiglu_limit (10.0 on the released model); the DENSE and SHARED
// paths read swiglu_clamp_shexp and the routed experts read swiglu_clamp_exp (two arrays in the
// conversion, usually equal) - the caller picks, this kernel does not care.
#pragma once

#include <cstdint>

namespace maya::kernels {

/// out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, +limit), elementwise over n.
void glm_swiglu_clamp(const float* gate, const float* up, float limit, int64_t n, float* out, void* stream);

}  // namespace maya::kernels
