// src/kernels/cpu/ik2_rows.hpp - the ik2 CPU lane's hand-tuned rows (see native_expert.hpp for the lane).
//
// ggml-cpu's own dot products are the lane's ground truth, but NVFP4's is the generic scalar loop (quants.c:330:
// a 16-value integer sum a sub-block), which is what a Q3_K_XL / Q4_K_XL expert takes on this PC.  These are the
// same sums with AVX2: the codes widened 16 at a time and maddups'd against the quantized activation.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// out[r - r0] = row r of a tensor of ggml `type` . the quantized activation `act`, through the hand-tuned AVX2 kernel.
/// False when this build has none for the type (the caller then takes ggml-cpu's dot).
bool hand_rows(int type, size_t row_bytes, const uint8_t* base, const void* act, int n, int r0, int r1, float* out);

}  // namespace strata::kernels::cpu
