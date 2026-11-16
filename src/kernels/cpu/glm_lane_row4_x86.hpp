// src/kernels/cpu/glm_lane_row4_x86.hpp - the glm-dsa port B14.9: what glm_lane_row4.cpp (the CPU expert lane's
// 4-row switch, CPU check and hooks; compiled for any x86-64) takes from glm_lane_row4_avx2.cpp (the kernels; AVX2 +
// FMA + F16C).  Only x86-64 builds with STRATA_GLM_LANE_ROW4_KERNELS compile either side of it
// (cmake/glm_dsa.d/70-glm-lane-row4.cmake).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::row4x86 {

constexpr int kQK = 256;       // values per super-block
constexpr size_t kQ8K = 292;   // block_q8_K: float d, int8 qs[256], int16 bsums[16]

/// out[k] = row w[k] . q8 over nb super-blocks (one Q8_K activation), each bit for bit ggml-cpu's AVX2 vec_dot.
using Kernel = void (*)(int nb, const uint8_t* const w[4], const uint8_t* q8, float out[4]);

/// The 4-row kernel of a ggml type, nullptr for a type without one.  IQ1_S (19): iq1s_fused picks the form of its
/// scalar delta term (one fmaf, or a multiply and an add), which glm_lane_row4.cpp's probe finds.  The kernels and this
/// function are AVX2 code: call it only once glm_lane_row4.cpp's CPU check has passed.
Kernel kernel(int ggml_type, bool iq1s_fused);

}  // namespace strata::kernels::cpu::row4x86
