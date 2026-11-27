// include/maya/kernels/cpu/iq_avx512_one.hpp - one-token AVX-512 dot products for the decode's CPU lane, bit-identical
// to ggml-cpu's dot (see src/kernels/cpu/iq_avx512_one.cpp). IQ2_XXS is always
// considered; IQ2_S, IQ3_XXS and IQ3_S additionally require MAYA_GLM_CPU512_1T=1.
#pragma once

#include <cstddef>
#include <cstdint>

namespace maya::kernels::cpu {

/// true when this CPU runs the one-token rows for `ggml_type`. The optional
/// IQ2_S/IQ3_XXS/IQ3_S forms additionally require AVX-512 DQ/VBMI/BITALG/VNNI
/// and BMI2; unsupported CPUs remain on the established fallback.
bool iq512_one_supported(int ggml_type) noexcept;
/// out[t][r] = w_r . a[t], rows [r0, r1), each token on its own (the same shape as iq512_rows)
void iq512_one_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                    float* const* out, int r0, int r1);

}  // namespace maya::kernels::cpu
