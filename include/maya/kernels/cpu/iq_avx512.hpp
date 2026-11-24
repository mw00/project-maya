// include/maya/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) and the K-quants Q4_K / Q5_K against Q8_K
// activations (ggml's block_q8_K).  Each token's result is bit-identical to ggml-cpu's AVX2 dot for that pair.
#pragma once

#include <cstddef>
#include <cstdint>

namespace maya::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);
/// Q5_K's per-block min term as one fmadd (GCC / Clang's contraction of ggml's line, the default) or a multiply
/// then an add (MSVC's) - set from native_expert.cpp's check against this build's ggml-cpu.
void iq512_set_q5k_fused(bool fused) noexcept;

}  // namespace maya::kernels::cpu
