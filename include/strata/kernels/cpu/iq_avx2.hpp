// include/strata/kernels/cpu/iq_avx2.hpp - AVX-2 multi-token dot products for the i-quant expert
// formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) and the K-quants Q4_K / Q5_K against Q8_K activations
// (ggml's block_q8_K).  Each token's result is bit-identical to ggml-cpu's AVX2 dot for that pair.
//
// The AVX-512 kernel's scheme on CPUs without it (AMD Zen 2/3, Intel Core 12th-14th gen): decoded weights are
// reused across activation rows. IQ1_M keeps a separate delta-correction accumulator to preserve ggml's rounding.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq256_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);
/// See iq512_set_q5k_fused.
void iq256_set_q5k_fused(bool fused) noexcept;

/// Decode-step rows for IQ1_M, IQ2_XS, IQ3_XXS and IQ3_S with cheaper decoding; iq256_rows' arguments and exactness.
bool iq256_rows1_supported(int ggml_type) noexcept;
void iq256_rows1(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0).
/// IQ4_NL is a 32-value-block format, so this does not go through iq256_rows (QK_K blocks, Q8_K acts).
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
