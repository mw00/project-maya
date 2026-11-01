// include/strata/kernels/dequant_bf16.hpp - plan v0.3 P5: native GGUF blocks -> BF16 on the device, for the
// tensor-core GEMMs of batched prompt processing.
//
// One thread per 32-element group, transcribed from the reference `dequantize_row_*` functions of the pinned
// llama.cpp (`ggml/src/ggml-quants.c`, MIT): Q4_0 (2), Q5_0 (6), Q8_0 (8), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14),
// IQ4_NL (20), IQ4_XS (23) and Q2_0 (42).  A row-major (n_rows, n_cols) tensor of blocks becomes a row-major BF16
// matrix; `n_cols` must be a multiple of the type's block size.
#pragma once

#include <cstdint>

namespace strata::kernels {

bool dequant_bf16_supported(int ggml_type) noexcept;
/// Whether dequant_f16 serves this type (dequant_bf16's types and the i-quants: IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S).
bool dequant_f16_supported(int ggml_type) noexcept;

/// `rows` rows of `cols` values starting at block row `row0` (so a caller can dequantize a slice of a tensor).
void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream);

/// The same into FP16 bits (the prompt path's quantized-weight GEMMs: Q2_0 values are exact in FP16).
void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream);

/// The same into FP32 (tests and small tensors).
void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream);
/// one row as dequant_f32, its index read on the device from *d_row (no host round trip; not the i-quant formats)
bool dequant_f32_dev_supported(int ggml_type) noexcept;
void dequant_f32_dev(int ggml_type, const void* blocks, const int32_t* d_row, int64_t cols, float* out, void* stream);
/// Rows d_rows[0 .. rows) (device ids) of a matrix to F32, one launch (an embedding lookup of a prompt chunk).
void dequant_f32_rows_dev(int ggml_type, const void* blocks, const int32_t* d_rows, int64_t rows, int64_t cols, float* out,
                          void* stream);

}  // namespace strata::kernels
