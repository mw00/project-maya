// include/strata/kernels/dequant_bf16.hpp - plan v0.3 P5: native GGUF blocks -> BF16 on the device, for the
// tensor-core GEMMs of batched prompt processing.
//
// One thread per 32-element group, transcribed from the reference `dequantize_row_*` functions of the pinned
// llama.cpp (`ggml/src/ggml-quants.c`, MIT): Q4_0 (2), Q5_0 (6), Q8_0 (8), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14),
// IQ4_NL (20), IQ4_XS (23), Q2_0 (42) and Q2_K (10). A row-major (n_rows, n_cols) tensor of blocks becomes a row-major BF16
// matrix; `n_cols` must be a multiple of the type's block size.
#pragma once

#include <cstdint>

namespace strata::kernels {

bool dequant_bf16_supported(int ggml_type) noexcept;

/// `rows` rows of `cols` values starting at block row `row0` (so a caller can dequantize a slice of a tensor).
void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream);

/// The same into FP16 bits (the prompt path's quantized-weight GEMMs: Q2_0 values are exact in FP16).
void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream);

/// The same into FP32 (tests and small tensors).
void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream);

/// Gather GGUF embedding rows using token IDs already on the device.
void dequant_embed_rows(int ggml_type, const void* table, int64_t row_bytes, const int32_t* tokens,
                        int64_t n_tok, int64_t n_embd, float* out, void* stream);

/// Dequantize gate and up matrices into interleaved FP16 rows for the prompt GEMM.
void dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd,
                    uint16_t* out, void* stream);

}  // namespace strata::kernels
