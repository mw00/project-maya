// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ1_S / IQ2_XS / IQ3_XXS model files.
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);

/// The grouped WMMA launches' by-value weight table (the Volta prompt experts, STRATA_PF_WMMA): expert e's gate/up
/// matrix at gu[e], its down matrix at dn[e].  The prompt path's group size (16) is the table's capacity.
constexpr int kGroupedWeightTableMax = 16;
struct GroupedWeightTable {
    const void* gu[kGroupedWeightTableMax] = {};
    const void* dn[kGroupedWeightTableMax] = {};
    int n = 0;
};

/// Y[r][0..n_out) = X[ids ? ids[r] : r][0..K) . W_e (expert e's matrix: `n_out` rows of `K` values of `ggml_type`,
/// expert e at W + e * expert_bytes, or at the by-value table's gu[e]/dn[e] when `indirect`) for the group's rows
/// [bounds[e], bounds[e+1]), r in [bounds[0], bounds[n_experts]).  FP16 activations (X, row stride `ldx` halves,
/// uint4-aligned), FP32 sums (Y, row stride `ldy` floats).  One grouped launch of FP16 tensor-core tiles over the
/// whole group, the weights dequantized into shared memory.  Volta (sm_70) and up; returns false when this device
/// or shape is not covered, having launched nothing.  `down` selects the table's dn half.
bool gemm_iq_f16_grouped(int ggml_type, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream,
                         const int32_t* ids = nullptr, const GroupedWeightTable* indirect = nullptr, bool down = false);

/// Whether device `dev` (an engine device index, as cudaGetDevice reports it) accepts the grouped WMMA launches at
/// all: its tensor-core capability and the shared-memory attributes the kernels need.  A caller deciding whether to
/// convert its activations for the products (a conversion the MMQ path would not need) probes once with this instead
/// of attempting a launch; `gemm_iq_f16_grouped` checks the same table per launch either way.
bool gemm_iq_f16_grouped_supported(int dev);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.  swiglu_limit
/// = +infinity (default) keeps the plain unclamped silu*up; a finite limit is GLM's clamped swiglu.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           float swiglu_limit = std::numeric_limits<float>::infinity());

}  // namespace strata::kernels
