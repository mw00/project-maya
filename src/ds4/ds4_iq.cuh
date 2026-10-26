// src/ds4/ds4_iq.cuh - a GGUF's routed experts in ggml's low-bit block types on the device for maya-ds4: the i-quants
// IQ1_S, IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and the K-quants Q2_K, Q3_K, Q4_K (256 values a block, ggml's
// layouts: ggml-common.h, llama.cpp MIT).  ds4_gguf.cu sends an expert layer here when its experts are not MXFP4
// (x3::ExpertLayout::fmt kFmtMXFP4 with K[r] = role r's ggml type; K[r] == kTypeMXFP4: ds4_gguf.cu's own kernels).
//
// Every weight is decoded exactly (ggml's dequantize_row_*: a block scale times an integer scale times a grid or code
// value) and every product takes the activations rounded to int8 per 32 values with an F32 scale (ds4_gguf.cu's
// q8_quant): a 32-value sub-block's sum is an exact integer dot (dp4a) times its scales in F32 - no integer division as
// llama.cpp's vecdotq takes (its sums truncate), so these are the double reference's sums to F32 rounding.  A window's
// entry gets the token path's bits (the same kernel, the same order).  IQ1_S / IQ1_M's +-1/8 shift of each grid value
// (IQ1S_DELTA, a sign a sub-block or a group of 8) rides in the integer dot: the codes are 8 g +- 1 (g in {-1, 0, 1}),
// the scale d / 8 - so the dot carries the int8 activations' group sums exactly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::gguf::iq {

// ggml type ids
constexpr int kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19, kIQ3_S = 21,
              kIQ2_S = 22, kIQ4_XS = 23, kIQ1_M = 29;

/// Whether the device kernels take this ggml type.
bool supported(int type);
/// Bytes of a block (256 values) of the type (0: not one of these).
int block_bytes(int type);
/// Bytes of a row of k values (k % 256 == 0).
inline size_t row_bytes(int type, int64_t k) { return (size_t) (k / 256) * (size_t) block_bytes(type); }
/// The type's name ("?" when unknown).
const char* name(int type);

/// The token / window experts (ds4_gguf.cu's MxArgs, with the roles' types): group g = an expert (grp_ptr null: expert g
/// of the plan, plan_ptr[g] its blob or 0, its entry g, input row 0; else blob grp_ptr[g], entries grp_ent[grp_start[g]
/// ..], entry e's input row e >> 3).  x / h as int8 rows with an F32 scale per 32 (xq, xd: rows of n_embd; hq, hd: row e
/// = entry e, rows of n_ff).
struct Args {
    const unsigned long long* plan_ptr = nullptr;
    const unsigned long long* grp_ptr = nullptr;
    const int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
    size_t tr[3] = {0, 0, 0};   // the roles' offsets in a blob
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    const int8_t* hq = nullptr;
    const float* hd = nullptr;
    float* h = nullptr;      // [entries][n_ff]: the clamped swiglu
    float* part = nullptr;   // [entries][n_embd]: each entry's down output
    int tg = 0, td = 0;      // gate / up's type, down's
};
/// grid (n_ff / 8, groups): h of every entry (ne_max: a group's most entries).  false: a type not covered.
bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s);
/// grid (n_embd / 8, groups): part of every entry.
bool down(const Args& a, int groups, int ne_max, cudaStream_t s);

/// The prompt path's expert sets (ds4_gguf.cu's SetK): expert i's blob at wbase + i stride owns the sorted rows
/// [bounds[i], bounds[i + 1]) (local row = sorted row - row0); gate / up: h16[local] = F16 swiglu over x[idx[local]]
/// (F32 rows, ldx); down: y[local] = down(h16[local]) (F16 rows in, ldx; F32 out, ldo).  A set's experts of at most 16
/// rows take the token kernels' form (each weight read once, the rows' inputs rounded to int8 per 32 as the token path
/// rounds them; STRATA_DS4_NO_LIGHT=1: never), the others GEMM tiles (tensor cores on CUDA, F16 operands).
struct SetArgs {
    const uint8_t* wbase = nullptr;
    size_t stride = 0;
    size_t tr[3] = {0, 0, 0};
    const int* bounds = nullptr;
    int row0 = 0, max_rows = 0, n_exp = 0;
    int n_embd = 0, n_ff = 0;
    float limit = 0.0f;
    const float* xf = nullptr;
    const int* idx = nullptr;
    const uint16_t* xh = nullptr;
    int ldx = 0;
    uint16_t* h = nullptr;
    float* y = nullptr;
    int ldo = 0;
    int tg = 0, td = 0;
    int skip_le = 0;   // (the tiles) the experts of at most this many rows are left out: the light kernels took them
    int xh_vec = 0;    // (launch_set's: xh 16-byte aligned rows - the tensor-core tiles load 8 values at once)
    int skip_ge = 0;   // (the tiles) > 0: the experts of at least this many rows are left out - the caller's GEMMs
};
bool set_gate_up(const SetArgs& a, cudaStream_t s);
bool set_down(const SetArgs& a, cudaStream_t s);
/// The busy experts' path (a prompt's experts of many rows): one expert's matrix - n rows of k values, row_bytes(k)
/// apart - as FP16 rows (ldd apart), the values the tensor-core tiles multiply (their loader's, rounded to FP16 the same
/// way), for the caller's GEMM: the tiles decode an expert's weights again for every 64 of its rows.
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, cudaStream_t s);
/// dst[r] = x[idx[r]] for r < n (FP16 rows of k values; x ldx apart, dst k apart)
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, cudaStream_t s);
/// h[r][c] = FP16 of the tiles' clamped SwiGLU of gu[r][c] (gate) and gu[r][ff + c] (up), F32 rows ld apart
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, cudaStream_t s);
/// The light form alone (the set's experts of at most 16 rows; the others untouched), for any of these types and MXFP4
/// (tg / td 39: ds4_gguf.cu's experts): false (nothing launched) when it is switched off - the caller's tiles take every
/// expert then; true: they take those of more rows (skip_le 16).
bool light_gate_up(const SetArgs& a, cudaStream_t s);
bool light_down(const SetArgs& a, cudaStream_t s);

/// GEMV jobs on such matrices (an expert's views on the host-routed path): y[c][r] = alpha (row r . x_c) + bias[r] for
/// ncol activations, x as int8 rows (xq + c k, scales xd + c k / 32).
struct MvJob {
    const uint8_t* w = nullptr;
    size_t ld = 0;   // bytes a row
    int type = 0, k = 0, n = 0, ncol = 1;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    float* y = nullptr;
    size_t ycs = 0;
    const float* bias = nullptr;
    float alpha = 1.0f;
};
bool mv(const MvJob* jobs, int n, cudaStream_t s);

/// (tests) every value of n rows of k (rows row_bytes(type, k) apart, on the device) as the kernels decode them: sub =
/// the token / window / light kernels' codes times their scales (d s q - m), tile = the prompt tiles' loader (F32,
/// before their F16) - both ggml's dequantize_row_* values to the bit.  false: a type not covered.
bool decode_values(int type, const uint8_t* w, int k, int n, float* sub, float* tile, cudaStream_t s);

}  // namespace strata::ds4::gguf::iq
