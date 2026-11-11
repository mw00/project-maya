// include/strata/prefill/moe_pascal.hpp - the prompt path's expert products for compute < 6.1 (Pascal P100).
// ggml's MMQ runs the i-quant dots on dp4a, which sm_60 lacks: the emulation (byte extracts + XMAD multiplies)
// holds it near 1 TMAC/s.  This kernel dequantizes a weight tile to FP32 in shared memory once and multiplies it
// with FP32 activations on the FMA pipe (a tile of tokens per weight tile), so the cost per MAC is one FFMA.
// Same grouping as mmq::Product: n experts a fixed stride apart, expert e's rows [bounds[e], bounds[e+1]).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::pascal {

/// The kernels cover this ggml type (IQ2_XXS, IQ2_S, IQ3_XXS) and the build has them.
bool supported(int ggml_type);

/// The current device is compute < 6.1, both expert types are covered and the shapes tile (n_embd and n_ff multiples
/// of 256): the prompt path takes this product instead of MMQ.  STRATA_GLM_PREFILL_PASCAL=0 / 1 forces it off / on.
bool usable(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);

/// dst[r, 0..w_rows) = x[row_map[r] (or r), 0..w_cols) . W_e^T for the rows r of expert e (W_e: [w_rows, w_cols],
/// the weights `expert_bytes` apart from `w`), all pointers addressed so that bounds[e] indexes them directly
/// (pre-offset by the set's first row, as for mmq::Product).  w_rows % 128 == 0, w_cols % 256 == 0, ld_x and
/// ld_dst % 4 == 0 and the pointers 16-byte aligned.  `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const float* x = nullptr;
    int64_t ld_x = 0;
    const int32_t* row_map = nullptr;
    const int32_t* bounds = nullptr;
    int64_t max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
    bool f16 = false;   // the HFMA2 kernel (see f16_default)
};

/// Whether the prompt path takes the FP16 kernel: half2 products (twice the FMA rate) summed in FP32 every 256
/// values of K; the weights and activations are rounded to half (error vs exact ~0.2%, MMQ's int8 activations ~0.5%).
/// On by default; STRATA_GLM_PREFILL_PASCAL_FP16=0 keeps the FP32 kernel.
bool f16_default();

void run(const Product& p, void* stream);

/// A dense FP16 product for the same cards: y[t][n] = sum_k x[t][k] * w[n][k] (FP32 out, no beta), the weights
/// already dequantized to FP16 (row n at w + n * k) and the activations FP16.  cuBLAS runs this at the FP32 rate on
/// sm_60 (about 3.3 TMAC/s); the HFMA2 kernel sums products in half for 256 values of K at a time and in FP32 across
/// them (error ~0.17% vs cuBLAS's FP32 sums), at 5.5-5.9 TMAC/s.
struct Dense {
    const void* w = nullptr;   // __half [n][k]
    int64_t n = 0, k = 0;
    const void* x = nullptr;   // __half [t][ld_x]
    int64_t ld_x = 0;
    int64_t t = 0;
    float* y = nullptr;        // [t][ld_y]
    int64_t ld_y = 0;
    int pieces = 0;            // K pieces: 0 picks from the tile grid, 1 / 2 force.  Two pieces add into y (zeroed first)
};

/// The device is compute < 6.1, the shapes and pointers fit (k % 32, n % 4, ld_x and ld_y % 4, aligned) and the grid
/// is big enough (under 20 tiles of 128 x 64 stays with cuBLAS; a grid of 20..80 or 113..139 tiles runs in two K pieces
/// that add into y, zeroed first on the stream: two sums add the same in either order, so the bits do not change).
/// STRATA_GLM_PREFILL_PASCAL_DENSE=0 turns it off, 1 forces it on whatever the card.
bool dense_usable(const Dense& d);

void dense_run(const Dense& d, void* stream);

}  // namespace strata::prefill::pascal
