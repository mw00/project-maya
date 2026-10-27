// src/ds4/ds4_lt.hpp - the prompt path's dense GEMMs on hipBLASLt (AMD): each shape's fastest kernel, measured here.
//
// hipBLASLt's own first pick for these shapes is often a slow kernel (gfx1151, a chunk of 2048 rows: ~5 TFLOP/s, the
// best of its candidates ~25-35): the first product of a shape times the heuristic's candidates once on this GPU and
// keeps the fastest, every later one runs it.
#pragma once
#include <cstdint>

namespace strata::ds4::lt {

/// y[r][c] = x[r] . W[c] for r < t, c < n: W FP16 [n][k] row-major, x FP16 rows of ldx, y FP32 rows of ldy (beta 0).
/// false: not taken (not an AMD build with hipBLASLt, DS4_LT=0, fewer than kMinRows rows, no candidate runs) - the
/// caller's cuBLAS / hipBLAS then.
bool gemm(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy);

constexpr int kMinRows = 64;   // (fewer: the BLAS's own GEMV-like kernels)

/// The file the picks are kept in (the pack's lt-picks.txt, or MAYA_DS4_LT_STORE's): each line a GPU kind and hipBLASLt
/// version's kernel for one shape, read at the first product on a device and taken where it still runs the shape; a
/// new pick is added. Unset or empty: measured each start.
void set_store(const char* path);

/// Whether this build has hipBLASLt (an AMD build with it; DS4_LT=0 counts as without).
bool available();

}  // namespace strata::ds4::lt
