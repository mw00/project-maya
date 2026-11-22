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

/// The same product for a family of shapes (an attention's scores: a count of keys that changes from one product to
/// the next): the kernel measured once for the bucket of n and k (powers of two), then taken by every shape of the
/// bucket that it runs.
bool gemm_loose(void* stream, const uint16_t* W, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy);

/// y[r][c] = sum over j < k of x[r][j] A[j][c], for r < t, c < n: A FP16 [k][la] row-major (its first n columns), x
/// FP16 rows of ldx, y FP32 rows of ldy.  loose: as gemm_loose.
bool gemm_n(void* stream, const uint16_t* A, int la, int n, int k, const uint16_t* x, int ldx, int t, float* y, int ldy,
            bool loose = false);

constexpr int kMinRows = 64;   // (fewer: the BLAS's own GEMV-like kernels)

/// The file the picks are kept in (the pack's lt-picks.txt, or MAYA_DS4_LT_STORE's): each line a GPU kind and hipBLASLt
/// version's kernel for one shape, read at the first product on a device and taken where it still runs the shape; a
/// new pick is added. Unset or empty: measured each start. setting: the engine's setting that names another file - a
/// store that does not open for writing is logged once, with its path and that setting.
void set_store(const char* path, const char* setting = nullptr);

/// Whether this build has hipBLASLt (an AMD build with it; DS4_LT=0 counts as without).
bool available();

}  // namespace strata::ds4::lt
