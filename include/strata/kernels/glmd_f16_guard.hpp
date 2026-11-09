// include/strata/kernels/glmd_f16_guard.hpp - glm-dsa B15.1: the FP16 activation range guard of the prompt path.
//
// Volta has FP16 tensor cores only (no BF16 MMA), so every prompt GEMM that runs on them takes its activation operand
// as FP16: an F32 activation above 65504 becomes inf there, and one inf poisons the whole output row (llama.cpp
// #27016 for Volta cuBLAS, #30260: MoE down-projection inputs reach 1e5 in real models).  The guard, per ROW (one
// token) of every F32 -> FP16 activation conversion that feeds an FP16 GEMM or WMMA:
//
//   amax_t = max |x[t][:]|;  s_t = f16_guard_scale(amax_t)       (1 when amax_t <= 65504, else a power of two with
//                                                                 amax_t / s_t <= 32768 - #27016's rule)
//   x16[t][:] = FP16(x[t][:] / s_t)                               (rewritten only when s_t != 1)
//   Y[t][:] *= s_t after the GEMM                                 (the epilogue; only rows with s_t != 1 are touched)
//
// Dividing by a power of two is exact in F32 and in FP16 (outside the FP16 subnormal range), and so is the product
// back, so a scaled row's result differs from an unbounded-range FP16 GEMM only by the subnormals it loses.  A row
// with amax <= 65504 has s = 1: neither its FP16 bits nor its output are written again, so every output is
// bit-identical to the unguarded path whenever no row overflows (Flash, and every normal glm-dsa prompt).
//
// Plumbing (src/kernels/cuda/glmd_guards.cu): a producer calls f16_guard_rows / f16_guard_swiglu right after it
// wrote the FP16 operand.  The call keeps the per-row scales on the device and BINDS them to the FP16 buffer
// (per stream: base, row stride, rows).  The GEMM wrapper calls f16_guard_after(x16, ldx, Y, ...) after the product:
// when x16 lies inside a binding (the latest one wins), the bound rows starting at (x16 - base) / ld are unscaled
// in Y.  A phase start (`phase = true`: the residual read that produces the layer's x16, at the start of each
// mixer and each FFN) drops every other binding of the stream; f16_guard_reset drops all of them.  Invariant for
// callers: inside a phase, an FP16 buffer that a GEMM wrapper consumes is either guarded (bound) or does not
// overlap a buffer guarded earlier in the same phase.
//
// Volta is where #27016 found it, but the glm-dsa prompt path feeds FP16 activations to its tensor-core products on
// every backend (cuBLAS / hipBLAS GemmEx, the absorbed MLA's strided-batched products, hipBLASLt), so the exposure is
// every device's.
//
// Switch: STRATA_GLM_F16_GUARD = auto (default: on, on every device - NVIDIA or AMD), 1 (on), 0 (off: no amax pass,
// no epilogue; the Maya path exactly).
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cstdint>
#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMD_GUARD_HD __host__ __device__
#else
#define GLMD_GUARD_HD
#endif

namespace glmd {

constexpr float kF16Max = 65504.0f;      // the largest finite FP16
constexpr float kF16Target = 32768.0f;   // a scaled row's bound (#27016: while amax > 32768 * s, s *= 2)

/// The power-of-two scale of a row whose largest magnitude is amax: 1 when amax <= 65504 (and for NaN / inf, which
/// no scale can repair), else the smallest 2^k (k >= 1) with amax / 2^k <= 32768.
GLMD_GUARD_HD inline float f16_guard_scale(float amax) {
    if (!(amax > kF16Max) || !(amax <= 3.402823466e38f)) return 1.0f;
    float s = 2.0f;
    while (amax > kF16Target * s) s *= 2.0f;
    return s;
}

/// The SwiGLU product the FP16 producers compute (glm_batch.cu swiglu_f16_kernel's arithmetic).
GLMD_GUARD_HD inline float swiglu_value(float gate, float up, float limit) {
    const float g = fminf(gate, limit);
    const float u = fminf(fmaxf(up, -limit), limit);
    return (g / (1.0f + expf(-g))) * u;
}

/// -1 auto, 0 off, 1 on: STRATA_GLM_F16_GUARD, read on every call (host only).
int f16_guard_mode();

// ---- GPU (src/kernels/cuda/glmd_guards.cu, CUDA and HIP; no-ops with the switch off) -----------------------------

/// Whether the guard runs: the switch (auto: on, whatever the device).
bool f16_guard_active();
/// Rows t < T of src (row stride src_row floats; n_head heads src_head floats apart, K values each) were converted
/// to dst[(t * n_head + h) * K + k] (glmd::pack_f16's layout; n_head 1 and src_head 0 for plain rows).  Rewrites the
/// rows that overflow, keeps the scales, binds them to dst (row stride n_head * K).  phase: drop the stream's other
/// bindings first.  false on a launch error (counted in f16_guard_errors()).
bool f16_guard_rows(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst,
                    void* stream, bool phase = false);
/// The SwiGLU producer's guard: h16[t * n_ff + c] = FP16(swiglu_value(gate, up, limit)), gate / up T x n_ff
/// contiguous (glm_batch.cu swiglu_f16).  Rewrites the rows that overflow; binds the scales to h16.
bool f16_guard_swiglu(const float* gate, const float* up, int n_ff, int T, float limit, uint16_t* h16, void* stream);
/// Drops every binding of this stream (before FP16 operands that are produced unguarded).
void f16_guard_reset(void* stream);
/// After Y[t][0..n) (row stride ldy) = alpha * X16[t] . W^T + beta * Y over T rows: when x16 lies inside a binding,
/// Y's rows are multiplied by their scales (rows whose scale is 1 are not written).  beta must be 0 for a bound x16
/// (counted as an error and left unscaled otherwise).  No binding: nothing happens.
bool f16_guard_after(const void* x16, int64_t ldx, float* Y, int64_t ldy, int n, int T, float beta, void* stream);
/// Refusals and launch errors so far (a binding table that is full, a bound x16 off a row start, beta != 0, ...).
uint64_t f16_guard_errors();
/// Device pointer of the scales the latest binding covering x16 holds for its first row, nullptr when unbound (tests).
const float* f16_guard_scales_for(const void* x16, void* stream);

// ---- CPU twins (host pointers; the kernels' exact results) -------------------------------------------------------

namespace guard_twin {

/// Plain conversion (no guard): dst[(t * n_head + h) * K + k] = FP16(src[t * src_row + h * src_head + k]).
void pack_f16(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst);
/// The guard over rows converted by pack_f16: scale[t] for every row, the overflowing rows rewritten.
void guard_rows(const float* src, int64_t src_row, int64_t src_head, int n_head, int K, int T, uint16_t* dst,
                float* scale);
/// Plain SwiGLU to FP16 (glm_batch.cu swiglu_f16) and its guard.
void swiglu_f16(const float* gate, const float* up, int64_t n, float limit, uint16_t* h16);
void guard_swiglu(const float* gate, const float* up, int n_ff, int T, float limit, uint16_t* h16, float* scale);
/// Y[t][0..n) *= scale[t] where scale[t] != 1.
void unscale_rows(float* Y, int64_t ldy, int n, int T, const float* scale);
/// Y[t][o] = sum_k f32(X16[t][k]) * f32(W16[o][k]) (an FP16 tensor-core GEMM with F32 accumulation; the order of the
/// sum is not cuBLAS's, so compare with a tolerance).
void gemm_f16(const uint16_t* X16, int64_t ldx, const uint16_t* W16, float* Y, int64_t ldy, int T, int N, int K);

}  // namespace guard_twin

}  // namespace glmd
