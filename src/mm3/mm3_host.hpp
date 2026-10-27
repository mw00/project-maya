// src/mm3/mm3_host.hpp - maya-mm3's x86 host helpers (AVX2), in a .cpp of their own: nvcc's front end cannot read
// <immintrin.h> with every host compiler (CUDA 12.4 with GCC 13: the AMX intrinsics' builtins are unknown to it), so
// the .cu files call these instead of including it.
#pragma once

#include <cstddef>

namespace mm3::host {

/// a . b over n floats in a fixed order (4 accumulators of 8 lanes, then the lanes, then the tail)
float dot(const float* a, const float* b, int n);
/// y [rows][D] = RMSNorm(x) . w (F32; the sum of squares in dot's order; D a multiple of 8)
void rms_rows(const float* x, const float* w, int rows, int D, float eps, float* y);
/// h [n] += (p[0] + p[1] + ... + p[np - 1]) (+ extra) - the parts added in member order (n a multiple of 8)
void sum_into(float* h, const float* const* p, int np, const float* extra, size_t n);
/// a spin-wait's pause (x86 PAUSE)
void pause();

}  // namespace mm3::host
