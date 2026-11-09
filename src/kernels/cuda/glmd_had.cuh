// src/kernels/cuda/glmd_had.cuh - the 128-point Walsh-Hadamard rotation of the glm-dsa indexer (glm-dsa B14.11).
//
// STRATA_GLM_IDX_HADAMARD=1 (read once, by the state carve: GlmStateLayout::ik_had) rotates every index key after
// its rope, before the cache store, and every indexer query head after its rope.  H is orthonormal and symmetric
// (H H = I), so (H q) . (H k) = q . k and the scores are those of the unrotated pair in exact arithmetic; what changes
// is the FP16 key store: a key whose energy sits in a few outlier channels is spread over all idx_dim channels, so
// no channel overflows FP16 (65504) or falls into its subnormal range, and fewer near-tie selections flip
// (ik_llama.cpp build_deepseek2.cpp:445-455 applies the same rotation, ggml_hadamard over head_size).
//
// The transform is ik's fast_ht (iqk_cpu_ops.cpp): in-place butterflies x' = x + y, y' = x - y for h = 1, 2, .., n/2,
// then one multiply by the float product of log2(n) factors 0.707106781f.  Every butterfly is the same pair of IEEE
// operations on the host and the device, so the CPU twin and the kernels write the same bits.
//
// Host + device, no CUDA types: the CPU twins include it too.  n is a power of two, 2 <= n <= 512.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMD_HAD_HD __host__ __device__ __forceinline__
#else
#define GLMD_HAD_HD inline
#endif

namespace glmd {

/// A Hadamard length the kernels serve: a power of two in [2, 512] (GLM-5.3 idx_key 128).
GLMD_HAD_HD bool had_len_ok(int n) { return n >= 2 && n <= 512 && (n & (n - 1)) == 0; }

/// The orthonormal scale of an n-point transform: 0.707106781f multiplied log2(n) times, in float (ik's order).
GLMD_HAD_HD float had_scale(int n) {
    float s = 1.0f;
    for (int h = 1; h < n; h <<= 1) s *= 0.707106781f;
    return s;
}

/// The serial transform (the CPU twins; a device thread may call it on a private row).
GLMD_HAD_HD void had_row(float* x, int n) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
    const float s = had_scale(n);
    for (int i = 0; i < n; ++i) x[i] *= s;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/// The block-wide transform of s[0, n) in shared memory: every thread of the block calls it (it synchronises).
/// The butterflies of one stage are independent, so a thread per pair gives the serial result exactly.
__device__ __forceinline__ void had_block(float* s, int n) {
    for (int h = 1; h < n; h <<= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
            const int j = (i / h) * 2 * h + (i % h);
            const float a = s[j], b = s[j + h];
            s[j] = a + b;
            s[j + h] = a - b;
        }
    }
    __syncthreads();
    const float sc = had_scale(n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] *= sc;
    __syncthreads();
}

/// Rotate `heads` rows of n floats per token in place: row (h, t) at x + t * t_stride + h * n.  Grid (heads, T),
/// any block size; dynamic shared memory n floats.
static __global__ void had_rows_kernel(float* __restrict__ x, int n, long long t_stride) {
    extern __shared__ float s_had[];
    float* row = x + (long long) blockIdx.y * t_stride + (long long) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_had[i] = row[i];
    had_block(s_had, n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] = s_had[i];
}
#endif

}  // namespace glmd

#undef GLMD_HAD_HD
