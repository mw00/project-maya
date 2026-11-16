#pragma once
// tests/cpu_cuda_shim/cuda_runtime.h - CPU stand-in for <cuda_runtime.h> on a host with no CUDA or
// ROCm toolkit (this build box has neither nvcc nor hipcc).
//
// WHY this exists: native_mmvq.cu and dequant_bf16.cu must be compiled and their host-side
// capability queries and device-side dequant/dot helpers run on CPU before the GPU window opens
//   This header supplies only what those two translation units name.  It is NOT a
// device: there is no scheduler, no stream ordering and no memory hierarchy.  The CPU test never
// launches a kernel, so no return value here is load-bearing for the test except __dp4a, which
// is implemented byte-exact because the Q5_1 dot test calls the real dot helper.
//
// Scratch for the CPU test only.  The repo's real builds use the toolkit header.

#include <cstddef>
#include <cstdint>
#include <math.h>   // fmaxf/fminf/fabsf/roundf: CUDA's math header brings these in, g++ does not

// Kernel-qualifier macros.  Stripped to nothing so a __global__/__device__ body compiles as an
// ordinary function; the CPU test calls the device helpers directly and never launches.
#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __restrict__
#define __restrict__
#endif
// CUDA's __align__(n) is the same alignment attribute under its own spelling.
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif

inline void __syncthreads() {}

// CUDA's packed integer vector types (vector_types.h).  Alignments match CUDA because the block
// layout static_asserts in native_mmvq.cu are checked against them.
struct __attribute__((aligned(8))) int2 {
    int x;
    int y;
};

struct __attribute__((aligned(16))) uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    return uint4{x, y, z, w};
}

inline int2 make_int2(int x, int y) { return int2{x, y}; }

// CUDA's byte permute (PTX prmt.b32).  The 8 input bytes are a.b0..b3 then b.b0..b3; each output
// byte takes the 4-bit selector field at bit 4*i, where 0..7 pick one of those bytes and 8..15
// replicate that byte's sign bit instead.  Implemented exactly, not stubbed, because the Q2_0,
// Q3_K and IQ4 paths use it and this harness must not quietly change what they mean.
inline unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int selector) {
    const unsigned int source[8] = {
        (a >> 0) & 0xffu, (a >> 8) & 0xffu, (a >> 16) & 0xffu, (a >> 24) & 0xffu,
        (b >> 0) & 0xffu, (b >> 8) & 0xffu, (b >> 16) & 0xffu, (b >> 24) & 0xffu,
    };
    unsigned int result = 0;
    for (int byte_slot = 0; byte_slot < 4; ++byte_slot) {
        const unsigned int selection = (selector >> (4 * byte_slot)) & 0xFu;
        const unsigned int chosen_byte = source[selection & 0x7u];
        const unsigned int result_byte =
            (selection < 8u) ? chosen_byte : ((chosen_byte & 0x80u) ? 0xFFu : 0x00u);
        result |= result_byte << (8 * byte_slot);
    }
    return result;
}

// CUDA's SIMD subtract of four signed bytes with saturation, packed in one int32.
inline int32_t __vsubss4(int32_t a, int32_t b) {
    const uint32_t ua = (uint32_t) a;
    const uint32_t ub = (uint32_t) b;
    uint32_t result = 0;
    for (int byte_slot = 0; byte_slot < 4; ++byte_slot) {
        const int32_t a_byte = (int32_t) (int8_t) ((ua >> (8 * byte_slot)) & 0xffu);
        const int32_t b_byte = (int32_t) (int8_t) ((ub >> (8 * byte_slot)) & 0xffu);
        int32_t difference = a_byte - b_byte;
        if (difference > 127) difference = 127;
        if (difference < -128) difference = -128;
        result |= ((uint32_t) (int8_t) difference) << (8 * byte_slot);
    }
    return (int32_t) result;
}

// Warp shuffle.  There is no warp here: one host thread stands in for the whole warp, so every
// lane reads its own value back.  The CPU test never launches the kernels that use this, so the
// value is only load-bearing for compilation, not for a measured number.
template <typename T>
inline T __shfl_xor_sync(unsigned int, T value, int, int = 32) { return value; }

// CUDA's byte-wise 4-way dot (signed int8 x signed int8, accumulated into int32 with the same
// wrap-around as the hardware instruction).  The Q5_1 dot test adds real weight codes with this,
// so it must match __dp4a exactly rather than approximate it.
inline int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    const uint32_t ua = (uint32_t) a;
    const uint32_t ub = (uint32_t) b;
    uint32_t accumulator = (uint32_t) c;
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
        const int32_t a_byte = (int32_t) (int8_t) ((ua >> (8 * byte_index)) & 0xffu);
        const int32_t b_byte = (int32_t) (int8_t) ((ub >> (8 * byte_index)) & 0xffu);
        accumulator += (uint32_t) (a_byte * b_byte);
    }
    return (int32_t) accumulator;
}

// Launch geometry.  threadIdx/blockIdx/blockDim are plain mutable globals so a CPU harness could
// step the indices if it ever needs to; the test leaves them at zero/one and calls the helpers
// directly instead of launching.
struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    explicit dim3(uint32_t a = 1, uint32_t b = 1, uint32_t c = 1) : x(a), y(b), z(c) {}
};

struct dim3_stub {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

inline dim3_stub& threadIdx_stub() { static dim3_stub value; return value; }
inline dim3_stub& blockIdx_stub() { static dim3_stub value; return value; }
inline dim3_stub& blockDim_stub() { static dim3_stub value = {1, 1, 1}; return value; }
#define threadIdx (threadIdx_stub())
#define blockIdx (blockIdx_stub())
#define blockDim (blockDim_stub())

typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
typedef struct CUstream_st* cudaStream_t;
// No cudaMemcpy* here on purpose: neither native_mmvq.cu nor dequant_bf16.cu calls one, and
// tools/check_hip_compat.py scans every header in the tree, so an unused name in this shim would
// fail the AMD build's mapping check for something the AMD build never sees.

inline const char* cudaGetErrorString(cudaError_t) { return "cuda-stub (no CUDA runtime on this host)"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
