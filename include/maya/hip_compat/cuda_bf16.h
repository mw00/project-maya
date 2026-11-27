#pragma once
// CUDA's <cuda_bf16.h> for the HIP build: the BF16 value type and the two conversions the engines use
// (maya-hy4's and maya-ling's vision towers make BF16 operands for hipBLAS' GemmEx, HIP_R_16BF).  Self-contained: HIP's own
// hip_bf16.h is not in every ROCm the build supports, and its type has another name.
#include <hip/hip_runtime.h>

#include <cstdint>

struct __nv_bfloat16 {
    uint16_t x;
};

// round to nearest, ties to even - as CUDA's __float2bfloat16
__host__ __device__ inline __nv_bfloat16 __float2bfloat16(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    __nv_bfloat16 r;
    if ((v.u & 0x7fffffffu) > 0x7f800000u) {   // (NaN: kept a NaN, quiet)
        r.x = (uint16_t) ((v.u >> 16) | 0x0040u);
        return r;
    }
    r.x = (uint16_t) ((v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16);
    return r;
}

// exact, as CUDA's __bfloat162float: the 16 bits are a float's top half (maya-ling's vision tower)
__host__ __device__ inline float __bfloat162float(__nv_bfloat16 h) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t) h.x << 16;
    return v.f;
}
