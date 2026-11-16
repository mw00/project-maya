#pragma once
// tests/cpu_cuda_shim/cuda_fp16.h - CPU stand-in for <cuda_fp16.h> on a host with no CUDA or ROCm
// toolkit (this build box has neither nvcc nor hipcc).
//
// WHY this exists: the Q5_1 work lives in .cu translation units whose HOST functions
// (native_mmvq_supported, native_mmvq_weight_bytes, dequant_bf16_supported) and whose device
// dequant/dot helpers must be exercised before any GPU window opens.  g++ can compile those bodies
// if the CUDA types resolve to something with the same numeric meaning, so this header maps the
// handful of names the two TUs use onto _Float16, which is IEEE 754 binary16 on x86-64 - the same
// format CUDA's `half` is.  Every conversion below is exact or round-to-nearest-even, matching the
// CUDA intrinsic it replaces; nothing here is a placeholder that could hide a wrong value.
//
// Scratch for the CPU test only.  The repo's real builds use the toolkit header.

#include <cstdint>
#include <cstring>

typedef _Float16 half;

// CUDA's `__half` is the same type under a second name; dequant_bf16.cu uses both spellings.
typedef _Float16 __half;

// CUDA's half2 is 4-byte aligned (it is designed to be loaded in one 32-bit access); the block
// layout static_asserts in native_mmvq.cu depend on that, so the shim must carry it too.
struct __attribute__((aligned(4))) half2 {
    half x;
    half y;
};

struct __attribute__((aligned(8))) float2 {
    float x;
    float y;
};

inline float __half2float(half value) { return (float) value; }

inline float2 __half22float2(half2 value) { return float2{(float) value.x, (float) value.y}; }

inline float __low2float(half2 value) { return (float) value.x; }

inline float __high2float(half2 value) { return (float) value.y; }

// _Float16's converting assignment is round-to-nearest-even, which is what __float2half_rn does.
inline half __float2half_rn(float value) { return (half) value; }

inline half __float2half(float value) { return (half) value; }

inline half2 make_half2(half low, half high) { return half2{low, high}; }

inline uint16_t __half_as_ushort(half value) {
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline half __ushort_as_half(uint16_t bits) {
    half value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t __float_as_uint(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float __uint_as_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
