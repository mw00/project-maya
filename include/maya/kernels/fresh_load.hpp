// include/maya/kernels/fresh_load.hpp - device loads of memory another block, device or the host writes, and the
// device's clock in ns, on CUDA and HIP alike.
//
// CUDA has cache-hint loads for this (__ldcg: past L1, from L2; __ldcv: not cached, fetched again) and a nanosecond
// %globaltimer.  HIP has neither for float / float4 (its __ldcg exists for __half only, as a plain load) and no
// %globaltimer.  On AMD a volatile load is a glc load: it skips the CU's L1, which is not coherent with other CUs or
// the host - the same answer as maya-ds4's ld_fresh (src/ds4/ds4_kernels.cu) and pull_clock (src/ds4/ds4_pcie.cuh),
// which this header is the shared form of.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)

#include <cstdint>

namespace maya::kernels {

#if defined(MAYA_USE_HIP)
typedef unsigned int fresh_u32x4 __attribute__((ext_vector_type(4)));
#endif

// written by another block of this kernel or an earlier one: past L1 (CUDA __ldcg, kept in L2)
__device__ __forceinline__ float ld_l2(const float* p) {
#if defined(MAYA_USE_HIP)
    return *(const volatile float*) p;
#else
    return __ldcg(p);
#endif
}

// written by another device or the host (pinned or peer memory): never from a stale cache line (CUDA __ldcv)
__device__ __forceinline__ float4 ld_fresh_v(const float4* p) {
#if defined(MAYA_USE_HIP)
    const fresh_u32x4 v = *(const volatile fresh_u32x4*) p;
    return make_float4(__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w));
#else
    return __ldcv(p);
#endif
}

// the device's clock in ns: NVIDIA's %globaltimer; HIP's wall clock (100 MHz on gfx11 / gfx12); 0 on other AMD GPUs,
// whose wall clock may tick at another rate - a caller's timeout then never fires
__device__ __forceinline__ unsigned long long clock_ns() {
#if !defined(MAYA_USE_HIP)
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#elif defined(__GFX11__) || defined(__GFX12__)
    return (unsigned long long) wall_clock64() * 10ull;
#else
    return 0ull;
#endif
}

}  // namespace maya::kernels

#endif  // defined(__CUDACC__) || defined(__HIPCC__)
