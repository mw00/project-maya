#pragma once

// Pascal (sm_60) has no tensor cores, so for FP16 x FP16 -> F32 cuBLAS's default algorithm is a slower legacy kernel
// on big products.  Measured through Gemm::f16 on a Tesla P100 (K = 2048..4096): from 2048 tokens the default gives
// 5.9-6.1 TFLOPS and ALGO6 7.2-7.3 (+20%), the very same output bits; around 1024 tokens they are even, and below
// 512 ALGO6 is SLOWER (T = 256: 24%, T = 16: 4x).  So it is used from 1024 tokens up.  Plain (non-batched) FP16
// GemmEx calls only: bf16 and the batched entry points refuse an explicit algo.  Only sm_60 is measured; every other
// arch keeps the algo the call site had.
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
#define STRATA_F16_GEMM_ALGO(dflt, tokens) (dflt)
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>

namespace strata::cuda_compat {

inline cublasGemmAlgo_t f16_gemm_algo(cublasGemmAlgo_t dflt, long long tokens) {
    static std::atomic<int> cc_of[64];   // compute capability a device, 0: not read yet
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return dflt;
    int cc = cc_of[dev].load(std::memory_order_relaxed);
    if (cc == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
            return dflt;
        cc = major * 10 + minor;
        cc_of[dev].store(cc, std::memory_order_relaxed);
    }
    return cc == 60 && tokens >= 1024 ? CUBLAS_GEMM_ALGO6 : dflt;
}

}  // namespace strata::cuda_compat
#define STRATA_F16_GEMM_ALGO(dflt, tokens) (::strata::cuda_compat::f16_gemm_algo(dflt, (long long) (tokens)))
#endif
