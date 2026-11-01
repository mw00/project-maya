// The HIP runtime names the peer all-reduce uses (init_peer_ar, calibrate_peer, its test), on CUDA (DS4_CUDA_PEER_AR)
#pragma once
#if defined(DS4_CUDA_PEER_AR) && !defined(STRATA_USE_HIP)
#include <cuda_runtime.h>
#define hipSuccess cudaSuccess
#define hipError_t cudaError_t
#define hipErrorNotReady cudaErrorNotReady
#define hipErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define hipStream_t cudaStream_t
#define hipEvent_t cudaEvent_t
#define hipGetDevice cudaGetDevice
#define hipSetDevice cudaSetDevice
#define hipGetDeviceCount cudaGetDeviceCount
#define hipStreamQuery cudaStreamQuery
#define hipGetLastError cudaGetLastError
#define hipPeekAtLastError cudaPeekAtLastError
#define hipGetErrorString cudaGetErrorString
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipStreamCreate cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipEventCreate cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventRecord cudaEventRecord
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventDestroy cudaEventDestroy
#define hipEventSynchronize cudaEventSynchronize
#define hipEventQuery cudaEventQuery
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define hipMemGetInfo cudaMemGetInfo
#define hipHostMalloc(p, sz, fl) cudaHostAlloc((p), (sz), (fl))
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostMallocPortable cudaHostAllocPortable
#define hipHostFree cudaFreeHost
#define hipDeviceMallocFinegrained 0u
// CUDA: %globaltimer counts nanoseconds -> a wall-clock rate of 1,000,000 kHz
#define hipDeviceAttributeWallClockRate (-4242)
static inline cudaError_t hipDeviceGetAttribute(int* v, int attr, int dev) {
    if (attr == hipDeviceAttributeWallClockRate) { *v = 1000000; return cudaSuccess; }
    return cudaDeviceGetAttribute(v, (cudaDeviceAttr) attr, dev);
}
// HIP's fine-grained memory: plain cudaMalloc on CUDA (peer stores + __threadfence_system + __ldcv)
static inline cudaError_t hipExtMallocWithFlags(void** p, size_t sz, unsigned) { return cudaMalloc(p, sz); }
#if defined(__CUDACC__)
__device__ __forceinline__ unsigned long long wall_clock64() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#endif
#endif
