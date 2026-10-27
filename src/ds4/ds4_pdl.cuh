// src/ds4/ds4_pdl.cuh - programmatic dependent launch (compute capability 9.0 on): a kernel launched with pdl_launch
// may be put on the device while the kernel before it in its stream still runs - its blocks wait in pdl_wait() until
// that kernel has ended and its writes are visible - so the launch's own latency hides behind the previous kernel
// instead of sitting between the two.  Every kernel pdl_launch starts calls pdl_wait() before it touches memory a
// kernel before it wrote or reads; pdl_trigger() (right after the wait) lets the next one be put on the device in turn.
// One kernel at most waits behind a running one.  A kernel that waits for a flag another stream, device or the host
// sets triggers only once that wait is over (or not at all): the blocks of the kernel behind it would otherwise sit on
// the multiprocessors while it spins, and the kernel that sets its flag on another stream of the same device could
// find none free - a deadlock (the prompt path's all-reduces).  Elsewhere (older GPUs, HIP, DS4_PDL=0): a plain
// launch, and the two calls do nothing.  The arithmetic is untouched: the same kernels in the same order.
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::ds4 {

__device__ __forceinline__ void pdl_wait() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

/// Whether the current device takes programmatic launches: compute capability 9.0 or later, unless DS4_PDL=0.
/// (MAYA_DS4_IQ_ONLY: ds4_iq.cu built into another engine without the rest of maya-ds4 - maya-mm3 and its tests: their
/// launches as before)
#if defined(MAYA_DS4_IQ_ONLY)
inline bool pdl_on() { return false; }
#else
bool pdl_on();
#endif

template <typename... P, typename... A>
inline void pdl_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
#if !defined(STRATA_USE_HIP)
    if (pdl_on()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = s;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        (void) cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...);
        return;
    }
#endif
    kernel<<<grid, block, smem, s>>>(std::forward<A>(args)...);
}

}  // namespace strata::ds4
