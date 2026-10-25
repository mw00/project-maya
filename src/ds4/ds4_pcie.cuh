// src/ds4/ds4_pcie.cuh - the decode path's PCIe lane on the device (see ds4_pcie.hpp): the GPU pulls the experts the
// host read into pinned memory into its staging slots itself (as GLM's moe_fetch_kernel does) - the host only reads and
// writes flags, never calls the GPU's API while the device waits on it (a thread that enqueues the layers can be
// blocked in the driver until the device moves on, and with it every other thread's call).  Included by ds4_model.cu
// alone (its one translation unit), and the tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::pcie {

/// A request's pull (host-mapped, a ring of them by seq): `seq` once src is set; src[i] the device address of pick i's
/// expert in pinned memory (0: not streamed), laid out as its blob (gate, up, down at the layer's offsets);
/// ready[i][r] == seq once its matrix r is read (bad[i]: a read failed - the pick is left out, its weight lost); done ==
/// seq once the device pulled every one (the host times the stream lane by it).
constexpr int kPullEnt = 16 * 8;   // a window's entries (kWin tokens x 8 picks)
constexpr int kSlots = 16;          // a pull's streamed experts at most (a window's; a token's: its picks)
struct Pull {
    unsigned seq;
    unsigned done;
    unsigned pad[2];
    unsigned long long src[kSlots];
    unsigned ready[kSlots][4];
    unsigned bad[kSlots];
    unsigned long long mat[kSlots][3];   // (0: src + the matrix's offset) pick i's matrix r where it lies - pulled in place
    signed char ent[kPullEnt];      // (a verify window) entry t * 8 + i's slot when its expert streams, else -1
    unsigned long long t_beg, t_end;   // the device's clock (ns) when it saw the pull, when its last block was done
    unsigned long long dst[kSlots];    // (0: its staging slot) where pick i lands - a recent slot it then stays in
};
// the fetch kernel's device flags: the pull, each pick's matrices, then the blocks done (the last one sets Pull::done)
constexpr int kPullCnt = 1 + kSlots * 3, kPullFlags = kPullCnt + 1;

/// A layer's matrices in its blob: offsets and bytes (each a multiple of 16 when rounded up: the slots' matrices start
/// 256-aligned, the copy may run into the padding after one)
struct Mats {
    size_t off[3], bytes[3];
};

// the device's clock in ns (NVIDIA's globaltimer); 0 elsewhere - the host then times the pull itself
__device__ __forceinline__ unsigned long long pull_clock() {
#if !defined(STRATA_USE_HIP)
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return 0ull;
#endif
}

__device__ __forceinline__ void pull_nap() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(500);
#endif
}

// grid: up to one block a multiprocessor.  Block 0's thread 0 polls the host's flags (the others wait on device copies
// of them in dflag [kPullFlags] - so the link is not flooded with polls); each streamed pick's matrices, each once read,
// copied from pinned memory into its slot (vram + i stride; copy 0: the kernels read them where they lie - an
// integrated GPU); then pick i's slot into plan2[i] (the second gate / up's plan: the streamed picks alone) and plan[i]
// (the down pass and the sum take a streamed expert as a VRAM one); the last block to finish sets the pull's done
__global__ void __launch_bounds__(256) stream_fetch_kernel(Pull* p, unsigned seq, unsigned* dflag, uint8_t* vram,
                                                           size_t stride, Mats m, int copy, unsigned long long* plan,
                                                           unsigned long long* plan2, int k, unsigned long long* wplan,
                                                           int went) {
    __shared__ unsigned long long s_src[kSlots], s_mat[kSlots][3], s_dst[kSlots];
    __shared__ unsigned s_bad[kSlots];
    const int tid = threadIdx.x;
    volatile const Pull* vp = (volatile const Pull*) p;
    volatile unsigned* df = dflag;
    if (tid == 0) {
        if (blockIdx.x == 0) {
            while (vp->seq != seq) pull_nap();
            __threadfence_system();
            ((volatile Pull*) p)->t_beg = pull_clock();
            df[0] = seq;
            __threadfence();
        } else {
            while (df[0] != seq) pull_nap();
            __threadfence();
        }
    }
    __syncthreads();
    if (tid < kSlots) {
        s_src[tid] = tid < k ? vp->src[tid] : 0ull;
        for (int r = 0; r < 3; ++r) s_mat[tid][r] = tid < k ? vp->mat[tid][r] : 0ull;
        s_dst[tid] = tid < k && vp->dst[tid] != 0ull ? vp->dst[tid] : (unsigned long long) (vram + (size_t) tid * stride);
        s_bad[tid] = 0u;
    }
    __syncthreads();
    for (int i = 0; i < k; ++i) {
        if (s_src[i] == 0ull) continue;
        for (int r = 0; r < 3; ++r) {
            const int f = 1 + 3 * i + r;
            if (tid == 0) {
                if (blockIdx.x == 0) {
                    while (vp->ready[i][r] != seq) pull_nap();
                    __threadfence_system();
                    df[f] = vp->bad[i] != 0u ? ~seq : seq;   // (a failed read: ~seq - no other request's seq soon)
                    __threadfence();
                } else {
                    while (df[f] != seq && df[f] != ~seq) pull_nap();
                    __threadfence();
                }
                if (df[f] != seq) s_bad[i] = 1u;
            }
            __syncthreads();
            if (s_bad[i] || !copy) continue;
            const uint4* src = (const uint4*) (s_mat[i][r] != 0ull ? s_mat[i][r] : s_src[i] + m.off[r]);
            uint4* dst = (uint4*) (s_dst[i] + m.off[r]);
            const size_t n16 = (m.bytes[r] + 15) / 16;
            for (size_t j = (size_t) blockIdx.x * blockDim.x + tid; j < n16; j += (size_t) gridDim.x * blockDim.x)
                dst[j] = src[j];
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&dflag[kPullCnt], 1u) == gridDim.x - 1) {
            dflag[kPullCnt] = 0u;
            ((volatile Pull*) p)->t_end = pull_clock();
            __threadfence_system();
            ((volatile Pull*) p)->done = seq;
            __threadfence_system();
        }
    }
    if (blockIdx.x == 0 && tid < k && wplan == nullptr) {
        const unsigned long long d = s_src[tid] == 0ull || s_bad[tid] ? 0ull : copy ? s_dst[tid] : s_src[tid];
        plan2[tid] = d;
        if (d != 0ull) plan[tid] = d;
    }
    // (a verify window) each entry whose expert streamed: its slot into the window's plan - the grouping that follows
    // takes it as a VRAM expert (a failed read: the entry stays the CPU lane's, which left it out - the token goes
    // without it, as a token does)
    if (blockIdx.x == 0 && wplan != nullptr)
        for (int e = tid; e < went; e += blockDim.x) {
            const int i = vp->ent[e];
            if (i < 0 || i >= k || s_src[i] == 0ull || s_bad[i]) continue;
            wplan[e] = copy ? s_dst[i] : s_src[i];
        }
}

inline void stream_fetch(Pull* p, unsigned seq, unsigned* dflag, uint8_t* vram, size_t stride, const Mats& m,
                         bool copy, unsigned long long* plan, unsigned long long* plan2, int k, int blocks,
                         cudaStream_t s) {
    stream_fetch_kernel<<<blocks, 256, 0, s>>>(p, seq, dflag, vram, stride, m, copy ? 1 : 0, plan, plan2, k, nullptr,
                                                0);
}
// (a verify window of T tokens) the streamed experts pulled as above, then their entries' slots into its plan
inline void stream_fetch_win(Pull* p, unsigned seq, unsigned* dflag, uint8_t* vram, size_t stride, const Mats& m,
                             bool copy, unsigned long long* wplan, int T, int blocks, cudaStream_t s) {
    stream_fetch_kernel<<<blocks, 256, 0, s>>>(p, seq, dflag, vram, stride, m, copy ? 1 : 0, nullptr, nullptr, kSlots,
                                                wplan, T * 8);
}

}  // namespace strata::ds4::pcie
