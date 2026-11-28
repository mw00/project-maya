// src/ik2s/ik2_hip_iq.cu - maya-ds4's ggml block-type kernels as maya-ik2s builds them: src/ds4/ds4_iq.cu itself (one
// translation unit with it, so its file-local decoders and dot products are in reach, unchanged), plus maya-ik2s's own
// GEMV on them, mv_lds (ik2_hip_iq.cuh).  cmake/ik2s.cmake compiles this file in place of ds4_iq.cu: one copy of
// ds4_iq in the binary.
#include "../ds4/ds4_iq.cu"

#include "ik2_hip_iq.cuh"

namespace maya::ds4::gguf::iq {
namespace {
// mv_kernel for one job whose activation columns sit in LDS (mv_lds): each block copies the ncol int8 rows and their
// scales into LDS once, then its warps take rows gridDim.x * warps apart - a lane's sub-blocks, the dot products, the
// fmaf order and the warp's sum are mv_kernel's line for line (the same bits); only where the activations are read
// from differs.  With 8 columns mv_kernel reads 8 x 32 activation bytes from the caches for every weight sub-block
// (measured, R9700, a Q6_K output matrix of 201,024 x 4,608: 1 column 1,197 us, 8 columns 1,515 us)
template <int T, int NC>
__global__ void __launch_bounds__(1024) mv_lds_kernel(const MvJob b) {
    pdl_wait();
    pdl_trigger();
    extern __shared__ __align__(16) uint8_t lds_x[];
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    load_tab<T>(tab);
    const int nsb = b.k / 32, nc = b.ncol;
    int8_t* sx = (int8_t*) lds_x;
    float* sd = (float*) (lds_x + (size_t) nc * b.k);
    for (int i = threadIdx.x; i < nc * b.k / 16; i += blockDim.x) ((int4*) sx)[i] = ((const int4*) b.xq)[i];
    for (int i = threadIdx.x; i < nc * nsb; i += blockDim.x) sd[i] = b.xd[i];
    __syncthreads();
    const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
    for (int row = blockIdx.x * nw + (threadIdx.x >> 5); row < b.n; row += gridDim.x * nw) {
        const uint8_t* w = b.w + (size_t) row * b.ld;
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
        for (int kb = lane; kb < nsb; kb += 32) {
            Sub s;
            decode<T>(w, kb, tab, s);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < nc) {
                    const int4* xp = (const int4*) (sx + (size_t) c * b.k + (size_t) kb * 32);
                    acc[c] = fmaf(sd[(size_t) c * nsb + kb], sub_dot<T, kMinMV>(s, xp[0], xp[1]), acc[c]);
                }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc) {
                const float v = warp_sum(acc[c]);
                if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
            }
    }
}
}  // namespace
}  // namespace maya::ds4::gguf::iq

namespace ik2hip {

bool mv_lds(const maya::ds4::gguf::iq::MvJob& b, int grid, int threads, cudaStream_t s) {
    using namespace maya::ds4::gguf::iq;
    using maya::ds4::pdl_launch;
    const size_t smem = (size_t) b.ncol * (size_t) b.k + (size_t) b.ncol * (size_t) (b.k / 32) * 4;
    if ((b.type != kQ6_K && b.type != kQ8_0) || b.n <= 0 || b.k % block_values(b.type) != 0 || b.k % 32 != 0 ||
        b.ncol < 1 || b.ncol > 8 || smem > 65536 || threads < 64 || threads > 1024 || threads % 32 != 0 ||
        ((uintptr_t) b.xq & 15) != 0)
        return false;
    if (grid <= 0) grid = (b.n + threads / 32 - 1) / (threads / 32);
#define MVL_(T, NC)                                                                                          \
    do {                                                                                                     \
        static bool attr = false;                                                                            \
        if (!attr) {                                                                                         \
            if (cudaFuncSetAttribute((const void*) mv_lds_kernel<T, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                     65536) != cudaSuccess) {                                                \
                (void) cudaGetLastError();                                                                   \
                return false;                                                                                \
            }                                                                                                \
            attr = true;                                                                                     \
        }                                                                                                    \
        pdl_launch(mv_lds_kernel<T, NC>, dim3((unsigned) grid), dim3((unsigned) threads), smem, s, b);       \
    } while (0)
    if (b.type == kQ6_K) {
        if (b.ncol <= 1) MVL_(kQ6_K, 1);
        else if (b.ncol <= 4) MVL_(kQ6_K, 4);
        else MVL_(kQ6_K, 8);
    } else {
        if (b.ncol <= 1) MVL_(kQ8_0, 1);
        else if (b.ncol <= 4) MVL_(kQ8_0, 4);
        else MVL_(kQ8_0, 8);
    }
#undef MVL_
    return true;
}

}  // namespace ik2hip
