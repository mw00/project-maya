// include/maya/kernels/rdna_wmma.hpp - the matrix cores of an AMD wave32 GPU (RDNA 3 / 3.5 gfx11, RDNA 4 gfx12):
// a 16 x 16 x 16 product of F16 operands with F32 sums, where each result value lies in a lane, and a block's barrier
// for its shared memory alone.  The shared form of maya-ds4's attn_wmma / set_wmma / lds_sync
// (src/ds4/ds4_kernels.cu, src/ds4/ds4_gguf.cu) for the engines that came after it.
//
// The operands: lane l (and l + 16 on gfx11, which takes both halves of the wave the same) gives row / column l % 16's
// 16 values of the k-step; gfx12 takes only values 8 * (l / 16) .. + 7 of them.  The result's 8 values a lane:
// value i is row wmma_row(i, l / 16) of column l % 16.
#pragma once

#if defined(__HIPCC__) && defined(MAYA_USE_HIP) && (defined(__GFX11__) || defined(__GFX12__))
#define MAYA_RDNA_WMMA 1

namespace maya::kernels {

using wmma_h16 = _Float16 __attribute__((ext_vector_type(16)));
using wmma_f8 = float __attribute__((ext_vector_type(8)));

/// c += a . b (a: 16 rows x 16, b: 16 x 16 columns, F16; c F32)
__device__ __forceinline__ wmma_f8 wmma_f16(wmma_h16 a, wmma_h16 b, wmma_f8 c) {
#if defined(__GFX12__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (int) (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aa[i] = a[k0 + i];
        bb[i] = b[k0 + i];
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}
/// the row of a lane's result value i (hi: the lane's half of the wave, l / 16)
__device__ __forceinline__ int wmma_row(int i, int hi) {
#if defined(__GFX12__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
/// a block's barrier for its shared memory alone: the threads' own global loads need not land at it (the next tile's
/// stay in flight under this one's sums)
__device__ __forceinline__ void lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace maya::kernels

#endif
