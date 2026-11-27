// include/maya/kernels/lane_xor.hpp - a 32-lane warp's value from lane ^ M, in registers on RDNA.
//
// HIP compiles __shfl_xor_sync to ds_bpermute_b32: an LDS round trip for every step of a warp sum, five dependent
// ones a sum.  On RDNA (gfx10 and later, 32-lane waves) the same value moves in registers: DPP quad_perm for 1 and 2,
// row_xmask for 4 and 8, v_permlanex16 for 16 (exl3.cu's xor_lane, commit 1662a28).  The same value is moved, so a sum
// built from it keeps its order of operations and its bits.  For whole-warp calls only (every lane active), which is
// what __shfl_xor_sync(0xffffffffu, ...) already asks for.  MAYA_XOR_BPERMUTE keeps the shuffle; CUDA, wave64 HIP
// builds and older AMD GPUs take the shuffle as before.
#pragma once

template <int M>
__device__ __forceinline__ float maya_lane_xor(float v) {
#if defined(MAYA_USE_HIP) && !defined(MAYA_HIP_WAVE64) && !defined(MAYA_XOR_BPERMUTE) && \
    (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    const int x = __builtin_bit_cast(int, v);
    int r;
    if constexpr (M == 1) r = __builtin_amdgcn_update_dpp(0, x, 0xB1, 0xF, 0xF, false);         // quad_perm [1,0,3,2]
    else if constexpr (M == 2) r = __builtin_amdgcn_update_dpp(0, x, 0x4E, 0xF, 0xF, false);    // quad_perm [2,3,0,1]
    else if constexpr (M == 4) r = __builtin_amdgcn_update_dpp(0, x, 0x164, 0xF, 0xF, false);   // row_xmask:4
    else if constexpr (M == 8) r = __builtin_amdgcn_update_dpp(0, x, 0x168, 0xF, 0xF, false);   // row_xmask:8
    else {
        static_assert(M == 16, "maya_lane_xor: 1, 2, 4, 8 or 16");
        r = (int) __builtin_amdgcn_permlanex16((unsigned) x, (unsigned) x, 0x76543210u, 0xFEDCBA98u, false, false);
    }
    return __builtin_bit_cast(float, r);
#else
    return __shfl_xor_sync(0xffffffffu, v, M);
#endif
}
