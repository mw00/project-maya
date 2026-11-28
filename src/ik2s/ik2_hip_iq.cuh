// src/ik2s/ik2_hip_iq.cuh - maya-ik2s's own GEMV on maya-ds4's ggml block-type kernels (src/ds4/ds4_iq.cuh): mv_lds,
// for the drafter's output head (ik2_hip_dspark.cu).  Built in ik2_hip_iq.cu, which is ds4_iq.cu plus it.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "../ds4/ds4_iq.cuh"

namespace ik2hip {

/// One Q6_K or Q8_0 job of up to 8 columns as ds4's iq::mv takes it (the same bits), its columns read from LDS: grid
/// blocks of `threads` (grid <= 0: a row a warp), each copying the columns in once (at most 64 KiB of LDS, xq 16-byte
/// aligned).  false: a job it does not take (nothing launched).
bool mv_lds(const maya::ds4::gguf::iq::MvJob& b, int grid, int threads, cudaStream_t s);

}  // namespace ik2hip
