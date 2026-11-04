// src/mm3/mm3_gv.cuh - maya-mm3's dense GEMVs over a GGUF's K-quant and Q8_0 matrices (the attention, the shared and dense
// MLPs, the output head of MiniMax's Maya quants): the sums of ds4_iq's mv to the bit, read faster.
//
// ds4_iq's mv (maya-ds4's, kept as it is for DeepSeek) gives each row a warp, each lane a 32-value sub-block a step
// (lane, lane + 32, ...), its integer dot times the sub-block's scales accumulated with fmaf, then the warp's butterfly.
// These kernels keep exactly that per row and per activation column - the same lane takes the same sub-blocks in the
// same order, the same roundings - and change only how the bytes arrive and how often the activations are read:
//   * a warp takes R rows (each lane the same sub-block of each), so an activation sub-block is read once for R rows;
//   * a lane's weight bytes come in a few wide loads (Q4_K / Q5_K: 16-byte, their blocks 16-byte aligned; Q6_K / Q8_0:
//     8-byte, their 210 / 34-byte blocks re-aligned in registers) instead of 2- and 4-byte ones;
//   * the K-quants' minimum term takes the activations' 16-value sums from the rounding kernel (q8 with sums) instead
//     of four dp4a a row - the same integers.
// A job these kernels do not take (another type, an alignment the wide loads cannot assume) goes to ds4_iq's mv.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace mm3::gv {

/// One matrix-vector job: y[c][row] = alpha * (W[row] . x[c]) (+ bias[row]) for c < ncol, x rounded to int8 per 32
/// (xq [ncol][k], xd [ncol][k / 32]) with xs [ncol][k / 32] its 16-value sums (low 16 bits the first half's, high the
/// second's, signed) - needed by Q4_K / Q5_K only.
struct Job {
    const uint8_t* w = nullptr;
    size_t ld = 0;   // bytes a row
    int type = 0, k = 0, n = 0, ncol = 1;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    const int* xs = nullptr;
    float* y = nullptr;
    size_t ycs = 0;
    const float* bias = nullptr;
    float alpha = 1.0f;
};

/// Whether these kernels take the job (its type, k, row bytes and alignment) - else ds4_iq's mv.
bool covered(const Job& j);
/// Whether the type's minimum term needs the activations' sums (Job::xs).
bool needs_sums(int type);
/// The jobs (each covered, ncol <= 16), grouped by type a launch.  false: one not covered (nothing launched for it).
bool run(const Job* jobs, int n, cudaStream_t s);
/// MAYA_MM3_GV=0: every job through ds4_iq's mv (as before).  On by default - CUDA only: on HIP the compiler contracts
/// these kernels' multiplies and adds otherwise than ds4_iq's mv (other bits), so HIP keeps ds4_iq's mv (covered false).
bool enabled();

}  // namespace mm3::gv
