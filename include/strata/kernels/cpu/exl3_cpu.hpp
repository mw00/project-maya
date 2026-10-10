// include/strata/kernels/cpu/exl3_cpu.hpp - EXL3 routed experts on the CPU (the fast path's CPU lane).
//
// The mul1 codebook is affine in the byte sum of state * 0x83DCD12D: w = fp16(1024 + s) * k_inv + k_bias (one FP16
// FMA on the device).  The CPU keeps the affine form without the FP16 rounding, so a 128-output band is
//   y' = k_inv * sum_k xh_k s_kc + (1024 k_inv + k_bias) * sum_k xh_k,   xh = H(x * suh)
// and y = H(y') * svh.  The byte sums come from AVX-512 (permutes and funnel shifts pull 16 columns' states at once,
// vpmaddubsw / vpmaddwd add the bytes up, an F32 FMA takes the activation); a CPU without AVX-512 runs a scalar loop.
// The format: include/strata/kernels/exl3.hpp.
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// One EXL3 matrix (mul1 codebook): k inputs, n outputs, K bits (1..8), tiles in exllamav3's natural layout.
struct Exl3Part {
    const uint32_t* trellis = nullptr;
    const uint16_t* suh = nullptr;   // FP16 bits [k]
    const uint16_t* svh = nullptr;   // FP16 bits [n]
    int k = 0, n = 0, K = 0;
};
/// Whether the vector kernel runs here (AVX-512F/BW); the scalar one otherwise.
bool exl3_cpu_vector();
/// xh = H(x * suh) (k values, k % 128 == 0); returns the sum of xh.
float exl3_cpu_prepare(const Exl3Part& m, const float* x, float* xh);
/// One 128-value block of that: xh[i] = H(x * suh)[i] for i < 128 (suh: the block's 128 scales); returns its sum.
float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh);
/// The 128 outputs from column c0 (a multiple of 128) of y = H(x W') * svh, from exl3_cpu_prepare's xh and sum.
void exl3_cpu_band(const Exl3Part& m, const float* xh, float xsum, int c0, float* y);
/// The same in parts: the byte-sum products of the 16-row slices [ks0, ks1) (xh from slice 0) ...
void exl3_cpu_band_acc(const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc);
/// ... and, with the parts' acc summed over every slice, the outputs (the codebook's affine form, H, svh).
void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y);

}  // namespace strata::kernels::cpu
