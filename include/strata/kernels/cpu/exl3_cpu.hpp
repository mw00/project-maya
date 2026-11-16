// include/strata/kernels/cpu/exl3_cpu.hpp - EXL3 routed experts on the CPU (the fast path's CPU lane).
//
// The mul1 codebook is affine in the byte sum of state * 0x83DCD12D: w = fp16(1024 + s) * k_inv + k_bias (one FP16
// FMA on the device).  The CPU keeps the affine form without the FP16 rounding, so a 128-output band is
//   y' = k_inv * sum_k xh_k s_kc + (1024 k_inv + k_bias) * sum_k xh_k,   xh = H(x * suh)
// and y = H(y') * svh.  The byte sums come from AVX-512 (permutes and funnel shifts pull 16 columns' states at once,
// vpmaddubsw / vpmaddwd add the bytes up, an F32 FMA takes the activation), or from AVX2 + FMA the same way 8 columns
// at a time - the same FMAs in the same order, so the two give the same bits; a CPU with neither runs a scalar loop.
// The format: include/strata/kernels/exl3.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// One EXL3 matrix (mul1 codebook): k inputs, n outputs, tiles in exllamav3's natural layout, the bitrate as k2 = 2 K:
/// 2..16 even (K 1..8) or 3 / 5 / 7 (exllamav3 1.6's half-integer K 1.5 / 2.5 / 3.5; exl3.hpp has the format).
struct Exl3Part {
    const uint32_t* trellis = nullptr;   // [k/16][n/16][4 k2] words
    const uint16_t* suh = nullptr;       // FP16 bits [k]
    const uint16_t* svh = nullptr;       // FP16 bits [n]
    int k = 0, n = 0, k2 = 0;
};
/// Whether the CPU lane takes this bitrate (k2 2..16 even, or 3 / 5 / 7).
inline bool exl3_cpu_k2_ok(int k2) { return (k2 >= 2 && k2 <= 16 && (k2 & 1) == 0) || k2 == 3 || k2 == 5 || k2 == 7; }
/// Whether a vector kernel runs here (AVX-512F/BW or AVX2 + FMA); the scalar one otherwise.
bool exl3_cpu_vector();
/// Whether the CPU lane's kernels exist here at all.  They are x86-only (the portable build has no version):
/// false means the GPU computes every EXL3 expert, so the CPU lane must stay off.
bool exl3_cpu_available();
/// The kernel the functions below run here: 2 AVX-512 (F, BW), 1 AVX2 + FMA (+ F16C), 0 scalar.  MAYA_FORCE_AVX2=1
/// keeps an AVX-512 CPU on AVX2 (to compare the two; they give the same bits).
int exl3_cpu_tier();
/// MAYA_EXL3_AVX2_SHUF=1: the AVX2 tier takes a state whose window shift is a whole number of
/// bytes (8, 16 or 24 bits) with byte shuffles instead of vector shifts - for Haswell / Broadwell, where the shifts
/// share the one port the multiplies and the byte sums are bound by.  The same states, so the same bits; off by
/// default.  Read at the first AVX2 call and checked there against the shifts (kept off if any sum differs, with a
/// line on stderr either way).  exl3_cpu_avx2_shuf says whether it is on; exl3_cpu_set_avx2_shuf sets it for the
/// process, unchecked (tests: both forms in one run).  It changes tier 1 alone (exl3_cpu_band_acc_on's too).
bool exl3_cpu_avx2_shuf();
void exl3_cpu_set_avx2_shuf(bool on);
/// xh = H(x * suh) (k values, k % 128 == 0); returns the sum of xh.
float exl3_cpu_prepare(const Exl3Part& m, const float* x, float* xh);
/// One 128-value block of that: xh[i] = H(x * suh)[i] for i < 128 (suh: the block's 128 scales); returns its sum.
float exl3_cpu_prepare_block(const uint16_t* suh, const float* x, float* xh);
/// The 128 outputs from column c0 (a multiple of 128) of y = H(x W') * svh, from exl3_cpu_prepare's xh and sum.
void exl3_cpu_band(const Exl3Part& m, const float* xh, float xsum, int c0, float* y);
/// The same in parts: the byte-sum products of the 16-row slices [ks0, ks1) (xh from slice 0) ...
void exl3_cpu_band_acc(const Exl3Part& m, const float* xh, int c0, int ks0, int ks1, float* acc);
/// exl3_cpu_band_acc for n activations of the same matrix at once (xh[i] -> acc[i], each 128 floats): each tile row is
/// decoded once for up to kExl3CpuMaxN of them, and acc[i] is exl3_cpu_band_acc's for xh[i] alone, bit for bit.
constexpr int kExl3CpuMaxN = 8;
void exl3_cpu_band_acc_n(const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                         float* const* acc);
/// The byte-sum products of the slices [ks0, ks1) for EVERY column (acc: m.n floats, overwritten) - a split-K piece,
/// whose tile rows lie in memory order where a band's are 1 KB runs a whole tile row apart (from DRAM, ~1.5x the
/// bands' speed).  Each column's sums are exl3_cpu_band_acc's over the same slices, bit for bit.
void exl3_cpu_rows_acc(const Exl3Part& m, const float* xh, int ks0, int ks1, float* acc);
/// ... for n activations at once (xh[i] -> acc[i]), each tile row decoded once; acc[i] is the call for xh[i] alone.
void exl3_cpu_rows_acc_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, float* const* acc);
/// ... over the columns [c0, c1) alone (multiples of 128; acc[i] indexed by column, only those overwritten): a piece cut
/// by columns too, for more jobs than pieces - each column's sums are the whole piece's, bit for bit.
void exl3_cpu_rows_acc_cols_n(const Exl3Part& m, const float* const* xh, int n, int ks0, int ks1, int c0, int c1,
                              float* const* acc);
/// The byte-sum products of nb bands from c0 (128 nb columns; acc[i]: 128 nb floats) over every input for any n
/// activations, as a GEMM: each 256-input chunk of a band decoded once into F32 in `wc` (exl3_cpu_gemm_scratch()
/// floats, 64-byte aligned, the caller's), then multiplied for all n, the chunk of the activations kept in L2 across
/// the nb bands - for many tokens an expert (the prompt path), where the band kernels decode a tile row again every
/// kExl3CpuMaxN.  The same sums as exl3_cpu_band_acc, in its order.
size_t exl3_cpu_gemm_scratch();
void exl3_cpu_band_gemm(const Exl3Part& m, const float* const* xh, int n, int c0, int nb, float* const* acc,
                        float* wc);
/// exl3_cpu_band_acc_n on a given tier (0, 1, 2, 3 as exl3_cpu_tier; 4: the AVX2 tier with exl3_cpu_set_w16's 16-bit
/// lanes) - tests and benchmarks; false (nothing computed) where this CPU cannot run it.
bool exl3_cpu_band_acc_on(int tier, const Exl3Part& m, const float* const* xh, int n, int c0, int ks0, int ks1,
                          float* const* acc);
/// The AVX2 tier's codebook product and byte sum in 16-bit lanes, a tile row's two halves at once: two vpmullw, one
/// vpmulhuw and two vpmaddubsw for 16 weights where the 32-bit path takes two vpmulld (two uops each), two vpmaddubsw
/// and two vpmaddwd - fewer uops on port 0, which every vector integer multiply and shift shares on Haswell and
/// Broadwell.  The same integer byte sums, so the same bits.  Calls of 1-3 activations and the band GEMM's decode; more
/// activations keep the 32-bit path.  Off by default; process-wide, for the calls that start after it; the other tiers
/// ignore it.
void exl3_cpu_set_w16(bool on);
/// Whether the kernels above take that path now: set, and this CPU's lane runs the AVX2 tier.
bool exl3_cpu_w16();
/// ... and, with the parts' acc summed over every slice, the outputs (the codebook's affine form, H, svh).
void exl3_cpu_band_out(const Exl3Part& m, const float* acc, float xsum, int c0, float* y);

}  // namespace strata::kernels::cpu
