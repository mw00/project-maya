// src/ds4/ds4_lane_parity.hpp - the CPU lane's low-bit experts in the device's arithmetic (DS4_LANE_PARITY,
// mw00/project-maya#644): every sum, rounding and its order as ds4_iq.cu's token kernels make them, so an expert gives
// the same bits on either lane and the answers no longer depend on where the placement, the expert cache or the PCIe
// lane puts each expert.
//
// The device's token path (gate_up_kernel / down_kernel, EPI 0): the activation rounded to int8 as Q8_K (an F32 scale a
// 256 values, gguf q8k_quant_kernel); a row's 32-value sub-blocks dealt to LPR lanes (gate / up 32, down 16: lane l
// takes sub-blocks l, l + LPR, ...), each lane summing fma(dx, d * (s0 lo + s1 hi), acc) in that order (lo / hi: the
// sub-block's exact int8 dots of values 0..15 and 16..31), then the lanes added pairwise (lane l + lane l + LPR/2, ...
// down to lane 0's sum); h = the clamped SwiGLU with exp_portable (ds4_par_exp.h), rounded again as Q8_K for the down.
// Here: the same operations on the host (this file is built with -ffp-contract=off), the integer dots by AVX2 where the
// CPU has it (exact either way), checked against the scalar form once at the first use.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::ds4::par {

/// ggml types with device-order rows here: IQ2_XXS, IQ2_S (gate / up), IQ3_XXS, IQ3_S (down) - and each in the other
/// role too.  MXFP4 (a GGUF's expert format, 32 values a block) has them for both roles.  k must be a multiple of 256
/// (MXFP4: of 32).
bool lane_type(int type);
/// The lanes a row of `type` is summed over: 32 for gate / up, 16 for a down - MXFP4: 32 for both, as its kernels.
int lane_lanes(int type, bool gate_up);
/// Bytes of an activation of k values: k int8, then k / 32 F32 scales (each 256-block's scale written 8 times, as the
/// device's xd / hd hold it).
inline size_t act_bytes(int k) { return (size_t) k + (size_t) (k / 32) * 4; }
/// x (k values) -> its activation, byte for byte q8k_quant_kernel's (amax's first index, iscale = -127 / that value,
/// q = min(127, nearest(iscale x)), d = 1 / iscale).
void quant(const float* x, int k, void* act);
/// The same for a weight type `type`: an IQ type's Q8_K (as above), MXFP4's per 32 values q8_quant's (sc = max|x| / 127,
/// q = round-half-even(x / sc), d = sc) - the rounding the device's MXFP4 kernels make.
void quant(int type, const float* x, int k, void* act);
/// h[t][r] = swiglu_portable(gate_r . act[t], up_r . act[t], limit) for rows [r0, r1) of matrices of `type` with k
/// inputs, nt <= 8 activations.
void gate_up(int type, const uint8_t* gate, const uint8_t* up, int k, const void* const* act, int nt, float* const* h,
             int r0, int r1, float limit);
/// y[t][r] = down_r . act[t] for rows [r0, r1) (k = n_ff inputs).
void down(int type, const uint8_t* down, int k, const void* const* act, int nt, float* const* y, int r0, int r1);
/// The same rows where a row is `rs` bytes in memory instead of k's own - a tensor-parallel member's FF band of a
/// whole expert's down: `down` points at the band's first block of row 0, the band's k = ffn values are summed from
/// that block on (the lanes b % 32 of the member's blocks, as the member's own kernel sums them).
void down_band(int type, const uint8_t* down, size_t rs, int k, const void* const* act, int nt, float* const* y,
               int r0, int r1);
/// (tests) the scalar forms of the same, and which form the two above take ("avx2" or "scalar") after their check.
/// The band's rows through the scalar form too (down_band_ref): same bits either form.
void gate_up_ref(int type, const uint8_t* gate, const uint8_t* up, int k, const void* const* act, int nt,
                 float* const* h, int r0, int r1, float limit);
void down_ref(int type, const uint8_t* down, int k, const void* const* act, int nt, float* const* y, int r0, int r1);
void down_band_ref(int type, const uint8_t* down, size_t rs, int k, const void* const* act, int nt, float* const* y,
                   int r0, int r1);
const char* form();
/// One row's sum with `lpr` lanes (the device's gate / up 32, down 16), scalar.
float row_ref(int type, const uint8_t* row, int k, const void* act, int lpr);

}  // namespace strata::ds4::par
