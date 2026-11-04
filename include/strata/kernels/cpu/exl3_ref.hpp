// include/strata/kernels/cpu/exl3_ref.hpp - the EXL3 format on the CPU, scalar: the reference the device kernels
// are tested against, and the CPU lane's fallback.  The format: include/strata/kernels/exl3.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The codebook value of one 16-bit trellis state (exactly the device's FP16 result, as F32).
float exl3_decode(uint32_t state, int cb);
/// Whether k2 (twice the bits per weight) is a bitrate of the format: 2..16 even (K 1..8), or 3 / 5 / 7 (exllamav3
/// 1.6's half-integer K 1.5 / 2.5 / 3.5, which only the mul1 codebook takes).
bool exl3_k2_valid(int k2, int cb);
/// The 16-bit states of one tile's 256 steps in step order (exllamav3's unpack_trellis): step p's state is the window
/// that ends at bit S(p) = floor((p + 1) k2 / 2) of the tile's 4 k2 words read MSB-first, around the tile - (p + 1) K
/// for an integer K, steps of KA and KA + 1 bits alternating (even p KA) for a half-integer K = KA + 0.5.
void exl3_tile_states_k2(const uint32_t* tile, int k2, uint16_t* st);
/// W'[r][c] (row-major k x n, F32) of the k x n matrix whose first tile is `trellis` (ld tiles per 16-row slice).
void exl3_inner_k2(const uint32_t* trellis, int ld, int k, int n, int k2, int cb, float* w);
/// The blockwise 128-point Sylvester transform over sqrt(128), in place (n % 128 == 0).
void exl3_had128(float* v, size_t n);
/// y = H(H(x * suh) W') * svh with the rotated input rounded to FP16 as the device holds it (suh, svh: FP16 bits).
/// w: W' from exl3_inner_k2.
void exl3_mv(const float* w, const uint16_t* suh, const uint16_t* svh, int k, int n, const float* x, float* y);

}  // namespace strata::kernels::cpu
