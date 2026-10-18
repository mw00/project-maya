// include/strata/kernels/cpu/exl3_ref.hpp - the EXL3 format on the CPU, scalar: the reference the device kernels
// are tested against, and the CPU lane's fallback.  The format: include/strata/kernels/exl3.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The codebook value of one 16-bit trellis state (exactly the device's FP16 result, as F32).
float exl3_decode(uint32_t state, int cb);
/// The 16-bit states of one tile's 256 steps in step order (exllamav3's unpack_trellis): step p's state is the window
/// that ends at bit (p + 1) K of the tile's 8 K words read MSB-first, around the tile.
void exl3_tile_states(const uint32_t* tile, int K, uint16_t* st);
/// W'[r][c] (row-major k x n, F32) of the k x n matrix whose first tile is `trellis` (ld tiles per 16-row slice).
void exl3_inner(const uint32_t* trellis, int ld, int k, int n, int K, int cb, float* w);
/// The blockwise 128-point Sylvester transform over sqrt(128), in place (n % 128 == 0).
void exl3_had128(float* v, size_t n);
/// y = H(H(x * suh) W') * svh with the rotated input rounded to FP16 as the device holds it (suh, svh: FP16 bits).
/// w: W' from exl3_inner.
void exl3_mv(const float* w, const uint16_t* suh, const uint16_t* svh, int k, int n, const float* x, float* y);

}  // namespace strata::kernels::cpu
