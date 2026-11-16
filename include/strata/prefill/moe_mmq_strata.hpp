// include/strata/prefill/moe_mmq_strata.hpp - glm-dsa's view of the vendored MMQ (B9.8, B14.1, B14.2), compiled into
// strata_mmq (src/prefill/moe_mmq_strata.cu) so it sees ggml-cuda's mmq.cuh exactly as moe_mmq.cu does.
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill::mmq_strata {

/// Bytes of one block_q8_1_mmq (144: 16 B of scales / sums, then 128 int8).
int block_bytes();
/// Blocks of `rows` activation rows of `cols` values as mmq::quantize lays them out (rows padded to 512 values).
int64_t blocks(int64_t rows, int64_t cols);
/// FP16 halves at the head of each block for weights of `ggml_type`: 8 for the DS4 / D2S6 layouts, 0 for D4
/// (f32 scales, nothing to clamp).  0 for a type MMQ does not cover.
int ds_halves(int ggml_type);

/// B14.2: for `ggml_type` on compute capability `cc` (700: V100) with `smpbo` bytes of opt-in shared memory, every
/// tile width J the host switch (mul_mat_q_switch_J) can pick, for both the aligned and the fallback (nrows % 128)
/// kernels and every column count 1..max_cols, has a real device instance in this build: the host config equals the
/// config the sm_70 device code compiles (no NO_DEVICE_CODE stub), it fits smpbo, and some J is always found.
/// False with the reason in `why`.
bool sm70_check(int ggml_type, int cc, int64_t smpbo, int max_cols, std::string& why);

}  // namespace strata::prefill::mmq_strata
