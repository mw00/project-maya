// include/strata/kernels/glmd_dec.hpp - glm-dsa L1-4: the decode attention of mla2 (glmd_fast.cu) shaped for the GPU.
//
// mla2's attention (mla2_split_kernel + mla2_combine_kernel) took ~85 us a layer on a V100 at a short context and
// ~240 us at the 2,048 selected cells (a warp_sum chain per head and cell, 16 heads of 32 cells a block, the latents
// staged as 2-byte loads): 6.8 ms of every token's GPU time at a 150-token context, ~19 ms at 2K+.  mla_dec:
//   - the selected cells in tiles of 32, one cell a lane for the scores (a dot of 576 a lane, no shuffle chain), the
//     staged latents FP16 at a 580-half row stride (the lanes' 8-byte reads fall in distinct banks);
//   - a block holds 16 heads (2 a warp) and walks a contiguous range of tiles with an online softmax (f32 m, l, O),
//     so the partials are S x heads whatever the context (S = min(tiles, splits): the device's live count decides,
//     the launch shape is fixed - CUDA-graph replay safe);
//   - the context O = P . L: a lane holds 16 of the 512 values of both its heads, each staged latent read once for
//     both; the combine merges the S partials per head (f32), as mla2_combine did its chunks.
// Arithmetic f32 throughout (FP16 latents as staged by mla2, INT8 records decoded to FP16 the same way); the sums are
// associated differently from mla2's (f32 rounding apart).  kv_lora 512, qk_rope 64, n_head % 16 == 0 (GLM-5.3);
// other shapes keep mla2's kernels.  Where it runs (glmd_dec.cu, auto_device): every CUDA device it fits; on HIP only
// the archs it was measured faster on (gfx1151) unless STRATA_GLMD_MLA_DEC=1; 8 heads a block where a block gets
// 64 KB of shared memory (the AMD GPUs).
#pragma once

#include <cstddef>

namespace glmd {

/// The shapes mla_dec serves, on a device that takes it (see above).
bool mla_dec_ok(int n_head, int kv_lora, int qk_rope);
/// The splits mla_dec uses at most on the current device (2 blocks a multiprocessor over the head groups, <= 64).
int mla_dec_splits(int n_head);
/// The partials' bytes for n_head heads (part: S x n_head x (512 + 2) floats; mla2_work_bytes's part covers it).
size_t mla_dec_part_bytes(int n_head);
/// ctx[h * 512 + c] = sum_s softmax_s(scale (q_abs[h] . lat[s] + q_pe[h] . k_pe[s])) lat[s][c] over the selected
/// cells (cells[0 .. n), -1 = padding; n = min(n_sel, *n_sel_dev) when n_sel_dev is set).  q_pe[h] = q + h q_stride +
/// q_pe_off.  part: scratch of mla_dec_part_bytes.  false (nothing launched) for a shape it does not serve.
bool mla_dec(const float* q_abs, const float* q, int q_stride, int q_pe_off, const void* lat, int lat_q8,
             const unsigned short* k_pe, const int* cells, int n_sel, const int* n_sel_dev, int n_head, int kv_lora,
             int qk_rope, float scale, float* part, float* ctx, void* stream);

}  // namespace glmd
