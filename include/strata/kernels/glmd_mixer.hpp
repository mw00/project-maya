// include/strata/kernels/glmd_mixer.hpp - the glm-dsa decode mixer's own kernels (glm-dsa B4.6, B14.3, B14.4).
//
// What the B4.3-B4.5 kernels (glmd_fast.hpp) leave to the mixer:
//   idx_score   the token-level lightning-indexer score of one decode token over the index-key cache, cells 0..p:
//               score[s] = sum_h w_h * relu(iq_h . ik_cache[s]).  The per-cell arithmetic is Maya's dsa_score_kernel
//               (glm_fast.cu: a warp per cell, lane-strided dot, xor butterfly, heads in order), which already fits
//               token-level keys (a "pool" of one position); it is repeated here because Maya's launch sizes its grid
//               from the host count, and a CUDA graph replayed at the next position would score one cell too few.
//               This one reads n = min(n_cap, *p_dev + 1) on the device, runs a fixed grid, and also writes the
//               attention's cell count min(n, sel_cap) for glmd::mla2's n_sel_dev (B14.4).
//               B10.8: the keys may be FP16 rows (key_f16; the same dot, each key widened to f32 as it is read).
//               glm-dsa wave 5: the decode policy (glmd_decode_policy.hpp) serves the same request with the WMMA scorer
//               at long context (wmma_args() derives its IdxScoreWmma from this struct; n_grid is its launch extent).
//   f32_to_bf16 round-to-nearest-even BF16 copies of F32 attn_k_b / attn_v_b (glmd::mla2 reads BF16 only; a pack
//               whose k_b / v_b are small enough to stay F32 gets them converted once at setup).
//   iota        cells 0..n-1 (the dense-attention switch, MAYA_GLM_DSA_DENSE=1, B14.3).
// CPU twins in glmd::twin take the same structs with host pointers (p_dev and n_sel_out are host ints there).
#pragma once

#include <cstddef>
#include <cstdint>

namespace glmd {

struct IdxScore {
    const float* iq = nullptr;        // (idx_dim, idx_heads), rotated
    const float* ik_cache = nullptr;  // F32 [n_cap][idx_dim], cell s = position s (FP16 rows cast when key_f16)
    const float* w = nullptr;         // (idx_heads), already scaled by 1 / sqrt(idx_heads * idx_dim)
    int idx_heads = 32, idx_dim = 128;
    int n_cap = 0;                    // cells the cache holds (max_ctx): fixes the grid
    int p = 0;                        // the position when p_dev is null; ALSO the host's copy when p_dev is set (the
                                      // decode policy's n = p + 1 and the WMMA scorer's extent read it)
    const int* p_dev = nullptr;       // device position (wins when set): n = min(n_cap, p + 1)
    float* score = nullptr;           // n_cap floats; [0, n) written (unless skipped)
    int skip_le = 0;                  // n <= skip_le: no score written (the selection is the identity, B4.5)
    int* n_sel_out = nullptr;         // optional: *n_sel_out = min(n, sel_cap)
    int sel_cap = 2048;
    int key_f16 = 0;                  // B10.8: ik_cache holds FP16 rows (uint16_t bits) - the state layout's dtype
    int n_grid = 0;                   // the WMMA scorer's launch extent (0: p + 1; a decode graph: its capacity);
                                      // the token scorer's grid is fixed by n_cap and ignores it
};
/// One launch of a fixed grid (graph-replay safe).  idx_dim <= 512, idx_heads * idx_dim <= 12288.
bool idx_score(const IdxScore& a, void* stream);
/// y[i] = BF16(x[i]), round to nearest even (NaN stays NaN).
bool f32_to_bf16(const float* x, uint16_t* y, size_t n, void* stream);
/// cells[i] = i for i < n.
bool iota(int* cells, int n, void* stream);
/// Launch errors and refused shapes of this file's kernels.
int mixer_launch_errors();

namespace twin {
void idx_score(const IdxScore& a);
uint16_t bf16_bits(float f);
}  // namespace twin

}  // namespace glmd
