// include/strata/kernels/q35_dflash.hpp - the kernels of the DFlash2 drafter for the dense qwen35 model.
//
// The drafter (z-lab's DFlash2 for Qwen3.8-27B; the GGUF arch "qwen35-dflash-draft") drafts K tokens in one pass of
// 5 small layers over the block [anchor, <mask> x K] at positions F .. F + K, attending non-causally over the block
// and over a ring of K/V rows made from the target's hidden states at the committed positions (a 2048-position
// sliding window).  Each sublayer is wrapped in a grouped dynamic convolution (2 taps, groups of 16 channels):
//
//   conv(x, dyn, base, side)[t][e] = sum_tap (base[side][tap][e] + dyn[t][side][tap][e / 16]) * x[t - tap][e]
//
// (x[t - tap] = 0 before row 0).  The target's head gives each mask row its top-16 candidates; a low-rank score
// between adjacent rows' candidates re-ranks them and a greedy walk from the anchor picks the drafts.
// Ported from upstream Maya's GLM DFlash2 drafter (src/core/glm_dflash.cu, needmorevram) - the same kernels for a
// head size of 128 - with NInfer's spec (docs/maintainer/qwen3.8-27b-dflash2.md) as the reference.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// rows of x (dense, `n` wide): per head of 128 (n_heads), rms * w then NeoX rope over all 128 dims at pos0 + row
void q35_df_head_norm_rope(float* x, int rows, int n_heads, const float* w, float eps, int64_t pos0, float theta,
                           void* stream);
/// out (rows x E, dense) = conv(x, dyn rows (stride ldd), base, side) over a block of `rows`
void q35_df_conv(const float* x, const float* dyn, int ldd, const float* base, int side, int rows, int E, int taps,
                 int group, float* out, void* stream);
/// context K / V rows (rows x kvd, dense; positions pos0 + r) into the ring (FP16, ring x kvd), slot = pos % ring,
/// and ring_pos[slot] = pos
void q35_df_ring_store(const float* K, const float* V, int rows, int kvd, int64_t pos0, int ring, uint16_t* Kr,
                       uint16_t* Vr, int64_t* ring_pos, void* stream);
/// the block's attention: row i (position n_past + i) over the ring positions [max(0, P - window + 1), n_past) that
/// the ring still holds, and every row of the block (Kb / Vb rows x kvd).  Q rows x n_head x 128 -> out the same.
void q35_df_attn(const float* Q, const uint16_t* Kr, const uint16_t* Vr, const int64_t* ring_pos, const float* Kb,
                 const float* Vb, int rows, int n_head, int n_kv, int64_t n_past, int ring, int window, float* out,
                 void* stream, float* scratch = nullptr);
/// the bytes of q35_df_attn's `scratch` (the split-K form: a kv head's rows together a chunk of the ring)
int64_t q35_df_attn_scratch(int n_head, int rows, int window);
/// the top k (k <= 32) of each of `rows` rows of x (row stride ld) among ids [0, n): ids and values, descending
int64_t q35_df_topk_scratch(int rows, int k);
void q35_df_topk(const float* x, int64_t ld, int n, int rows, int k, int* ids, float* vals, void* scratch,
                 void* stream);
/// the selector walk: tok[i], conf[i] for the rows 0..n-1 (see the header); tables FP16 rows of `rank`
void q35_df_walk(const int* cand, const float* lg, const float* gate, int k, int rank, const uint16_t* pred_tab,
                 const uint16_t* succ_tab, int anchor, int n, int* tok, float* conf, void* stream);
/// the walk drawing each token from q_i = softmax(score / temperature) over its k candidates (u: n device uniforms):
/// tok, conf (q of the drawn one) and qprob [n][k] (q_i in cand's order) - a sampled request's proposal distribution
void q35_df_walk_sample(const int* cand, const float* lg, const float* gate, int k, int rank, const uint16_t* pred_tab,
                        const uint16_t* succ_tab, int anchor, int n, const float* u, float temperature, int* tok,
                        float* conf, float* qprob, void* stream);

/// out (rank floats) = row tok[0] (a device int) of an FP16 table of rows of `rank` (DSpark's Markov W1)
void q35_df_gather_f16(const uint16_t* tab, int rank, const int* tok, float* out, void* stream);

}  // namespace strata::kernels
