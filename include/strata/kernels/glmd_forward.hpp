// include/strata/kernels/glmd_forward.hpp - the host-side pieces of the glm-dsa F32 reference forward (B3.5).
//
// src/core/glm_dsa_ref.cu (Glm5Model::step_layers_dsa, the MAYA_GLM_SLOW path for glm-dsa) runs every layer with
// the glmd::ref_* kernels (glmd_ref.hpp).  What it decides on the host lives here, in plain C++ that the CPU test
// (tests/glm_dsa/glm_dsa_ref_cpu_test.cpp) runs without a GPU:
//
//   * the MoE router (HF numerics): p = sigmoid(logits); the top-k by p + exp_probs_b (the bias picks experts and
//     is never a weight); w = p[sel] / (sum p[sel] + 1e-20) * scale when normalized; ties by the lower expert id;
//   * the attention selection policy: dense (every cell 0..p) while n_kv <= top_k, the indexer's top-k after;
//     shared layers reuse the selection of the previous full layer of the same token;
//   * the attention scale (1/16 for GLM-5.3, glmd::kGlm53KqScale; 1/sqrt(qk_head) for other synthetic heads);
//   * the dequant tile plan (row chunks of a packed tensor that fit an F32 tile);
//   * the F32 KV carve: one layer's latent (kv_lora) + k_pe (qk_rope) rows and, on full layers, the F32 index keys,
//     placed inside the span Maya's state arena already reserves for that layer (latent | ik | ig | pool).
#pragma once

#include <cstdint>

namespace glmd {

/// The HF / reference MoE router over one token.  ids and w hold k entries (selection order: biased score
/// descending, ties by the lower id).  Accumulates in double.
void route_sigmoid(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale, int* ids,
                   float* w);

/// Attention over every cached cell (no indexer selection) while the context holds at most top_k positions.
inline bool attend_dense(int64_t n_kv, int top_k) { return n_kv <= (int64_t) top_k; }

/// The softmax scale of the absorbed attention for a head of qk_head = qk_nope + qk_rope dims: exactly
/// kGlm53KqScale (1/16) at GLM-5.3's 256, 1/sqrt(qk_head) otherwise (synthetic geometries).
float attn_scale(int qk_head);

/// Rows of `cols` floats that fit a tile of tile_floats (0: not even one row fits).
inline int64_t tile_rows(int64_t cols, int64_t tile_floats) { return cols > 0 ? tile_floats / cols : 0; }

/// Where one layer's F32 reference KV lives, in floats into the state arena.  Maya's arena reserves, per DSA layer,
/// [latent kv_lora x max_ctx | ik idx_key x ring | ig idx_key x ring | pool idx_key x n_pools] back to back; the
/// reference needs kv_row x max_ctx (+ idx_key x max_ctx on a full layer).  kv_carve checks the reserved span is
/// contiguous and large enough and returns the two offsets (ik = -1 on a shared layer); false: allocate elsewhere.
struct KvCarve {
    int64_t kv = -1;   // (kv_lora + qk_rope) floats per position: [c_kv | k_pe]
    int64_t ik = -1;   // idx_key floats per position (full layers)
};
bool kv_carve(int64_t lat, int64_t ik, int64_t ig, int64_t pool, int64_t lat_floats, int64_t ring, int64_t n_pools,
              int idx_key, int kv_row, int64_t max_ctx, bool full, KvCarve& out);

}  // namespace glmd
