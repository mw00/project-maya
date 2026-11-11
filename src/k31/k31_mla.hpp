// src/k31/k31_mla.hpp - Kimi K3.1's MLA attention core (the 24 full-attention layers), absorbed form, NoPE.
//
// Per query token t and head h, over the keys the lightning indexer selected for t (top-2048, or every visible key
// while there are fewer):
//
//     s_j   = scale * ( q_abs[t,h] . latent[j]  +  q_pe[t,h] . k_pe[j] )      scale = 1/sqrt(192) (key_length_mla)
//     p     = softmax_j(s)
//     ctx   = sum_j p_j * latent[j]                                          the value is the 512-wide latent
//
// with the query packed per head as one 576-wide row [q_abs (kv_lora 512) | q_pe (64)] and the cache row as
// [kv_cmpr (512) | k_pe (64)]: the key is the whole row, the value its first 512.  The 64 "rope" dims are never
// rotated (K3.1 has no RoPE).  Scores and the value sum accumulate in float64 from the float32 inputs, as
// ref/kimi_k31.py mla_layer does - this is the M1 (correct, slow) kernel; the fast path is an M3 microbenchmark.
// The per-head value projection (attn_v_b, 512 -> 128) and attn_output stay ordinary GEMMs; the output gate
// kqv * sigmoid(attn_gate x) is mla_gate_launch.
// Gated by tests/k31_mla_test.cu against tools/kimi_k31_mla_dump.py.
#pragma once

#include <cstdint>

namespace strata::k31 {

inline constexpr int kMlaLatent = 512;        // kv_lora_rank
inline constexpr int kMlaRope = 64;           // the un-rotated key part
inline constexpr int kMlaRow = kMlaLatent + kMlaRope;   // 576
inline constexpr int kMlaMaxSel = 2048;       // index_topk

struct MlaAttnArgs {
    const float* q = nullptr;       // [n_tokens][n_head][576]
    const float* kv = nullptr;      // [n_kv][576]: kv_cmpr then k_pe
    const int32_t* sel = nullptr;   // [n_tokens][k_sel]: selected key positions, valid entries first, then -1
    int n_tokens = 0, n_head = 0, n_kv = 0, k_sel = 0;
    double scale = 0.0;             // 1/sqrt(key_length_mla), in float64 as the oracle's (a float32 1/sqrt(192)
                                    // moves every score by ~3e-8 relative)
    float* ctx = nullptr;           // [n_tokens][n_head][512]
};

/// Returns false, launching nothing, outside the contract: k_sel outside 1..kMlaMaxSel, n_head < 1, n_kv < 1,
/// n_tokens < 0, missing buffers.  Every valid sel entry must lie in [0, n_kv) and be causal for its query (the
/// indexer's job); a token with no valid entry gets ctx = 0.
bool mla_attn_launch(const MlaAttnArgs& a, void* stream);

/// out[i] = kqv[i] * float32(sigmoid(gate_pre[i]))  (the sigmoid in float64, as the oracle's); out may alias kqv.
bool mla_gate_launch(const float* kqv, const float* gate_pre, float* out, int64_t n, void* stream);

}  // namespace strata::k31
