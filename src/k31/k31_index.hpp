// src/k31/k31_index.hpp - Kimi K3.1's DSA lightning indexer (the 24 MLA layers): the per-token key, the scores and
// the top-2048 selection the MLA attention reads (k31_mla.hpp's `sel`).
//
// ref/kimi_k31.py indexer(), per query t at position p = p0 + t, over the keys at positions 0..p (NoPE, no rotation):
//
//     key[s]    = layer_norm(indexer.attn_k x_s; k_norm w, b, eps)             one 128-wide key a token, all heads
//     score[s]  = sum_h relu(q[t,h] . key[s]) * w[t,h] * (H*D)^-0.5            q = attn_q_b qr, w = indexer.proj x
//     sel       = the top 2048 of score[0..p], ties to the lower position       (all of 0..p while p < 2048)
//
// The keys live in a per-sequence cache, float32 or BF16 (the plan's format: 256 B a token a layer); the key kernel
// writes this call's rows at p0..p0+T-1 (and optionally the float32 seam).  Scores accumulate in float32 (128
// products a head, then the weighted sum of the relu'd heads); the oracle's are float64.  The top-k is a radix select
// on the scores' order keys: the picks above the k-th score in position order, then its ties in position order (the
// lowest positions win the ties, as top_k_desc's stable sort), then -1 to k.  The MLA kernel reads the picks as a
// set (any order, valid prefix).
//
// index_score and topk are DeepSeek's (src/ds4/ds4_kernels.cu index_score_kernel / topk_kernel), copied (plan D2)
// and adapted: K3.1's causal rows (no compressor, no candidate blocks), a float32 key format beside BF16, the
// (H*D)^-0.5 prescale folded into w in float64, head dims 32/64/128.  The segmented top-k for decode rows past 32K
// positions (topk_seg_kernel + topk_merge_kernel, the same picks faster) is M3 work.
// Gated by tests/k31_index_test.cu against tools/kimi_k31_indexer_dump.py.
#pragma once

#include <cstdint>

namespace strata::k31 {

inline constexpr int kIdxHeads = 32;          // attention.indexer.head_count
inline constexpr int kIdxDim = 128;           // attention.indexer.key_length
inline constexpr int kIdxTopK = 2048;         // attention.indexer.top_k
inline constexpr int kIdxMaxHeads = 32;       // the score kernel's per-thread accumulators

enum class IdxKeyFmt : int { kF32 = 0, kBf16 = 1 };

struct IndexKeyArgs {
    const float* kpre = nullptr;    // [n_tokens][dim]: indexer.attn_k x
    const float* ln_w = nullptr;    // [dim] indexer.k_norm
    const float* ln_b = nullptr;    // [dim] indexer.k_norm bias
    double eps = 1e-5;              // attention.layer_norm_epsilon
    int n_tokens = 0, dim = 0;
    long long p0 = 0;               // this call's rows are cache rows p0 .. p0 + n_tokens - 1
    IdxKeyFmt fmt = IdxKeyFmt::kBf16;
    void* cache = nullptr;          // [n_ctx][dim] float32 or BF16
    float* k_out = nullptr;         // optional [n_tokens][dim]: the float32 keys (the indexer_k seam)
};

/// Returns false, launching nothing, outside the contract: n_tokens < 0, dim outside 1..1024, p0 < 0, eps not > 0,
/// missing buffers, or a CUDA error already pending.  n_tokens == 0 is a no-op.
bool index_key_launch(const IndexKeyArgs& a, void* stream);

struct IndexScoreArgs {
    const float* q = nullptr;       // [n_tokens][n_head][dim]
    const float* w = nullptr;       // [n_tokens][n_head]: indexer.proj x, before the prescale
    const void* keys = nullptr;     // the cache: [>= n_kv][dim] in fmt
    IdxKeyFmt fmt = IdxKeyFmt::kBf16;
    int n_tokens = 0, n_head = 0, dim = 0, n_kv = 0;   // n_kv: the score rows' width (keys 0 .. n_kv - 1)
    long long p0 = 0;               // query t is at position p0 + t and sees keys 0 .. p0 + t
    double prescale = 0.0;          // (n_head * dim)^-0.5, folded into w in float64 and rounded once
    float* score = nullptr;         // [n_tokens][score_stride]: positions past p0 + t are -inf
    long long score_stride = 0;
};

/// Returns false, launching nothing, outside the contract: n_head outside 1..kIdxMaxHeads, dim not 32, 64 or 128,
/// n_tokens < 0, n_kv < 1, p0 < 0, p0 + n_tokens > n_kv (a query past the cache), score_stride < n_kv, prescale not
/// > 0, missing buffers, or a CUDA error already pending.  n_tokens == 0 is a no-op.
bool index_score_launch(const IndexScoreArgs& a, void* stream);

struct IndexTopkArgs {
    const float* score = nullptr;   // [n_tokens][score_stride], as index_score writes it
    long long score_stride = 0;
    int n_tokens = 0, n_kv = 0, k = kIdxTopK;
    long long p0 = 0;               // query t sees positions 0 .. p0 + t
    int32_t* sel = nullptr;         // [n_tokens][sel_stride]: min(k, p0 + t + 1) positions, then -1 to k
    int sel_stride = 0;
};

/// Returns false, launching nothing, outside the contract: k < 1, n_tokens < 0, n_kv < 1, p0 < 0,
/// p0 + n_tokens > n_kv, score_stride < n_kv, sel_stride < k, missing buffers, or a CUDA error already pending.
bool index_topk_launch(const IndexTopkArgs& a, void* stream);

}  // namespace strata::k31
