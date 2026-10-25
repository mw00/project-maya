// src/mm3/mm3_dspark.hpp - maya-mm3's DSpark draft (MiniMax-M3.1-Flash's dspark/: vLLM's MiniMaxM3DSparkForCausalLM).
//
// Context: the target's residual streams after layers 3, 17, 31, 45 and 59 of every token, concatenated, through fc
// and the hidden norm; each of the 5 draft layers keeps the K / V of that (its k / v projections, k normed and
// rotated at the token's position) for the last `window` positions, a ring.  A proposal: the block [anchor, mask x 6]
// at the positions after the context through the 5 layers (Gemma norms, per-head QK norm, partial RoPE, attention
// over the ring's window and the whole block - not causal -, a dense swigluoai MLP), the final norm, the target's
// head; then left to right each position's logits plus the Markov bias of the token before it (markov_w2 .
// markov_w1[prev], prev the anchor first), greedy, and its confidence sigmoid(proj . [hidden ; markov_w1[prev]] + b).
//
// The weights: the MXFP8 matrices expanded to BF16 (exact), fc / the Markov tables / the confidence head as stored
// (BF16), on one GPU with the target's head.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "mm3_exl3.hpp"
#include "mm3_pack.hpp"

namespace mm3 {

struct DraftGeo {
    int dim = 0, n_layers = 0, n_head = 0, n_kv = 0, hd = 0, n_rot = 0, ff = 0, vocab = 0, rank = 0;
    int block = 7, window = 4096, mask_id = 0;
    std::vector<int> taps;   // the target layers whose output streams the context takes
    float eps = 1e-6f, alpha = 1.702f, limit = 7.0f, theta = 1e7f;
};

class Draft {
public:
    Draft();
    ~Draft();
    Draft(const Draft&) = delete;
    Draft& operator=(const Draft&) = delete;

    /// Loads the draft (the pack's dspark.* tensors) onto device dev, sharing the target's head (on dev) and its
    /// embeddings (emb: the mapped BF16 table).  max_rows: the most context rows one call takes.
    bool load(const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas, const strata::mm3::exl3::Mat* head,
              const uint16_t* emb, int max_rows, std::string& err);
    const DraftGeo& geo() const;
    /// The context of positions pos0 .. pos0 + T - 1 from the taps (device, on dev: [T][taps x dim] F32).
    bool context(const float* taps, int T, int pos0, std::string& err);
    /// The same in steps, T <= max_rows: tap j's rows (device [T][dim] F32) into the context - the taps in order, 0
    /// first - then the rest; on the draft's stream (a forward's taps can go in as its layers pass them).
    bool context_tap(int j, const float* rows, int T, std::string& err);
    bool context_finish(int T, int pos0, std::string& err);
    /// Proposes n (<= block) tokens after `anchor` at position pos (the context holds pos - 1 and before): drafts
    /// and their confidences (host).  logits (if given): each position's biased logits [n][vocab] (host), the
    /// distribution the drafts were taken from.
    bool propose(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, float* logits,
                 std::string& err);
    /// Device bytes the draft holds.
    size_t bytes() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
