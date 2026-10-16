// include/strata/core/glm_dflash.hpp - the DFlash2 block-diffusion drafter for the GLM trunk (src/core/glm_dflash.cu).
//
// A DFlash GGUF (general.architecture "dflash", llama.cpp's / beellama's schema) is a small dense transformer that
// drafts a whole block of tokens in one pass: its input is [the last accepted token, <mask> x (block - 1)] at the
// positions after the context, it attends non-causally over that block and over a K/V cache built from the TARGET's
// hidden states (dflash.target_layers: the residual entering those trunk layers - for GLM-5.3-Flash's 4-stream mHC
// residual, the mean of the streams, as sglang's capture), and its outputs at the mask positions are the drafts.
// DFlash2 adds a grouped dynamic convolution around each sublayer and a candidate SELECTOR: the target's output head
// gives each position its top-k candidates, a low-rank transition score between adjacent positions' candidates
// re-ranks them, and the greedy walk through that lattice is the draft (beellama's draft-dflash, src/models/dflash.cpp
// and common/speculative.cpp).  The drafter has no embedding or head of its own: it reads the target's.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// A split's DFlash verify window: pipelined (consecutive tokens on the parts at once, glm_window.cu's win_pipe_*) or
/// batched (all of the window a part at a time, staggered, each part's CPU lane on both parts' pools).
/// STRATA_GLM_WIN_PIPE=1 / 0 forces either; by default pipelined from three parts up.  Maya-L, 3090 + 5060 Ti:
/// batched 21.8 tok/s against MTP's 20.5 (pipelined 17.3); nine GPUs: pipelined 59.8-62.0 against MTP's 42.5.
inline bool dflash_pipelined(int n_parts) {
    if (n_parts < 2) return false;
    const char* v = std::getenv("STRATA_GLM_WIN_PIPE");
    if (v != nullptr && v[0] != '\0') return std::atoi(v) != 0;
    return n_parts >= 3;
}

struct WSlot;

class GlmDflash {
public:
    GlmDflash();
    ~GlmDflash();
    GlmDflash(const GlmDflash&) = delete;
    GlmDflash& operator=(const GlmDflash&) = delete;

    /// Reads the GGUF and puts its weights, its K/V ring and its buffers on CUDA device `dev` (the device of the
    /// target's last part: its output head).  Before the target's expert pool is sized, so the pool leaves room.
    bool load(const std::string& path, int dev, std::string& err);
    /// Checks the drafter against the target it serves (hidden size, vocabulary, the trunk layers it reads).
    bool bind(int target_embd, int target_vocab, int target_layers, std::string& err);
    int dev() const;
    cudaStream_t stream();                         // its own stream on dev() (created on first use; null on failure)
    size_t bytes() const;                          // device bytes it holds
    int block() const;                             // dflash.block_size: the anchor and up to block - 1 drafts
    int ctx() const;                               // the context positions a draft attends to (its sliding window)
    const std::vector<int>& target_layers() const; // the trunk layers whose inputs it reads, in its feature order
    std::string describe() const;

    /// The pinned staging the trunk's passes copy their captured rows into: n_embd floats for target layer k of
    /// position p.  staged(p, k) records that the copy was queued (any thread).
    float* stage_row(int64_t p, int k) const;
    size_t stage_pitch() const;                    // bytes from a position's row of layer k to the next position's
    int ring_size() const;                         // the staging's and the ring's positions (p goes to p % ring)
    void staged(int64_t p, int k);
    /// Positions [lo, hi) - staged, their passes done - into the K/V ring (on `s`, the drafter's device's stream).
    /// Positions already there or not staged are skipped.  Returns false on a CUDA error.
    bool inject(int64_t lo, int64_t hi, cudaStream_t s, std::string& err);
    /// A draft: the block [anchor, <mask>...] at positions n_past.. (the context: the ring's positions before
    /// n_past), up to n_max tokens; the walk stops at a position whose selector confidence is below p_min.  `head`:
    /// the target's output head on this device; `emb` / `emb_type`: the target's token embedding rows on the host.
    bool draft(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head, const uint8_t* emb,
               int emb_type, cudaStream_t s, std::vector<int32_t>& out, std::string& err,
               std::vector<float>* conf = nullptr);   // conf: every walked position's selector confidence
    /// draft() in two calls: draft_start queues the pass on `s` (the caller's work runs meanwhile), draft_finish waits
    /// for it and returns the drafts
    bool draft_start(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head, const uint8_t* emb,
                     int emb_type, cudaStream_t s, std::string& err);
    // (n_min: the first n_min drafts kept whatever their confidence; p_cum > 0: a draft only while the product of the
    //  confidences up to it is at least p_cum)
    bool draft_finish(std::vector<int32_t>& out, std::string& err, std::vector<float>* conf = nullptr, int n_min = 0,
                      float p_cum = 0.0f);
    /// The target's head split for the drafts' head (Glm5Model::head_split_link): its columns [0, c0) here (`lo`) and
    /// [c0, vocab) on device `hdev` (`hi`, on its stream `hs`) at once, each side its own slices' top-k candidates,
    /// the helper's copied in for the merge.  On the helper: `hx` (rows x n_embd floats), `hlg` (rows x (vocab - c0)
    /// floats), `hcand` (cand_bytes).  A draft of more rows, or a c0 off the top-k's slices, keeps the whole head here.
    /// STRATA_GLM_DFLASH_HEAD_SPLIT=0: off.  false: its pinned rows did not allocate
    bool split_head(const WSlot* lo, const WSlot* hi, int c0, int hdev, cudaStream_t hs, float* hx, float* hlg,
                    void* hcand, size_t cand_bytes, int rows, std::string& err);
    /// Forget every cached position at or past `from` (0: a new sequence, or a state loaded from elsewhere; else the
    /// sequence went back to `from` and the positions after it will hold other tokens).
    void reset(int64_t from = 0);
    /// The last draft's mask rows (tests): their final normed hidden states (n x hidden), their candidates and logits
    /// (n x top_k), the walk's picks and confidences (n).
    bool last_rows(int n, std::vector<float>& hidden, std::vector<int>& cand, std::vector<float>& logit,
                   std::vector<int>& pick, std::vector<float>& conf) const;
    int top_k() const;

    struct Impl;

private:
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::core
