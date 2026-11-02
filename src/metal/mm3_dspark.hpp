// src/metal/mm3_dspark.hpp - the DSpark draft model (MiniMax's block drafter) on Metal, the
// counterpart of src/mm3/mm3_dspark.hpp for the maya-metal-mm3 engine.  The algorithm is the
// CUDA engine's: the draft reads the target's residual streams after layers 3, 17, 31, 45, 59
// (its taps) through an fc into a context whose per-layer K/V it keeps; a proposal runs the
// block [anchor, mask x (block-1)] through its 5 layers attending (not causally) over the last
// `window` context positions and the block itself, then a Markov head (rank 256) picks the
// drafts left to right, each with a confidence.  The target verifies; nothing here changes what
// the target would say - it only guesses sooner.
//
// Weights: the HF safetensors (peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF, dspark/) read
// directly; the MXFP8 matrices are expanded to BF16 at the load (each value exact: an E4M3
// times a power of two), the norms are Gemma-style (the stored weight + 1 at use).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace maya::metal {

struct Mm3DraftGeo {
    int dim = 0, vocab = 0, n_layers = 0, n_head = 0, n_kv = 0, hd = 0, n_rot = 0;
    int ff = 0, block = 0, rank = 0, mask_id = 0, window = 0;
    float eps = 1e-6f, theta = 0.0f, alpha = 0.0f, limit = 0.0f;
    std::vector<int> taps;   // the target layers whose residual streams feed the context
};

class Mm3Draft {
public:
    Mm3Draft();
    ~Mm3Draft();
    Mm3Draft(const Mm3Draft&) = delete;
    Mm3Draft& operator=(const Mm3Draft&) = delete;

    // dir holds config.json and model-dspark-00001.safetensors.  emb/head are the target's
    // embedding and output tensors (any ggml type; owned by the caller, must outlive this).
    bool load(const std::string& dir, ggml_tensor* emb, ggml_tensor* head, int64_t max_ctx,
              std::string& err);
    bool loaded() const { return p != nullptr; }
    const Mm3DraftGeo& geo() const;
    int64_t max_ctx() const;

    // feed the context with the taps of T committed rows at positions pos0..pos0+T-1.
    // taps: [n_taps][T][dim] F32, tap-major (tap j's rows T*dim after tap j-1's).
    bool context(const float* taps, int T, int64_t pos0, std::string& err);
    void reset_context();   // a new conversation: the cache rows are position-indexed garbage
                            // until overwritten, so this only rewinds the high-water mark
    int64_t context_hi() const;   // one past the highest position the context has been fed

    // up to n (<= block) drafts after `anchor` (at position pos; the context must hold pos-1
    // and before).  Drafts and their confidences out.  False only on a hard error; an empty
    // drafts vector with no error means "no room" (pos + block past the cache).
    bool propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts,
                 std::vector<float>& confs, std::string& err);

    // the last proposal's final hidden states [block][dim] and logits [block][vocab] (tests)
    const std::vector<float>& last_hid() const;
    const std::vector<float>& last_logits() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace maya::metal
