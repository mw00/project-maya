// src/metal/mm3_metal.hpp - MiniMax-M3.1-Flash on Apple Silicon: Maya's GGUF set (peasantsmith's Maya-XS /
// Maya-S quants) through ggml's Metal backend, beside the GLM engine (glm_metal.hpp).
//
// The forward follows tools/mm3/ref.py (transformers' MiniMaxM3VL arithmetic) op for op: Gemma RMSNorms
// (the GGUF's norm weights already hold 1 + w), per-head q/k norms, NeoX partial RoPE, MiniMax Sparse
// Attention on every layer (a 4-head indexer over one shared key per token, 128-token blocks, the top
// blocks kept), swigluoai MLPs - layers under leading_dense_block_count dense, the rest 128 experts top-4
// plus a shared expert, sigmoid routing with the correction bias.  One slice runs two graphs per layer:
// the first writes the BF16 KV and index-key caches and the indexer's scores, the host picks the kept
// blocks from those scores exactly as ref.py's msa_select does, and the second runs the sparse attention
// and the MLP with every expert resident (the Maya quants are built to fit a 128 GB Mac whole).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace maya::metal {

struct Mm3Options {
    bool use_gpu = true;      // false: ggml-cpu (the oracle backend)
    int cpu_threads = 0;      // 0: the performance cores
    int n_ubatch = 512;       // prompt tokens per slice
    int attn_chunk = 64;      // query rows per attention chunk inside a layer graph
};

struct Mm3Geometry {
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0, n_kv = 0, head_dim = 0, n_rot = 0;
    float rope_theta = 0.0f, rms_eps = 0.0f;
    int n_vocab = 0;
    int n_ff_dense = 0, n_ff_exp = 0, n_ff_shared = 0;
    int n_dense_lead = 0;
    int n_expert = 0, n_expert_used = 0;
    float routed_scale = 1.0f, swiglu_alpha = 1.702f, swiglu_limit = 7.0f;
    int ix_heads = 0, ix_dim = 0, ix_blk = 128, ix_topk = 16, ix_local = 1;
    int64_t ctx_train = 0;
    std::vector<int32_t> eos;   // tokenizer.ggml.eos_token_id (the serve loop's stop tokens)
};

struct Mm3Stats {
    size_t weight_bytes = 0, state_bytes = 0;
    uint64_t graphs = 0, tokens = 0;
    double gpu_seconds = 0, host_seconds = 0;
};

class Mm3Metal {
  public:
    Mm3Metal();
    ~Mm3Metal();
    Mm3Metal(const Mm3Metal&) = delete;
    Mm3Metal& operator=(const Mm3Metal&) = delete;

    void set_options(const Mm3Options& o);
    // path: one .gguf of the set (a split set's any shard names them all); the dense layers' indexers may
    // sit in the set's <stem>-maya-indexers.bin beside the shards, which is read too
    bool load(const std::string& path, int64_t max_ctx, std::string& err);
    void reset();
    // the caches are positional (a token's entries sit at its position and are never read past the
    // current position), so a snapshot is just a position: restoring rewinds to it
    void snapshot_save();
    bool snapshot_restore();
    int64_t snapshot_position() const;
    int ubatch() const;
    // appends ids at the current position; logits (n_vocab) of the last token
    bool forward(const std::vector<int32_t>& ids, std::vector<float>& logits, std::string& err);
    int64_t position() const;
    int64_t max_context() const;
    const Mm3Geometry& geometry() const;
    const Mm3Stats& stats() const;
    std::string describe() const;

    struct Impl;

  private:
    std::unique_ptr<Impl> p_;
};

}  // namespace maya::metal
