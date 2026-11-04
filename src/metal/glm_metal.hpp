// src/metal/glm_metal.hpp - GLM-5.3-Flash (glm5-next) on Apple Silicon through ggml's Metal backend.
//
// The whole forward is a ggml graph: ggml-metal already has every op this architecture needs (gated delta net for
// KDA, the mHC hyper-connection ops, MoE mul_mat_id, the i-quant matvecs), so the numerics are the pinned
// llama.cpp's - the same code Maya's CUDA kernels were validated against.  The graph follows
// ref/upstream-llama.cpp/glm5-next.cpp op for op; where ref/glm.py and that file differ, glm.py's
// fixtures (data/glm5-synth*.gguf + dumps) decide.
//
// What is Maya's own is the memory plan.  A Mac has one pool of unified memory, usually smaller than the model:
// dense weights, caches and state stay resident, the routed experts (95 % of the bytes) live in a resident slot
// pool that the CPU fills from the SSD.  The graph is cut after each MoE router: the host reads the layer's top-k,
// makes every selected expert resident (parallel pread into the pool), and the next graph runs the experts by
// slot.  A GPU never faults on file pages: on the M4 Max that path measured 12-29 MB/s, against 5+ GB/s for
// parallel pread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace maya::metal {

struct GlmGeometry {
    int n_vocab = 0;
    int n_embd = 0;
    int n_layers = 0;           // trunk layers (the nextn block is not counted)
    bool mtp = false;           // the NextN block (layer n_layers) is loaded: GlmOptions::mtp and the GGUF has one
    int n_head = 0;
    int hc = 4;
    int hc_iters = 20;
    float hc_eps = 1e-6f;
    float rms_eps = 1e-5f;
    float norm_eps = 1e-5f;
    // KDA
    int kda_head_dim = 128;
    int d_conv = 4;
    float kda_lower_bound = -5.0f;
    // MLA + indexer
    int q_lora = 0;
    int kv_lora = 0;
    int head_k_mla = 0;         // == qk nope dim (nope-only MLA)
    int head_v_mla = 0;
    int idx_heads = 0;
    int idx_key = 0;
    int idx_top_k = 0;
    int idx_kpool = 0;
    bool idx_select_tail = false;
    // FFN / MoE
    int n_ff = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    int n_dense_lead = 0;
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = true;
    std::vector<int> recurrent;       // per layer, the NextN block included: 1 = KDA, 0 = DSA
    std::vector<float> clamp_exp;     // per layer swiglu limits
    std::vector<float> clamp_shexp;
};

struct GlmOptions {
    bool use_gpu = true;              // false: ggml-cpu only (the oracle the Metal path is checked against)
    int n_ubatch = 2048;              // tokens per forward slice: 8 of 288 experts get ~57 tokens each, past
                                      // mul_mm_id's 32-column tiles (at 512, ~14: +20-37% prefill measured, ~0.3 GB)
    int attn_chunk = 32;              // tokens per sparse-attention gather inside a slice
    bool dsa_mm = true;               // DSA products with the heads as matrix columns (ggml-metal's mul_mm, half
                                      // tiles); false: one f32 matvec per head (MAYA_METAL_DSA_MM=0)
    bool dsa_kernel = true;           // with dsa_mm and a ggml built with metal-sparse-mla.patch: the
                                      // sparse attention as one kernel reading the cache rows in place (no gather, no
                                      // transpose; MAYA_METAL_DSA_KERNEL=0 keeps the gather + mul_mm)
    bool kv_f16 = false;              // MLA latent cache in F16 (as llama.cpp and the CUDA engine; half the bytes,
                                      // ~1e-4 logit error on the fixtures); default F32: ~1e-6.  On the GPU with
                                      // dsa_mm the latent is F16 unless kv_f32 (mul_mm rounds it to half anyway)
    bool kv_f32 = false;              // keep the latent F32 on the GPU with dsa_mm too: the fused kernel reads F32 rows
                                      // (--kv-f32 / MAYA_METAL_KV_F32=1; twice the cache bytes, for A/B runs)
    size_t expert_budget = 0;         // bytes for the routed-expert pool; 0: derive from the GPU's working set
    int pool_slots = 0;               // >0: force this many slots per expert group (tests)
    int io_threads = 16;
    std::string expert_profile;       // a ranked expert profile (data/expert-profile-glm.bin): warms the pool at load
    std::string expert_profile_save;  // this machine's learned profile: read instead of expert_profile when it exists,
                                      // written by save_profile() (the experts its conversations picked, most first)
    int cpu_threads = 0;              // 0: all performance cores
    int max_window = 0;               // >0: forward_window() takes up to this many tokens (speculative decoding)
    bool mtp = false;                 // load the NextN block (the MTP drafter)
    std::string dflash;               // a DFlash2 drafter GGUF (GLM-5.3-Flash-DFlash2-*.gguf): load it
    std::string dump_dir;             // non-empty: write every named seam as raw f32 (debug)
};

struct GlmStats {
    uint64_t tokens = 0;
    uint64_t expert_hits = 0;
    uint64_t expert_misses = 0;
    uint64_t bytes_read = 0;
    double io_seconds = 0.0;          // wall time the forward waited on expert reads
    double gpu_seconds = 0.0;         // wall time in graph compute
    double host_seconds = 0.0;        // wall time building and allocating graphs, uploading their inputs
    uint64_t graphs = 0;
    size_t pool_slots = 0;
    size_t pool_bytes = 0;
    size_t dense_bytes = 0;
    size_t state_bytes = 0;
    bool experts_all_resident = false;
};

class GlmMetal {
public:
    GlmMetal();
    ~GlmMetal();
    GlmMetal(const GlmMetal&) = delete;
    GlmMetal& operator=(const GlmMetal&) = delete;

    void set_options(const GlmOptions& o);
    // path: a GGUF, or the first shard of a split GGUF ("-00001-of-0000N.gguf"; the rest are found beside it)
    bool load(const std::string& path, int64_t max_ctx, std::string& err);
    // appends `tokens` at the current position; `logits` receives the LAST token's logits (n_vocab)
    bool forward(const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err);
    void reset();

    // a verify window: `tokens` at the current position in one pass, `logits` receives every row's (T x n_vocab), the
    // recurrent state of every row is kept so that accept(n) can keep the first n tokens (1..T) and drop the rest.
    // Until accept() the position is past all T tokens.
    bool forward_window(const std::vector<int32_t>& tokens, std::vector<float>& logits, std::string& err);
    bool accept(int n, std::string& err);
    // the trunk's final hidden state (output_norm of the streams' mean, what the head and the MTP block read) of the
    // last forward: its last row, or every row after forward_window (n_embd x rows)
    void hidden(std::vector<float>& out) const;
    // MTP (the NextN block, GlmOptions::mtp): one draft step at `pos` with `token` (the token at pos + 1 in the
    // block's convention: the trunk's hidden state of pos and the next token's embedding).  from_trunk: the input is
    // the trunk's hidden state of pos (mtp_hidden_pos() must be pos), else the previous step's own.  draft = the
    // argmax of the step's logits.  The step writes the block's cache entry at pos.
    bool has_mtp() const;
    int64_t mtp_hidden_pos() const;
    bool mtp_step(int32_t token, int64_t pos, bool from_trunk, int32_t& draft, std::string& err);

    // DFlash2 (GlmOptions::dflash): up to n drafts after `token` (the token at the current position, not fed yet),
    // already cut by the confidence rule - kept while i < n_min, or while each one's confidence >= p_min and their
    // product >= p_cum
    bool has_dflash() const;
    int dflash_block() const;   // the drafter's block: at most block - 1 drafts a round
    bool dflash_draft(int32_t token, int n, float p_min, int n_min, float p_cum, std::vector<int32_t>& drafts,
                      std::string& err);

    // images: the rows (n_embd each, maya-vision's) that replace the token embeddings at these absolute positions;
    // empty: none.  Kept until the next call.
    void set_image_rows(std::vector<int64_t> pos, std::vector<float> rows);
    // the recurrent state (KDA conv + S) at the current position; restore rewinds to it.  The MLA latent and
    // indexer caches are indexed by position and need no copy: rewinding only means overwriting from there.
    // write the learned profile (options.expert_profile_save): the experts picked so far, most first, then the
    // starting profile's order for the rest
    bool save_profile(std::string& err) const;

    // grow the expert pool into memory freed since it was sized (it starts from what was free then): up to the
    // GPU's working-set budget, keeping the headroom free.  Returns the bytes added; cheap when nothing changed.
    size_t grow_pool();
    bool pool_holds_all() const;   // every routed expert has a slot (grown slots fill on first use, then nothing misses)

    void snapshot_save();
    bool snapshot_restore();          // false: no snapshot
    int64_t snapshot_position() const;  // -1: none

    const GlmGeometry& geometry() const;
    int64_t position() const;
    int64_t max_context() const;
    int ubatch() const;   // tokens per forward slice (GlmOptions::n_ubatch)
    const GlmStats& stats() const;
    std::string describe() const;     // one line: backend, memory plan

    struct Impl;   // glm_metal_impl.hpp

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace maya::metal
