// include/strata/core/dense_model.hpp - a DENSE qwen35 model on Strata's kernels: Qwen3.8-27B (unsloth GGUF).
//
// WHY THIS IS A SEPARATE RUNNER AND NOT A MODE OF `strata`.  The MoE engine is specialised end to end
// (check_architecture: `qwen4exp`, 2560 wide, 48 blocks; expert arena; QSA indexer; hyper-connections; PLE;
// n-gram table).  A dense model has none of it, so `strata` refuses its GGUF at the first guard.  What the two
// models DO share is the GDN mixer (S = 128, 16 QK heads, 48 V heads, 10240 conv channels, d_conv 4 - the same
// numbers), the quantized GEMVs, RoPE and the attention output gate.  Those kernels are reused as they are.
//
// THE MODEL (qwen35, 64 layers, n_embd 5120, n_ff 17408), per layer:
//
//     xn = rms_norm(x, attn_norm)
//     GDN layer  (3 of every 4):  qkv, z = attn_qkv xn, attn_gate xn
//                                 conv+silu -> l2(q), l2(k) -> alpha, beta = ssm_alpha xn, ssm_beta xn
//                                 gate = softplus(alpha + dt) * ssm_a ; beta = sigmoid(beta)
//                                 o = delta-rule(state, q, k, v) ; y = rms_norm(o) * ssm_norm * silu(z)
//                                 mix = ssm_out y
//     attention layer (1 of 4):   q_full, k, v = attn_q xn, attn_k xn, attn_v xn      q_full = [q | gate] per head
//                                 q, k = rope(rms_norm_head(q), rms_norm_head(k))       24 Q heads, 4 KV heads, 256
//                                 attn = softmax(q K^T / sqrt(256)) V ; attn *= sigmoid(gate)
//                                 mix = attn_output attn
//     x += mix
//     x += ffn_down( silu(ffn_gate xn2) * ffn_up xn2 ),   xn2 = rms_norm(x, post_attention_norm)
//
// then logits = output( rms_norm(x, output_norm) ).  Which of the two mixers a layer has is read from the tensors
// it carries (attn_qkv.weight -> GDN, attn_q.weight -> attention), not from a formula.
//
// FORWARD PASSES.  forward() runs T tokens at once at positions position() .. position() + T - 1.  Up to 8 tokens
// (a decode step, a speculative verify window) the projections are native_mmvq GEMVs over T columns; more (a prompt
// chunk) are tensor-core GEMMs on weights dequantized to FP16 a matrix at a time (prefill::Gemm::native).  The GDN
// recurrence walks the tokens in one kernel.  A VERIFY pass writes the attention caches but leaves the recurrent
// state as it was; commit(n) then keeps the first n tokens (it replays the recurrence over them from their stashed
// inputs) and moves the position by n.
//
// Weights stay in their GGUF blocks on the GPU and are multiplied by native_mmvq, which takes every type an
// Unsloth UD file uses - Q4_K, Q5_K, Q6_K, Q3_K, Q8_0 AND IQ3_S / IQ3_XXS / IQ4_XS / IQ2_*.  There is no pack step
// and no per-type conversion; a GGUF with a type outside that list is refused with the list.
//
// Origin: Niko1221/Strata PR #314 (perronemirko), the one-token DenseModel; batched passes, verify / commit and the
// MTP block added for Maya.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct DenseConfig {
    std::string arch;            ///< general.architecture ("qwen35", "qwen35moe", "gemma4")
    int n_layer = 0, n_embd = 0, n_ff = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int ssm_d_conv = 0, ssm_state = 0, ssm_k_heads = 0, ssm_v_heads = 0;
    int n_nextn = 0;             ///< MTP (nextn) blocks stored after the trunk's layers
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;   ///< a MoE FFN (qwen35moe); 0: dense
    float expert_weights_scale = 1.0f;
    float rms_eps = 1e-6f, rope_base = 1e7f;
    int32_t eos_id = -1;         ///< tokenizer.ggml.eos_token_id: the id that ends an answer
    int conv_channels() const { return 2 * ssm_state * ssm_k_heads + ssm_state * ssm_v_heads; }
    int value_dim() const { return ssm_state * ssm_v_heads; }
};

class DenseModel {
public:
    DenseModel();
    ~DenseModel();
    DenseModel(const DenseModel&) = delete;
    DenseModel& operator=(const DenseModel&) = delete;

    /// Reads the GGUF (single file or the first shard of a split), checks every tensor it needs, uploads the
    /// weights and allocates the KV cache for `max_context` cells and the recurrent state.  `chunk`: the largest
    /// prompt chunk a forward pass takes (its activations are allocated for it; <= 0: the model picks - see chunk()).
    bool load(const std::string& gguf_path, int64_t max_context, std::string& err, int chunk = 0);

    const DenseConfig& config() const { return cfg_; }
    int64_t max_context() const { return max_context_; }
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// Weights that did not fit in VRAM and sit in pinned host memory (read over PCIe each token).
    uint64_t host_weight_bytes() const { return host_bytes_; }
    uint64_t kv_bytes() const { return kv_bytes_; }
    int chunk() const;

    /// Forgets the conversation: zeroes the recurrent state and the position.
    void reset();
    /// Number of tokens the state has consumed (= the next position).
    int64_t position() const { return pos_; }

    enum Logits { kNone = 0, kLast = 1, kAll = 2 };
    /// T tokens (host ids) at position() ..; verify = false commits them (the position moves by T), verify = true
    /// leaves the recurrent state and the position for commit() (T <= 8).  Logits: none, the last row's, or every
    /// row's (T <= 8) into logits() as rows of n_vocab.
    bool forward(const int32_t* tokens, int T, bool verify, int logits, std::string& err);
    /// After a verify pass of T tokens: keep the first n (1 <= n <= T).
    bool commit(int n, std::string& err);
    /// One token (forward of 1, committed).
    bool step(int32_t token, bool want_logits, std::string& err) {
        return forward(&token, 1, false, want_logits ? kLast : kNone, err);
    }
    /// Checkpoints (pinned host memory): the recurrent state at position() - every GDN layer's state and conv history,
    /// and the DFlash2 drafter's context ring - into slot `slot`; restore() puts it back and sets the position.  The
    /// K/V caches are position-indexed and stay valid for the prefix, so they are not copied.
    bool ckpt_save(int slot, std::string& err);
    bool ckpt_restore(int slot, int64_t pos, std::string& err);
    uint64_t ckpt_bytes() const;

    /// Images (GENI, STRATA_DENSE_VISION=1 at load): an image's tokens sit at cells cell .. cell + nx * ny - 1 (row
    /// i at grid x = i % nx, y = i / nx); their embedding rows (E floats each, the images' in order) replace the
    /// token embeddings, and every attention layer rotates them by interleaved M-RoPE (t, h, w) = (p, p + y, p + x),
    /// the text after an image going on at p + max(nx, ny).  An empty list: text only (the positions back to the
    /// cell index).  Applies to the forward passes until the next call.
    struct ImagePlace { int64_t cell; int nx, ny; };
    bool vision_enabled() const;
    bool set_images(const std::vector<ImagePlace>& places, std::vector<float> rows, std::string& err);
    /// Blocks until everything queued has run.
    bool sync(std::string& err);

    float* logits() const;             ///< device pointer: rows of n_vocab after a forward with logits
    void* stream() const;
    /// The final hidden rows of the last forward (output_norm applied), device, rows of n_embd - what the MTP block
    /// reads (llama.cpp's t_h_nextn).
    const float* hidden_norm() const;

    // ---- the MTP (nextn) block: drafts the token after next from (that token's embedding, this position's hidden)
    bool has_mtp() const;
    /// The MTP block over T rows at positions p0 ..: row t joins token toks[t] (the token AT position p0 + t + 1) with
    /// hidden row t of `hidden` (device, rows of n_embd).  Its own K/V cache gets cells p0 .. p0 + T - 1; the logits
    /// of the last row (or every row: all) go to mtp_logits(), its normed hidden rows to mtp_hidden().
    bool mtp_forward(const int32_t* toks, const float* hidden, int64_t p0, int T, bool all_logits, std::string& err);
    /// one MTP row whose token is on the device (*d_tok - a draft's argmax): no host round trip between drafts
    bool mtp_forward_dev(const int32_t* d_tok, const float* hidden, int64_t p0, std::string& err);
    // the MTP block's K / V of T rows into its cache only (rows whose MTP output nobody reads: a prompt's, a lookup
    // round's) - the same cache as mtp_forward's
    bool mtp_fill(const int32_t* toks, const float* hidden, int64_t p0, int T, std::string& err);
    bool mtp_dev_ok() const;
    float* mtp_logits() const;         ///< rows of draft_vocab()
    /// The vocabulary drafts come from: the first N ids of the head (STRATA_DENSE_DRAFT_VOCAB, NInfer's reduced
    /// proposal head), else all of it.  MTP's logits rows and DFlash2's candidates use it.
    int draft_vocab() const;
    /// The MTP draft head's rows: slot j of mtp_logits() is token draft_ids()[j] (device, draft_vocab() entries;
    /// nullptr: slot j is token j).  mtp_vocab_select() picks them for a request: the first token ids, then the
    /// prompt's most frequent tokens beyond them (STRATA_DENSE_MTP_VOCAB_EXTRA slots, default 4096; 0: off).
    const int32_t* draft_ids() const;
    int32_t draft_id(int slot) const;
    bool mtp_vocab_select(const int32_t* prompt, size_t n, std::string& err) {
        return mtp_vocab_select(prompt, n, nullptr, 0, err);
    }
    /// ... and the answer so far (`gen`, each counted 4 times: what is being written now is the better guess).
    bool mtp_vocab_select(const int32_t* prompt, size_t n, const int32_t* gen, size_t n_gen, std::string& err);
    const float* mtp_hidden() const;

    // ---- the DFlash2 drafter (a separate GGUF, arch qwen35-dflash-draft): K drafts in one pass of its 5 layers
    /// Loads the drafter; from then on every forward pass keeps the target's hidden rows the drafter reads.
    bool dflash_load(const std::string& path, std::string& err);
    bool has_dflash() const;
    int dflash_max() const;            ///< the most drafts a pass makes (its block size - 1)
    /// The last forward's rows 0 .. n - 1 (positions p0 ..) into the drafter's context ring.
    bool dflash_inject(int n, int64_t p0, std::string& err);
    /// K drafts after `anchor` (the token at position(), not yet fed); out gets them (host).
    bool dflash_draft(int32_t anchor, int K, std::vector<int32_t>& out, std::string& err);
    /// The last dflash_draft's confidence of each draft (its top-1 probability as the drafter scores it; empty: the
    /// drafter gives none) - what STRATA_DENSE_P_CUM's floor reads.
    const std::vector<float>& dflash_conf() const;
    /// DFlash2 for a sampled request: the next dflash_draft draws its tokens (u: a uniform a draft, temperature) and
    /// keeps their proposal distributions - dflash_proposals: q [K][k] over the candidates cand [K][k]
    bool dflash_samples() const;
    void dflash_sample(const std::vector<float>& u, float temperature);
    void dflash_proposals(std::vector<float>& q, std::vector<int32_t>& cand, int& k) const;
    /// "DSpark" or "DFlash2" (the loaded drafter's kind)
    const char* dflash_kind() const;

    struct Impl;

private:
    bool load_gemma4(int64_t max_context, std::string& err);                 ///< (after the files are indexed)
    bool forward_gemma4(const int32_t* tokens, int T, bool verify, int want, std::string& err);
    bool load_muse(int64_t max_context, std::string& err);
    bool forward_muse(const int32_t* tokens, int T, bool verify, int want, std::string& err);
    DenseConfig cfg_;
    int64_t max_context_ = 0, pos_ = 0;
    int pending_ = 0;                  ///< tokens of the last verify pass, waiting for commit()
    uint64_t weight_bytes_ = 0, kv_bytes_ = 0, host_bytes_ = 0;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
