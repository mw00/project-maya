// include/strata/core/glm_model.hpp - a glm5-next model runner: GGUF in, streaming logits out.
//
// Phase 3 of docs/GLM5-FLASH.md: the kernel family (glm_hc, glm_kda, glm_dsa, glm_ffn,
// router_sigmoid) composed into a MODEL, with the state a real decode needs living between calls:
//
//   * the mHC residual stack R;
//   * per KDA layer: the recurrent state S and the three conv histories;
//   * per DSA layer: the LATENT cache (kv_lora floats per token - no per-head K/V exists), the
//     indexer key/gate buffers, and the completed pools' pooled keys.
//
// `forward` APPENDS tokens: a batched call and a sequence of single-token calls on the same state
// produce identical logits (glm_model_test holds both to the oracle's dumps - the streaming path
// IS the decode path, so that equivalence is the gate that matters for the engine).
//
// CORRECTNESS-FIRST SCOPE, and the two things that follow from it:
//   * weights are consumed as F32 exactly as the GGUF stores them, in the ggml layouts the kernels
//     document (features fastest everywhere, no transposes anywhere between file and kernel);
//   * a QUANTIZED glm5-next GGUF is refused with a pointer to the packer - riding the engine's
//     quantized GEMV/expert machinery is the pack-integration step, not this runner's.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_dflash.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::kernels::glmf {
struct MoeRequest;
struct CpuAnswer;
}  // namespace strata::kernels::glmf

namespace strata::core {

namespace glmfast {
class Workers;
}  // namespace glmfast

/// One served weight: F32 (dequantized at load) or a ggml-quantized row tensor uploaded RAW to
/// VRAM and dotted by the device MMVQ.  type 0 = F32.
struct WSlot {
    const float* f32 = nullptr;
    int type = 0;
    const void* q = nullptr;
    int64_t n_in = 0, n_out = 0;
};

/// Every number the runner's kernels depend on, parsed from the GGUF's glm5-next.* metadata with
/// the released GLM-5.3-Flash values as defaults (mirrors ref/glm.py::Glm5Config - the Python and
/// C++ readers must agree key for key, which glm_model_test checks against the dumps).
struct Glm5Geometry {
    int n_embd = 4096, n_layers = 45, n_head = 64, n_vocab = 154880;
    int nextn = 0;                     // NextN/MTP layers counted INSIDE block_count (unsloth
                                       // ships 46 with nextn 1); tensors of blk.{n_layers}+ are
                                       // skipped at load and MTP stays deferred
    std::vector<int> n_head_kv;        // 0 marks a KDA layer
    int q_lora = 1536, kv_lora = 512, qk_nope = 256, v_head = 256;
    int kda_head_dim = 128, d_conv = 4;
    float kda_lb = -5.0f;
    int idx_heads = 32, idx_key = 128, idx_top_k = 2048, idx_kpool = 4;
    int idx_select_tail = 1;
    std::vector<int> indexer_full;     // per layer; the arch allows a non-full DSA layer to reuse
                                       // the previous full layer's selection (prev_sel) - this
                                       // runner implements full indexers only and REFUSES the rest
    int hc = 4, sinkhorn_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-5f;
    int n_expert = 288, n_exp_used = 8, n_ff_exp = 2048, n_shared = 1, n_ff_dense = 12288, dense_lead = 3;
    float w_scale = 1.0f;
    int norm_w = 1;
    float swiglu_exp = 10.0f, swiglu_shexp = 10.0f;

    int d_inner() const { return kda_head_dim * n_head; }
    int top_pools_max() const { return idx_top_k / idx_kpool; }
    int n_sel_max() const { return idx_top_k + (idx_select_tail ? idx_kpool - 1 : 0); }
    bool is_recr(int il) const { return n_head_kv[(size_t) il] == 0; }

    static bool from_gguf(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
};

class Glm5Model {
public:
    Glm5Model() = default;
    ~Glm5Model();
    Glm5Model(const Glm5Model&) = delete;
    Glm5Model& operator=(const Glm5Model&) = delete;

    /// Parses the metadata, uploads every tensor (F32 only - see the header note) and sizes the
    /// state for `max_ctx` tokens.  `max_ctx` caps the latent/indexer caches, nothing else.
    bool load(const std::string& gguf_path, int64_t max_ctx, std::string& err);
    /// Loads a PACK (tools/iq_pack.py output: index.txt + dense.bin + native_experts.txt beside
    /// the GGUF shards).  Dense tensors are dequantized into the same F32 device arena at load;
    /// the routed experts stay quantized on the CPU path (native_expert), read from the shards'
    /// mmaps per token.  docs/GLM5-FLASH.md §9.  `dev` picks the CUDA device; `l0`/`l1` restrict
    /// this instance to layers [l0, l1) (l1 == 0 means "to the end") - the layer split's halves.
    bool load_pack(const std::string& pack_dir, int64_t max_ctx, std::string& err, int dev = 0, int l0 = 0,
                   int l1 = 0);
    /// The split entry the driver/test use: STRATA_GLM_SPLIT (else `layer_split`, the driver's --layer-split) =
    /// "0" one device; "<layer>" two parts - layers [0,layer) on device 0, the rest on STRATA_GLM_DEV1 (default
    /// 1); "K1,K2,.." one part per visible device, each later one starting at its K; "auto"/absent: every visible
    /// device (STRATA_GLM_DEVS="0,2,.." picks and orders them).  The parts are joined by a per-token 64 KB host
    /// hop of the residual at each boundary (no peer access needed).
    bool load_pack_env(const std::string& pack_dir, int64_t max_ctx, std::string& err,
                       const std::string& layer_split = "");
    bool load_pack_place(const std::string& pack_dir, int64_t max_ctx, std::string& err, const std::string& layer_split);
    /// The explicit split form behind load_pack_env: devs[i] runs layers [bounds[i-1], bounds[i]) (bounds holds
    /// the first layer of each later part, devs.size() - 1 of them).
    bool load_pack_split(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& bounds,
                         const std::vector<int>& devs, std::string& err);
    static constexpr int kMaxParts = 16;
    /// The fast decode path (src/core/glm_fast_path.cu) is the default for packs; STRATA_GLM_SLOW=1
    /// keeps the correctness-first per-op path (the reference the fast path is checked against).
    bool fast() const { return fast_ != nullptr; }
    /// Skip the per-token logits download (the driver samples on the device).
    void set_host_logits(bool on) { host_logits_ = on; }
    /// Live counters of the fast path (both halves summed), for the driver's STAT lines.
    struct FastStats {
        uint64_t tokens = 0, hits = 0, misses = 0, miss_layers = 0;
        uint64_t ram_hits = 0, disk_reads = 0, promotions = 0, cpu_experts = 0, disk_skips = 0;
        double ms = 0, miss_ms = 0, disk_ms = 0, cpu_ms = 0;
        int64_t pool_slots = 0, pool_used = 0, ram_slots = 0, ram_used = 0;
        double pool_gb = 0, ram_gb = 0;
    };
    FastStats fast_stats() const;
    /// What a part holds on its GPU, in bytes (generate's VRAM lines, the Monitor's map of a GPU): the device's total
    /// and free memory now, its PCI address (the server finds the GPU NVML numbers by it), the dense weights, the
    /// sequence state (the DSA latent cache, KDA states, indexer keys; kv_ram: KV streaming's pinned copy in RAM), the
    /// activations, a drafter's weights (DFlash or the NextN block) and the verify window's buffers, the expert pool
    /// with its lendable tail and what a prompt (borrow, at most) or the vision encoder borrows of that tail.  The
    /// CUDA context and the small buffers are not counted: total - free - the parts.
    struct VramReport {
        int dev = 0;
        std::string pci;
        uint64_t total = 0, free = 0;
        uint64_t weights = 0, kv = 0, kv_ram = 0, act = 0, drafter = 0, window = 0;
        uint64_t pool = 0, tail = 0, borrow = 0, vision = 0;
        int64_t slots = 0, ctx = 0;
        int l0 = 0, l1 = 0;
    };
    std::vector<VramReport> vram_report() const;
    /// Every GPU's expert pool as it was sized (part order): the device, the VRAM kept free beside it (MiB), whether the
    /// default reserve was lowered so a small card's pool holds its floor, and --expert-cache N (0: auto) - for the
    /// driver's free-VRAM lines, which name the setting that leaves room
    struct PoolVram {
        int dev = 0, reserve_mib = 0;
        bool adapted = false;
        int64_t fixed = 0;
    };
    std::vector<PoolVram> pool_vram() const;
    /// Windows on an APU (AMD's Strix / Gorgon Halo): the bytes of experts the commit (RAM + page file) kept off the
    /// GPU over every part, and the page file (GiB) that keeps them on it - {0, 0} elsewhere (pool_check)
    std::pair<size_t, size_t> commit_capped() const;
    /// The CPU LANE (a decode's RAM-tier experts): the share of them copied over PCIe to the GPU instead of computed
    /// on the CPU - 0: the CPU takes every one, 1: none (of f such experts in a route, f - round(share f) go to the
    /// CPU).  pcie_share(): the split in use, as a share; set_pcie_share(s): this split from the next token on, s < 0
    /// the one the engine started with (STRATA_GLM_PCIE_SHARE, else measured at start).  cpu_lane_threads(): the
    /// threads taking its jobs; set_cpu_lane_threads(n): at most n of the pool's, n <= 0 all of them.  Both are for
    /// setup's calibration (per request, no restart); false when this engine has no CPU lane.
    double pcie_share() const;
    /// STRATA_GLM_CPU_CAL: once a request (2000+ lane experts a window), drop a calibration decode stays >15% from
    void lane_drift_check();
    /// Between requests: over decode's first windows of lane work, a split the start-up timing chose is planned again
    /// from the lane's own cost in decode when that is >15% off (at most three times; STRATA_GLM_CPU_RETIME=0: never)
    void lane_retime();
    bool set_pcie_share(double share);
    int cpu_lane_threads() const;
    bool set_cpu_lane_threads(int n);
    /// The decode's disk reads: each expert read from the SSD in 3 x n pieces at once (STRATA_GLM_READ_CHUNKS, 8).
    /// set_read_chunks(n): n (1-16) from the next read on, n <= 0 the engine's own - for setup's calibration (per
    /// request, no restart): the best size depends on the drive, its driver and the CPU
    void set_read_chunks(int n);
    /// The decode's light misses (STRATA_GLM_MISS_SKIP_W / _FROM, changes the answers): a routed expert that is not in
    /// VRAM and carries less than w of the route's weights is left out on layers from `from` on.  set_miss_skip(w, from):
    /// this request's bar and first layer (w < 0 / from < 0: the engine's own) - for sweeps without a restart.
    void set_miss_skip(double w, int from);
    /// Conversation reuse (the fast path): save the sequence state at the current position - the recurrent KDA
    /// states and conv histories (the DSA caches are append-only, so the position alone restores them) - and
    /// restore it later to continue a prompt that extends the saved one.  false when unsupported.
    bool snapshot_save();
    // the expert usage file (opt-in, STRATA_GLM_USAGE=<file>: the warm-up follows it ahead of the expert profile):
    // written after each request.  Without it nothing is kept between sessions but --expert-profile-save's profile.
    bool save_usage();
    std::string usage_path() const;
    /// The expert profile (Strata's `profile.bin`, tools/make_profile.py --glm; setup passes the shipped
    /// data/expert-profile-glm.bin): its ranking of every (layer, expert) pair decides how many VRAM slots each layer
    /// gets and which experts the warm-up puts in them, in place of the pack's expert_counts.txt / expert_prior.txt.
    /// Set before load_pack*: every half reads it.  Its layer axis is profile_layers(): the decoder layers, then the
    /// NextN block.
    void set_expert_profile(const std::string& path) { profile_path_ = path; }
    int64_t profile_layers() const { return g_.n_layers + 1; }
    /// --expert-profile-save: what the tiers learned, as a profile --expert-profile reads (rank_learned_profile over
    /// every half: the experts in VRAM now, then the routing counted, then the profile this run started from).
    bool save_expert_profile(const std::string& path, std::string& err);
    /// --dump-routing: every route (the decode's and the prompt's) appended to `path` as Strata's trace records -
    /// int32 layer, int32 k, then k int32 expert ids and k float weights - tools/make_profile.py's input.
    static bool set_routing_dump(const std::string& path, std::string& err);
    bool snapshot_restore();
    int64_t snapshot_pos() const { return snap_pos_; }
    /// Conversation CHECKPOINTS (host RAM, `slot` 0, 1, ...): every half's KDA states + conv histories and its open
    /// indexer pool rows at the current position, beside the snapshot - a prompt that leaves the conversation before
    /// the snapshot's position (an edited last message, another chat with the same system prompt) restores the
    /// nearest earlier one instead of reading from token 0.  The DSA rows before its position must still be the
    /// conversation's: the driver drops a checkpoint once a prompt writes at or before it.  ckpt_restore sets the
    /// position; false (+ err) when that slot holds none.  ~150 MB a checkpoint for GLM-5.3-Flash's 34 KDA layers.
    bool ckpt_save(int slot, std::string& err);
    bool ckpt_restore(int slot, std::string& err);
    /// The snapshot (the previous prompt's state, before its last token) copied into checkpoint `slot` on a side
    /// thread, while the next prompt runs: the snapshot stays intact until that prompt's snapshot_save, which waits
    /// for the copy (as every checkpoint call does).  A turn that starts within 512 tokens of the snapshot gets no
    /// ckpt_save at its own start; this one costs the prompt nothing.  False (+ err) when it cannot start.
    bool ckpt_save_snapshot(int slot, std::string& err);
    /// Conversation slots: the snapshot (its KDA states) and every DSA cache's rows up to its position, every half,
    /// written to a file; slot_load puts them back as the live state and the snapshot, so a conversation set aside
    /// while others ran continues where it was.  The bytes written, or 0 + err.
    uint64_t slot_save(const std::string& path, std::string& err);
    bool slot_load(const std::string& path, int64_t n_pos, std::string& err);
    /// The same, in host RAM: the exact bytes the file version writes, in a caller-held buffer whose
    /// lifetime is the conversation's (engine-side: the run's).  Nothing reaches the disk.  The bytes
    /// written, or 0 + err (out of RAM, no snapshot).  slot_load takes the buffer back as the live
    /// state and the snapshot; false + err on a mismatch or a truncated buffer.
    uint64_t slot_save(std::vector<uint8_t>& data, std::string& err);
    bool slot_load(const std::vector<uint8_t>& data, int64_t n_pos, std::string& err);
    /// Runs the engine's sampler on the last forward's logits and returns the picked token id
    /// (-1 + err on failure).  Handles the device bookkeeping: with a split the logits (and the
    /// sampler buffer) live on the tail half's device, where peer reads are not available.
    int sample_token(strata::kernels::SamplerParams& sp, std::string& err);
    /// A request's logit_bias (token id, bias) as the dense n_vocab table SamplerParams::logit_bias takes, on the
    /// sampler's device (the tail half's): one table per conversation slot (0 .. seq_count()-1), rewritten for each
    /// request that names one.  nullptr for an empty list (and nullptr + err on failure).
    const float* logit_bias_table(int slot, const std::vector<std::pair<int32_t, float>>& bias, std::string& err);
    /// EVERY pick of the engine's sampler (sample_token, the windows, the several-conversation steps): T rows of
    /// `logits` (n_vocab each, this part's device) -> out[0..T) on stream s.  A request with penalties (#273) gets each
    /// row's history: row t counts the last penalty_last_n tokens of *sp.host_history (what the state was fed, the
    /// window's first token last) followed by win[1 .. n0 + t] - the drafts before that row's position (win: the
    /// window's tokens, nullptr for a lone pick; n0: the drafts before row 0, a pipelined window's token).  So a draft
    /// stands only where it is the penalized pick: drafting counts exactly the history plain decode counts.  Without
    /// penalties it is exactly the sample_tokens call the sites made before.
    void pick_tokens(const float* logits, int T, const strata::kernels::SamplerParams& sp, const int32_t* win, int n0,
                     int* out, cudaStream_t s);
    /// Zeroes R, every KDA state and every DSA cache: the next `forward` starts a fresh sequence.
    void reset();

    /// Appends `tokens` to the current sequence and writes the LAST token's logits (n_vocab f32)
    /// to `logits_out`.  State persists across calls; call `reset` between sequences.
    bool forward(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err);
    /// The DEVICE pointer of the last forward()'s logits (n_vocab floats) - for
    /// strata::kernels::sample_tokens, which is a kernel and takes device memory (see sampler.hpp).
    const float* device_logits() const { return sc_ + sc_logits; }

    /// The position the next token will occupy (tokens consumed since the last reset).
    int64_t position() const { return pos_; }
    const Glm5Geometry& geometry() const { return g_; }
    /// The next decoded token is `tok` instead of the sampled one (once; -1 clears): the serve loop's thinking budget
    /// closes the reasoning block this way inside the running decode - the speculative loop takes it as the truth
    /// for its position, like a rejected draft, so nothing is read again.
    void force_next(int32_t tok) { force_tok_ = tok; }
    /// The on-demand vision encoder (serve VLEND / VRECLAIM; STRATA_GLM_VISION_LEND_MB at load): the first GPU's
    /// lendable pool tail is emptied and freed so the encoder's process can use that VRAM, and allocated again
    /// once it has ended (false while the memory is still taken).  vision_lend_bytes(): 0 when unavailable.
    bool vision_lend(size_t& bytes, std::string& err);
    bool vision_reclaim(std::string& err);
    size_t vision_lend_bytes() const;
    int32_t forced(int32_t sampled) {
        int32_t t = force_tok_ >= 0 ? force_tok_ : sampled;
        force_tok_ = -1;
        if (follow_at_ < follow_.size()) {   // --spec-follow: the continuation's token, whatever was picked
            const int32_t f = follow_[follow_at_++];
            follow_differ_ += f != t;
            t = f;
        }
        return t;
    }
    /// --spec-follow (STRATA_GLM_SPEC_FOLLOW): the next decoded tokens are `ids`, one a position, instead of the picked
    /// ones - every decode path takes them through forced(), so the speculative ones keep exactly the drafts that match
    /// them.  Two builds or settings then decode the same text (the same windows, drafts and routes), which a speed
    /// A/B needs: a near-tie flip otherwise changes the text and with it the draft acceptance.  Empty: off.
    void set_follow(std::vector<int32_t> ids) {
        follow_ = std::move(ids);
        follow_at_ = 0;
        follow_differ_ = 0;
    }
    /// The continuation's tokens decoded so far, and how many of them differ from the model's own pick.
    size_t follow_used() const { return follow_at_; }
    size_t follow_differ() const { return follow_differ_; }
    /// The continuation's token k places after the one forced() returns next (-1: past its end, or no continuation) -
    /// a decode that checks a whole window of drafts before it emits any (DFlash2) compares them with these.
    int32_t follow_peek(size_t k) const { return follow_at_ + k < follow_.size() ? follow_[follow_at_ + k] : -1; }

private:
    bool step(int32_t token, std::string& err);   // one token through all layers

    Glm5Geometry g_;
    std::map<std::string, const float*> w_;       // name -> device pointer (the weights arena)
    float* w_arena_ = nullptr;
    uint64_t w_arena_bytes_ = 0;                           // (the VRAM report at load)
    // state and scratch arenas (device), carved in load(); see glm_model.cpp for the layout
    float* state_ = nullptr;
    uint64_t state_bytes_ = 0;
    float* sc_ = nullptr;
    uint64_t sc_bytes_ = 0;
    int* d_pos_ = nullptr;                        // the select kernel reads the position on device
    int64_t pos_ = 0;
    int64_t max_ctx_ = 0;
    bool loaded_ = false;
    std::vector<int32_t> follow_;                 // set_follow: the continuation; follow_at_ its next token
    size_t follow_at_ = 0, follow_differ_ = 0;

    // per-layer state offsets, in floats into state_ (-1: the layer has no such state)
    std::vector<int64_t> kda_S_, kda_conv_, dsa_lat_, dsa_ik_, dsa_ig_, dsa_pool_;

public:
    // ---- per-weight serving slots (strata::core::WSlot below): F32 pointers for the
    // dequantized tensors, typed raw pointers for the quantized ones (device MMVQ)
    std::map<std::string, WSlot> ws_map_;
    std::map<std::string, std::string> row_kind_;

    // ---- the layer split (speed): this instance owns layers [l0_, l1_) on device dev_.  When
    // split_next_ is set, step() runs this half, hands the residual over with one 64 KB host hop
    // (peer access is closed on this driver and not needed for a single boundary copy) and lets the
    // next half continue through its own layers (and, if it is the last, the head).
    int l0_ = 0, l1_ = 0;
    int dev_ = 0;
    int n_parts_ = 1;                                      // the devices the layers are split across (the CPU lane's share)
    int part_ = 0;                                         // ... and which of them this one is (0 = the first layers)
    // THE HEAD SPLIT (an EXL3 pack on 3+ parts; STRATA_GLM_HEAD_SPLIT=0: off): a pipelined window's pick on the last
    // part reads the head's columns [0, head_c0_) there and [head_c0_, n_vocab) on a helper part at once (its own
    // copy), the two argmaxes (or the helper's logits, sampling) joined on the last part - the same pick as the whole
    // row's.  The helper: the part of the fewest layers between the first and the last (their head GEMV ~1.2 ms of
    // a token on an RTX 5060 Ti)
    bool head_half_ = false;          // this part (the helper) loads the head's columns [head_c0_, n_vocab)
    int head_c0_ = 0;
    Glm5Model* head_helper_ = nullptr;   // (the last part) the helper
    float* hh_x_ = nullptr;            // (the helper) kPipeSlots head inputs (n_embd), their logits (n_vocab - c0)
    float* hh_lg_ = nullptr;
    int* hh_pair_ = nullptr;           // ... and their argmax pairs (argmax_pair)
    cudaStream_t hh_s_ = nullptr;      // ... the stream they run on (the highest priority)
    cudaEvent_t hh_ev_[64] = {};       // ... a slot's result out (pair or logits copied to the host)
    float* hd_x_ = nullptr;            // (the helper) a draft's head rows (kWinMax x n_embd), their logits (n_vocab - c0)
    float* hd_lg_ = nullptr;
    void* hd_cand_ = nullptr;          // ... and their slices' top-k candidates (kHdCandBytes)
    cudaStream_t hd_s_ = nullptr;      // ... the stream they run on (not hh_s_: behind a window's late picks there)
    int* hh_skip_d_ = nullptr;         // (the helper) its skip flag for a pick's half (a token past a wrong draft)
    static constexpr size_t kHdCandBytes = (size_t) 1 << 20;
    float* ht_x_h_ = nullptr;          // (the last part) pinned: the slots' head inputs, the helper's logits, pairs
    float* ht_lg_h_ = nullptr;
    int* ht_pair_h_ = nullptr;
    int* ht_pair_d_ = nullptr;         // ... on the device: [slot][2] own pair, [slot][2] the helper's
    cudaEvent_t ht_ev_[64] = {};       // ... a slot's head input out to the host
    bool head_split_ready() const { return head_helper_ != nullptr && ht_x_h_ != nullptr && ht_pair_d_ != nullptr; }
    bool head_split_link(Glm5Model* helper, std::string& err);   // (the last part, after the load) the helper's buffers
    bool win_head_split(int t, int sl, const float* hx, float* lg, std::string& err);
    std::vector<int> split_devs_;                          // ... and the CUDA devices of all of them, part by part
    std::unique_ptr<Glm5Model> split_next_;
    // parallel split load: parts take the RAM budget in part order, then pin their RAM tiers in part order too
    // (wait_pin / pass_pin); CPU lane setup waits until all have loaded
    struct SplitLoad {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<bool> passed, pinned;
        int turn = 0, pin_turn = 0;
        explicit SplitLoad(int n) : passed((size_t) n, false), pinned((size_t) n, false) {}
        void wait(int part) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return turn >= part; });
        }
        void pass(int part) {
            {
                std::lock_guard<std::mutex> lk(mu);
                passed[(size_t) part] = true;
                while (turn < (int) passed.size() && passed[(size_t) turn]) ++turn;
            }
            cv.notify_all();
        }
        void wait_pin(int part) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return pin_turn >= part; });
        }
        void pass_pin(int part) {
            {
                std::lock_guard<std::mutex> lk(mu);
                pinned[(size_t) part] = true;
                while (pin_turn < (int) pinned.size() && pinned[(size_t) pin_turn]) ++pin_turn;
            }
            cv.notify_all();
        }
    };
    SplitLoad* split_load_ = nullptr;
    bool defer_lane_ = false;
    std::vector<float> hop_;                               // the boundary residual staging buffer
    int* d_tok_ = nullptr;                                 // the sampler's one-int output, on dev_
    std::vector<float*> bias_tab_;                         // logit_bias_table's tables (on the tail's device)
    int bias_dev_ = -1;                                    // ... that device
    bool step_layers(int64_t p, std::string& err);
    // ---- the cross-token pipeline: with a split and more than one token to feed (a teacher-forced
    // prompt), the next token's first half is queued on device 0 BEFORE the host waits for this
    // token's residual, so half-A(t+1) runs while half-B(t) runs on device 1.
    void* stream_hop_ = nullptr;                           // the residual D2H stream on dev_
    void* ev_a_pipe_ = nullptr, *ev_h_pipe_ = nullptr;
    void* hop_side_ = nullptr;                             // 2x hop floats on dev_ (the end-copy slots)
    void* ev_h2d_[2] = {nullptr, nullptr};                 // dev1-side H2D completion, per host slot
    bool step_issue(int32_t token, int64_t& p, std::string& err);   // embed + this half's layers
    bool forward_pipeline(const std::vector<int32_t>& tokens, std::vector<float>& logits_out,
                          std::string& err);

    // ---- the pack (set by load_pack; empty when load()ing a plain F32 GGUF)
    bool pack_ = false;
    struct Shard {
        std::string path;
        uint8_t* base = nullptr;
        uint64_t size = 0, data_start = 0;
        int fd = -1;                                   // kept open for posix_fadvise reads (prefetch)
        int fd_direct = -1;                            // O_DIRECT: expert reads that bypass the page cache
        void* h_direct = nullptr;                      // Windows: the same, a FILE_FLAG_NO_BUFFERING HANDLE
    };
    std::vector<Shard> pack_shards_;
    struct NativeLayer {
        int layer = -1;
        strata::kernels::cpu::NativeFmt fmt;
        int gate_shard = 0, up_shard = 0, down_shard = 0;
        uint64_t gate_off = 0, up_off = 0, down_off = 0;   // absolute file offsets of the tensors
        // EXL3 experts (exl3.txt, x_cb >= 0): expert e's part r (gate, up, down) piece j (trellis, suh, svh) lies at
        // x_off[9 e + 3 r + j] of shard x_shard[...]; the blob is [gate | up | down], a part [trellis | suh | svh]
        int x_cb = -1;
        int x_k2[3] = {0, 0, 0}, x_k[3] = {0, 0, 0}, x_n[3] = {0, 0, 0};   // x_k2: each part's bitrate as 2 K
        std::vector<int> x_shard;
        std::vector<uint64_t> x_off;
        size_t x_piece(int r, int j) const {
            return j == 0 ? (size_t) x_k[r] * (size_t) x_n[r] * (size_t) x_k2[r] / 16
                          : 2 * (size_t) (j == 1 ? x_k[r] : x_n[r]);
        }
        size_t x_part(int r) const { return x_piece(r, 0) + x_piece(r, 1) + x_piece(r, 2); }
    };
    std::vector<NativeLayer> pack_layers_;                 // indexed by layer id (dense lead = null)
    // ---- an EXL3 pack (tools/exl3_pack.py): the trellis-coded matrices, read from the model's safetensors (shards
    // exl3_shard0_ on, after the GGUF), uploaded per piece; the WSlots of type exl3::kTypeEXL3 point at exl3_mats_
    bool exl3_ = false;
    int exl3_shard0_ = -1;
    std::map<std::string, strata::kernels::exl3::Mat> exl3_mats_;
    std::vector<float> host_x_, host_moe_, host_h_;        // per-token CPU MoE scratch
    std::vector<float> host_probs_b_;                      // the per-layer router bias (prefetch ranking)
    std::vector<float> host_ff_, host_dn_;                 // per-expert gate/up and down outputs
    std::vector<uint8_t> host_act_, host_hq_;              // quantized activations
    cudaError_t pack_moe_tail(std::string& err, const std::vector<int>& ids,
                              const std::vector<float>& rw, float* A, int layer);
    // token_embd stays QUANTIZED in the pack arena; step() dequantizes the one row it needs per
    // token from the host mapping (2.2 GB of F32 embedding the expert pool now gets instead).  A row's
    // bytes come from ggml_row_size, which knows every GGUF type: iq_row_bytes knows only the device
    // kernels' and gave 0 for F16 / BF16 / Q2_K / Q4_1 / Q5_1, so every token read row 0
    const uint8_t* pack_emb_src_ = nullptr;
    int pack_emb_type_ = -1;
    std::vector<uint8_t> pack_emb_copy_;                   // Windows: token_embd once the shard views are gone
    void pack_release_views();                             // Windows, fast path: unmap the shards after the load

    // ---- M2b: the device expert pool.  Slots of [gate|up|down] rows (the native blob layout),
    // filled on miss by streaming the rows straight out of the mmap'd shards; LFU-with-aging
    // residency when the working set exceeds VRAM.  SIZE CLASSES: one class per distinct
    // (gu_row, d_row) pair - uniform slots at the largest blob would pad ~20% of the arena
    // (8.78 MB max vs 7.03 MB average).  Class c's slots live at pool_ + class_off_[c] with
    // GLOBAL ids [class_base_[c], class_base_[c] + class_slots_[c]); eviction scans only its own
    // range.  Empty unless load_pack found headroom.
    bool dev_experts_ = false;
    uint8_t* pool_ = nullptr;                              // the compute device's arena
    size_t pool_bytes_ = 0, slot_bytes_ = 0;               // slot_bytes_ = the LARGEST class (staging sizes)
    int pool_nclasses_ = 0;
    size_t class_off_[8] = {};                             // byte offset of each class's sub-arena
    int64_t class_base_[8] = {}, class_slots_[8] = {};     // global slot-id range per class
    size_t class_stride_[8] = {};                          // bytes per slot of that class
    std::vector<int> layer_class_;                         // absolute layer -> class
    std::vector<int32_t> pool_slot_of_;                    // layer*n_expert+expert -> GLOBAL slot, -1 absent
    std::vector<int64_t> pool_key_of_;                     // global slot -> key, -1 free
    std::vector<uint64_t> pool_tick_;                      // recency per global slot
    std::vector<uint32_t> pool_count_;                     // per-slot use count (LFU with aging, see pool_evict)
    uint64_t pool_evictions_ = 0;
    uint64_t pool_clock_ = 0, pool_hits_ = 0, pool_misses_ = 0;
    // ---- STRATA_GLM_TIMING=1: cudaEvent phase accumulators (dense mixer / expert tail / rest),
    // printed once at destruction; pool_host accumulates the host-side miss staging (memcpy from
    // the mmap + issue), which shows up as stream idle inside the expert tail.
    bool timing_ = false;
    bool tail_rec_ = false;   // per layer: the MoE tail recorded b and c (dense layers do not)
    void *tev_a_ = nullptr, *tev_b_ = nullptr, *tev_c_ = nullptr, *tev_d_ = nullptr;
    double t_dense_ = 0, t_tail_ = 0, t_rest_ = 0, t_pool_host_ = 0;
    int64_t t_tokens_ = 0;
    uint8_t* dev_xq_ = nullptr;                            // one q8_1 activation row
    void* dev_scratch_ = nullptr;                          // native_expert_grouped scratch
    float* dev_hit_out_ = nullptr;                         // n_exp_used x n_embd expert outputs
    // the grouped plan, host-written and read by the kernels over UVA (host-mapped)
    struct DevPlan {
        unsigned long long ptr[8];
        int32_t start[9];
        int32_t n_groups[1];
        int32_t dst[8];
        int32_t tok[8];
    };
    DevPlan* plan_host_ = nullptr;
    uint8_t* plan_dev_ = nullptr;
    uint8_t* bounce_ = nullptr;                            // pinned staging ring for miss copies
    int bounce_i_ = 0;
    // ---- the miss staging pool (speed pass): one worker per in-flight miss reads the expert's three
    // ranges IN PARALLEL (the NVMe at multi-queue rate instead of one faulting memcpy on the main
    // thread), then the main thread DMAs each staged blob to its slot with a single pinned copy.
    // STRATA_GLM_NO_STAGE=1 keeps the serial pageable path (A/B).
    bool stage_on_ = false;
    struct StageJob {
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};
        uint64_t len[3] = {0, 0, 0};
        uint8_t* buf = nullptr;
    };
    std::vector<StageJob> stage_q_;
    std::vector<std::thread> stage_workers_;
    std::mutex stage_mu_;
    std::condition_variable stage_cv_, stage_done_cv_;
    int stage_pending_ = 0;
    bool stage_quit_ = false;
    std::vector<uint8_t*> stage_bufs_;                     // one pinned buffer per job row (n_exp_used <= 8)
    uint8_t* staged_dst_[8] = {};                          // per-entry: where the DMA goes once staged
    uint64_t staged_len_[8] = {};
    void stage_worker();
    void stage_join();
    void pool_ensure(int layer, const int* ids, int k);
    int32_t pool_evict(int cls);
    bool pack_moe_tail_device(std::string& err, const std::vector<int>& ids,
                              const std::vector<float>& rw, float* A, int layer);

    // ---- the fast path (src/core/glm_fast_path.cu)
    struct FastState;
    FastState* fast_ = nullptr;
    bool fast_mode_ = false;                               // decided at load (STRATA_GLM_SLOW=1 turns it off)
    bool host_logits_ = true;
    int last_tok_ = -1;                                    // the greedy argmax of the last fast forward
    const strata::kernels::SamplerParams* mtp_sample_ = nullptr;   // decode_spec, STRATA_GLM_SPEC_SAMPLE: fast_mtp DRAWS
    int32_t force_tok_ = -1;                               // force_next()
    // pick_tokens' penalty rows (this part's, allocated with the first request that has penalties): a ring of
    // kPenRows rows of pen_w_ tokens, pinned + device, each row's event recorded behind the pick that read it (the
    // picks of several conversations, a pipelined window's tokens, can be in flight at once)
    static constexpr int kPenCap = 4096;                   // the penalty window at most (penalty_last_n beyond it: 4096)
    static constexpr int kPenRows = 64;
    int32_t* pen_h_ = nullptr;
    int32_t* pen_d_ = nullptr;
    int pen_w_ = 0, pen_next_ = 0;
    cudaEvent_t pen_ev_[kPenRows] = {};
    std::mutex pen_mu_;                                    // (a pipelined window picks from its last part's thread)
    bool pen_alloc(int w, std::string& err);
    void pen_free();
    std::map<std::string, const uint16_t*> w16_;           // the pack's big BF16 rows, kept BF16 (fast mode)
    bool fast_setup(std::string& err);
    bool fast_setup_finish(std::string& err);
    bool dual_prefill_ready();   // STRATA_GLM_DUAL_PREFILL (glm_prefill.cu)
    int dual_pf_state_ = 0;      // 0 not tried, 1 on, -1 off
    bool calc_prefill_ready();   // STRATA_GLM_PREFILL_CALC (glm_prefill.cu)
    int calc_pf_state_ = 0;      // 0 not tried, 1 on, -1 off
    void ram_left_check() const;
    void fast_destroy();
    bool forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err);
    bool fast_token(int32_t token, std::string& err);
    bool fast_layers(int64_t p, bool hop_in, std::string& err);
    bool fast_dsa(int il, int64_t p, std::string& err);     // one DSA mixer (x -> mixer)
    bool fast_moe(int il, bool& pf_pending, std::string& err, bool cpu_wait = true);   // one MoE FFN (x -> ffn)
    bool moe_defer_wait_ = false;                            // (fast_moe's cpu_wait = false: the caller waits)
    // ... its routed experts, once the router and the shared expert's gate / up have run: the token path's own
    // activations, or one row of a batched step's (multi_rows)
    struct MoeIo {
        const float* x;
        const void* xq;
        const float* rlog;
        const float* sh_g;
        const float* sh_u;
        float* ffn;
    };
    bool fast_moe_experts(int il, const MoeIo& io, bool pred, int n_ah, const float* const* ah_bias, bool& pf_pending,
                          std::string& err);
    void fast_service();
    // gate / up / down of one expert from the shards (or its chunk-th of n_chunks pieces)
    void fast_read_part(int il, int e, int role, uint8_t* blob, int chunk = 0, int n_chunks = 1);
    void fast_read_expert_x3(int il, int e, uint8_t* blob);   // a whole EXL3 expert in as few reads as it allows
    void fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[8]);   // LOOKAHEAD
    void fast_ahead_reader();
    bool fast_boundary(std::string& err);                  // between tokens: apply finished promotions
    int fast_sample(strata::kernels::SamplerParams& sp, std::string& err);   // this half's logits, sampled on its stream
    int fast_sample_verify(strata::kernels::SamplerParams& sp, std::string& err);   // speculative sampling's verify
    int64_t ram_budget_ = -1;                              // bytes of pinned RAM tier for this half (-1: derive)
    float* snap_ = nullptr;                                // the saved KDA states + conv histories (this half)
    // the DSA indexer's key / gate caches (ik, ig) are read only to pool each kpool-position block, so the fast path
    // keeps a RING of ik_ring_ positions (a prompt sub-batch, <= 8192, + one open pool; the NextN block's cache fill goes
    // in pieces of the ring) - 8 KB/position/layer, 1 GB per card saved at 128K; a snapshot keeps the open pool's rows
    // (snap_pool_: per DSA layer kpool-1 ik rows, then kpool-1 ig rows), since later positions overwrite their slots
    int ik_ring_ = 0;
    // the fast path's latent cache in INT8 records (--kv int8 / STRATA_GLM_KV_INT8=1; glm_fast.hpp lat8_rec_bytes):
    // 544 bytes a position and layer against FP16's 1024
    bool lat_q8_ = false;
    // KV streaming (--kv-resident N / STRATA_GLM_KV_RESIDENT; strata/kernels/glm_kv_stream.hpp): the fast path's DSA
    // layers keep their latent cache in pinned RAM and N positions of it in VRAM - state_ + dsa_lat_[il] is then the
    // layer's VRAM slots.  Null: every position in VRAM.  (glm_fast_state.hpp)
    struct KvStream;
    KvStream* kv_ = nullptr;
    bool kv_stream_setup(int64_t cells, std::string& err);   // load_pack, once the state arena is in
    // a stream for one sequence's caches of a context of ctx tokens (null: no DSA layer here, or err), and its free
    KvStream* kv_stream_make(int64_t cells, int64_t ctx, std::string& err);
    void kv_stream_free(KvStream* K);
    uint64_t kv_stage_bytes_ = 0;                            // the prompt path's staging copy (the first sequence's)
    void kv_stream_reset_maps();                             // nothing resident (reset, slot_load)
    float* snap_pool_ = nullptr;
    std::vector<int> dsa_layers() const;                   // this half's DSA layers (NextN block first), slot order
    void snap_pool_copy(bool restore);
    int64_t snap_pos_ = -1;
    std::vector<std::vector<float>> ckpt_buf_;             // ckpt_save's checkpoints (this half): KDA runs, pool rows
    std::vector<int64_t> ckpt_pos_;                        // ... their positions (-1: none)
    void ckpt_pool_copy(int64_t pos, float* host, bool restore);   // the open pool's rows at pos <-> host
    std::thread ckpt_th_;                                  // ckpt_save_snapshot's copy (every half's, on this one)
    void ckpt_join() { if (ckpt_th_.joinable()) ckpt_th_.join(); }
    std::string pack_dir_;
    std::string profile_path_;                             // --expert-profile (empty: the pack's own files)
    std::vector<std::pair<int32_t, int32_t>> profile_;     // its ranked (layer, expert) pairs, read by fast_setup
    bool fast_warm(std::string& err);                      // load-time: stream every expert into the tiers
    bool fast_cpu_lane_setup(std::string& err);            // STRATA_GLM_CPU_LANE: the pool, its calibration, the split
    bool fast_helper_setup(std::string& err);              // STRATA_GLM_EXPERT_HELPER: complementary GPU cache
    bool fast_helper_adapt(std::string& err);              // helper-only, at an idle boundary
    void fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out);
    // the same on `pool` with the rows act / hq / ff / dn, or x3 for an EXL3 pack (a verify window's CPU lanes:
    // FastState::win_lane, each with its own rows)
    void lane_experts(glmfast::Workers& pool, uint8_t* act, uint8_t* hq, float* ff, float* dn, float* x3, int il,
                      int ne, const uint8_t* const* blob, const float* w, const float* x, float* out);
    void win_lanes_stop();
    void win_lanes_make(std::shared_ptr<glmfast::Workers> p0, std::shared_ptr<glmfast::Workers> p1,
                        const std::vector<int>& c0, const std::vector<int>& c1);
    // a batched verify window's layer on both window lanes, each distinct expert read once (glm_fast_path.cu)
    void win_cpu_merged(int il, int nb, const strata::kernels::glmf::MoeRequest* const* rq,
                        strata::kernels::glmf::CpuAnswer* const* ans, const unsigned int* seq);
    // ---- the NextN (MTP) draft block (the last half carries it; layer index n_layers)
    int mtp_il_ = -1;                                      // its layer index here, -1: not on this half
    int mtp_src_ = -1;                                     // the shard of STRATA_GLM_MTP_GGUF's block, -1: the model's
    int lt_ = 0;                                           // the tier layers: [l0_, lt_) = the trunk's + the draft's
    void* mtp_arena_ = nullptr;
    uint64_t mtp_arena_bytes_ = 0;                         // (the VRAM report)
    bool load_mtp(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs, std::string& err);
    // the draft block at position p -> mtp_tok[step] (see glm_fast_path.cu: the chained drafts and the draft head)
    bool fast_mtp(int64_t p, int32_t next_tok, std::string& err, int step = 0, bool sub = false, bool read_back = true);
    void mtp_draft_setup();                                // fast_setup: the draft head's rows, the chain's embeddings
    // the draft vocabulary (load_pack, from the GGUF's tokenizer - see glm_mtp.cu): ids [0, mtp_vlead_) and
    // mtp_vextra_; mtp_vin_[id] != 0 for each (empty: the whole head)
    int mtp_vlead_ = 0;
    std::vector<int32_t> mtp_vextra_;
    std::vector<uint8_t> mtp_vin_;
    void mtp_vocab_scan(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs);
public:
    /// The NextN draft for the token after next: after forward() of position p, given the token chosen for p + 1,
    /// returns the block's greedy proposal for p + 2 (-1: no draft block).  Fills the block's caches at p.
    int mtp_draft(int32_t next_tok, std::string& err);
    bool has_mtp() const;
    bool kv_int8() const { return lat_q8_; }   // the latent cache in INT8 (STRATA_GLM_KV_INT8=1)
    /// KV streaming: the positions of each DSA layer kept in VRAM (0: all of them - not streamed); and what
    /// STRATA_GLM_KV_RESIDENT gives for a context: rounded up to whole pools and to glmf::kKvResidentMin, 0 when unset
    /// or when the context fits in that many anyway
    int64_t kv_resident() const;
    static int64_t kv_resident_cells(int kpool, int64_t max_ctx);
    /// ... and its block reads since the start, every part summed: those the selections named, those read from RAM
    struct KvStreamStats {
        uint64_t lookups = 0, misses = 0;
        bool overflow = false;   // a resolve could not place every block (never, with a legal slot count)
        double host_gb = 0;      // the pinned host copies
        int block_bytes = 0;     // one page of latent rows: what a miss reads
    };
    KvStreamStats kv_stream_stats() const;
    /// Images (the vision path): rows of n_embd floats that stand in for the token embeddings at these absolute
    /// positions - the prompt's <|image|> tokens, in order; an empty call clears them.  Read wherever a token is
    /// embedded (the prompt path - the NextN block's cache fill included - and the token path).
    void set_image_rows(std::vector<int64_t> positions, std::vector<float> rows) {
        img_pos_ = std::move(positions);
        img_rows_ = std::move(rows);
    }
    const float* image_row(int64_t p) const {
        const auto it = std::lower_bound(img_pos_.begin(), img_pos_.end(), p);
        if (it == img_pos_.end() || *it != p) return nullptr;
        return img_rows_.data() + (size_t) (it - img_pos_.begin()) * (size_t) g_.n_embd;
    }
private:
    std::vector<int64_t> img_pos_;                         // ascending
    std::vector<float> img_rows_;
public:
    /// The pipelined speculative decode (src/core/glm_fast_path.cu): needs a two-half split whose tail carries the
    /// NextN block.  After the prompt's last forward(), emits up to max_new tokens (emit returns false to stop) - the
    /// same tokens the token-at-a-time loop would produce for the same samples.  The sequence state is left for the
    /// next request to restore (snapshot) or reset.
    bool spec_ready() const;
    bool decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                     int64_t& produced, std::string& err);
    uint64_t spec_steps_ = 0, spec_hits_ = 0;              // speculative positions, accepted
    double spec_t_[4] = {0, 0, 0, 0};                      // STRATA_GLM_SPEC_PROF host phases of spec_head/tail
    int64_t spec_n_ = 0;
    cudaEvent_t spec_ev_[3] = {nullptr, nullptr, nullptr}; // ... and the device span of each token
    bool spec_ev_live_ = false;
    double spec_dev_ms_ = 0, spec_gap_ms_ = 0;
    int64_t spec_dev_n_ = 0;
    float* kda_bak_ = nullptr;                             // the head's saved recurrent states
    bool kda_bak_cur_ = false;   // win_commit_part left kda_bak_ equal to the live states (the next pipelined begin's
                                 // backup is not needed)
    float* spec_hop_h_[2] = {nullptr, nullptr};            // pinned hop slots by position parity
    cudaEvent_t spec_ev_hop_[2] = {nullptr, nullptr};
    bool spec_kda_copy(bool restore);
    size_t spec_kda_bytes() const;                         // the head's saved recurrent states on this part
    bool spec_kda_reserve(std::string& err);               // ... taken at setup, before the expert pool
    // the speculative decode on a split of more than two parts: the HEAD GROUP is this part and the ones after it
    // up to spec_head_last() (STRATA_GLM_SPEC_HEAD=<parts>, default half), the TAIL GROUP the rest; each group runs
    // its parts one after another and the two groups overlap as the two halves of a two-part split do
    Glm5Model* spec_head_last();
    Glm5Model* spec_tail_last();
    // this part's residual into the next part B (one host hop through this part's pinned hop_h), then B's layers at p
    bool hop_to_next(Glm5Model* B, int64_t p, std::string& err);
    bool spec_head(int64_t p, int32_t token, std::string& err);
    bool spec_tail(Glm5Model* head, int64_t p, std::string& err);
    /// The one-GPU MTP decode (src/core/glm_mtp.cu): each round the NextN block drafts up to n tokens in a chain (its
    /// own hidden state feeding the next step), one verify window (window()) checks [last token, drafts...], the
    /// target's samples accept the drafts while they match (window_commit keeps them), and the NextN block's cache takes
    /// the kept rows with the trunk's hidden states.  The same tokens the token-at-a-time loop gives for the same
    /// samples.  On by default on the Strix / Gorgon Halo APUs (gfx115x, mtp_decode_default); STRATA_GLM_MTP_DECODE=1
    /// elsewhere, =0 off.  mtp_ready(): this model was set up for it.
    static bool mtp_decode_wanted(int dev, int n_parts);
    bool mtp_ready() const;
    bool decode_mtp(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                    int64_t& produced, std::string& err);
    uint64_t mtp_rounds_ = 0;                              // verify passes (spec_steps_ / spec_hits_: drafts, accepted)
    uint64_t mtp_round_n_[9] = {};                         // ... by the draft length the round used
    bool mtp_fill_prev(int32_t token, int64_t p, std::string& err);   // the token path: entry p - 1 from head_x and token
private:
    bool mtp_dec_ = false;                                 // the one-GPU MTP decode is set up on this part
    int64_t mtp_hx_pos_ = -1;                              // the position whose trunk hidden state head_x holds
    // the adaptive draft length: per length n, the rounds' wall time (EMA, ms) and how many; per draft position, how
    // often it was reached (its earlier drafts all accepted) and accepted - decayed counts, the recent text's
    // (STRATA_GLM_MTP_ACC_WINDOW)
    double mtp_ms_[9] = {};
    uint64_t mtp_seen_[9] = {};
    double mtp_reach_[9] = {}, mtp_hit_[9] = {};
    uint64_t mtp_round_no_ = 0;
    int mtp_best_ = -1;                                    // the length in use, the probe gap and the next probe
    uint64_t mtp_gap_ = 32, mtp_next_probe_ = 0;
    bool mtp_probe_up_ = false;
    int mtp_stint_n_ = 0, mtp_stint_left_ = 0;             // a length's run of rounds (exploring, probing)
    int mtp_prev_n_ = -1, mtp_same_ = 0;                   // the last round's length and the rounds of it in a row
    double mtp_out_ = 0.0;                                 // the share of emitted tokens outside the draft vocabulary
                                                           // (an EMA over ~64): above 2%, drafts take the whole head
    int mtp_choose(int n_cap);
    bool mtp_cache_setup(std::string& err);                // fast_setup: the cache fill's row buffers (kWinMax rows)
    void mtp_cache_free();
    // the NextN block's cache-writing half for n entries from p0: entry p0 + i from h + i * n_embd (device, this part)
    // and the token next[i] - eh_proj, the attention norm, kv_a and the indexer's key/gate, no query, no FFN
    bool mtp_cache_rows(int64_t p0, int n, const float* h, const int32_t* next, std::string& err, bool sync = true);
public:
    // ---- several conversations at once (src/core/glm_multi.cu, STRATA_GLM_SEQS=<n>): each part keeps n sequence
    // states (the state arena: residual, KDA states, DSA caches; the position, the snapshot) and binds one at a time -
    // every path that reads `state_` (the token path, the prompt path, snapshots, slots) then works on the bound one.
    // The pipelined decode issues a token of one sequence through every part without waiting, so with n sequences in
    // flight each part works on a different one (a layer split's cards no longer wait for each other).
    struct SeqCtx {
        float* state = nullptr;
        int64_t pos = 0, snap_pos = -1;
        // its own context (STRATA_GLM_SEQ_CTX: the extra sequences can be smaller than the first) and so its own
        // arena layout
        int64_t max_ctx = 0;
        uint64_t bytes = 0;
        int ik_ring = 0;
        std::vector<int64_t> kda_S, kda_conv, dsa_lat, dsa_ik, dsa_ig, dsa_pool;
        float *snap = nullptr, *snap_pool = nullptr, *kda_bak = nullptr;
        int last_tok = -1;
        std::vector<int64_t> img_pos;
        std::vector<float> img_rows;
        // the pipeline's per-sequence buffers on this part: the boundary hop (a sequence's next token waits for its
        // last one, so one slot each is enough), the embedding row (head part), the sampled token (tail part)
        float* hop_h = nullptr;
        cudaEvent_t ev_hop = nullptr;
        float* emb_h = nullptr;
        int* tok_h = nullptr;
        cudaEvent_t ev_done = nullptr;
        KvStream* kv = nullptr;   // KV streaming: its latent caches' host copy and maps (null: not streamed, or bound)
    };
    std::vector<SeqCtx> seq_;                              // empty: one sequence (the classic engine)
    int cur_seq_ = 0;
    int seq_count() const { return seq_.empty() ? 1 : (int) seq_.size(); }
    int seq_bound() const { return cur_seq_; }
    void seq_bind(int s);                                  // this part
    void seq_bind_all(int s);                              // every part of the split
    bool seq_alloc_arenas(std::string& err);               // load: the extra arenas (STRATA_GLM_SEQS), before the pool
    uint64_t seq_layout(int64_t ctx, SeqCtx& c) const;     // load_pack's arena layout for a context of ctx tokens
    int64_t seq_max_ctx(int s) const {                     // a sequence's context (the bound one's is max_ctx_)
        return seq_.empty() || s == cur_seq_ ? max_ctx_ : seq_[(size_t) s].max_ctx;
    }
    bool seq_pipe_ready(std::string& err);                 // the pipeline buffers (once)
    /// The pipelined decode: `token` of sequence s at its next position through every part and the head, sampled with
    /// `sp` (greedy when sp.greedy) on the tail's stream - nothing is waited for.  multi_take(s) waits for that token.
    bool multi_issue(int s, int32_t token, const strata::kernels::SamplerParams& sp, std::string& err);
    int multi_take(int s, std::string& err);
    /// Every part idle, then the between-token tier work (fast_boundary) on each: once every few rounds.
    bool multi_boundary(std::string& err);
    /// ONE GPU (no split; src/core/glm_rows.cu): one token of each of n sequences through the model together.  The
    /// dense weights - the attention projections, the shared experts, the dense FFN, the head - are read once for all
    /// n; the residual mixing, the attention against each sequence's caches and the routed experts run per sequence
    /// through the token path's own kernels, so each sequence gets the token the token path would give it.
    /// tokens_out[i]: sequence seqs[i]'s next token after tokens[i] (greedy, or sampled with sps[i]); every
    /// sequence's position advances by one.  n <= rows_max().
    bool rows_ok() const { return rows_ != nullptr && n_parts_ == 1 && split_next_ == nullptr && fast_ != nullptr; }
    static int rows_max();
    bool multi_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n,
                    int* tokens_out, std::string& err);
    /// LAYER SPLIT (glm_rows.cu): one token of each of n sequences through every part as one batched step per part,
    /// queued like multi_issue (nothing waited for); multi_take(seqs[i]) returns each one's token.  n <= rows_max().
    bool split_rows_ok() const;
    bool multi_issue_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n,
                          std::string& err);
private:
    struct RowSeq;
    bool rows_bind(const int* seqs, int n, RowSeq* rw, std::string& err);
    bool rows_embed(const RowSeq* rw, const int32_t* tokens, int n, std::string& err);
    bool rows_layers(RowSeq* rw, int n, std::string& err);
    bool rows_head(const RowSeq* rw, const strata::kernels::SamplerParams* sps, int n, std::string& err);
public:
    struct RowsState;
    RowsState* rows_ = nullptr;
    bool rows_alloc(std::string& err);                     // load, several sequences on one GPU: before the pool
    void rows_destroy();

    // ---- the verify window (src/core/glm_window.cu)
    static constexpr int kWinMax = 16;                     // its tokens at most (gf::kMaxMvCols)
    bool win_alloc(int tmax, std::string& err);            // this part's window buffers, for up to tmax tokens
    void win_free();
    bool fast_window_layers(int64_t p0, int T, std::string& err);   // T rows through this part's layers
    bool fast_dsa_window(int il, int64_t p0, int t0, int T, std::string& err);
    bool fast_moe_window(int il, int t0, int T, int part, int gid, std::string& err);
    /// The verify WINDOW: `tokens[0..T)` at positions position()..+T-1 (the last accepted token and T-1 drafts) through
    /// every part's layers in one pass; `out[t]` = the pick after token t - the greedy argmax, or with `sp` (not
    /// greedy) row t's sample at counter sp->counter + t: what token-at-a-time decode would pick there.  Leaves the
    /// position at +T; window_commit(n) keeps the first n tokens (1 <= T <= kWinMax).
    bool window(const int32_t* tokens, int T, int32_t* out, std::string& err,
                const strata::kernels::SamplerParams* sp = nullptr);
    /// The same window PIPELINED over a split: the tokens go through the token path one after another, each part
    /// starting token t + 1 as soon as it has handed token t on, so the parts work on different tokens at once (a
    /// window costs about one token's pass plus (T - 1) of the slowest part's, where window() runs the parts one after
    /// the other).  Same picks and the same window_commit.
    bool window_pipe(const int32_t* tokens, int T, int32_t* out, std::string& err,
                     const strata::kernels::SamplerParams* sp = nullptr);
    /// window_pipe in steps: win_pipe_begin queues the first token's pass (`sp` not greedy: the picks are draws; not
    /// `fresh`: it follows the previous window's late tokens without waiting for the devices), win_pipe_push each
    /// further token's (kPipeMax in all), win_pipe_pick(t) waits for the pick after token t; then win_pipe_settle
    /// (every part's share enqueued) before window_commit, or win_pipe_end (every pick, the devices drained)
    // (its tokens at most: the anchor and a DFlash block's 15 drafts; the rows a token crosses with alternate between two
    // banks of kPipeMax, kPipeSlots in all - a window's token index stays below kWinMax, its KDA input rows')
    static constexpr int kPipeMax = 16;
    static constexpr int kPipeSlots = 2 * kPipeMax;
    static_assert(kPipeSlots <= 64, "the head split's events (hh_ev_, ht_ev_)");
    static_assert(kPipeMax <= kWinMax, "a pipelined token's KDA input rows are the window's");
    bool win_pipe_begin(int32_t first, std::string& err, const strata::kernels::SamplerParams* sp = nullptr,
                        bool fresh = true);
    bool win_pipe_begin_part(bool fresh, std::string& err);   // a part's own share of win_pipe_begin
    void win_commit_part(int n_keep);                          // ... and of window_commit
    std::chrono::steady_clock::time_point win_pipe_begun_at(int k) const;
    bool win_pipe_push(int32_t token, std::string& err);
    bool win_pipe_pick(int t, int32_t& pick, std::string& err);
    bool win_pipe_settle(std::string& err);
    void win_pipe_cancel_from(int t);   // tokens t.. of this window are past a wrong draft: their work is skipped
    void win_pipe_drain();              // every part's queued work (a window's cancelled tokens, a commit) done
    bool win_pipe_end(int32_t* out, std::string& err);
    /// window_pipe's enqueue threads (the head part owns them): one a part, each issuing its part's share of every
    /// token - a token's launches on nine parts took the one host thread ~3-5 ms, longer than any part's GPU time
    struct PipeCrew;
    PipeCrew* crew_ = nullptr;
    bool win_pipe_part(int k, int t, std::string& err);   // part k's share of window token t (a crew thread's job)
    void win_crew_stop();
    /// window_pipe's token t: KDA layer il's projections go to its window input rows (window_commit replays from them)
    void win_kda_rows(int il, int t, float** proj, float** fa, float** ga, float** beta);
    // (wait = false, a pipelined window: the parts' commit enqueues are not waited for - the next window's begin is
    //  queued behind each on its crew thread; win_commit_wait before anything else reads the state)
    bool window_commit(int n_keep, std::string& err, bool wait = true);
    void win_commit_wait();
    bool win_reserve(std::string& err);                    // the window's buffers and state backup, before the pool
    void win_prof_report();                                // STRATA_GLM_WIN_PROF: each part's window time, then reset
    // ---- the DFlash drafter (src/core/glm_dflash.cu): the part with the head holds it, on its device
    std::unique_ptr<GlmDflash> dflash_;
    std::string dflash_path_;                              // STRATA_GLM_DFLASH: its GGUF (every part knows it)
    bool dflash_on_ = false;                               // a drafter will be bound: the window's buffers at setup
    GlmDflash* cap_d_ = nullptr;                           // every part: the drafter its captures go to ...
    std::vector<int> cap_k_;                               // ... absolute layer -> its feature index (-1: not read)
    int64_t cap_floor_ = 0;                                // ... positions below it are not captured (a long prompt)
    bool dflash_load(const std::string& path, int dev, std::string& err);
    bool dflash_bind(std::string& err);                    // after the parts loaded: the captures wired
    void dflash_capture(int il, const float* R, int n, int64_t p0, float* scratch, int rows, cudaStream_t s);
    bool dflash_ready() const;
    void dflash_reset();
    /// The DFlash decode (Strata's serial speculative loop with a block drafter): each round the drafter's block for
    /// the last token (up to n_max drafts, fewer where the selector's confidence is below p_min; n_max < 0: the
    /// drafter's block, p_min < 0: 0.5 on a split, 0.9 on one GPU) is checked by one verify window - pipelined on a split (window_pipe's steps, stopped at
    /// the first wrong draft), batched on one GPU - the same tokens as token-at-a-time decode for the same samples.
    /// After the prompt's last forward().
    /// n_min / p_cum: GlmDflash::draft_finish's (< 0: STRATA_GLM_DFLASH_MIN / _CUM, else 0)
    bool decode_dflash(strata::kernels::SamplerParams& sp, int64_t max_new, int n_max, float p_min,
                       const std::function<bool(int)>& emit, int64_t& produced, std::string& err, int n_min = -1,
                       float p_cum = -1.0f);
    // ---- the batched prompt path (src/core/glm_prefill.cu)
    struct PrefillState;
    PrefillState* pf_ = nullptr;
    // STRATA_GLM_PREFILL_CALC: this (waiting) part's share of another part's MoE layer, enqueued on this GPU
    struct CalcJob;
    bool prefill_calc(const CalcJob& job, std::string& err);
    bool prefill_setup(std::string& err);                  // in fast_setup: sizes, host buffers (no device memory)
    size_t prefill_borrow_bytes() const;                   // device bytes the prompt path borrows from the pool's tail
    size_t prefill_trim_prestage(size_t max_borrow);       // ... at most this many: a smaller prestage buffer (the new bytes)
    bool prefill_bind(uint8_t* region, size_t bytes, std::string& err);   // its buffers inside the borrowed region
    // ... laid out for chunks of T (the bytes it uses); end: the prompt's last position + 1, which the KV streaming stage
    // must hold (-1: the whole context)
    size_t prefill_carve(int T, int64_t end = -1);
    bool prefill_lend(std::string& err);                    // the tail slots -> the prompt path (drops their experts)
    bool prefill_return(std::string& err);                  // ... and back to the expert pool
    void prefill_destroy();
    void prefill_cap(int T, const char* why);              // its pinned staging for chunks of at most T
    void prefill_settle(double pinned_share);              // --prefill auto's lend cap from the pinned share (85/90%)
    // a chunk's device buffers {kept rows, scratch}; kv_stage: the KV streaming stage's bytes (SIZE_MAX: the whole
    // context's, kv_stage_bytes_)
    std::pair<size_t, size_t> prefill_bytes_for(size_t T, size_t kv_stage = SIZE_MAX) const;
    // the pinned host bytes a part's prompt path takes at its start (the token rows and the disk landing ring) and the
    // residual rows a part that hands on adds - this part's sizes
    struct PinnedPrompt {
        size_t staging = 0, hop = 0;
    };
    PinnedPrompt prefill_pinned() const;
    // the expert pool's bytes on part `part` from its free VRAM (glm_fast_path.cu): --expert-cache auto | N and the VRAM
    // reserve (STRATA_GLM_EXPERT_CACHE, STRATA_GLM_RESERVE_MB / _RESERVE_LATER_MB; the driver sets them from the
    // config's args).  floor_b: the pool's floor (a route's experts and the spares, every layer), which a small card's
    // default reserve gives way to; slot_b: the part's largest expert slot (--expert-cache N is N of them)
    struct PoolBudget {
        size_t avail = 0, reserve = 0;
        int64_t fixed = 0;              // --expert-cache N (the first GPU), 0: auto
        bool reserve_given = false;     // a reserve set by hand is kept as it is
        bool adapted = false;           // the default reserve lowered so the pool holds its floor
    };
    static PoolBudget pool_budget(size_t free_b, size_t total_b, int part, size_t floor_b, size_t slot_b,
                                  bool unified = false);
    PoolBudget part_budget(size_t free_b, size_t total_b, size_t floor_b, size_t slot_b) const;   // ... less pool_short_
    void pool_check(size_t floor_b, size_t slot_b, size_t all_b);   // auto: the budget written once, free VRAM read again
    size_t pool_short_ = 0;                                // what that check found short of the reserve
    size_t commit_kept_ = 0, commit_limit_ = 0, commit_ram_ = 0;   // (a Windows APU: what the commit kept off, as read)
    PoolBudget pool_b_;                                    // the budget the pool was carved from
    bool lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err);
    bool vis_lend_ok_ = false;                             // this half's tail can go to the vision encoder
    bool vis_lent_ = false;                                // ... and is with it now (freed)
    // this half's layers over a chunk (rows in pf_->R); next_ids / next_img: the NextN block's next tokens and, where
    // one is an <|image|>, the image row standing for it (null: its token embedding)
    bool prefill_half(int64_t p0, int T, std::string& err, const int32_t* next_ids = nullptr,
                      const float* const* next_img = nullptr);
    int32_t prefill_next_ = -1;                            // the token after the prompt's last prefilled position
    /// The prompt path: `tokens` at positions [position(), position() + n) through every layer in chunks, layer by
    /// layer (GEMMs for the projections, MMQ for the routed experts), leaving the state the token path would have
    /// left; no logits.  false with `err` empty when the fast path has no prompt path (feed them to forward()).
    bool prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token = -1);
    bool prefill_run(const std::vector<int32_t>& tokens, std::string& err);   // prefill() between lend and return
    int prefill_chunk() const;                             // tokens per chunk (0: no prompt path)
    /// Called (from any thread) when a chunk has gone through every layer: (positions done, total); false cancels
    /// the prompt (prefill then returns false with err "cancelled" and the sequence must be reset).
    std::function<bool(int64_t, int64_t)> prefill_progress;
    /// Debug: every half's sequence state (KDA states + conv histories, DSA caches) to <path>.<half>.bin, with the
    /// per-layer offsets in <path>.<half>.txt - two runs that should agree are compared per layer and component.
    bool dump_state(const std::string& path);

private:
    // scratch offsets, in floats into sc_
    int64_t sc_mixed = 0, sc_x = 0, sc_pre = 0, sc_post = 0, sc_comb = 0, sc_inv = 0, sc_proj = 0;
    int64_t sc_emb = 0;
    int64_t sc_conv[3] = {0, 0, 0};
    int64_t sc_g = 0, sc_g1 = 0, sc_beta = 0, sc_tmp = 0, sc_scan = 0, sc_g2 = 0, sc_gated = 0;
    int64_t sc_qr = 0, sc_kv = 0, sc_q = 0, sc_qabs = 0, sc_iq = 0, sc_iw = 0;
    int64_t sc_score = 0, sc_cells = 0, sc_attn = 0, sc_gate = 0, sc_up = 0, sc_h = 0, sc_dn = 0;
    int64_t sc_moe = 0, sc_sh = 0, sc_ffn = 0, sc_mixer = 0, sc_head = 0, sc_logits = 0;
    int64_t sc_ids = 0, sc_rw = 0;
};

/// What a part with these layer kinds lends the prompt path for a chunk of T tokens, its prestage buffer aside
/// (src/core/glm_prefill.cu) - the layer split search's startability gate.  kv_stage: KV streaming's staging copy of
/// one layer's latent cache (0: not streamed).
size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride, size_t kv_stage = 0);

}  // namespace strata::core
