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
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

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
        uint64_t ram_hits = 0, disk_reads = 0, promotions = 0, cpu_experts = 0;
        double ms = 0, miss_ms = 0, disk_ms = 0, cpu_ms = 0;
        int64_t pool_slots = 0, pool_used = 0, ram_slots = 0, ram_used = 0;
        double pool_gb = 0, ram_gb = 0;
    };
    FastStats fast_stats() const;
    /// The CPU LANE (a decode's RAM-tier experts): the share of them copied over PCIe to the GPU instead of computed
    /// on the CPU - 0: the CPU takes every one, 1: none (of f such experts in a route, f - round(share f) go to the
    /// CPU).  pcie_share(): the split in use, as a share; set_pcie_share(s): this split from the next token on, s < 0
    /// the one the engine started with (STRATA_GLM_PCIE_SHARE, else measured at start).  cpu_lane_threads(): the
    /// threads taking its jobs; set_cpu_lane_threads(n): at most n of the pool's, n <= 0 all of them.  Both are for
    /// setup's calibration (per request, no restart); false when this engine has no CPU lane.
    double pcie_share() const;
    /// STRATA_GLM_CPU_CAL: once a request (2000+ lane experts a window), drop a calibration decode stays >15% from
    void lane_drift_check();
    bool set_pcie_share(double share);
    int cpu_lane_threads() const;
    bool set_cpu_lane_threads(int n);
    /// Conversation reuse (the fast path): save the sequence state at the current position - the recurrent KDA
    /// states and conv histories (the DSA caches are append-only, so the position alone restores them) - and
    /// restore it later to continue a prompt that extends the saved one.  false when unsupported.
    bool snapshot_save();
    // the expert usage profile kept between sessions (the warm-up follows it): written after each request
    bool save_usage();
    std::string usage_path() const;
    bool snapshot_restore();
    int64_t snapshot_pos() const { return snap_pos_; }
    /// Conversation slots: the snapshot (its KDA states) and every DSA cache's rows up to its position, every half,
    /// written to a file; slot_load puts them back as the live state and the snapshot, so a conversation set aside
    /// while others ran continues where it was.  The bytes written, or 0 + err.
    uint64_t slot_save(const std::string& path, std::string& err);
    bool slot_load(const std::string& path, int64_t n_pos, std::string& err);
    /// Runs the engine's sampler on the last forward's logits and returns the picked token id
    /// (-1 + err on failure).  Handles the device bookkeeping: with a split the logits (and the
    /// sampler buffer) live on the tail half's device, where peer reads are not available.
    int sample_token(strata::kernels::SamplerParams& sp, std::string& err);
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
        const int32_t t = force_tok_ >= 0 ? force_tok_ : sampled;
        force_tok_ = -1;
        return t;
    }

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
    std::vector<int> split_devs_;                          // ... and the CUDA devices of all of them, part by part
    std::unique_ptr<Glm5Model> split_next_;
    std::vector<float> hop_;                               // the boundary residual staging buffer
    int* d_tok_ = nullptr;                                 // the sampler's one-int output, on dev_
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
    };
    std::vector<NativeLayer> pack_layers_;                 // indexed by layer id (dense lead = null)
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
    int32_t force_tok_ = -1;                               // force_next()
    std::map<std::string, const uint16_t*> w16_;           // the pack's big BF16 rows, kept BF16 (fast mode)
    bool fast_setup(std::string& err);
    void fast_destroy();
    bool forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err);
    bool fast_token(int32_t token, std::string& err);
    bool fast_layers(int64_t p, bool hop_in, std::string& err);
    bool fast_dsa(int il, int64_t p, std::string& err);     // one DSA mixer (x -> mixer)
    bool fast_moe(int il, bool& pf_pending, std::string& err);   // one MoE FFN (x -> ffn)
    void fast_service();
    // gate / up / down of one expert from the shards (or its chunk-th of n_chunks pieces)
    void fast_read_part(int il, int e, int role, uint8_t* blob, int chunk = 0, int n_chunks = 1);
    void fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[8]);   // LOOKAHEAD
    void fast_ahead_reader();
    bool fast_boundary(std::string& err);                  // between tokens: apply finished promotions
    int fast_sample(strata::kernels::SamplerParams& sp, std::string& err);   // this half's logits, sampled on its stream
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
    float* snap_pool_ = nullptr;
    std::vector<int> dsa_layers() const;                   // this half's DSA layers (NextN block first), slot order
    void snap_pool_copy(bool restore);
    int64_t snap_pos_ = -1;
    std::string pack_dir_;
    bool fast_warm(std::string& err);                      // load-time: stream every expert into the tiers
    bool fast_cpu_lane_setup(std::string& err);            // STRATA_GLM_CPU_LANE: the pool, its calibration, the split
    void fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out);
    // ---- the NextN (MTP) draft block (the last half carries it; layer index n_layers)
    int mtp_il_ = -1;                                      // its layer index here, -1: not on this half
    int mtp_src_ = -1;                                     // the shard of STRATA_GLM_MTP_GGUF's block, -1: the model's
    int lt_ = 0;                                           // the tier layers: [l0_, lt_) = the trunk's + the draft's
    void* mtp_arena_ = nullptr;
    bool load_mtp(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs, std::string& err);
    bool fast_mtp(int64_t p, int32_t next_tok, std::string& err);   // the draft block at position p -> mtp_tok
public:
    /// The NextN draft for the token after next: after forward() of position p, given the token chosen for p + 1,
    /// returns the block's greedy proposal for p + 2 (-1: no draft block).  Fills the block's caches at p.
    int mtp_draft(int32_t next_tok, std::string& err);
    bool has_mtp() const;
    bool kv_int8() const { return lat_q8_; }   // the latent cache in INT8 (STRATA_GLM_KV_INT8=1)
    /// Images (the vision path): rows of n_embd floats that stand in for the token embeddings at these absolute
    /// positions - the prompt's <|image|> tokens, in order; an empty call clears them.  Read wherever a token is
    /// embedded (the prompt path and the token path); the draft block keeps the token embedding (drafts only).
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
    float* spec_hop_h_[2] = {nullptr, nullptr};            // pinned hop slots by position parity
    cudaEvent_t spec_ev_hop_[2] = {nullptr, nullptr};
    bool spec_kda_copy(bool restore);
    // the speculative decode on a split of more than two parts: the HEAD GROUP is this part and the ones after it
    // up to spec_head_last() (STRATA_GLM_SPEC_HEAD=<parts>, default half), the TAIL GROUP the rest; each group runs
    // its parts one after another and the two groups overlap as the two halves of a two-part split do
    Glm5Model* spec_head_last();
    Glm5Model* spec_tail_last();
    // this part's residual into the next part B (one host hop through this part's pinned hop_h), then B's layers at p
    bool hop_to_next(Glm5Model* B, int64_t p, std::string& err);
    bool spec_head(int64_t p, int32_t token, std::string& err);
    bool spec_tail(Glm5Model* head, int64_t p, std::string& err);
    // ---- the batched prompt path (src/core/glm_prefill.cu)
    struct PrefillState;
    PrefillState* pf_ = nullptr;
    bool prefill_setup(std::string& err);                  // in fast_setup: sizes, host buffers (no device memory)
    size_t prefill_borrow_bytes() const;                   // device bytes the prompt path borrows from the pool's tail
    size_t prefill_trim_prestage(size_t max_borrow);       // ... at most this many: a smaller prestage buffer (the new bytes)
    bool prefill_bind(uint8_t* region, size_t bytes, std::string& err);   // its buffers inside the borrowed region
    size_t prefill_carve(int T);                           // ... laid out for chunks of T (the bytes it uses)
    bool prefill_lend(std::string& err);                    // the tail slots -> the prompt path (drops their experts)
    bool prefill_return(std::string& err);                  // ... and back to the expert pool
    void prefill_destroy();
    void prefill_cap(int T, const char* why);              // its pinned staging for chunks of at most T
    void prefill_settle(double pinned_share);              // --prefill auto's lend cap from the pinned share (85/90%)
    std::pair<size_t, size_t> prefill_bytes_for(size_t T) const;   // a chunk's device buffers {kept rows, scratch}
    // the pinned host bytes a part's prompt path takes at its start (the token rows and the disk landing ring) and the
    // residual rows a part that hands on adds - this part's sizes
    struct PinnedPrompt {
        size_t staging = 0, hop = 0;
    };
    PinnedPrompt prefill_pinned() const;
    static size_t pool_avail(size_t free_b, size_t total_b);   // the expert pool's bytes from the free VRAM
    bool lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err);
    bool vis_lend_ok_ = false;                             // this half's tail can go to the vision encoder
    bool vis_lent_ = false;                                // ... and is with it now (freed)
    bool prefill_half(int64_t p0, int T, std::string& err,
                      const int32_t* next_ids = nullptr);   // this half's layers over a chunk (rows in pf_->R)
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
/// (src/core/glm_prefill.cu) - the layer split search's startability gate.
size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride);

}  // namespace strata::core
