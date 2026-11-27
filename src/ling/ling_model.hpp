// src/ling/ling_model.hpp - Ling-3.2-air (llama.cpp bailingmoe3) inference engine.
// Supports 1..N GPUs (layer split with verified peer copies), adaptive per-expert VRAM/RAM/NVMe-SSD tiers,
// GPU PCIe fetch (CUDA-graphable), AVX2 CPU lane with GPU-parity mode, double-buffered batched prefill,
// FP16 / INT8 MLA latent KV cache up to 262K context, persistent restore points, vision, and blk.42 MTP + lookup.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "hy4_kernels.cuh"
#include "hy4_lane.hpp"
#include "ling_kernels.cuh"
#include "ling_restore.hpp"
#include "ling_sys.hpp"
#include "ling_vision.hpp"

namespace maya {
class GgufFile;
struct TensorInfo;
}

namespace ling {

class GgufSet;   // ling_model.cu: a GGUF whole or in llama.cpp's shards

// Recoverable request-scoped routing/input error (fails the request, engine stays alive).
struct RouteError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Fatal sticky GPU error (names the first faulting kernel; engine exits cleanly for server restart).
struct GpuFaultError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class OffloadMode {
    kAuto = 0,   // measured per-PC split between GPU PCIe fetch, mix, and CPU lane
    kGpu  = 1,   // GPU adaptive VRAM cache + GPU fetch_copy (CUDA graph)
    kMix  = 2,   // GPU adaptive VRAM cache + up to mix_k_gpu misses on GPU + CPU lane for rest
    kCpu  = 3,   // GPU VRAM hits + CPU lane for all misses (no pinning required)
};

struct Config {
    int n_layers = 0;        // trunk blocks (block_count - nextn)
    int n_layer_all = 0;     // including MTP block
    int n_nextn = 0;
    int n_embd = 0, n_head = 0, vocab = 0;
    int ctx_train = 262144;  // bailingmoe3.context_length from GGUF
    int eos_id = -1;
    std::vector<int> stop_ids;  // EOS / EOT ids read from GGUF metadata & vocab
    bool is_stop_id(int tok) const {
        for (int s : stop_ids) if (s == tok) return true;
        return false;
    }
    float eps = 1e-6f;
    bool kda[64] = {};       // head_count_kv[il] == 0
    int d_conv = 0, kda_head = 0;
    float kda_lb = 0.0f;
    int kv_lora = 0, n_rot = 0, qk_head = 0, v_head = 0, qk_nope = 0;
    float rope_base = 0.0f;
    int n_expert = 0, n_used = 0, n_group = 0, n_group_used = 0;
    int n_ff = 0, n_ff_exp = 0, n_ff_shexp = 0, n_dense_lead = 0;
    float w_scale = 0.0f;
    bool w_norm = false;
    float clamp_exp[64] = {}, clamp_shexp[64] = {};
    int d_inner() const { return n_head * kda_head; }
};

struct Options {
    int ctx = 8192;
    int prefill_chunk = 2048;        // 1: token-by-token prefill; >1: attention/tile chunk size
    int prefill_block = 0;           // <= 0: derive from GPU VRAM pool & expert count; > 0: explicit macro-block cap
    bool ep_prefill = false;         // opt-in multi-GPU EP prefill relay
    double expert_vram_gib = -1.0;   // < 0: free VRAM minus headroom per GPU
    double reserve_gb = 0.0;         // extra VRAM held free per GPU (also used for 8/12/16 GB emulation)
    double pin_ram_gib = -1.0;       // < 0: derive from host available RAM; 0: off; > 0: explicit cap in GiB
    double ram_cap_gib = -1.0;       // < 0: auto (avail - 8 GiB); >= 0: cap RAM expert tier (remainder goes to NVMe SSD)
    std::vector<int> gpus = {0};     // CUDA device ordinals (layer split when size > 1)
    bool auto_sort_gpus = true;      // sort GPUs non-display & larger VRAM first by default (--no-auto-sort-gpus opts out)
    OffloadMode offload = OffloadMode::kAuto;
    int mix_k_gpu = 1;               // in kMix mode, max misses fetched on GPU per layer
    int cpu_threads = -1;            // hy4::Lane threads (-1: physical cores in affinity mask - 2)
    bool kv_q8 = false;              // INT8 MLA latent cache (672 B/layer/tok vs 1152 B FP16)
    bool use_graph = true;           // CUDA graph for single-token / windowed decode
    bool cache_warm = true;          // warm VRAM expert cache from prompt routing frequencies
    bool no_promo = false;           // disable per-token async VRAM promotions (for isolated placement checks)
    bool load_mtp = false;           // also load blk.<n_layers> MTP block
    bool repeatable = false;         // --repeatable only: same answers for 1 or N GPUs and any VRAM/CPU expert split (Q8_K lane parity, no fused cross-layer norm); a normal run can differ in the last digit between 1 and N GPUs
    bool lane_parity = false;        // CPU = GPU Q8_K per-256 rounding & exact warp-lane reduction
    int lookahead = -1;              // -1: auto (measure & keep if faster), 0: off, 1: on
    bool direct_io = false;          // buffered SSD reads by default (wins on NVMe under cgroup caps); --direct opts in
    bool recalibrate = false;        // ignore cached per-PC calibration and re-measure
    bool no_profile = false;         // disable default expert placement profile
    std::string profile_path;        // binary LNGP expert profile path
    std::string profile_out;         // optional output path to save accumulated routing profile
    std::string cal_file;            // keyed per-PC calibration file (default data/ling-calibration.txt)
    std::string restore_dir;         // opt-in directory for persistent disk restore points (empty = OFF by default)
    size_t restore_dir_max_mb = 2048; // total disk cap for restore_dir in MiB (oldest .lingrp evicted)
    std::string seam_dir;
    std::string mmproj_path;
};

struct TierStats {
    unsigned long long picks = 0;
    unsigned long long vram_hits = 0;
    unsigned long long copied = 0;
    unsigned long long zero_copy = 0;
    unsigned long long cpu_lane = 0;
    unsigned long long admitted = 0;
    unsigned long long ram_fetch = 0;
    unsigned long long ssd_fetch = 0;
    unsigned long long graph_captures = 0;
    unsigned long long launches_per_tok = 0;
};

class Model {
public:
    Model(const std::string& path, const Options& opt);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& cfg() const { return c_; }
    const std::string& info() const { return info_; }
    const std::string& start_plan_line() const { return start_plan_line_; }
    const std::string& cal_key_str() const { return cal_key_; }
    const std::string& cal_file_path() const { return cal_file_; }
    const std::string& gguf_path() const { return gguf_path_; }
    const Options& opts() const { return o_; }
    int output_dev() const { return gpus_.back().dev; }
    cudaStream_t output_stream() const { return gpus_.back().s; }
    bool has_mtp() const { return mtp_loaded_; }
    bool kv_q8() const { return o_.kv_q8; }
    const float* last_h_nextn() const { return gpus_.back().hn; }
    double last_warm_s() const { return last_warm_s_; }
    int cpu_threads_used() const { return lane_ ? lane_->threads() + 1 : 0; }
    double pcie_share() const { return pcie_share_; }
    const std::string& cal_file() const { return cal_file_; }
    const std::string& cal_key() const { return cal_key_; }
    int cpu_threads() const { return o_.cpu_threads; }
    bool direct_io_active() const {   // every shard's reader (a shard can fall back to buffered reads on its own)
        return !ssd_readers_.empty() && std::all_of(ssd_readers_.begin(), ssd_readers_.end(),
                                                    [](const auto& r) { return r->direct_active(); });
    }
    bool lookahead_active() const { return lookahead_on_; }
    double kv_gb() const { return total_kv_gb_; }
    double vram_gb() const { return total_vram_gb_; }
    double ram_gb() const { return total_ram_gb_; }
    double ssd_gb() const { return total_ssd_gb_; }
    size_t experts_all() const { return total_experts_; }
    size_t experts_vram() const { return total_vram_experts_; }
    size_t experts_ram() const { return total_ram_experts_; }
    size_t experts_ssd() const { return total_ssd_experts_; }
    int active_offload_mode() const { return (int) active_offload_; }

    // Wait for background library/kernel warmup to complete (idempotent, called automatically before compute).
    void wait_warmup();

    struct Image {
        int64_t start = 0;
        int width = 0;
        int height = 0;
        std::vector<float> rows;
    };
    const VisionGeo* vision() const { return (vision_ && vision_->loaded()) ? &vision_->geo() : nullptr; }
    const char* vision_room() const;
    size_t vision_room_bytes() const { return (vision_ && vision_->loaded()) ? vision_->max_room_bytes() : 0; }
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& out_rows, std::string& err);
    bool vlend(size_t& lent_mib, std::string& err);
    bool vreclaim(std::string& err);
    bool is_vlent() const { return vlent_; }
    void set_images(std::vector<Image> imgs);
    int rope_delta() const { return rope_delta_; }
    void set_rope_delta(int d) { rope_delta_ = d; }
    int max_ctx() const { return o_.ctx; }
    int prefill_block() const { return prefill_block_; }
    size_t kda_snapshot_bytes() const;
    int prefill_chunk_size() const { return gpus_.empty() ? o_.prefill_chunk : gpus_[0].pf_cap; }

    using KdaSnapshot = sys::RestoreSnapshotData;
    const sys::RestoreExpect& restore_expect() const { return restore_expect_; }
    void save_kda_snapshot(KdaSnapshot& snap, const std::vector<int>& tokens,
                           const std::vector<std::pair<int64_t, uint64_t>>& pics, bool include_mla = true) const;
    void restore_kda_snapshot(const KdaSnapshot& snap);
    bool save_restore_disk(const KdaSnapshot& snap, const std::string& dir, std::string& err) const;
    bool load_restore_disk(const std::string& dir, std::vector<KdaSnapshot>& out,
                           std::vector<std::string>* skipped_logs = nullptr) const;
    bool save_profile_now(const std::string& path, std::string& err) const;

    void reset();

    const float* step(int token, int pos);

    const float* prefill(const int* tokens, int n, int pos0, std::vector<float>* all_logits_host = nullptr,
                         const std::function<bool(int, int)>& on_progress = nullptr,
                         const std::atomic<bool>* stop_flag = nullptr);

    const float* mtp_step(const float* h_nextn_dev, int next_token, int pos);

    static constexpr int kMaxWin = 4;
    const float* step_window(const int* tokens, int W, int pos0);
    void commit_window(int n_accept, int W, int pos0, const int* tokens);

    // Propose up to max_k draft tokens using blk.42 MTP with per-token floor + cumulative confidence cut.
    // Optionally outputs full draft distributions (for exact Leviathan rejection sampling when temp > 0).
    void mtp_draft(int base_tok, int base_pos, int max_k, float floor, std::vector<int>& drafts,
                   std::vector<float>& confs, float floor0 = 0.0f, float cum_min = 0.0f,
                   std::vector<std::vector<float>>* draft_logits_out = nullptr);

    int argmax_logits(const float* logits_dev);
    void argmax_rows(const float* logits_dev, int W, int* out_ids);

    double vram_res_frac() const { return vram_res_frac_; }
    TierStats stats() const;
    std::string detokenize_one(int id) const;
    std::string detokenize(const std::vector<int>& ids) const;

private:
    static constexpr int kPfBatch = 32;
    static constexpr int kPfPmapRing = 8;

    struct Mat {
        const uint8_t* p = nullptr;
        int type = 0;
        int k = 0, n = 0;
        size_t ld = 0;
    };

    struct Layer {
        int dev_idx = 0;
        bool kda = false;
        const float* attn_norm = nullptr;
        const float* ffn_norm = nullptr;
        // KDA
        Mat wq, wk, wv, wo, wf, wbeta, wg;
        const float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr;
        const float *dt_bias = nullptr, *ssm_a = nullptr, *ssm_a_neg = nullptr, *ssm_norm = nullptr;
        float* conv_state = nullptr;
        float* state = nullptr;
        float* conv_bak = nullptr;
        float* state_tmp = nullptr;
        float* w_kda_buf = nullptr;
        // MLA
        Mat wqm, wkva, wkb, wvb, wgate, wom;
        const float* kv_a_norm = nullptr;
        void* lat = nullptr;
        uint16_t* k_pe = nullptr;
        // Dense or Shared FFN
        Mat fg, fu, fd;
        // MoE
        const void* router_w = nullptr;
        int router_type = 0;
        const float* rb_dev = nullptr;
        int etg = 0, etu = 0, etd = 0;
        // gate and up experts of different types: the kernels' two passes (Args / SetArgs::tu), never the CPU lane
        bool mixed_gu() const { return etu != etg; }
        int tu_arg() const { return mixed_gu() ? etu : 0; }
        size_t eg_ld = 0, eu_ld = 0, ed_ld = 0;
        size_t tr[3] = {0, 0, 0};
        size_t blob = 0;
        // GGUF file offsets for SSD-backed / file-order reads
        uint64_t gate_file_off = 0, up_file_off = 0, down_file_off = 0;
        int gate_file = 0, up_file = 0, down_file = 0;   // the shard of each (split GGUF)
        size_t eg_stride = 0, eu_stride = 0, ed_stride = 0;
        bool ssd_backed = false;
        // Expert memory tiers
        uint8_t* host_blobs = nullptr;
        size_t host_bytes = 0;
        bool host_pinned = false;
        bool host_registered = false;
        bool host_nohuge = false;
        int n_res = 0;
        size_t vram_pool_off = 0;
        hy4::k::FetchLayer fl;
        hy4::CpuExperts cex;
        std::vector<int> h_slot;
        std::vector<int> h_owner;
        std::vector<float> h_score;
        std::vector<uint32_t> pf_counts;
        std::vector<uint32_t> profile_counts;  // cumulative routing counts across session
        bool cpu_cache_ready = false;
        int promo_pending_e = -1;
        int promo_pending_v = -1;
        int promo_pending_sl = -1;
        cudaEvent_t ev_promo = nullptr;
        // Lookahead prefetch transition table: next_top[prev_e] -> predicted expert in next layer
        std::vector<int16_t> la_next;
        int last_top_e = -1;
    };

    struct Gpu {
        int dev = 0;
        int numa_node = -1;
        int l0 = 0, l1 = 0;
        cudaStream_t s = nullptr;
        cudaStream_t s_copy = nullptr;
        cudaStream_t s_d2h = nullptr;
        cublasHandle_t blas = nullptr;
        cudaEvent_t ev_xfer = nullptr;
        cudaEvent_t ev_route = nullptr;
        cudaEvent_t ev_d2h = nullptr;
        cudaEvent_t ev_norm = nullptr;
        cudaEvent_t ev_sh = nullptr;
        cudaEvent_t ev_copy[2] = {nullptr, nullptr};
        cudaEvent_t ev_comp[2] = {nullptr, nullptr};
        uint8_t* vram_pool = nullptr;
        size_t vram_pool_bytes = 0;
        std::vector<void*> dev_allocs;
        std::vector<void*> host_allocs;
        // decode scratch
        float *x = nullptr, *cur = nullptr, *out = nullptr, *hn = nullptr;
        int8_t *xq = nullptr, *xq2 = nullptr;
        float *xd = nullptr, *xd2 = nullptr;
        float *q = nullptr, *k = nullptr, *v = nullptr, *qc = nullptr, *kc = nullptr, *vc = nullptr;
        float *f = nullptr, *g2 = nullptr, *g1 = nullptr, *beta = nullptr, *raw_o = nullptr;
        float *qm = nullptr, *kvall = nullptr, *q_abs = nullptr, *ctxo = nullptr, *o128 = nullptr, *gate = nullptr;
        int8_t *q8n = nullptr, *q8c = nullptr;
        float *d8n = nullptr, *d8c = nullptr;
        float* mla_part = nullptr;
        float *gu = nullptr, *eh = nullptr, *shact = nullptr, *part = nullptr, *shout = nullptr, *moe = nullptr;
        float *rlog = nullptr, *d_w = nullptr, *d_cpu_out = nullptr, *logits = nullptr;
        int *d_ids = nullptr, *d_step = nullptr, *d_iota = nullptr;
        int* d_route_err = nullptr;
        int* h_route_err = nullptr;
        unsigned long long* h_stats = nullptr;
        int* d_mrope_pos = nullptr;
        int mrope_cap = 0;
        unsigned long long* d_plan = nullptr;
        int8_t *ehq = nullptr, *shq = nullptr;
        float *ehd = nullptr, *shd = nullptr;
        float* d_top = nullptr;
        float* d_arg_val = nullptr;
        int* d_arg_idx = nullptr;
        int* d_arg_out = nullptr;
        int* h_arg_out = nullptr;
        k::RoutePack* d_route_pack = nullptr;
        k::RoutePack* h_route_pack = nullptr;
        k::RoutePackWin* d_route_win = nullptr;
        k::RoutePackWin* h_route_win = nullptr;
        int* h_neg1 = nullptr;
        int* d_pos_win = nullptr;
        int* h_pos_win = nullptr;
        int* d_tok_win = nullptr;
        int* h_tok_win = nullptr;
        int8_t* x2q = nullptr;
        float* x2d = nullptr;
        unsigned long long* grp_ptr = nullptr;
        int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
        int* d_arg_rows = nullptr;
        int* h_arg_rows = nullptr;
        float* d_top_pair = nullptr;
        float* h_top_pair = nullptr;
        hy4::k::FetchScratch fxs;
        int* h_step = nullptr;
        int* h_ids = nullptr;
        float* h_w = nullptr;
        int* h_who = nullptr;
        float* h_cur = nullptr;
        float* h_cpu_out = nullptr;
        float* h_top = nullptr;
        float* h_xfer = nullptr;
        // prefill scratch
        int pf_cap = 0;
        int* pf_tok_ids = nullptr;
        float *pf_x = nullptr, *pf_pipe_x = nullptr, *pf_cur = nullptr, *pf_out = nullptr, *pf_moe = nullptr;
        uint16_t *pf_x16 = nullptr, *pf_w16 = nullptr;
        float *pf_tmp = nullptr;
        size_t pf_tmp_bytes = 0;
        int *pf_ids = nullptr, *pf_bounds = nullptr, *pf_tok = nullptr, *pf_row_of = nullptr;
        int *pf_pmap[2] = {nullptr, nullptr};
        float *pf_w = nullptr;
        uint8_t* pf_stage[2] = {nullptr, nullptr};
        int* h_pf_ids = nullptr;
        float* h_pf_w = nullptr;
        int h_pf_cap = 0;
        int* h_pf_pmap[kPfPmapRing] = {};
        cudaEvent_t ev_pmap[kPfPmapRing] = {};
        uint64_t pf_pmap_seq = 0;
        int* d_max_rows = nullptr;
        int* h_max_rows = nullptr;
        uint8_t* h_pf_stage[2] = {nullptr, nullptr};
        // CUDA graphs
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        bool graph_ready = false;
        cudaGraph_t wgraph[kMaxWin] = {};
        cudaGraphExec_t wgraph_exec[kMaxWin] = {};
        bool wgraph_ready[kMaxWin] = {};
    };

    struct MtpWeights {
        const float* enorm = nullptr;
        const float* hnorm = nullptr;
        const float* out_norm = nullptr;
        Mat eh_proj;
        Layer layer;
        float* emb_buf = nullptr;
        int8_t* eh_q = nullptr;
        float* eh_d = nullptr;
        float* hring = nullptr;
        float* h_tmp = nullptr;
        static constexpr int kRing = 256;
    };

    struct VlendSaved {
        Layer* l = nullptr;
        std::vector<int> owners;
        int sl_first = 0;
    };

    Config c_;
    Options o_;
    std::string gguf_path_;
    OffloadMode active_offload_ = OffloadMode::kGpu;
    std::string info_;
    std::string start_plan_line_;
    std::string cal_key_;
    std::string cal_file_;
    sys::CpuTopology cpu_topo_;
    sys::RestoreExpect restore_expect_;
    std::unique_ptr<hy4::Lane> lane_;
    std::vector<std::unique_ptr<sys::DirectGgufReader>> ssd_readers_;   // one a shard
    std::vector<Gpu> gpus_;
    std::vector<std::vector<bool>> p2p_mat_;
    std::vector<Layer> layers_;
    Mat emb_, out_w_;
    const float* out_norm_ = nullptr;
    size_t max_blob_ = 0;
    int prefill_block_ = 0;
    bool mtp_loaded_ = false;
    MtpWeights mtp_;
    bool any_host_unpinned_ = false;
    std::vector<std::string> vocab_tokens_;
    unsigned long long cpu_lane_picks_ = 0;
    unsigned long long cpu_admitted_ = 0;
    mutable unsigned long long ram_fetch_picks_ = 0;
    mutable unsigned long long ssd_fetch_picks_ = 0;
    unsigned long long graph_captures_ = 0;
    unsigned long long launches_per_tok_ = 0;
    int tok_promo_budget_ = 0;
    int promo_rr_ = 0;
    double last_warm_s_ = 0.0;
    double vram_res_frac_ = 1.0;
    double pcie_share_ = 0.0;
    double total_vram_gb_ = 0.0;
    double total_ram_gb_ = 0.0;
    double total_ssd_gb_ = 0.0;
    double total_kv_gb_ = 0.0;
    size_t total_experts_ = 0;
    size_t total_vram_experts_ = 0;
    size_t total_ram_experts_ = 0;
    size_t total_ssd_experts_ = 0;
    bool lookahead_on_ = false;
    double base_cpu_ms_per_exp_ = 0.0;
    double cpu_ema_ms_per_exp_ = 0.0;
    int busy_pc_shift_ = 0;
    int mtp_done_ = 0;
    std::vector<int> mtp_tokens_;
    std::vector<float> mtp_pf_h_host_;  // prompt final hidden states for full-prompt MTP catch-up
    int mtp_pf_pos0_ = 0;
    std::unique_ptr<VisionTower> vision_;
    std::vector<Image> images_;
    int rope_delta_ = 0;
    bool vlent_ = false;
    std::vector<VlendSaved> vlend_saved_;
    std::thread warmup_thread_;
    std::atomic<bool> warmup_done_{true};
    std::string last_kernel_;

    void* dev_alloc(Gpu& g, size_t bytes);
    void* host_alloc(Gpu& g, size_t bytes);
    void set_dev(int dev);
    void check_token_id(int tok, const char* where) const;
    void check_route_and_fault(const char* stage);
    bool read_expert_blob(const Layer& l, int e, uint8_t* dst) const;
    void run_cpu_entries(Layer& l, const std::vector<hy4::CpuEntry>& entries, const float* x, int T, float* out);
    void peer_setup_verified();
    void start_warmup_async();
    void calibrate_split_and_threads();
    void load_layer_weights(Layer& l, int il, Gpu& g, const GgufSet& gf,
                            const std::map<std::string, const maya::TensorInfo*>& want);
    void transfer_x(int g_from, int g_to, int T, bool prefill_buf);
    void transfer_buf(int g_from, int g_to, const float* p_src, float* p_dst, size_t floats);
    void transfer_bytes(int g_from, int g_to, const void* p_src, void* p_dst, size_t bytes);

    void decode_segment(int gi, bool capture_safe);
    void kda_decode_win(Gpu& gp, Layer& l, int il, int W, bool add_to_x);
    void mla_decode_win(Gpu& gp, Layer& l, int il, int W, bool add_to_x);
    void ffn_dense_decode_win(Gpu& gp, Layer& l, int il, int W, const float* next_norm);
    void ffn_moe_decode_win(Gpu& gp, Layer& l, int il, int W, const float* next_norm);
    void decode_segment_win(int gi, int W, bool capture_safe);
    void kda_decode(Gpu& gp, Layer& l, int il, bool seam, bool add_to_x = true);
    void mla_decode(Gpu& gp, Layer& l, int il, bool seam, bool add_to_x = true);
    void ffn_dense_decode(Gpu& gp, Layer& l, int il, bool seam, const float* next_norm = nullptr);
    void ffn_moe_decode(Gpu& gp, Layer& l, int il, bool seam, const float* next_norm = nullptr);
    void mtp_prefill_prompt(const int* tokens, int n, int pos0);

    void gemm_mat(Gpu& gp, const Mat& m, const uint16_t* a16, int T, float* out_f32);
    struct MacroBufs {
        float *x = nullptr, *cur = nullptr, *moe = nullptr, *w = nullptr;
        float *pipe_x = nullptr;
        int *ids = nullptr, *tok = nullptr, *row_of = nullptr;
    };
    void kda_prefill(Gpu& gp, Layer& l, int T, float* x_io);
    void mla_prefill(Gpu& gp, Layer& l, int T, int pos0, float* x_io, const int* mrope_pos = nullptr, int mrope_stride = 0);
    void ffn_dense_prefill(Gpu& gp, Layer& l, int il, int T, float* x_io);
    void ffn_moe_prefill_macro(int gi, Layer& l, int il, int B, bool is_last_block, bool layer_borrowed,
                               const std::vector<MacroBufs>& mbufs);
    void warm_vram_cache();

    void dump_seam_f32(Gpu& gp, const std::string& name, const float* dev, size_t n);
    void dump_seam_i32(Gpu& gp, const std::string& name, const int* dev, size_t n);
};

}  // namespace ling
