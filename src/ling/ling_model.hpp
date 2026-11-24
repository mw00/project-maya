// src/ling/ling_model.hpp - Ling-3.2-air (llama.cpp bailingmoe3) inference engine.
// Supports 1..N GPUs (layer split), adaptive per-expert VRAM cache, GPU PCIe fetch (CUDA-graphable),
// optional AVX2 CPU lane for offloaded experts, double-buffered batched prefill (KDA + MLA WMMA + grouped MoE),
// FP16 / INT8 MLA latent KV cache up to 262K context, and optional MTP block (blk.42).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "hy4_kernels.cuh"
#include "ling_kernels.cuh"
#include "ling_vision.hpp"
#include "hy4_lane.hpp"

namespace maya {
class GgufFile;
struct TensorInfo;
}

namespace ling {

enum class OffloadMode {
    kAuto = 0,   // gpu (CUDA graph) when host experts fit pinned RAM, else cpu lane
    kGpu  = 1,   // GPU adaptive VRAM cache + GPU fetch_copy (CUDA graph)
    kMix  = 2,   // GPU adaptive VRAM cache + up to mix_k_gpu misses on GPU + CPU lane for rest
    kCpu  = 3,   // GPU VRAM hits + CPU lane for all misses (no pinning required)
};

struct Config {
    int n_layers = 0;        // trunk blocks (block_count - nextn)
    int n_layer_all = 0;     // including MTP block
    int n_nextn = 0;
    int n_embd = 0, n_head = 0, vocab = 0;
    int eos_id = 156895;
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
    int prefill_block = 16384;       // macro-block size for MoE expert reuse (borrows tail of VRAM pool when > prefill_chunk)
    bool ep_prefill = false;         // opt-in multi-GPU EP prefill relay (default off; stage pipeline wins on multi-block)
    double expert_vram_gib = -1.0;   // < 0: free VRAM minus headroom per GPU
    double pin_ram_gib = 54.0;       // max pinned host RAM for offloaded experts
    std::vector<int> gpus = {0};     // CUDA device ordinals (layer split when size > 1)
    OffloadMode offload = OffloadMode::kAuto;
    int mix_k_gpu = 1;               // in kMix mode, max misses fetched on GPU per layer
    int cpu_threads = -1;            // hy4::Lane threads (-1: physical cores - 1)
    bool kv_q8 = false;              // INT8 MLA latent cache (544 B/cell vs 1024 B FP16)
    bool use_graph = true;           // CUDA graph for single-token decode (in kGpu / full-VRAM mode)
    bool cache_warm = true;          // warm VRAM expert cache from prompt routing frequencies
    bool load_mtp = false;           // also load blk.<n_layers> MTP block
    std::string seam_dir;
    std::string mmproj_path;            // optional per-layer seam dump dir (parity testing)
};

struct TierStats {
    unsigned long long picks = 0;
    unsigned long long vram_hits = 0;
    unsigned long long copied = 0;
    unsigned long long zero_copy = 0;
    unsigned long long cpu_lane = 0;
    unsigned long long admitted = 0;
};

class Model {
public:
    Model(const std::string& path, const Options& opt);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& cfg() const { return c_; }
    const std::string& info() const { return info_; }
    int output_dev() const { return gpus_.back().dev; }
    cudaStream_t output_stream() const { return gpus_.back().s; }
    bool has_mtp() const { return mtp_loaded_; }
    const float* last_h_nextn() const { return gpus_.back().hn; }
    double last_warm_s() const { return last_warm_s_; }

    struct Image {
        int64_t start = 0;   // prompt index of first <|image_pad|>
        int width = 0;
        int height = 0;
        std::vector<float> rows;  // [tokens * n_embd] F32
    };
    const VisionGeo* vision() const { return (vision_ && vision_->loaded()) ? &vision_->geo() : nullptr; }
    const char* vision_room() const;
    size_t vision_room_bytes() const { return (vision_ && vision_->loaded()) ? vision_->max_room_bytes() : 0; }
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& out_rows, std::string& err);
    void set_images(std::vector<Image> imgs);
    int rope_delta() const { return rope_delta_; }
    void set_rope_delta(int d) { rope_delta_ = d; }

    // Reset KDA states and prompt frequency counters for a fresh sequence.
    void reset();

    // One token at position pos -> logits (vocab F32 on output_dev(), valid until next step/prefill).
    const float* step(int token, int pos);

    // Batched prefill for tokens[0..n-1] starting at pos0.
    // Returns logits of the final token (vocab F32 on output_dev()).
    // If all_logits_host != nullptr, appends all n positions' F32 logits (n * vocab floats).
    const float* prefill(const int* tokens, int n, int pos0, std::vector<float>* all_logits_host = nullptr);

    // Run MTP block (blk.<n_layers>) on trunk h_nextn (E F32 on output_dev()) paired with next_token at pos.
    // Returns MTP logits (vocab F32 on output_dev()).
    const float* mtp_step(const float* h_nextn_dev, int next_token, int pos);

    // Batched token-window verify (1 <= W <= kMaxWin): returns [W][vocab] logits on output_dev().
    static constexpr int kMaxWin = 4;
    const float* step_window(const int* tokens, int W, int pos0);
    void commit_window(int n_accept, int W, int pos0, const int* tokens);

    // Batched greedy argmax for W rows of logits [W][vocab] -> fills out_ids[0..W-1].
    void argmax_rows(const float* logits_dev, int W, int* out_ids);

    // Autoregressively draft up to max_k tokens via blk.42 MTP after accepted (base_tok, base_pos).
    void mtp_draft(int base_tok, int base_pos, int max_k, float floor, std::vector<int>& drafts,
                   std::vector<float>& confs, float floor0 = 0.0f);

    // Fraction of trunk MoE experts resident in VRAM (0.0 .. 1.0).
    double vram_res_frac() const { return vram_res_frac_; }

    // Greedy argmax on output_dev() logits -> token id on host.
    int argmax_logits(const float* logits_dev);

    // Read cumulative expert-tier statistics across all GPUs.
    TierStats stats() const;

    // Decode token IDs using GGUF tokenizer.ggml.tokens (GPT-2 byte-level BPE).
    std::string detokenize(const std::vector<int>& ids) const;
    std::string detokenize_one(int id) const;

private:
    static constexpr int kPfBatch = 64;   // experts per double-buffered prefill staging half (2 x 64 = 128 slots)
    static constexpr int kPfPmapRing = 32; // host pmap slots so pinned DMA never stalls host

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
        const float* conv_q = nullptr;
        const float* conv_k = nullptr;
        const float* conv_v = nullptr;
        const float* dt_bias = nullptr;
        const float* ssm_a = nullptr;       // positive exp(A_log)
        const float* ssm_a_neg = nullptr;   // -exp(A_log) for glmb::kda_rec_split
        const float* ssm_norm = nullptr;
        float* conv_state = nullptr;        // [3][d_inner][d_conv - 1]
        float* state = nullptr;
        float *conv_bak = nullptr, *state_tmp = nullptr, *w_kda_buf = nullptr;             // [n_head][kda_head][kda_head]
        // MLA
        Mat wqm, wkva, wkb, wvb, wgate, wom;
        const float* kv_a_norm = nullptr;
        void* lat = nullptr;                // FP16 [ctx][kv_lora] or INT8 [ctx][544 B]
        uint16_t* k_pe = nullptr;           // FP16 [ctx][n_rot]
        // FFN / MoE
        Mat fg, fu, fd;
        const void* router_w = nullptr;
        int router_type = 0;                // 0: F32, 30: BF16
        const float* rb_dev = nullptr;      // exp_probs_b on GPU
        int etg = 0, etu = 0, etd = 0;
        size_t eg_ld = 0, eu_ld = 0, ed_ld = 0;
        size_t tr[3] = {0, 0, 0};           // offsets inside packed blob
        size_t blob = 0;                    // 256-aligned bytes per packed expert [gate|up|down]
        int n_res = 0;                      // experts resident in VRAM (0..n_expert)
        size_t vram_pool_off = 0;           // byte offset inside Gpu::vram_pool
        bool host_pinned = false;           // GPU-accessible pinned RAM (cudaHostRegister or cudaMallocHost)
        bool host_registered = false;       // allocated via ram_alloc + cudaHostRegister
        uint8_t* host_blobs = nullptr;      // [n_expert][blob] packed experts in RAM (null if n_res == n_expert)
        size_t host_bytes = 0;
        hy4::k::FetchLayer fl;              // GPU adaptive cache metadata
        hy4::CpuExperts cex;                // CPU lane view of host_blobs
        std::vector<int> h_slot;            // host mirror of fl.slot for prefill
        std::vector<int> h_owner;           // host mirror of fl.owner for async warm
        std::vector<float> h_score;         // host mirror of fl.score for async warm
        std::vector<uint32_t> pf_counts;    // prompt pick counts for cache warm
        // Async background cache promotion in kCpu decode mode
        bool cpu_cache_ready = false;
        int promo_pending_e = -1;
        int promo_pending_v = -1;
        int promo_pending_sl = -1;
        cudaEvent_t ev_promo = nullptr;
    };

    struct Gpu {
        int dev = 0;
        int l0 = 0, l1 = 0;                 // assigned trunk layers [l0, l1)
        cudaStream_t s = nullptr;
        cudaStream_t s_copy = nullptr;      // dedicated DMA / shared-expert overlap stream
        cudaStream_t s_d2h = nullptr;       // dedicated tiny D2H stream for kCpu routing readback
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
        // pinned host buffers
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
        float *pf_x = nullptr, *pf_cur = nullptr, *pf_out = nullptr, *pf_moe = nullptr;
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
        // CUDA graphs for single-token and windowed verify decode (index W - 1 for W = 1..kMaxWin)
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

    Config c_;
    Options o_;
    OffloadMode active_offload_ = OffloadMode::kGpu;
    std::string info_;
    std::unique_ptr<hy4::Lane> lane_;
    std::vector<Gpu> gpus_;
    std::vector<std::vector<bool>> p2p_mat_;
    std::vector<Layer> layers_;
    Mat emb_, out_w_;
    const float* out_norm_ = nullptr;
    size_t max_blob_ = 0;
    bool mtp_loaded_ = false;
    MtpWeights mtp_;
    bool any_host_unpinned_ = false;
    std::vector<std::string> vocab_tokens_;
    unsigned long long cpu_lane_picks_ = 0;
    unsigned long long cpu_admitted_ = 0;
    int tok_promo_budget_ = 0;
    int promo_rr_ = 0;
    double last_warm_s_ = 0.0;
    double vram_res_frac_ = 1.0;
    int mtp_done_ = 0;
    std::vector<int> mtp_tokens_;
    std::unique_ptr<VisionTower> vision_;
    std::vector<Image> images_;
    int rope_delta_ = 0;

    void* dev_alloc(Gpu& g, size_t bytes);
    void* host_alloc(Gpu& g, size_t bytes);
    void set_dev(int dev);
    void load_layer_weights(Layer& l, int il, Gpu& g, const maya::GgufFile& gf,
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

    void gemm_mat(Gpu& gp, const Mat& m, const uint16_t* a16, int T, float* out_f32);
    struct MacroBufs {
        float *x = nullptr, *cur = nullptr, *moe = nullptr, *w = nullptr;
        float *pipe_x = nullptr;     // second ping-pong x buffer for multi-GPU stage pipelining
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
