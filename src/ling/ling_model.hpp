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
    std::string seam_dir;            // optional per-layer seam dump dir (parity testing)
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
        float* state = nullptr;             // [n_head][kda_head][kda_head]
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
    };

    struct Gpu {
        int dev = 0;
        int l0 = 0, l1 = 0;                 // assigned trunk layers [l0, l1)
        cudaStream_t s = nullptr;
        cudaStream_t s_copy = nullptr;      // dedicated DMA stream for overlapped prefill staging
        cublasHandle_t blas = nullptr;
        cudaEvent_t ev_xfer = nullptr;
        cudaEvent_t ev_route = nullptr;
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
        unsigned long long* d_plan = nullptr;
        int8_t *ehq = nullptr, *shq = nullptr;
        float *ehd = nullptr, *shd = nullptr;
        float* d_top = nullptr;
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
        int* d_max_rows = nullptr;
        int* h_max_rows = nullptr;
        uint8_t* h_pf_stage[2] = {nullptr, nullptr};
        // CUDA graph for single-token decode
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        bool graph_ready = false;
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
    double last_warm_s_ = 0.0;

    void* dev_alloc(Gpu& g, size_t bytes);
    void* host_alloc(Gpu& g, size_t bytes);
    void set_dev(int dev);
    void load_layer_weights(Layer& l, int il, Gpu& g, const maya::GgufFile& gf,
                            const std::map<std::string, const maya::TensorInfo*>& want);
    void transfer_x(int g_from, int g_to, int T, bool prefill_buf);
    void transfer_buf(int g_from, int g_to, const float* p_src, float* p_dst, size_t floats);
    void transfer_bytes(int g_from, int g_to, const void* p_src, void* p_dst, size_t bytes);

    void decode_segment(int gi, bool capture_safe);
    void kda_decode(Gpu& gp, Layer& l, int il, bool seam);
    void mla_decode(Gpu& gp, Layer& l, int il, bool seam);
    void ffn_dense_decode(Gpu& gp, Layer& l, int il, bool seam);
    void ffn_moe_decode(Gpu& gp, Layer& l, int il, bool seam);

    void gemm_mat(Gpu& gp, const Mat& m, const uint16_t* a16, int T, float* out_f32);
    struct MacroBufs {
        float *x = nullptr, *cur = nullptr, *moe = nullptr, *w = nullptr;
        float *pipe_x = nullptr;     // second ping-pong x buffer for multi-GPU stage pipelining
        int *ids = nullptr, *tok = nullptr, *row_of = nullptr;
    };
    void kda_prefill(Gpu& gp, Layer& l, int T, float* x_io);
    void mla_prefill(Gpu& gp, Layer& l, int T, int pos0, float* x_io);
    void ffn_dense_prefill(Gpu& gp, Layer& l, int il, int T, float* x_io);
    void ffn_moe_prefill_macro(int gi, Layer& l, int il, int B, bool is_last_block, bool layer_borrowed,
                               const std::vector<MacroBufs>& mbufs);
    void warm_vram_cache();

    void dump_seam_f32(Gpu& gp, const std::string& name, const float* dev, size_t n);
    void dump_seam_i32(Gpu& gp, const std::string& name, const int* dev, size_t n);
};

}  // namespace ling
