// src/metal/glm_metal_impl.hpp - GlmMetal's private state, shared by the loader (glm_metal_load.cpp) and the
// graph (glm_metal.cpp).
#pragma once

#include "expert_pool.hpp"
#include "glm_metal.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace maya::metal {

struct TensorLoc {
    int fd = -1;
    uint64_t offset = 0;      // file offset of the tensor data
    ggml_tensor* meta = nullptr;
};

size_t free_memory();   // glm_metal_load.cpp: what this Mac could give the process now

// tensors read from GGUF files into GPU-mapped arenas: one buffer per 8 GiB, the weights by name
struct Weights {
    ggml_context* ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<std::pair<void*, size_t>> mem;
    std::unordered_map<std::string, ggml_tensor*> by_name;
    size_t bytes = 0;
    ~Weights();
    ggml_tensor* get(const std::string& name) const;   // nullptr if absent
    bool load(const std::vector<std::pair<std::string, TensorLoc>>& tensors, ggml_backend_dev_t dev, int io_threads,
              std::string& err);
};

// the DFlash2 drafter (glm_metal_dflash.cpp): its weights, hyperparameters and KV ring
struct DFlash {
    int n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 128, n_ff = 0, block = 8, taps = 2, group = 16;
    int rank = 0, top_k = 16, window = 2048, ring = 0;
    float eps = 1e-5f, freq_base = 10000.0f;
    int32_t mask_id = -1;
    std::vector<int> targets;            // trunk layers whose input streams' mean the drafter reads
    gguf_context* gguf = nullptr;
    ggml_context* meta = nullptr;
    int fd = -1;
    Weights w;
    ggml_context* sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    std::vector<ggml_tensor*> ring_k, ring_v;   // per layer [head_dim, n_head_kv, ring] F16
    std::vector<int64_t> kv_pos;                // per ring slot: the position it holds, -1 none
    ~DFlash();
};

struct LayerState {             // what a layer keeps between tokens
    ggml_tensor* conv = nullptr;    // KDA: [d_conv-1, d_inner, 3] q/k/v conv history, oldest first
    ggml_tensor* ssm = nullptr;     // KDA: [S, S, H] recurrent state
    ggml_tensor* lat = nullptr;     // DSA: [kv_lora, n_ctx + 1] latent per cell (+ the zero "null" cell)
    ggml_tensor* ikg = nullptr;     // DSA: [2*idx_key, n_ctx] indexer key | pool gate per cell
    ggml_tensor* pooled = nullptr;  // DSA: [idx_key, n_ctx/kpool] pooled key per completed pool
    ggml_tensor* conv_snap = nullptr;   // KDA: snapshot copies
    ggml_tensor* ssm_snap = nullptr;
    ggml_tensor* conv_win = nullptr;    // KDA, verify window: [dc + W, d_inner, 3] the conv inputs of the window
    ggml_tensor* ssm_win = nullptr;     // KDA, verify window: [S*S*H, W] the state after each row, newest first
};

struct GlmMetal::Impl {
    GlmOptions o;
    GlmGeometry g;
    GlmStats st;

    // files
    std::vector<gguf_context*> ggufs;
    std::vector<ggml_context*> metas;
    std::vector<int> fds;
    std::unordered_map<std::string, TensorLoc> loc;

    // backends
    ggml_backend_dev_t gpu_dev = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // dense weights
    Weights wts;
    TensorLoc embd;                 // read row by row on the CPU (as the CUDA engine does)
    std::vector<int64_t> img_pos;   // absolute positions whose embedding is an image row
    std::vector<float> img_rows;

    // state + carries between the graphs of one slice
    ggml_context* sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    std::vector<LayerState> ls;
    ggml_tensor* pool_tab = nullptr;    // [kpool, n_ctx/kpool] I32: pool p's member cells
    ggml_tensor* c_res = nullptr;       // [n_embd, hc, n_ubatch] residual streams entering the FFN half
    ggml_tensor* c_post = nullptr;      // [hc, n_ubatch]
    ggml_tensor* c_comb = nullptr;      // [hc, hc, n_ubatch]
    ggml_tensor* c_x = nullptr;         // [n_embd, n_ubatch] ffn_norm output
    ggml_tensor* c_w = nullptr;         // [1, n_expert_used, n_ubatch] routing weights
    ggml_tensor* c_shexp = nullptr;     // [n_embd, n_ubatch] shared expert output
    ggml_tensor* hid = nullptr;         // [n_embd, max(W, 1)] the trunk's final hidden state of the last forward
    int hid_rows = 0;
    std::vector<float> hid_host;        // the same on the host, read after every forward
    int64_t hx_pos = -1;                // the position whose trunk hidden state is hid_host column hx_col (-1: none)
    int hx_col = 0;
    std::vector<float> hid_snap;        // hx's row at snapshot_save()
    int64_t hx_snap = -1;
    std::vector<float> mtp_hx;          // the last MTP step's own hidden state (the next chained step's input)
    ggml_tensor* c_mres = nullptr;      // [n_embd] the MTP step's residual across its expert cut
    ggml_tensor* slot_tab = nullptr;    // [n_expert, held] I32: each expert's pool slot, latched once the pool holds them all
    bool slot_tab_live = false;         // slot_tab holds the pool's final slot map (no slot can change hands after)
    int win_T = 0;                      // the open window's rows (0: none)
    int64_t win_p0 = 0;

    std::vector<uint8_t> arena;         // the graphs' ggml context (metadata only), kept between forwards
    ExpertPool pool;
    // DFlash2: the drafter, the trunk features it reads (the streams' mean entering its target layers, every row of
    // the last slice: [n_embd * targets, n_ubatch]) and which layer fills which slot (-1: none)
    std::unique_ptr<DFlash> df;
    ggml_tensor* feat = nullptr;
    std::vector<int> feat_slot;
    std::vector<std::pair<int, int>> ranked;   // the profile the pool started from (layer, expert), most used first
    int64_t n_ctx = 0;
    int64_t pos = 0;
    int64_t snap_pos = -1;
    // restore-point slots (snapshot_save_slot & friends): one entry per slot id, grown on first save.
    // state = every KDA layer's conv then ssm tensor bytes in layer order (empty when the model has none);
    // hid = hx's row at the save, as hid_snap.  pos -1: the slot holds nothing.
    struct SnapSlot {
        int64_t pos = -1;
        int64_t hx_pos = -1;
        std::vector<float> hid;
        std::vector<uint8_t> state;
    };
    std::vector<SnapSlot> snap_slots;

    // Decode plan cache: the recurring T=1 trunk slice and the MTP part-0 graph, kept built and allocated
    // on schedulers of their own (the two alternate every round, so a shared scheduler would re-plan on
    // every alternation), with their uploads re-derived from the position on every hit instead of rebuilt.
    // A plan graph is built with its cache writes as position-indexed ggml_set_rows (the row indices are
    // uploads), so the one graph is exact for every position its signature covers; the signature therefore
    // carries everything structural, and anything that can change structure behind it - a reset, a snapshot
    // restore, the slot-table latch - frees the plans.  MAYA_GLM_PLAN_CACHE=0 disables the cache entirely.
    struct PlanSig {
        int branch = 0;                 // 1: born-resident slot ids, 2: latched table ids (0 never cached)
        int64_t n_pool = 0;             // the DSA layers' complete-pool count at the graph's position
        bool pool_write = false;        // the graph's own cache write completes a pool
        int fill = 0;                   // trunk: the NextN fill the head performs (1: from the row before)
        bool fill_pool_write = false;
        bool operator==(const PlanSig& o) const {
            return branch == o.branch && n_pool == o.n_pool && pool_write == o.pool_write && fill == o.fill &&
                   fill_pool_write == o.fill_pool_write;
        }
    };
    struct DecodePlan {
        struct Up {
            ggml_tensor* t = nullptr;
            std::vector<uint8_t> data;
            std::function<void(std::vector<uint8_t>&)> regen;
        };
        ggml_context* ctx = nullptr;         // keep-alive: the graph's tensors live here
        ggml_cgraph* gf = nullptr;
        ggml_backend_sched_t sched = nullptr;
        std::vector<uint8_t> arena;          // the context's memory, owned with it
        std::vector<Up> uploads;
        ggml_tensor* out = nullptr;          // trunk: the logits; MTP: the argmax
        ggml_tensor* aux = nullptr;          // MTP: the step's hidden state
        int hid_rows = 0;                    // trunk: hidden rows the head wrote at build
        PlanSig sig;
        bool valid = false;
    };
    DecodePlan trunk_plan, mtp_plan;
    bool plan_cache_on = true;               // MAYA_GLM_PLAN_CACHE, read at load()
    const float* pc_emb = nullptr;           // the running slice's embeddings (plan upload regen source)
    const float* pc_e = nullptr;             // MTP: the step's token embedding
    const float* pc_h = nullptr;             // MTP: the step's hidden-state input
    int64_t pc_p0 = 0;                       // the running graph's first position (regen source)
    void plans_invalidate();
    void plan_free(DecodePlan& p);
    ggml_backend_sched_t make_sched();
    PlanSig trunk_sig(int branch, int64_t at) const;
    bool plan_hit_trunk(std::vector<float>& logits, std::string& err);

    ~Impl();
    bool open_files(const std::string& path, std::string& err);
    bool read_geometry(std::string& err);
    bool init_backends(std::string& err);
    bool load_weights(std::string& err);
    bool init_experts(std::string& err);
    bool init_state(std::string& err);
    ggml_tensor* w(const std::string& name) const;     // nullptr if absent
    ggml_tensor* wl(int il, const char* suffix) const; // blk.<il>.<suffix>
    bool embed(const int32_t* tok, int n, std::vector<float>& out, std::string& err) const;   // rows pos .. pos+n-1
    bool embed_raw(const int32_t* tok, int n, std::vector<float>& out, std::string& err) const;   // no image rows
    size_t graph_size(int T) const;   // tensors a T-token slice's graph may hold
    // once the pool holds every expert its slot map can never change again: copy it into slot_tab so the
    // graphs look slots up on the GPU (the born-resident path's twin for a pool that started partial).
    // False while any expert is still out; a no-op once latched.
    bool arm_slot_tab();
    bool run_slice(const int32_t* tok, int n, std::vector<float>& logits, std::string& err, bool window = false);
    bool load_dflash(std::string& err);
    bool dflash_inject(int64_t p0, int n, std::string& err);   // rows 0..n-1 of feat = positions p0..p0+n-1
    void dflash_forget(int64_t from);                         // positions >= from leave the drafter's ring
    bool compute(ggml_cgraph* gf, const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
                 std::string& err);
};

}  // namespace maya::metal
