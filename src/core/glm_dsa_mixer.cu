// src/core/glm_dsa_mixer.cu - the glm-dsa decode attention mixer on the fast path (glm-dsa B4.6, B11.1, B14.3-B14.5).
//
// Glm5Model::fast_dsa_glm is what Maya's fast_dsa hands a glm-dsa layer to (one dispatch line): the trunk's DSA layers
// from fast_layers_plain and the NextN block from fast_mtp (a full layer with its own indexer and key cache).  It
// runs glmdsa::mixer_layer (src/core/glm_dsa_mixer.hpp) with the GPU backend below: gf::mv through glmdsa::mv for the
// projections (native, BF16 or F32 weights) and the glmd kernels (dsa_prep, dsa_rope_q, idx_score, topk, mla2).
//
// glmdsa::mixer_setup (called by fast_dsa_arena) makes the per-part state: the rope tables (theta from the GGUF,
// NORM pairs), a ring of position words (pinned host -> device, one per token and part; the kernels read the position
// on the device, B14.4), the device cell count, mla2's scratch, BF16 copies of F32 attn_k_b / attn_v_b, and with
// STRATA_GLM_DSA_DENSE=1 (B14.3) a static iota of every cell.
//
// glm-dsa wave 5 (B10.1-B10.8 wiring): the projections and kernels are chosen by ONE dispatch policy
// (include/strata/kernels/glmd_decode_policy.hpp): the WMMA scorer above 16K cells and the radix top-k above 32K
// (STRATA_GLMD_IDX_WMMA / STRATA_GLMD_TOPK_MB: auto, 1 forces, 0 disables), the fused MLA with STRATA_GLMD_FUSED_MLA=1,
// the index-key dtype of the state layout (B10.8), W_kb / W_vb BF16 or native Q8_0 as the binder picked them (B10.7).
// Every position and count the kernels read is device-resident in a glmd::DevPos (B14.4), so with
// STRATA_GLM_DSA_GRAPHS=1 a part's whole layer loop is captured once and replayed per token (B10.1, graph_prepare /
// graph_run below, called by fast_layers_plain).
//
// Glm5Model::prefill_tokens_dsa (one hook line in prefill()) takes a glm-dsa prompt the batched path does not
// (glmpfs::mtp_token_fill: some part without batched state, or fewer than STRATA_GLM_PREFILL_MIN tokens) when the model
// carries the NextN block: token by token, and the block at every prompt position with the next prompt token (B11.1),
// so its latent, k_pe and index-key caches cover the prompt as the batched path's mtp_fill fills them.  Any other
// prompt it leaves to prefill() (false, err empty).
#include "strata/core/glm_env.hpp"
#include "glm_dsa_fast.hpp"
#include "glm_dsa_graphs.hpp"
#include "glm_cb_dump.hpp"
#include "glm_dsa_mixer.hpp"
#include "glm_dsa_prefill_stream.hpp"
#include "glm_fast_state.hpp"

#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_forward.hpp"
#include "strata/kernels/glmd_graph.hpp"
#include "strata/kernels/glmd_mixer.hpp"
#include "strata/kernels/glmd_random_route.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include "../kernels/cuda/glmd_rope.cuh"

#include <algorithm>
#include <chrono>
#include <functional>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

namespace glmdsa {

constexpr int kPosRing = 64;   // DevPos records per part (a token takes one per part, the NextN block another)

struct GlmDsaFast::Mixer {
    int dev = 0;
    MixerDims d;                   // d.pol: the dispatch policy (idx_fp16 is refreshed from the layout per call)
    bool dense = false;
    void* mem = nullptr;           // one device block: rope tables, DevPos ring + graph DevPos, mla work, topk_mb work,
                                   // the dense iota
    glmd::DevPos* pos_ring = nullptr;   // kPosRing records (B14.4: the position and the cell counts on the device)
    glmd::DevPos* pos_graph = nullptr;  // the captured part's record (written by graph_prepare outside the capture)
    int pos_next = 0;
    const glmd::DevPos* p_cur = nullptr;   // the current token's record
    int64_t p_last = -1;
    float* inv_freq = nullptr;
    float* idx_inv_freq = nullptr;
    int* cells_dense = nullptr;
    int* slot_cells = nullptr;     // L1-6 KV streaming: a selection in slot rows (top_k)
    void* work = nullptr;
    size_t work_bytes = 0;
    void* topk_work = nullptr;     // glmd::topk_mb's (zeroed once; the kernel leaves it zeroed)
    std::vector<const uint16_t*> wk_b, wv_b;   // per layer: BF16, or Q8_0 blocks (w_type 8)
    std::vector<int> w_type;                   // per layer: glmd::kWTypeBF16 or kWTypeQ8_0 (B10.7)
    std::vector<void*> owned;                  // BF16 copies made from F32 (or a lone Q8_0) weights
    SelState st;
    // B10.1: the per-part decode graph
    DsaGraphs graphs;
    bool graph_live = false;       // fast_dsa_glm is running inside graph_run's body (capture)
    int64_t graph_n_cap = 0;
    int graph_last_full = -1;      // the part's last full layer (the host selection record after a replay)
    uint64_t graph_eager = 0;      // part-tokens run without a graph although STRATA_GLM_DSA_GRAPHS=1 asked for one
    const char* graph_off_why = nullptr;   // ... the host-side reason (the last one met, for the teardown line)
    const char* graph_off_last = nullptr;  // the previous part-token's reason (null: its host side allowed a graph)
    // L1-4: a verify window's rows (the one-GPU MTP decode, mixer_rows): per-row buffers, selections and position
    // records, allocated at the first window (win_rows_begin)
    struct Win {
        void* mem = nullptr;
        int cap = 0;
        MixerBufs b;                   // row 0's buffers (x, xq, mixer: the window's, set per call)
        RowStrides rs;
        std::vector<int*> cells;
        std::vector<int*> n_sel;
        std::vector<const int*> p;
        std::vector<SelState> st;
        glmd::DevPos* pos = nullptr;   // cap records: row t's position p0 + t
    } win;
    ~Mixer() {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != dev) cudaSetDevice(dev);
        if (win.mem) cudaFree(win.mem);
        if (graphs.captures || graphs.replays || graph_eager)
            std::fprintf(stderr, "glm-dsa decode graphs: CUDA%d %llu captures, %llu replays, %llu invalidations, %llu "
                                 "evictions, %llu part-tokens without a graph%s%s\n", dev,
                         (unsigned long long) graphs.captures, (unsigned long long) graphs.replays,
                         (unsigned long long) graphs.invalidations, (unsigned long long) graphs.evictions,
                         (unsigned long long) graph_eager, graph_off_why ? ": " : "", graph_off_why ? graph_off_why : "");
        graphs.clear();            // before the buffers it replays are freed
        for (void* p : owned) cudaFree(p);
        if (mem) cudaFree(mem);
        if (cur != dev) cudaSetDevice(cur);
    }
};

namespace {

// the GPU backend of mixer_layer: the dispatch policy picks each kernel (glmd_decode_policy.hpp)
struct GpuOps {
    GlmDsaFast* X = nullptr;
    cudaStream_t s = nullptr;
    glmd::DecodePolicy pol;
    int64_t p = 0;                 // the token's position (the policy's n = p + 1)
    void* topk_work = nullptr;
    const WSlot* slot(int il, Proj w) const {
        const GlmDsaLayerW& W = X->L[(size_t) il];
        switch (w) {
        case Proj::QA: return &W.q_a;
        case Proj::KVA: return &W.kv_a;
        case Proj::IdxK: return &W.idx_k;
        case Proj::IdxProj: return &W.idx_proj;
        case Proj::QB: return &W.q_b;
        case Proj::IdxQB: return &W.idx_q_b;
        case Proj::Out: return &W.out;
        }
        return nullptr;
    }
    bool gemv(int il, const Gemv* j, int n) {
        gf::MvJob m[gf::kMaxMvJobs];
        for (int i = 0; i < n; ++i) {
            const WSlot* w = slot(il, j[i].w);
            const void* wq = w == nullptr ? nullptr : (w->q != nullptr ? w->q : (const void*) w->f32);
            if (wq == nullptr || w->n_in != j[i].n_in || w->n_out != j[i].n_out) {
                std::fprintf(stderr, "glm-dsa mixer: layer %d %s is not bound as %d -> %d\n", il, proj_name(j[i].w),
                             j[i].n_in, j[i].n_out);
                return false;
            }
            m[i] = {wq, j[i].xq, j[i].xf, j[i].y, nullptr, j[i].alpha, w->type, j[i].n_in, j[i].n_out};
            m[i].ncol = j[i].ncol;   // (L1-4 mixer_rows: a window's rows as columns)
            m[i].xq_cstride = j[i].xq_cs;
            m[i].xf_cstride = j[i].xf_cs;
            m[i].y_cstride = j[i].y_cs;
        }
        return mv(X, m, n, s);
    }
    bool prep(const glmd::DsaPrep& a) { return glmd::dsa_prep(a, s); }
    bool rope_q(const glmd::DsaRopeQ& a) { return glmd::dsa_rope_q(a, s); }
    bool score(const glmd::IdxScore& a) { return glmd::decode_score(pol, a, p, s); }
    bool topk(const float* score, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        return glmd::decode_topk(pol, score, ld, n_kv, pos, p, top_k, sel, topk_work, s);
    }
    bool mla2(const glmd::Mla2& a) { return glmd::decode_mla(pol, a, s); }
    void set_pos(int64_t q) { p = q; }   // (L1-4 mixer_rows: each row's own position)
};

size_t up256(size_t b) { return (b + 255u) & ~(size_t) 255u; }

}  // namespace

namespace {

// a host-side BF16 copy of a native Q8_0 W_kb / W_vb (the default pack's rounding: dequantize, then BF16 RN-even) -
// used only when the two weights of a layer disagree in type (glmd::mla2 / mla_fused take one w_type for both)
bool q8_0_to_bf16_device(const void* q, size_t n, void*& out, std::string& err) {
    std::vector<uint8_t> blocks(n / 32 * 34);
    std::vector<float> f(n);
    std::vector<uint16_t> h(n);
    if (cudaMemcpy(blocks.data(), q, blocks.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "glm-dsa mixer: reading a Q8_0 attn_k_b / attn_v_b back";
        return false;
    }
    glmd::vtwin::q8_0_dequantize(blocks.data(), (int) n, f.data());
    for (size_t i = 0; i < n; ++i) h[i] = glmd::twin::bf16_bits(f[i]);
    if (cudaMalloc(&out, n * 2) != cudaSuccess || cudaMemcpy(out, h.data(), n * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        err = "glm-dsa mixer: a BF16 copy of a Q8_0 attn_k_b / attn_v_b";
        return false;
    }
    return true;
}

const char* switch_name(int v) { return v == glmd::kSwitchAuto ? "auto" : v == glmd::kSwitchOn ? "on" : "off"; }

}  // namespace

bool mixer_setup(GlmDsaFast& X, const Glm5Geometry& g, int64_t max_ctx, std::string& err) {
    (void) glmd::random_routing_seed();   // B15.7: its bench-mode warning at load, not at the first token
    auto M = std::make_shared<GlmDsaFast::Mixer>();
    M->dev = X.dev;
    MixerDims& d = M->d;
    d.n_embd = g.n_embd;
    d.n_head = g.n_head;
    d.q_lora = g.q_lora;
    d.kv_lora = g.kv_lora;
    d.qk_nope = g.qk_nope;
    d.qk_rope = g.qk_rope;
    d.v_head = g.v_head;
    d.idx_heads = g.idx_heads;
    d.idx_key = g.idx_key;
    d.idx_rope = g.idx_rope;
    d.top_k = g.idx_top_k;
    d.max_ctx = max_ctx;
    d.eps = g.norm_eps;
    d.idx_ln_eps = g.idx_ln_eps;
    d.kq_scale = glmd::attn_scale(g.qk_head());
#if defined(STRATA_USE_HIP)
    d.pol = glmd::decode_policy_built(glmd::decode_policy_env());   // (no WMMA kernels in a HIP build)
#else
    d.pol = glmd::decode_policy_env();
#endif
    d.pol.idx_fp16 = glm_state_idx_fp16();   // the carve's rule; fast_dsa_glm re-reads the bound layout per call
    {   // A GPU without FP16 tensor cores (compute capability below 7.0: Pascal) has no WMMA scorer and no fused
        // attention (glmd_volta_idx.cu / glmd_volta_mla.cu compile them as traps there): the CUDA-core scorer and mla2
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, X.dev) != cudaSuccess) {
            cudaGetLastError();
            major = 99;   // (unknown: the policy stays as the switches set it)
        }
        if (major < 7 && (d.pol.idx_wmma != glmd::kSwitchOff || d.pol.fused_mla)) {
            d.pol.idx_wmma = glmd::kSwitchOff;
            d.pol.fused_mla = false;
            std::fprintf(stderr, "glm-dsa mixer: CUDA%d has no FP16 tensor cores (compute capability %d.x): the "
                                 "CUDA-core index scorer and mla2 run\n", X.dev, major);
        }
    }
    if (const char* s = getenv("STRATA_GLMD_KQ_SCALE"); s != nullptr && s[0] != '\0') {   // mutation runs only (the reference reads it too)
        d.kq_scale = (float) strata::glmenv::num_value("STRATA_GLMD_KQ_SCALE", s, d.kq_scale);
        std::fprintf(stderr, "glm-dsa mixer: CUDA%d attention scale overridden to %g (STRATA_GLMD_KQ_SCALE)\n", X.dev,
                     (double) d.kq_scale);
    }
    M->dense = strata::glmenv::env_flag("STRATA_GLM_DSA_DENSE", false);
    if (max_ctx <= 0 || max_ctx >= INT_MAX || d.top_k <= 0 || d.qk_rope % 2 || d.idx_rope % 2 ||
        d.idx_rope > d.idx_key || (d.qk_rope != 0 && d.qk_rope != 64) || d.kv_lora > 512 || d.kv_lora % 32 ||
        d.q_lora > 2048 || d.q_lora % 32 || d.idx_key > 512 || d.v_head % 32) {
        err = "glm-dsa mixer: geometry outside the decode kernels (qk_rope 0 or 64, kv_lora <= 512, q_lora <= 2048, "
              "multiples of 32, even rope dims, idx_key <= 512)";
        return false;
    }
    // the device block
    const size_t b_inv = up256((size_t) std::max(1, d.qk_rope / 2) * 4), b_idx = up256((size_t) std::max(1, d.idx_rope / 2) * 4);
    const size_t b_pos = up256((kPosRing + 1) * sizeof(glmd::DevPos));
    M->work_bytes = mixer_work_bytes(d, M->dense);
    const size_t b_work = up256(M->work_bytes);
    const size_t b_topk = d.pol.topk_mb != glmd::kSwitchOff ? up256(glmd::topk_mb_work_bytes(1)) : 0;
    const size_t b_iota = M->dense ? up256((size_t) max_ctx * sizeof(int)) : 0;
    const size_t b_slot = up256((size_t) std::max(1, d.top_k) * sizeof(int));   // L1-6
    const size_t total = b_inv + b_idx + b_pos + b_work + b_topk + b_iota + b_slot;
    cudaSetDevice(X.dev);
    if (cudaMalloc(&M->mem, total) != cudaSuccess) {
        cudaGetLastError();
        err = "glm-dsa mixer: its state (" + std::to_string(total) + " B) did not allocate";
        return false;
    }
    glmd::graph_generation_bump("glm-dsa mixer scratch");   // B15.6: the mla / top-k work and DevPos block moved
    uint8_t* at = (uint8_t*) M->mem;
    M->inv_freq = (float*) at;
    at += b_inv;
    M->idx_inv_freq = (float*) at;
    at += b_idx;
    M->pos_ring = (glmd::DevPos*) at;
    M->pos_graph = M->pos_ring + kPosRing;
    at += b_pos;
    M->work = at;
    at += b_work;
    M->topk_work = b_topk ? at : nullptr;
    at += b_topk;
    M->cells_dense = M->dense ? (int*) at : nullptr;
    at += b_iota;
    M->slot_cells = (int*) at;
    cudaMemset(M->mem, 0, total);   // (topk_mb's work starts zeroed: topk_mb_init's contract)
    std::vector<float> inv((size_t) std::max(d.qk_rope, d.idx_rope) / 2 + 1);
    glmd::rope_inv_freq(inv.data(), d.qk_rope, g.rope_theta);
    cudaMemcpy(M->inv_freq, inv.data(), (size_t) d.qk_rope / 2 * 4, cudaMemcpyHostToDevice);
    glmd::rope_inv_freq(inv.data(), d.idx_rope, g.rope_theta);
    cudaMemcpy(M->idx_inv_freq, inv.data(), (size_t) d.idx_rope / 2 * 4, cudaMemcpyHostToDevice);
    if (M->dense && !glmd::iota(M->cells_dense, (int) max_ctx, nullptr)) {
        err = "glm-dsa mixer: the dense cell list";
        return false;
    }
    // W_kb / W_vb for the attention: the pack's BF16 copies or native Q8_0 rows as they are (B10.7), F32 ones converted
    // to BF16 once; a layer whose two weights disagree (one Q8_0, one BF16) gets a BF16 copy of the Q8_0 one
    M->wk_b.assign(X.L.size(), nullptr);
    M->wv_b.assign(X.L.size(), nullptr);
    M->w_type.assign(X.L.size(), glmd::kWTypeBF16);
    int converted = 0, n_q8 = 0, n_bf16 = 0;
    for (size_t il = 0; il < X.L.size(); ++il) {
        const GlmDsaLayerW& W = X.L[il];
        if (!W.bound) continue;
        const bool q8 = W.k_b.type == kTypeQ8_0 && W.v_b.type == kTypeQ8_0 && W.k_b.q != nullptr && W.v_b.q != nullptr;
        M->w_type[il] = q8 ? glmd::kWTypeQ8_0 : glmd::kWTypeBF16;
        if (q8) ++n_q8;
        else ++n_bf16;
        for (int k = 0; k < 2; ++k) {
            const WSlot& w = k == 0 ? W.k_b : W.v_b;
            const uint16_t*& dst = k == 0 ? M->wk_b[il] : M->wv_b[il];
            const size_t n = (size_t) w.n_in * (size_t) w.n_out;
            if (q8 || w.type == kTypeBF16) {
                dst = (const uint16_t*) w.q;
            } else if (w.type == kTypeQ8_0 && w.q != nullptr) {
                void* b16 = nullptr;
                if (!q8_0_to_bf16_device(w.q, n, b16, err)) return false;
                M->owned.push_back(b16);
                dst = (const uint16_t*) b16;
                ++converted;
            } else if (w.type == kTypeF32 && (w.q != nullptr || w.f32 != nullptr)) {
                void* b16 = nullptr;
                if (cudaMalloc(&b16, n * 2) != cudaSuccess) {
                    cudaGetLastError();
                    err = "glm-dsa mixer: a BF16 copy of layer " + std::to_string(il) + "'s attn_k_b / attn_v_b";
                    return false;
                }
                M->owned.push_back(b16);
                if (!glmd::f32_to_bf16(w.f32 != nullptr ? w.f32 : (const float*) w.q, (uint16_t*) b16, n, nullptr)) {
                    err = "glm-dsa mixer: the BF16 conversion of layer " + std::to_string(il);
                    return false;
                }
                dst = (const uint16_t*) b16;
                ++converted;
            }
        }
        if (q8 && (d.qk_nope % 32 || d.kv_lora % 32)) {
            err = "glm-dsa mixer: native Q8_0 attn_k_b / attn_v_b need qk_nope and kv_lora multiples of 32";
            return false;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("glm-dsa mixer setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const glmd::DecodePolicy& P = d.pol;
    const bool fused_ok = glmd::mla_fused_supported(d.n_head, d.qk_nope, d.qk_rope, d.kv_lora, d.v_head,
                                                    n_q8 > 0 ? glmd::kWTypeQ8_0 : glmd::kWTypeBF16);
    std::fprintf(stderr,
                 "glm-dsa mixer: CUDA%d top-%d over token keys%s, scale %g, attention scratch %.1f MB, %d F32/Q8_0 "
                 "k_b/v_b converted to BF16; W_kb/W_vb native Q8_0 on %d layers, BF16 on %d\n",
                 X.dev, d.top_k, M->dense ? " - DENSE attention (STRATA_GLM_DSA_DENSE: every cell 0..p)" : "",
                 (double) d.kq_scale, M->work_bytes / 1048576.0, converted, n_q8, n_bf16);
    std::fprintf(stderr,
                 "glm-dsa mixer: CUDA%d decode policy: attention %s, scorer WMMA %s (above %d cells%s), top-k radix %s "
                 "(above %d cells), index keys %s, decode graphs %s\n",
                 X.dev, P.fused_mla ? (fused_ok ? "mla_fused" : "mla2 (mla_fused refuses this shape)") : "mla2",
                 switch_name(P.idx_wmma), glmd::policy_wmma_min(P, d.top_k),
                 glmd::idx_wmma_shape(d.idx_heads, d.idx_key) ? "" : "; this indexer shape stays on the token scorer",
                 switch_name(P.topk_mb), glmd::policy_mb_min(P, d.top_k), P.idx_fp16 ? "FP16" : "F32",
                 glmd::dsa_graphs_enabled() ? (M->dense ? "off (the dense switch is not graph-safe)"
                                                        : "requested (not with profiling, dumps, prefetch or lookahead)")
                                            : "off");
    X.mix = M;
    return true;
}

void mixer_rollback(GlmDsaFast& X) {
    X.n_sel_last = 0;
    X.sel_layer = -1;
    X.sel_pos = -1;
    if (!X.mix) return;
    GlmDsaFast::Mixer& M = *X.mix;
    sel_rollback(M.st);
    M.p_cur = nullptr;
    M.p_last = -1;
}

}  // namespace glmdsa

// GLM_CB_DIR seam dumps (debug only, synchronous; the reference forward's names).  GLM_CB_APPEND=1 appends per token
// and GLM_CB_ONLY=<seams> filters (src/core/glm_cb_dump.hpp, B7.6); pad_n pads an appended selection row with -1.
static void mixer_dump(cudaStream_t s, const std::string& name, const void* dev, int64_t n, size_t elem = 4,
                       int64_t pad_n = 0) {
    if (glmcb::dir() == nullptr || !glmcb::wanted(name)) return;
    cudaStreamSynchronize(s);
    std::vector<uint8_t> buf((size_t) std::max<int64_t>(n, 0) * elem);
    if (!buf.empty()) cudaMemcpy(buf.data(), dev, buf.size(), cudaMemcpyDeviceToHost);
    glmcb::write(name, buf.data(), buf.size(), elem == 4 ? ".f32" : ".bin", (size_t) std::max<int64_t>(pad_n, 0) * elem);
}

// L1-6 KV streaming: MixerCache::kv_resolve for one streamed layer - the latent's map resolves the selection (and copies
// the missed latent rows), the same misses are copied from the k_pe host copy into the k_pe slots
namespace {
struct DsaKvCtx {
    const strata::kernels::KvStreamMap* map = nullptr;
    const uint8_t* lat_host = nullptr;
    uint8_t* lat_slots = nullptr;
    const uint8_t* kpe_host = nullptr;
    uint8_t* kpe_slots = nullptr;
    int lat_row = 0, kpe_row = 0, page = 1;
    int* slot_cells = nullptr;
    cudaStream_t s = nullptr;
};
const int* dsa_kv_resolve(void* ctx, const int* cells, int n_cap, const int* n_sel_dev) {
    const DsaKvCtx& k = *(const DsaKvCtx*) ctx;
    if (k.slot_cells == nullptr || cells == nullptr) return nullptr;
    if (n_sel_dev != nullptr)
        gf::glm_kv_resolve_dev(*k.map, k.lat_host, k.lat_slots, k.lat_row, k.page, cells, n_cap, n_sel_dev,
                               k.slot_cells, k.s);
    else
        gf::glm_kv_resolve(*k.map, k.lat_host, k.lat_slots, k.lat_row, k.page, cells, n_cap, k.slot_cells, k.s);
    if (k.kpe_host != nullptr) gf::glm_kv_copy_misses(*k.map, k.kpe_host, k.kpe_slots, k.kpe_row, k.page, k.s);
    return k.slot_cells;
}
}  // namespace

// the cache of layer il, streamed or not (k: the resolve context it points at, alive for the layer's call)
void Glm5Model::dsa_mixer_cache(int il, void* cache, void* kctx, int* slot_cells, void* stream) const {
    glmdsa::MixerCache& c = *(glmdsa::MixerCache*) cache;
    c.lat = state_ + lay_.dsa_lat[(size_t) il];
    c.lat_q8 = lat_q8_ ? 1 : 0;
    c.k_pe = g_.qk_rope > 0 ? (uint16_t*) (state_ + lay_.dsa_kpe[(size_t) il]) : nullptr;
    if (!kv_streamed(kv_, il)) return;
    DsaKvCtx& k = *(DsaKvCtx*) kctx;
    k.map = &kv_->map[(size_t) il];
    k.lat_host = kv_->host[(size_t) il];
    k.lat_slots = (uint8_t*) c.lat;
    k.kpe_host = kv_->kpe_host.empty() ? nullptr : kv_->kpe_host[(size_t) il];
    k.kpe_slots = (uint8_t*) c.k_pe;
    k.lat_row = kv_->row_bytes;
    k.kpe_row = kv_->kpe_row_bytes;
    k.page = kv_->page;
    k.slot_cells = slot_cells;
    k.s = (cudaStream_t) stream;
    c.lat_host = kv_->host[(size_t) il];
    c.kpe_host = (uint16_t*) k.kpe_host;
    c.kv_table = kv_->map[(size_t) il].page_table;
    c.kv_page = kv_->page;
    c.kv_resolve = dsa_kv_resolve;
    c.kv_ctx = kctx;
}

bool Glm5Model::fast_dsa_glm(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !F->dsa || !F->dsa->mix) {
        err = "glm-dsa mixer: no mixer state on this part (fast_dsa_arena did not run)";
        return false;
    }
    glmdsa::GlmDsaFast& X = *F->dsa;
    glmdsa::GlmDsaFast::Mixer& M = *X.mix;
    const glmdsa::GlmDsaLayerW& W = X.L[(size_t) il];
    cudaStream_t s = F->cs;
    if (!W.bound || !lay_.is_dsa(il) || (g_.qk_rope > 0 && !lay_.has_kpe(il))) {
        err = "glm-dsa mixer: layer " + std::to_string(il) + " is not bound or has no latent / k_pe cache on this part";
        return false;
    }
    // the position record (B14.4): inside a captured part the one graph_prepare wrote before the capture; else a fresh
    // ring slot for each token on this part (its first layer) and for the NextN block, written by one tiny launch on
    // this stream (kernel arguments, no host buffer: nothing for a later token to overwrite under a pending copy)
    if (M.graph_live) {
        M.p_cur = M.pos_graph;
    } else if (il == l0_ || il == mtp_il_ || p != M.p_last || M.p_cur == nullptr || M.p_cur == M.pos_graph) {
        const int slot = M.pos_next;
        M.pos_next = (M.pos_next + 1) % glmdsa::kPosRing;
        if (!glmd::set_dev_pos(M.pos_ring + slot, (int) p, M.d.top_k, s)) {
            err = "glm-dsa mixer: the position record";
            return false;
        }
        M.p_cur = M.pos_ring + slot;
    }
    M.p_last = p;
    glmdsa::MixerDims d = M.d;
    d.pol.idx_fp16 = lay_.ik_bytes == 2;   // B10.8: the BOUND layout's key dtype (a sequence may carry its own)
    d.pol.idx_had = lay_.ik_had != 0;      // B14.11 (the graph key's bit)
    glmdsa::MixerBufs b;
    b.x = F->x;
    b.xq = F->xq;
    b.qr_raw = X.qr_raw;
    b.qr = X.qr;
    b.qr_q = X.qr_q;
    b.kv_raw = X.kv_raw;
    b.ik_raw = X.ik_raw;
    b.iw = X.iw;
    b.q = X.q;
    b.iq = X.iq;
    b.score = X.score;
    b.cells = X.cells;
    b.cells_dense = M.cells_dense;
    b.n_sel_dev = const_cast<int*>(&M.p_cur->n_sel);   // min(p + 1, top_k), written with the position
    b.n_sel_preset = true;
    b.attn_out = X.attn_out;
    b.attn_out_q = X.attn_out_q;
    b.mixer = F->mixer;
    b.work = M.work;
    b.work_bytes = M.work_bytes;
    b.inv_freq = M.inv_freq;
    b.idx_inv_freq = M.idx_inv_freq;
    b.p_dev = &M.p_cur->p;
    b.n_grid = M.graph_live ? (int) M.graph_n_cap : 0;
    glmdsa::MixerCache c;
    DsaKvCtx kctx;   // L1-6 KV streaming (unused when the layer is not streamed)
    dsa_mixer_cache(il, &c, &kctx, M.slot_cells, F->cs);
    c.ik = lay_.has_index_cache(il) ? state_ + lay_.dsa_ik[(size_t) il] : nullptr;
    c.ik_f16 = d.pol.idx_fp16 ? 1 : 0;
    c.ik_had = lay_.ik_had;   // B14.11: the BOUND layout's rotation (keys and queries)
    glmdsa::MixerLayer w;
    w.full = W.full;
    w.idx_src = W.idx_src;
    w.q_a_norm = W.q_a_norm;
    w.kv_a_norm = W.kv_a_norm;
    w.ik_ln_w = W.idx_k_norm_w;
    w.ik_ln_b = W.idx_k_norm_b;
    w.wk_b = M.wk_b[(size_t) il];
    w.wv_b = M.wv_b[(size_t) il];
    w.w_type = M.w_type[(size_t) il];
    if (w.wk_b == nullptr || w.wv_b == nullptr) {
        err = "glm-dsa mixer: layer " + std::to_string(il) + " has no BF16 / Q8_0 attn_k_b / attn_v_b";
        return false;
    }
    glmdsa::GpuOps ops;
    ops.X = &X;
    ops.s = s;
    ops.pol = d.pol;
    ops.p = p;
    ops.topk_work = M.topk_work;
    if (!glmdsa::mixer_layer(ops, d, b, c, w, il, p, M.dense, M.st, err)) return false;
    X.n_sel_last = M.st.n_sel_last;
    X.sel_layer = M.st.layer;
    X.sel_pos = M.st.pos;
    if (F->prof_on) F->mark("dsa_glm");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps && !M.graph_live) {
        const std::string L = std::to_string(il);
        mixer_dump(s, "qr-" + L, X.qr, g_.q_lora);
        mixer_dump(s, "q-" + L, X.q, (int64_t) g_.n_head * g_.qk_head());
        if (W.full && !M.dense) mixer_dump(s, "top_k-" + L, X.cells, M.st.n_sel_last, 4, g_.idx_top_k);
        mixer_dump(s, "kqv_out-" + L, X.attn_out, (int64_t) g_.n_head * g_.v_head);
        mixer_dump(s, "attn_out-" + L, F->mixer, g_.n_embd);
    }
    return true;
}

namespace glmdsa {

// L1-4: a verify window starts on this part - its rows' buffers (allocated once, cap rows), each row's position record
// written (a launch a row on s, ahead of the window's kernels) and its selection record cleared
bool win_rows_begin(GlmDsaFast& X, int64_t p0, int T, int cap, cudaStream_t s, std::string& err) {
    if (!X.mix || T < 1 || T > cap) {
        err = "glm-dsa window: no mixer on this part, or " + std::to_string(T) + " rows of " + std::to_string(cap);
        return false;
    }
    GlmDsaFast::Mixer& M = *X.mix;
    auto& W = M.win;
    const MixerDims& d = M.d;
    if (W.mem == nullptr || W.cap < cap) {
        if (W.mem) cudaFree(W.mem);
        W = GlmDsaFast::Mixer::Win{};
        const auto up = [](size_t b) { return (b + 255) & ~(size_t) 255; };
        const size_t f = sizeof(float), n = (size_t) cap, q8 = 36;
        const size_t qrq = (size_t) d.q_lora / 32 * q8, aq = (size_t) d.n_head * d.v_head / 32 * q8;
        const size_t sz[12] = {up(n * d.q_lora * f), up(n * d.q_lora * f), up(n * qrq), up(n * d.kv_row() * f),
                               up(n * d.idx_key * f), up(n * d.idx_heads * f), up(n * d.n_head * d.qk_head() * f),
                               up(n * d.idx_heads * d.idx_key * f), up(n * d.n_head * d.v_head * f), up(n * aq),
                               up(n * d.top_k * sizeof(int)), up(n * sizeof(glmd::DevPos))};
        size_t total = 0;
        for (const size_t b : sz) total += b;
        if (cudaMalloc(&W.mem, total) != cudaSuccess) {
            cudaGetLastError();
            W.mem = nullptr;
            err = "glm-dsa window: " + std::to_string(total >> 20) + " MB of row buffers did not allocate";
            return false;
        }
        uint8_t* at = (uint8_t*) W.mem;
        void* p[12];
        for (int i = 0; i < 12; ++i) {
            p[i] = at;
            at += sz[i];
        }
        W.b.qr_raw = (float*) p[0];
        W.b.qr = (float*) p[1];
        W.b.qr_q = p[2];
        W.b.kv_raw = (float*) p[3];
        W.b.ik_raw = (float*) p[4];
        W.b.iw = (float*) p[5];
        W.b.q = (float*) p[6];
        W.b.iq = (float*) p[7];
        W.b.attn_out = (float*) p[8];
        W.b.attn_out_q = p[9];
        W.rs.qr_raw = W.rs.qr = d.q_lora;
        W.rs.qr_q = (int64_t) qrq;
        W.rs.kv_raw = d.kv_row();
        W.rs.ik_raw = d.idx_key;
        W.rs.iw = d.idx_heads;
        W.rs.q = (int64_t) d.n_head * d.qk_head();
        W.rs.iq = (int64_t) d.idx_heads * d.idx_key;
        W.rs.attn_out = (int64_t) d.n_head * d.v_head;
        W.rs.attn_out_q = (int64_t) aq;
        W.pos = (glmd::DevPos*) p[11];
        for (int t = 0; t < cap; ++t) {
            W.cells.push_back((int*) p[10] + (size_t) t * d.top_k);
            W.n_sel.push_back(&W.pos[t].n_sel);
            W.p.push_back(&W.pos[t].p);
        }
        W.st.assign((size_t) cap, SelState{});
        W.cap = cap;
    }
    for (int t = 0; t < T; ++t) {
        if (!glmd::set_dev_pos(W.pos + t, (int) (p0 + t), d.top_k, s)) {
            err = "glm-dsa window: a row's position record";
            return false;
        }
        W.st[(size_t) t] = SelState{};
    }
    return true;
}

}  // namespace glmdsa

// L1-4: rows t0 .. t0 + n - 1 of a run of rows (positions p0 + t0 ..; win_rows_begin ran for the run) through layer
// il's mixer: x / xq rows in (row 0 of the run at x / xq, xq_s bytes a row), mixer rows out - glmdsa::mixer_rows on the
// GPU backend fast_dsa_glm uses.  The verify window's rows (fast_window_layers_plain) and the NextN block's cache rows
// (mtp_cache_rows).
bool Glm5Model::fast_dsa_glm_window(int il, int64_t p0, int t0, int n, const float* x, const void* xq, size_t xq_s,
                                    float* mixer, std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !F->dsa || !F->dsa->mix) {
        err = "glm-dsa window: no mixer state on this part";
        return false;
    }
    glmdsa::GlmDsaFast& X = *F->dsa;
    glmdsa::GlmDsaFast::Mixer& M = *X.mix;
    auto& Wd = M.win;
    const glmdsa::GlmDsaLayerW& L = X.L[(size_t) il];
    if (!L.bound || !lay_.is_dsa(il) || (g_.qk_rope > 0 && !lay_.has_kpe(il)) || Wd.mem == nullptr ||
        t0 + n > Wd.cap) {
        err = "glm-dsa window: layer " + std::to_string(il) + " is not bound, or the rows were not set up";
        return false;
    }
    const int E = g_.n_embd;
    glmdsa::MixerDims d = M.d;
    d.pol.idx_fp16 = lay_.ik_bytes == 2;
    d.pol.idx_had = lay_.ik_had != 0;
    const auto& rs = Wd.rs;
    glmdsa::MixerBufs b;
    b.x = x + (size_t) t0 * E;
    b.xq = (const uint8_t*) xq + (size_t) t0 * xq_s;
    b.mixer = mixer + (size_t) t0 * E;
    b.qr_raw = Wd.b.qr_raw + t0 * rs.qr_raw;
    b.qr = Wd.b.qr + t0 * rs.qr;
    b.qr_q = (uint8_t*) Wd.b.qr_q + t0 * rs.qr_q;
    b.kv_raw = Wd.b.kv_raw + t0 * rs.kv_raw;
    b.ik_raw = Wd.b.ik_raw + t0 * rs.ik_raw;
    b.iw = Wd.b.iw + t0 * rs.iw;
    b.q = Wd.b.q + t0 * rs.q;
    b.iq = Wd.b.iq + t0 * rs.iq;
    b.attn_out = Wd.b.attn_out + t0 * rs.attn_out;
    b.attn_out_q = (uint8_t*) Wd.b.attn_out_q + t0 * rs.attn_out_q;
    b.score = X.score;
    b.cells_dense = M.cells_dense;
    b.work = M.work;
    b.work_bytes = M.work_bytes;
    b.inv_freq = M.inv_freq;
    b.idx_inv_freq = M.idx_inv_freq;
    b.n_sel_preset = true;
    glmdsa::RowStrides r = rs;
    r.x = E;
    r.xq = (int64_t) xq_s;
    r.mixer = E;
    // the caches and the layer's small weights, as fast_dsa_glm passes them
    glmdsa::MixerCache c;
    DsaKvCtx kctx;   // L1-6 KV streaming (unused when the layer is not streamed)
    dsa_mixer_cache(il, &c, &kctx, M.slot_cells, F->cs);
    c.ik = lay_.has_index_cache(il) ? state_ + lay_.dsa_ik[(size_t) il] : nullptr;
    c.ik_f16 = d.pol.idx_fp16 ? 1 : 0;
    c.ik_had = lay_.ik_had;
    glmdsa::MixerLayer w;
    w.full = L.full;
    w.idx_src = L.idx_src;
    w.q_a_norm = L.q_a_norm;
    w.kv_a_norm = L.kv_a_norm;
    w.ik_ln_w = L.idx_k_norm_w;
    w.ik_ln_b = L.idx_k_norm_b;
    w.wk_b = M.wk_b[(size_t) il];
    w.wv_b = M.wv_b[(size_t) il];
    w.w_type = M.w_type[(size_t) il];
    if (w.wk_b == nullptr || w.wv_b == nullptr) {
        err = "glm-dsa window: layer " + std::to_string(il) + " has no BF16 / Q8_0 attn_k_b / attn_v_b";
        return false;
    }
    glmdsa::GpuOps ops;
    ops.X = &X;
    ops.s = F->cs;
    ops.pol = d.pol;
    ops.p = p0 + t0;
    ops.topk_work = M.topk_work;
    return glmdsa::mixer_rows(ops, d, b, r, n, Wd.cells.data() + t0, Wd.n_sel.data() + t0, Wd.p.data() + t0, c, w, il,
                              p0 + t0, M.dense, Wd.st.data() + t0, err);
}

namespace glmdsa {

bool graph_prepare(GlmDsaFast& X, const GlmStateLayout& lay, int l0, int l1, int64_t p, unsigned long long cpu_plan,
                   const void* residual, const void* state, const char* host_off, uint64_t requests, cudaStream_t s,
                   PartGraph& pg, std::string& err, int seqs) {
    pg = PartGraph{};
    if (!X.mix || !glmd::dsa_graphs_enabled() || X.mix->dense || l1 <= l0) return true;
    GlmDsaFast::Mixer& M = *X.mix;
    if (host_off != nullptr) {   // requested, but the host side is not pure device work: say why, once a part
        // (once more whenever the reason changes or comes back after graphs ran: a request's knobs, GEN prefetch_n= /
        // graphs=, turn graphs off for that request only - glm_req_knobs.hpp)
        if (M.graph_off_last == nullptr || std::strcmp(M.graph_off_last, host_off) != 0)
            std::fprintf(stderr, "glm-dsa decode graphs: CUDA%d layers %d-%d requested but off: %s\n", X.dev, l0, l1 - 1,
                         host_off);
        M.graph_off_why = M.graph_off_last = host_off;
        ++M.graph_eager;
        return true;
    }
    M.graph_off_last = nullptr;   // (the host side allows a graph again: the next reason met is logged)
    if (p < 0 || p >= M.d.max_ctx) return true;   // (the mixer refuses it with a message)
    M.graphs.set_per_part(seqs);   // a graph per sequence's state arena: switching sequences replays (GraphIndex)
    static const bool mv_nw2 = strata::glmenv::env_flag("STRATA_GLMD_MV_NW2", false);     // (as glmd_mv_ext.cuh reads them)
    static const bool mv_typed = strata::glmenv::env_flag("STRATA_GLMD_MV_TYPED", false);
    glmd::DecodePolicy pol = M.d.pol;
    pol.idx_fp16 = lay.ik_bytes == 2;
    pol.idx_had = lay.ik_had != 0;   // B14.11
    glmd::GraphKey& k = pg.key;
    k.part = l0;
    k.l0 = l0;
    k.l1 = l1;
    k.cpu_plan = cpu_plan;
    k.residual = residual;
    k.arena = X.arena;
    k.state = state;
    k.bits = glmd::policy_bits(pol, lay.mode == GlmStateMode::Int8, mv_nw2, mv_typed);
    k.regime = glmd::policy_regime(pol, p, M.d.top_k);
    k.n_cap = glmd::graph_n_cap(p, M.d.max_ctx);
    k.gen = glmd::graph_generation();   // B15.6: any arena reallocation since the capture forces a new one
    // the position is DATA: one launch before the replay (outside any capture)
    if (!glmd::set_dev_pos(M.pos_graph, (int) p, M.d.top_k, s)) {
        err = "glm-dsa graphs: the position record";
        return false;
    }
    M.graph_last_full = -1;
    for (int il = l0; il < l1 && (size_t) il < X.L.size(); ++il)
        if (X.L[(size_t) il].full) M.graph_last_full = il;
    pg.requests = requests;
    pg.on = true;
    return true;
}

bool graph_run(GlmDsaFast& X, PartGraph& pg, int64_t p, cudaStream_t s, uint64_t& expected,
               const std::function<bool()>& body, std::string& err) {
    GlmDsaFast::Mixer& M = *X.mix;
    M.graph_live = true;
    M.graph_n_cap = pg.key.n_cap;
    bool replayed = false;
    const bool ok = M.graphs.enqueue(pg.key, s, expected, pg.requests, body, replayed, err);
    M.graph_live = false;
    if (!ok) return false;
    if (replayed) {   // no host code ran: the records the body would have left
        M.st.layer = M.graph_last_full;
        M.st.pos = p;
        M.st.dense = false;
        M.st.n_sel_last = sel_count(p, M.d.top_k, false);
        X.n_sel_last = M.st.n_sel_last;
        X.sel_layer = M.st.layer;
        X.sel_pos = M.st.pos;
    }
    M.p_cur = M.pos_graph;   // the next token outside a graph takes a ring slot
    M.p_last = p;
    return true;
}

void graph_stats(const GlmDsaFast& X, uint64_t& captures, uint64_t& replays, uint64_t& invalidations) {
    captures = replays = invalidations = 0;
    if (!X.mix) return;
    captures = X.mix->graphs.captures;
    replays = X.mix->graphs.replays;
    invalidations = X.mix->graphs.invalidations;
}

}  // namespace glmdsa

int Glm5Model::dsa_n_sel_last() const {
    return fast_ != nullptr && fast_->dsa ? fast_->dsa->n_sel_last : -1;
}

bool Glm5Model::prefill_tokens_dsa(const std::vector<int32_t>& tokens, int32_t next_token, std::string& err) {
    bool batched = true;   // prefill()'s batched path can run on every part
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        batched = batched && m->pf_ != nullptr && m->fast_ != nullptr && !m->vis_lent_;
    if (!glmpfs::mtp_token_fill(g_.arch == GlmArch::GlmDsa, fast_ != nullptr && has_mtp(), batched,
                                (int64_t) tokens.size(), glmpfs::prefill_min_tokens(getenv("STRATA_GLM_PREFILL_MIN"))))
        return false;   // Maya's prefill(): the batched path (mtp_fill) or the generic token path
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!fast_token(tokens[i], err)) {
            if (err.empty()) err = "glm prompt: the token path failed at position " + std::to_string(i);
            return false;
        }
        const int32_t nx = i + 1 < tokens.size() ? tokens[i + 1] : next_token;
        if (nx >= 0 && mtp_draft(nx, err) < 0 && !err.empty()) return false;   // the NextN block's caches at this position
        fast_->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++fast_->tokens;
        if (prefill_progress && !prefill_progress((int64_t) i + 1, (int64_t) tokens.size())) {
            err = "cancelled";
            return false;
        }
    }
    return true;
}

}  // namespace strata::core
