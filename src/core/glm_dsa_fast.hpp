// src/core/glm_dsa_fast.hpp - the glm-dsa fast binder's device side (glm-dsa B2.4): the per-layer weights the glm-dsa
// attention mixer consumes, its activation arena, and the gf::mv wrappers that let a BF16 / F32 weight sit where the
// Flash path only ever saw native rows.
//
// Owned by Glm5Model::FastState::dsa (a shared_ptr, null on glm5-next): filled by Glm5Model::fast_bind_dsa (one call
// per layer from fast_setup's weight loop) and Glm5Model::fast_dsa_arena (after the shared arena is carved), both in
// src/core/glm_dsa_fast.cu.  Not a public header.
#pragma once

#include "glm_dsa_fast_plan.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core::glmdsa {

/// One glm-dsa layer's attention weights, resolved at load (device pointers; never looked up per token).
///
/// Every matrix is a WSlot: {q: the weight rows, type: its gf::mv type, n_in, n_out; f32 is set for type F32}.  type is
/// a native ggml type (mv dots it against a q8_1 input), kTypeBF16 or kTypeF32 (mv reads the FLOAT input: pass xf, or
/// call glmdsa::mv, which dequantizes the q8_1 input when only that is at hand).  Layouts are ggml's (first dimension
/// contiguous, the same as include/strata/kernels/glmd_ref.hpp documents): a weight W[o * n_in + i].
struct GlmDsaLayerW {
    bool bound = false;       // this part holds the layer (l0_ <= il < lt_)
    bool mtp = false;         // the NextN block (il == n_layers)
    bool full = false;        // Glm5Geometry::idx_full(il): the layer runs its own indexer
    int idx_src = -1;         // the layer whose top-k selection this one uses: il when full, else the previous full
                              // layer (a part boundary is always full, so it is on this part)

    const float* attn_norm = nullptr;   // [n_embd] RMSNorm weight of the attention input (also FastState::Layer)
    WSlot q_a;                          // attn_q_a       n_embd -> q_lora (2048)
    const float* q_a_norm = nullptr;    // [q_lora] attn_q_a_norm (RMSNorm, eps norm_eps)
    WSlot q_b;                          // attn_q_b       q_lora -> n_head * qk_head (64 x 256 = 16384), per head
                                        //                [q_nope 192 | q_pe 64]; upcast set: may be BF16 / F32
    WSlot kv_a;                         // attn_kv_a_mqa  n_embd -> kv_row (576 rows out): [c_kv 512 | k_pe 64]
    const float* kv_a_norm = nullptr;   // [kv_lora] attn_kv_a_norm (RMSNorm of c_kv only; k_pe is roped, not normed)
    WSlot k_b;                          // attn_k_b  BF16 (or F32, or native Q8_0: B10.7 pick_headwise)
                                        //           GGUF {qk_nope, kv_lora, n_head}: n_in qk_nope,
                                        //           n_out kv_lora * n_head, element [(h * kv_lora + r) * qk_nope + c]
    WSlot v_b;                          // attn_v_b  BF16 (or F32, or native Q8_0)
                                        //           GGUF {kv_lora, v_head, n_head}: n_in kv_lora,
                                        //           n_out v_head * n_head, element [(h * v_head + v) * kv_lora + c]
    WSlot out;                          // attn_output    n_head * v_head (16384) -> n_embd; upcast set

    // the lightning indexer: bound on full layers only (null / empty on shared layers, which reuse idx_src's top-k)
    WSlot idx_k;                        // indexer.attn_k    BF16 / F32  n_embd -> idx_key (128)
    WSlot idx_q_b;                      // indexer.attn_q_b  BF16 / F32  q_lora -> idx_heads * idx_key (32 x 128)
    WSlot idx_proj;                     // indexer.proj      BF16 / F32  n_embd -> idx_heads (32)
    const float* idx_k_norm_w = nullptr;   // [idx_key] indexer.k_norm.weight (LayerNorm, eps idx_ln_eps)
    const float* idx_k_norm_b = nullptr;   // [idx_key] indexer.k_norm.bias
    // no compressor tensors: glm-dsa has none (token-level keys, idx_kpool 1)
};

/// The glm-dsa side of one part's FastState.  The arena buffers are described in glm_dsa_fast_plan.hpp (enum Buf);
/// the pointers below are carved from `arena` by Glm5Model::fast_dsa_arena.
struct GlmDsaFast {
    std::vector<GlmDsaLayerW> L;   // indexed by absolute layer, n_layers + 1 entries (like FastState::L)
    ArenaPlan plan;
    void* arena = nullptr;
    int dev = 0;
    float* q = nullptr;            // kQ        n_head * qk_head
    float* kv_raw = nullptr;       // kKvRaw    kv_row (576)
    float* q_abs = nullptr;        // kQAbs     n_head * kv_row
    float* score = nullptr;        // kScore    max_ctx
    int* cells = nullptr;          // kCells    idx_top_k
    float* qr_raw = nullptr;       // kQrRaw    q_lora
    float* qr = nullptr;           // kQr       q_lora
    void* qr_q = nullptr;          // kQrQ      q_lora q8_1
    float* ik_raw = nullptr;       // kIkRaw    idx_key
    float* iq = nullptr;           // kIq       idx_heads * idx_key
    float* iw = nullptr;           // kIw       idx_heads
    float* attn_out = nullptr;     // kAttnOut  n_head * v_head
    void* attn_out_q = nullptr;    // kAttnOutQ n_head * v_head q8_1
    float* act = nullptr;          // kAct      kMaxMvJobs rows of plan.act_floats (glmdsa::mv's fallback scratch)
    // what the binder resolved (logged once by fast_dsa_arena)
    int n_router_f32 = 0, n_router_bf16 = 0, n_upcast_bf16 = 0, n_upcast_f32 = 0;
    // B4.6: the attention mixer's own state (rope tables, the DevPos ring, the attention scratch, the k_b / v_b the
    // attention reads, the dispatch policy, the decode graph cache; src/core/glm_dsa_mixer.cu), made by mixer_setup at
    // the end of fast_dsa_arena
    struct Mixer;
    std::shared_ptr<Mixer> mix;
    int n_sel_last = 0;            // the cells the last full layer's attention read (host record, B4.6)
    int sel_layer = -1;            // ... which layer made that selection, at which position
    int64_t sel_pos = -1;
    GlmDsaFast() = default;
    GlmDsaFast(const GlmDsaFast&) = delete;
    GlmDsaFast& operator=(const GlmDsaFast&) = delete;
    ~GlmDsaFast();   // frees the arena (FastState is destroyed with its device current)
};

/// Whether a gf::mv type reads the float input (BF16 / F32) rather than q8_1.
inline bool float_type(int type) { return type == kTypeF32 || type == kTypeBF16; }

/// gf::mv with the float-input fallback: a job whose weight is BF16 / F32 but which carries only a q8_1 input (xf null)
/// gets that input dequantized into ext->act first (one row per distinct input).  Jobs that need no fallback pass
/// through unchanged, so on glm5-next (ext null) this IS gf::mv.  False when gf::mv refuses, or a fallback is needed
/// and there is no ext.
bool mv(GlmDsaFast* ext, const strata::kernels::glmf::MvJob* jobs, int n, cudaStream_t s);

/// The shared expert's down when moe_gate_up's fused shared down cannot dot its type (!shexp_fused_ok: BF16 / F32, or a
/// B10.7 type - F16, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL): the down runs here instead (glmdsa::mv into sh_out: BF16 / F32 on
/// sh_hq dequantized, the B10.7 types on sh_hq through ext_mv) and `apart` is set: the caller then passes a null sh_down
/// to moe_gate_up and still hands sh_out to moe_down.  apart = false (nothing launched) for a type the fused dot serves.
/// ext may be null (glm5-next): the B10.7 types need no fallback; BF16 / F32 then fail (no scratch).
bool shexp_down_apart(GlmDsaFast* ext, const WSlot& sh_down, const void* sh_hq, int n_ff_sh, int n_embd, float* sh_out,
                      cudaStream_t s, bool& apart);

/// B4.6: the mixer's state for this part (glm_dsa_mixer.cu; a refusing stub without CUDA).  Called once by
/// Glm5Model::fast_dsa_arena after the arena and every layer are bound.
bool mixer_setup(GlmDsaFast& X, const Glm5Geometry& g, int64_t max_ctx, std::string& err);

/// B10.1 / B14.4: the per-part decode graph of the glm-dsa layer loop (MAYA_GLM_DSA_GRAPHS=1; glm_dsa_mixer.cu, a
/// no-op stub without CUDA).  Per token, fast_layers_plain calls graph_prepare; when pg.on it runs its layer loop as
/// graph_run's body (captured the first time and whenever pg.key changes, replayed otherwise), else directly.
struct PartGraph {
    bool on = false;               // this token's part runs as a graph
    glmd::GraphKey key;            // part / bounds / CPU-lane plan / fixed addresses / kernel bits / regime / capacity
    uint64_t requests = 0;         // the CPU-lane requests one pass of the body posts (one per MoE layer)
};
/// Decide for the token at p: on only with MAYA_GLM_DSA_GRAPHS=1, host_off null (the caller's host-side conditions:
/// no profiling, dumps, expert prefetch or lookahead; else it names the one that keeps graphs off, logged once per
/// part and counted, see host_off_reason) and without the dense switch.  When on, it writes the part's glmd::DevPos
/// for p (one launch on s, outside any capture) and fills the key from the dispatch policy and the bound layout.
/// seqs: the model's sequence count (Glm5Model::seq_count): the part keeps a graph per sequence's state.
/// false + err only on a launch failure.
bool graph_prepare(GlmDsaFast& X, const GlmStateLayout& lay, int l0, int l1, int64_t p, unsigned long long cpu_plan,
                   const void* residual, const void* state, const char* host_off, uint64_t requests, cudaStream_t s,
                   PartGraph& pg, std::string& err, int seqs = 1);
/// fast_layers_plain's host-side conditions for a decode graph: null when the body is pure device work, else the
/// switch that keeps graphs off (the first of: profiling, GLM_CB_DIR dumps, expert prefetch, lookahead, GLM_DUMP_H,
/// and a request that asks for none: GEN graphs=0, glm_req_knobs.hpp).  max_pf is the request's prefetch count
/// (GEN prefetch_n=, else MAYA_GLM_PREFETCH_N), so a prefetch request runs eagerly while a plain one replays.
inline const char* host_off_reason(bool prof_on, bool dumps, int max_pf, int n_ahead, bool dump_h,
                                   bool request_off = false) {
    return prof_on ? "profiling (MAYA_GLM_PROF)" : dumps ? "GLM_CB_DIR dumps" :
           max_pf > 0 ? "expert prefetch (MAYA_GLM_PREFETCH_N > 0, or the request's GEN prefetch_n=)" :
           n_ahead > 0 ? "lookahead (MAYA_GLM_AHEAD > 0)" :
           dump_h ? "GLM_DUMP_H" : request_off ? "the request (GEN graphs=0)" : nullptr;
}
/// Capture or replay the body (every layer of the part at p) on s; expected advances by pg.requests either way.
bool graph_run(GlmDsaFast& X, PartGraph& pg, int64_t p, cudaStream_t s, uint64_t& expected,
               const std::function<bool()>& body, std::string& err);
/// The part's graph counters (all 0 without the mixer).
void graph_stats(const GlmDsaFast& X, uint64_t& captures, uint64_t& replays, uint64_t& invalidations);
/// B11.2: a rejected speculative position on this part (the pipelined decode's head redoes it): forgets the
/// selection the rejected run made (the shared layers' SelState, the host record n_sel_last / sel_layer / sel_pos) and
/// the position word, so the redo's first layer takes a fresh ring slot and every shared layer must find its full
/// layer's selection made again at the redo.  The caches need nothing: the redo rewrites the same rows (cell s =
/// position s).  Host state only (the device may still run the rejected position).  No-op without a mixer.
void mixer_rollback(GlmDsaFast& X);

/// L1-4: a verify window of T rows (positions p0 ..) starts on this part: the mixer's row buffers for up to cap rows
/// (allocated on first use), each row's position record (glmd::DevPos) written on s and its selection record cleared.
/// Then Glm5Model::fast_dsa_glm_window runs the rows through each layer (glm_dsa_mixer.cu; refuses without CUDA).
bool win_rows_begin(GlmDsaFast& X, int64_t p0, int T, int cap, cudaStream_t s, std::string& err);

/// y[i] = q8_1 value i (n a multiple of 32): the device twin of q8_1_to_f32_host.
void q8_1_to_f32(const void* xq, int n, float* y, cudaStream_t s);

}  // namespace strata::core::glmdsa
