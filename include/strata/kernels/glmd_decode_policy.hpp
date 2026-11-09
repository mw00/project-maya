// include/strata/kernels/glmd_decode_policy.hpp - the glm-dsa decode kernel dispatch policy (glm-dsa B4.6 x B10.1-B10.8).
//
// ONE place decides which decode kernel serves each step of the mixer (glmdsa::mixer_layer, src/core/glm_dsa_mixer.hpp)
// for a token at position p (n = p + 1 visible cells, top_k = 2048 at GLM-5.3):
//
//   step        kernel                         when
//   ----------  -----------------------------  -------------------------------------------------------------------
//   scorer      glmd::idx_score (IdxScore)     the default: the token scorer (F32 or FP16 keys, fixed grid, n read on
//               (the CUDA-core token scorer)   the device); also every n <= top_k (it skips: the selection is the
//                                              identity) and every shape the WMMA kernel refuses
//               glmd::idx_score_wmma           n > top_k and the WMMA shape (idx_heads % 16 == 0, <= 32, idx_dim % 16,
//               (B10.3, IdxScoreWmma)          <= 256) and STRATA_GLMD_IDX_WMMA: unset/"auto" -> n > 16384
//                                              (kIdxWmmaMinDecode), 1 -> always, 0 -> never
//   selector    glmd::topk                     the default (one block per row)
//               glmd::topk_mb (B10.4)          n > top_k and STRATA_GLMD_TOPK_MB: unset/"auto" -> n > 32768
//                                              (kTopkMbMin), 1 -> always, 0 -> never
//               glmd::topk_hist (B14.15)       n > top_k, STRATA_GLMD_TOPK_HIST=1 (default off) and topk_mb not
//                                              chosen: the bucket selector, the same cells as glmd::topk
//   attention   glmd::mla2                     the default
//               glmd::mla_fused (B10.2)        STRATA_GLMD_FUSED_MLA=1 and mla_fused_supported (kv_lora 512, <= 64
//                                              heads, qk_rope 0 / 64); split count STRATA_GLMD_FUSED_SPLIT
//   key dtype   F32 / FP16 (B10.8)             the STATE LAYOUT decides (GlmStateLayout::ik_bytes == 2, carved from
//                                              STRATA_GLM_IDX_FP16): dsa_prep, both scorers, the prompt path and the
//                                              slot fingerprint all read the same layout, never the env on their own
//   W_kb / W_vb BF16 / native Q8_0 (B10.7)     the binder's pick (glmdsa::pick_headwise): Mla2 / FusedMla w_type
//   key rotation off / Walsh-Hadamard (B14.11) the STATE LAYOUT decides (GlmStateLayout::ik_had, carved from
//                                              STRATA_GLM_IDX_HADAMARD): dsa_prep and dsa_rope_q rotate keys and queries
//
// The two indexer argument structs are reconciled here: the mixer always fills ONE glmd::IdxScore (the token
// scorer's request: iq, keys + dtype, head weights, the device position, the cache capacity, the identity skip, the
// optional device cell count); wmma_args() derives the glmd::IdxScoreWmma of the same request (T = 1, pos = the device
// position, ld = the cache capacity, n_kv = the launch extent), so neither scorer's struct leaks into the other's code.
// Likewise fused_args() derives a glmd::FusedMla from the mixer's glmd::Mla2.
//
// Every decision is a pure function of (policy, n, shapes): decode_score / decode_topk / decode_mla (CUDA,
// glmd_mixer.cu) and their twins below take the same inputs, so the CPU twin runs the GPU's choice.  The CUDA-graph
// key (glmd_graph.hpp) is derived from the same policy (policy_bits, policy_regime), so a captured part never
// straddles a kernel switch: within one regime every decision above is constant.
//
// Defaults (Flash glm5-next never reaches this code): the token scorer, single-block top-k and mla2 below 16K / 32K
// cells, WMMA scorer and radix top-k above (auto), mla2 always unless STRATA_GLMD_FUSED_MLA=1.  A default-constructed
// DecodePolicy is all-off (the wave-4 behaviour: the CPU tests' baseline); decode_policy_env() is the engine's.
#pragma once

#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_graph.hpp"
#include "strata/kernels/glmd_mixer.hpp"
#include "strata/kernels/glmd_volta.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace glmd {

/// A tri-state switch value.
enum : int { kSwitchAuto = -1, kSwitchOff = 0, kSwitchOn = 1 };

/// Parse a switch (strata::glmenv::parse): null / "" -> dflt; "auto" -> kSwitchAuto; 0 / off / false / no -> off;
/// 1 / on / true / yes (any case, or a non-zero integer) -> on; anything else -> dflt.
inline int parse_switch(const char* v, int dflt) {
    const int r = strata::glmenv::parse(v);
    return r == strata::glmenv::kInvalid ? dflt : r;
}
/// The switch `name` from the environment: parse_switch, with a warning for a value it does not know.
inline int env_switch(const char* name, int dflt) { return strata::glmenv::env_switch(name, dflt, true); }
static_assert((int) kSwitchAuto == (int) strata::glmenv::kAuto && (int) kSwitchOff == (int) strata::glmenv::kOff &&
                  (int) kSwitchOn == (int) strata::glmenv::kOn, "glmd switch values mirror glmenv's");

struct DecodePolicy {
    bool fused_mla = false;          // STRATA_GLMD_FUSED_MLA
    bool mla_dec = true;             // STRATA_GLMD_MLA_DEC (L1-4, default on): mla2's attention through mla_dec
    int fused_split = kFusedSplitDefault;   // STRATA_GLMD_FUSED_SPLIT (> 0)
    int idx_wmma = kSwitchOff;       // STRATA_GLMD_IDX_WMMA (engine default auto)
    int topk_mb = kSwitchOff;        // STRATA_GLMD_TOPK_MB (engine default auto)
    bool idx_fp16 = false;           // the state layout's index-key dtype (GlmStateLayout::ik_bytes == 2)
    bool topk_hist = false;          // STRATA_GLMD_TOPK_HIST (B14.15, default off)
    bool idx_had = false;            // the state layout's key rotation (GlmStateLayout::ik_had, B14.11)
};

/// The engine's policy: the switches as the environment sets them (read on every call: the mixer reads it once per
/// load).  idx_fp16 is NOT read here: the caller sets it from the state layout.
inline DecodePolicy decode_policy_env() {
    DecodePolicy p;
    p.fused_mla = strata::glmenv::env_flag("STRATA_GLMD_FUSED_MLA", false);
    p.mla_dec = strata::glmenv::env_flag("STRATA_GLMD_MLA_DEC", true);
    p.fused_split = (int) std::clamp<long long>(strata::glmenv::env_int("STRATA_GLMD_FUSED_SPLIT", kFusedSplitDefault),
                                                1, 1 << 20);
    p.idx_wmma = env_switch("STRATA_GLMD_IDX_WMMA", kSwitchAuto);
    p.topk_mb = env_switch("STRATA_GLMD_TOPK_MB", kSwitchAuto);
    p.topk_hist = strata::glmenv::env_flag("STRATA_GLMD_TOPK_HIST", false);
    return p;
}

/// The policy a build can run: a HIP build has neither nvcuda::wmma kernel (glmd_volta_mla.cu / glmd_volta_idx.cu), so
/// there the fused MLA and the WMMA scorer stay off whatever the switches say (mla2 and the CUDA-core scorer run; the
/// radix top-k is portable).  A CUDA build: p unchanged.
inline DecodePolicy decode_policy_built(DecodePolicy p) {
#if defined(STRATA_USE_HIP)
    p.fused_mla = false;
    p.idx_wmma = kSwitchOff;
#endif
    return p;
}

/// The WMMA scorer's shapes (the rule of idx_score_wmma_supported, glmd_volta_idx.cu; static here so the CPU side
/// decides without the CUDA object).
inline bool idx_wmma_shape(int idx_heads, int idx_dim) { return idx_wmma_shape_rule(idx_heads, idx_dim); }
/// The cell count above which the WMMA scorer / the radix top-k take over (the forced mode: as soon as selecting).
inline int policy_wmma_min(const DecodePolicy& p, int top_k) {
    return p.idx_wmma == kSwitchOn ? top_k : kIdxWmmaMinDecode;
}
inline int policy_mb_min(const DecodePolicy& p, int top_k) { return p.topk_mb == kSwitchOn ? top_k : kTopkMbMin; }

/// n = p + 1 visible cells.
inline bool use_idx_wmma(const DecodePolicy& p, int64_t n, int top_k, int idx_heads, int idx_dim) {
    return p.idx_wmma != kSwitchOff && n > top_k && n > policy_wmma_min(p, top_k) && idx_wmma_shape(idx_heads, idx_dim);
}
inline bool use_topk_mb(const DecodePolicy& p, int64_t n, int top_k) {
    return p.topk_mb != kSwitchOff && n > top_k && n > policy_mb_min(p, top_k);
}
/// B14.15: the bucket selector where the single-block selector would run (topk_mb keeps its range).
inline bool use_topk_hist(const DecodePolicy& p, int64_t n, int top_k) {
    return p.topk_hist && n > top_k && !use_topk_mb(p, n, top_k);
}
inline bool use_fused_mla(const DecodePolicy& p, int n_head, int qk_nope, int qk_rope, int kv_lora, int v_head,
                          int w_type) {
    return p.fused_mla && mla_fused_supported(n_head, qk_nope, qk_rope, kv_lora, v_head, w_type);
}

/// The attention scratch for max_sel cells under this policy (mla2's, or the larger of mla2's and mla_fused's).
inline size_t decode_mla_work_bytes(const DecodePolicy& p, int n_head, int kv_lora, int qk_rope, int max_sel) {
    const size_t a = mla2_work_bytes(n_head, kv_lora, max_sel);
    if (!p.fused_mla) return a;
    return std::max(a, mla_fused_work_bytes(n_head, kv_lora, qk_rope, max_sel, p.fused_split));
}

/// The CUDA-graph key words of this policy (glmd_graph.hpp): the kernel bits and the regime of position pos.
inline uint32_t policy_bits(const DecodePolicy& p, bool lat_int8, bool mv_nw2, bool mv_typed) {
    return graph_kernel_bits(p.fused_mla, p.idx_wmma != kSwitchOff, p.idx_fp16, p.topk_mb != kSwitchOff, lat_int8,
                             mv_nw2, mv_typed) |
           (p.topk_hist ? kGkTopkHist : 0u) | (p.idx_had ? kGkIdxHad : 0u);
}
inline int policy_regime(const DecodePolicy& p, int64_t pos, int top_k) {
    return graph_regime(pos, top_k, p.idx_wmma != kSwitchOff, policy_wmma_min(p, top_k), p.topk_mb != kSwitchOff,
                        policy_mb_min(p, top_k));
}

/// The scorer's launch extent: IdxScore::n_grid when set (a graph's fixed capacity), else the live n; at most the
/// cache capacity n_cap.
inline int score_extent(const IdxScore& a) {
    const int64_t n = a.n_grid > 0 ? (int64_t) a.n_grid : (int64_t) a.p + 1;
    return (int) std::min<int64_t>(a.n_cap, std::max<int64_t>(n, 0));
}

/// The WMMA scorer's arguments for the mixer's token-scorer request (T = 1).
inline IdxScoreWmma wmma_args(const IdxScore& a) {
    IdxScoreWmma w;
    w.iq = a.iq;
    w.w = a.w;
    w.keys = (const void*) a.ik_cache;
    w.key_f16 = a.key_f16;
    w.idx_heads = a.idx_heads;
    w.idx_dim = a.idx_dim;
    w.n_kv = score_extent(a);
    w.T = 1;
    w.pos = a.p_dev;     // null: the last cell of the extent is the position
    w.score = a.score;
    w.ld = std::max(a.n_cap, w.n_kv);
    return w;
}

/// The fused kernel's arguments for the mixer's Mla2 (the -1 padded cells; n_sel_dev is not needed: padding cells
/// are skipped by the kernel).
inline FusedMla fused_args(const Mla2& a, int n_split) {
    FusedMla f;
    f.q = a.q;
    f.q_stride = a.q_stride;
    f.wk_b = a.wk_b;
    f.wv_b = a.wv_b;
    f.w_type = a.w_type;
    f.lat = a.lat;
    f.lat_q8 = a.lat_q8;
    f.k_pe = a.k_pe;
    f.cells = a.cells;
    f.n_sel = a.n_sel;
    f.n_head = a.n_head;
    f.qk_nope = a.qk_nope;
    f.qk_rope = a.qk_rope;
    f.kv_lora = a.kv_lora;
    f.v_head = a.v_head;
    f.scale = a.scale;
    f.n_split = n_split;
    f.work = a.work;
    f.work_bytes = a.work_bytes;
    f.out = a.out;
    f.out_q8_1 = a.out_q8_1;
    return f;
}

// ---- the GPU dispatchers (glmd_mixer.cu; device pointers).  p is the token's position (host): the decisions use
// n = p + 1, the launches read the position on the device (graph-replay safe).
/// The scorer: idx_score_wmma or idx_score.  With the WMMA kernel a non-null n_sel_out is written by one extra
/// one-thread launch (the engine passes null: its cell count lives in the device DevPos).
bool decode_score(const DecodePolicy& pol, const IdxScore& a, int64_t p, void* stream);
/// The selector: topk_mb (work: topk_mb_work_bytes(1), zeroed once), topk_hist (B14.15) or topk.  n_kv: the launch
/// extent.
bool decode_topk(const DecodePolicy& pol, const float* score, int ld, int n_kv, const int* pos, int64_t p, int top_k,
                 int* sel, void* work, void* stream);
/// The attention: mla_fused or mla2.
bool decode_mla(const DecodePolicy& pol, const Mla2& a, void* stream);

// ---- their CPU twins (host pointers; the same decisions, the twin kernels)
namespace twin {

inline void decode_score(const DecodePolicy& pol, const IdxScore& a, int64_t p) {
    if (use_idx_wmma(pol, p + 1, a.skip_le, a.idx_heads, a.idx_dim)) {
        vtwin::idx_score_wmma(wmma_args(a));
        if (a.n_sel_out) {
            const int n = (int) std::min<int64_t>(a.n_cap, (a.p_dev ? *a.p_dev : a.p) + 1);
            *a.n_sel_out = std::min(n, a.sel_cap);
        }
        return;
    }
    idx_score(a);
}
inline void decode_topk(const DecodePolicy& pol, const float* score, int ld, int n_kv, const int* pos, int64_t p,
                        int top_k, int* sel) {
    if (use_topk_mb(pol, p + 1, top_k))
        vtwin::topk_mb(score, ld, n_kv, 1, pos, top_k, sel);
    else if (use_topk_hist(pol, p + 1, top_k))
        topk_hist(score, ld, n_kv, 1, pos, top_k, sel);
    else
        topk(score, ld, n_kv, 1, pos, top_k, sel);
}
inline void decode_mla(const DecodePolicy& pol, const Mla2& a) {
    if (use_fused_mla(pol, a.n_head, a.qk_nope, a.qk_rope, a.kv_lora, a.v_head, a.w_type)) {
        vtwin::mla_fused(fused_args(a, pol.fused_split));
        return;
    }
    mla2(a);
}

}  // namespace twin

}  // namespace glmd
