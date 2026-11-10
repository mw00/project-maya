// include/strata/kernels/glmd_graph.hpp - glm-dsa B10.1 / B14.4: CUDA-graph rules for the glm-dsa decode part.
//
// A part (the layers l0..l1 of one GPU) is captured once and replayed every token while its GraphKey holds.  The
// B14.4 replay-safety rules, which every kernel launched inside a captured part must follow:
//   1. nothing that changes per token is a kernel ARGUMENT (arguments are frozen at capture): the position, n_kv
//      and n_sel live in a device DevPos written by set_dev_pos() before each replay (one launch, outside the
//      graph); the glmd kernels read it through their p_dev / pos pointers (DsaPrep::p_dev, DsaRopeQ::p_dev,
//      IdxScoreWmma::pos = &dev->p, topk / topk_mb pos = &dev->p);
//   2. grids are fixed: the scorer and the selector run over n_cap = graph_n_cap(p, max_ctx) cells (the cells past
//      p are -inf / not selected on the device), mla2 / mla_fused over n_sel = top_k with the selector's -1
//      padding (identity while p + 1 <= top_k);
//   3. every kernel choice that depends on a switch or on n (fused MLA, WMMA scorer, FP16 keys, radix top-k, INT8
//      latents, the mv variants) is part of the key: graph_kernel_bits() + graph_regime(), and n_cap too, so a
//      token that crosses a regime or capacity boundary re-captures (a handful of times per session);
//   4. the latent / k_pe / index-key write positions come from the device position (dsa_prep's p_dev), so a
//      replayed graph writes row p + 1 on the next token, never the captured p (the stale-graph bug of
//      ik_llama.cpp 9da563d0);
//   5. fixed addresses: every buffer a captured part touches is allocated before the capture and never moved
//      (the arena, the state, the DevPos block); the key carries the arena and residual base addresses;
//   6. (B15.6) every (re)allocation of a buffer a captured part may touch (the DSA arena with its score buffer, the
//      mixer's scratch block, the MoE activation / staging arena, the NextN rows) bumps the process-wide graph
//      generation (graph_generation_bump / graph_generation_note), and the key carries it: a graph captured before
//      a reallocation never replays over freed or re-carved memory, even when cudaMalloc hands back the same base
//      address (llama.cpp #26771: ggml_cuda_pool_leg::invalidate_cuda_graphs on free / realloc).
// STRATA_GLM_DSA_GRAPHS=1 enables the decode graphs (default off until validated on the test PC).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glmd {

/// The device-resident per-token values (16 B, one per part's device).
struct DevPos {
    int p = 0;       // the token's position
    int n_kv = 0;    // p + 1: the visible cells
    int n_sel = 0;   // min(p + 1, top_k): the cells the selector keeps
    int pad = 0;
};
static_assert(sizeof(DevPos) == 16, "DevPos is one 16-byte store");

/// One tiny launch (OUTSIDE any capture): *dev = {p, p + 1, min(p + 1, top_k), 0}.
bool set_dev_pos(DevPos* dev, int p, int top_k, void* stream);
/// STRATA_GLM_DSA_GRAPHS=1 (read once).
bool dsa_graphs_enabled();

/// Kernel-choice bits of a captured part.
enum : uint32_t {
    kGkFusedMla = 1u,
    kGkIdxWmma = 2u,
    kGkIdxFp16 = 4u,
    kGkTopkMb = 8u,
    kGkLatInt8 = 16u,
    kGkMvNw2 = 32u,
    kGkMvTyped = 64u,
    kGkTopkHist = 128u,   // B14.15 (glmd::policy_bits ORs it in)
    kGkIdxHad = 256u,     // B14.11 rotated index keys (glmd::policy_bits ORs it in)
};
constexpr uint32_t graph_kernel_bits(bool fused_mla, bool idx_wmma, bool idx_fp16, bool topk_mb, bool lat_int8,
                                     bool mv_nw2, bool mv_typed) {
    return (fused_mla ? kGkFusedMla : 0u) | (idx_wmma ? kGkIdxWmma : 0u) | (idx_fp16 ? kGkIdxFp16 : 0u) |
           (topk_mb ? kGkTopkMb : 0u) | (lat_int8 ? kGkLatInt8 : 0u) | (mv_nw2 ? kGkMvNw2 : 0u) |
           (mv_typed ? kGkMvTyped : 0u);
}

/// The regime of a token at position p: bit 0 selection active (p + 1 > top_k), bit 1 the WMMA scorer's range
/// (idx_wmma and p + 1 > wmma_min), bit 2 the radix top-k's range (topk_mb and p + 1 > mb_min).  A captured part
/// is valid for one regime only.
constexpr int graph_regime(int64_t p, int top_k, bool idx_wmma, int wmma_min, bool topk_mb, int mb_min) {
    return (p + 1 > top_k ? 1 : 0) | (idx_wmma && p + 1 > wmma_min ? 2 : 0) | (topk_mb && p + 1 > mb_min ? 4 : 0);
}

/// The fixed cell capacity of the scorer / selector grids for a token at position p: the next power of two
/// >= p + 1, at least 4096, at most max_ctx.
constexpr int64_t graph_n_cap(int64_t p, int64_t max_ctx) {
    int64_t c = 4096;
    while (c < p + 1) c <<= 1;
    return c < max_ctx ? c : max_ctx;
}

/// B15.6: the process-wide graph generation (starts at 1).  A captured graph is valid for one generation only.
uint64_t graph_generation();
/// Bumps the generation (every graph re-captures before its next replay) and returns the new value; `why` names the
/// buffer for the counters (graph_generation_bumps).  Thread-safe.
uint64_t graph_generation_bump(const char* why);
/// Bumps only when the buffer `name` is new or its (base, bytes) differ from the previous note of that name (a
/// setup that runs again with the same allocation does not force a re-capture).  Returns the generation.
uint64_t graph_generation_note(const char* name, const void* base, size_t bytes);
/// How many bumps `why` / `name` caused so far (0 for an unknown name); all of them with nullptr.
uint64_t graph_generation_bumps(const char* why);

/// Everything a captured part depends on besides device data.
struct GraphKey {
    int part = 0, l0 = 0, l1 = 0;
    unsigned long long cpu_plan = 0;   // the MoE CPU-lane plan (glm_layer_graphs.hpp's key)
    const void* residual = nullptr;    // fixed addresses
    const void* arena = nullptr;
    const void* state = nullptr;
    uint32_t bits = 0;                 // graph_kernel_bits()
    int regime = 0;                    // graph_regime()
    int64_t n_cap = 0;                 // graph_n_cap()
    uint64_t gen = 0;                  // B15.6: graph_generation() at capture
    bool abl = false;                  // --ablate: the bound conversation projects (its launches are in the graph)
    bool operator==(const GraphKey& b) const {
        return part == b.part && l0 == b.l0 && l1 == b.l1 && cpu_plan == b.cpu_plan && residual == b.residual &&
               arena == b.arena && state == b.state && bits == b.bits && regime == b.regime && n_cap == b.n_cap &&
               gen == b.gen && abl == b.abl;
    }
    bool operator!=(const GraphKey& b) const { return !(*this == b); }
};

/// The decode-graph cache's bookkeeping (src/core/glm_dsa_graphs.hpp DsaGraphs; CUDA-free, so the CPU test runs the
/// same code).  A slot per (part, state): with several sequences at once (STRATA_GLM_SEQS / --parallel) every sequence
/// binds its own state arena (Glm5Model::seq_bind), so each keeps its own graph and a switch between sequences
/// replays instead of re-capturing.  At most per_part slots per part (set_per_part: the sequence count), the least
/// recently used dropped beyond that (an eviction).  A key that differs from its slot's - same part and state, but a
/// regime, a capacity, a switch, a moved arena or a newer generation - drops that slot (an invalidation) and
/// captures anew, as before.
class GraphIndex {
public:
    struct Pick {
        bool replay = false;       // the slot's graph matches: replay it
        int slot = -1;             // the graph to replay, or the slot to capture into
        int drop = -1;             // a slot whose graph is destroyed first (-1: none)
        bool invalidated = false;  // ... because its key changed
        bool evicted = false;      // ... because the part had per_part slots
    };
    void set_per_part(int n) { per_part_ = n < 1 ? 1 : n; }
    int per_part() const { return per_part_; }
    Pick pick(const GraphKey& k) {
        Pick r;
        int same = -1, n_part = 0, lru = -1;
        for (int i = 0; i < (int) s_.size(); ++i) {
            const Slot& e = s_[(size_t) i];
            if (!e.live || e.key.part != k.part) continue;
            ++n_part;
            if (e.key.state == k.state) same = i;
            if (lru < 0 || e.used < s_[(size_t) lru].used) lru = i;
        }
        if (same >= 0 && s_[(size_t) same].key == k) {
            s_[(size_t) same].used = ++clock_;
            r.replay = true;
            r.slot = same;
            return r;
        }
        if (same >= 0) {
            r.drop = r.slot = same;
            r.invalidated = true;
        } else if (n_part >= per_part_) {
            r.drop = r.slot = lru;
            r.evicted = true;
        } else {
            for (int i = 0; i < (int) s_.size() && r.slot < 0; ++i)
                if (!s_[(size_t) i].live) r.slot = i;
            if (r.slot < 0) {
                r.slot = (int) s_.size();
                s_.push_back(Slot{});
            }
        }
        s_[(size_t) r.slot].live = false;   // until commit(): a failed capture leaves the slot free
        return r;
    }
    /// A capture into `slot` succeeded.
    void commit(int slot, const GraphKey& k) {
        Slot& e = s_[(size_t) slot];
        e.live = true;
        e.key = k;
        e.used = ++clock_;
    }
    /// The most recently used live slot of a part (-1: none).
    int latest(int part) const {
        int best = -1;
        for (int i = 0; i < (int) s_.size(); ++i)
            if (s_[(size_t) i].live && s_[(size_t) i].key.part == part &&
                (best < 0 || s_[(size_t) i].used > s_[(size_t) best].used))
                best = i;
        return best;
    }
    int live() const {
        int n = 0;
        for (const Slot& e : s_) n += e.live ? 1 : 0;
        return n;
    }
    void clear() { s_.clear(); }

private:
    struct Slot {
        bool live = false;
        GraphKey key;
        uint64_t used = 0;
    };
    std::vector<Slot> s_;
    uint64_t clock_ = 0;
    int per_part_ = 1;
};

}  // namespace glmd
