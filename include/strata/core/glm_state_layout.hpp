// include/strata/core/glm_state_layout.hpp - the glm-dsa port B4.1 / B4.2: the per-layer state layout descriptor.
//
// A part's state arena (Glm5Model::state_, one cudaMalloc of floats) holds the residual pair, then the NextN block's
// caches (when the part carries it), then each of its layers' caches.  Maya laid it out by hand in five places (load,
// load_pack, the split search's state price, the multi-sequence layout, the slot run builder); they all ask this
// descriptor now, so one function says what a layer keeps:
//
//   glm5-next (Flash), byte for byte Maya's layout:
//     KDA layer  S [d_inner * kda_head_dim] F32, conv [3 * d_inner * (d_conv - 1)] F32;
//     DSA layer  latent [ctx] records (F32 kv_lora | FP16 kv_lora | INT8 lat8_rec_bytes), indexer key and gate rings
//                [ik_ring][idx_key] F32 each, pooled keys [ctx / kpool][idx_key] F32.
//   glm-dsa (GLM-5.3 744B), every trunk layer and the NextN block are DSA layers:
//     latent [ctx] records in Maya's format unchanged (FP16 1024 B / INT8 544 B at kv_lora 512),
//     k_pe [ctx][qk_rope] FP16 (128 B a position, kept apart so the rope tail is never INT8-quantized),
//     index keys [ctx][idx_key] F32 on full-indexer layers and the NextN block only (cell s holds position s);
//     no ring, no gate cache, no pools.  The reference path (SlowF32) keeps one F32 row [c_kv | k_pe] (kv_row) a
//     position in the latent region instead, the layout glmd::ref_mla_attention reads; dsa_kpe is then kNone.
//
// Offsets are in floats from the arena start (state_ + off).  A region a layer does not have is 0 on Flash (Maya's
// convention, the per-layer vectors start zeroed) and kNone (-1) on glm-dsa, so a stray use faults instead of
// silently reading the residual.
#pragma once

#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_mtp_state.hpp"   // B15.3: the NextN carrier (Glm5Model members use its types)

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

enum class GlmStateMode : int {
    SlowF32 = 0,   // the reference path (STRATA_GLM_SLOW, load()): F32 latent
    Fp16 = 1,      // the fast path: FP16 latent rows
    Int8 = 2,      // the fast path with STRATA_GLM_KV_INT8: INT8 latent records (glm_fast.hpp lat8_rec_bytes)
};

struct GlmStateLayout {
    static constexpr int64_t kNone = -1;

    GlmArch arch = GlmArch::Glm5Next;
    GlmStateMode mode = GlmStateMode::SlowF32;
    int n_layers = 0, l0 = 0, l1 = 0, mtp_il = -1;
    int64_t max_ctx = 0;
    int64_t lat_rows = 0;       // L1-6: latent / k_pe rows a DSA layer holds (KV streaming: the VRAM slots; else max_ctx)
    int ik_ring = 0;            // positions of the indexer key / gate rings (glm-dsa: max_ctx, no ring)
    int64_t max_pools = 0;      // pooled-key rows (Flash ctx / kpool; glm-dsa 0)
    int idx_key = 0, idx_kpool = 1;

    // per-position strides in floats (a 4-byte unit; FP16 and INT8 records are whole floats wide)
    int64_t lat_pp = 0;         // latent record
    int64_t kpe_pp = 0;         // k_pe (glm-dsa fast path), 0 when absent or inline
    int64_t ik_pp = 0;          // one index key (idx_key)
    int ik_bytes = 4;           // bytes per index-key element: 4 (F32), 2 on the glm-dsa fast path with FP16 keys (B10.8)
    int ik_had = 0;             // 1: the index keys (and the queries scored against them) are Walsh-Hadamard rotated
                                // (B14.11, STRATA_GLM_IDX_HADAMARD on the glm-dsa fast path; idx_key a power of two)
    int64_t kda_s_floats = 0, kda_conv_floats = 0;   // one KDA layer's recurrent state and conv history
    int64_t residual_floats = 0;                     // the residual and its ping-pong at the arena's start

    // per layer, n_layers + 1 entries (the last is the NextN block's when mtp_il == n_layers)
    std::vector<int64_t> kda_S, kda_conv, dsa_lat, dsa_kpe, dsa_ik, dsa_ig, dsa_pool;
    std::vector<uint8_t> carried;   // 1 = this arena holds the layer's caches (l0 <= il < l1, or the NextN block)
    // what each layer of the model keeps (carried here or not): KDA, DSA, or glm-dsa DSA with its own index keys
    enum : uint8_t { kKda = 0, kDsa = 1, kDsaFull = 2 };
    std::vector<uint8_t> kind;

    int64_t floats = 0;
    uint64_t bytes = 0;

    bool glm_dsa() const { return arch == GlmArch::GlmDsa; }
    bool holds(int il) const { return il >= 0 && il < (int) carried.size() && carried[(size_t) il] != 0; }
    bool is_dsa(int il) const;          // a carried DSA layer
    bool has_kpe(int il) const { return holds(il) && dsa_kpe[(size_t) il] != kNone && kpe_pp > 0; }
    bool has_index_cache(int il) const; // keeps indexer keys (Flash: every DSA layer; glm-dsa: full layers + NextN)
    bool pooled() const { return !glm_dsa(); }

    /// One KDA layer's snapshot run in floats (state + conv history, contiguous).
    int64_t kda_run() const { return kda_s_floats + kda_conv_floats; }
    /// The open pool's rows per indexer cache and DSA layer a snapshot keeps (Flash (kpool - 1) * idx_key; 0 on a
    /// pool-less layout, which then needs no snap_pool_).
    int64_t open_rows() const { return pooled() ? (int64_t) (idx_kpool - 1) * idx_key : 0; }
    /// The DSA layers this arena carries, NextN block first (Glm5Model::dsa_layers order).
    std::vector<int> dsa_layers() const;
    /// floats of the snapshot's open-pool buffer: every carried DSA layer's two caches (ik, ig).
    int64_t snap_pool_floats() const { return (int64_t) dsa_layers().size() * 2 * open_rows(); }
    /// The state one layer keeps at this layout's context (any layer of the model, carried here or not).
    uint64_t bytes_per_layer(int il) const;
    /// The arena less the residual pair: what the per-layer formula sums to.
    uint64_t cache_bytes() const { return bytes - (uint64_t) residual_floats * 4u; }
    /// (state offset, floats) of each cache run a conversation slot of n positions holds, in file order.  Flash
    /// (MAYALST2): per DSA layer the latents, then the pooled keys.  glm-dsa (MAYALST3): per DSA layer latents and
    /// k_pe, then the index keys of the full layers.  NextN block first in both passes.
    std::vector<std::pair<int64_t, int64_t>> slot_runs(int64_t n) const;
    /// A one-line summary for the load log (bytes, and on glm-dsa the per-position formula).
    std::string describe() const;
};

/// Lays out one part's state arena: layers [l0, l1) of g, plus the NextN block at mtp_il (>= 0) placed first, for
/// max_ctx positions.  ik_ring: Flash's indexer ring length (the fast path's 8192 + 64 or max_ctx); ignored on
/// glm-dsa.  false + err for a geometry the layout cannot serve (glm-dsa: an INT8 record not 16-byte aligned, an odd
/// rope tail, a missing indexer schedule).
/// lat_rows: the latent rows a DSA layer holds in the arena (KV streaming: kv_resident_cells; <= 0: max_ctx).
bool glm_state_carve(const Glm5Geometry& g, int l0, int l1, int mtp_il, int64_t max_ctx, GlmStateMode mode,
                     int ik_ring, GlmStateLayout& out, std::string& err, int64_t lat_rows = -1);

/// The fast path's indexer ring for Flash (load_pack, seq_layout): the prompt path's largest sub-batch + 64.
inline int glm_state_fast_ik_ring(int64_t max_ctx) { return (int) (max_ctx < 8192 + 64 ? max_ctx : 8192 + 64); }

/// The slot fingerprint (MAYALST3): what a saved state must agree on to be read back by this model.
/// {format (| kGlmFpRandomRoute under random routing), arch, n_layers, n_embd, kv_lora, qk_rope, idx_key, n_full, latent mode, k_pe bytes a position,
///  index-key bytes an element (| kGlmFpIdxHad when rotated, B14.11), latent row bytes a position (B14.5: dtype and
///  row size of every cache a slot holds), the NextN block's identity (B15.3, glmmtp::block_id; 0 without one), the
///  carrier row bytes of the MTP1 slot trailer (B15.3; 0 without a NextN block)}.
/// This function leaves words 12 and 13 at 0: the NextN block lives on the chain's last part, so
/// Glm5Model::state_fp() (src/core/glm_mtp_state_model.cu) fills them for the whole chain; slots and the prompt cache
/// use that.
/// B10.8: STRATA_GLM_IDX_FP16 set to a non-zero value (read on every call): the glm-dsa fast-path carve (FP16 /
/// INT8 modes) stores the index keys as FP16 (ik_pp = idx_key / 2, ik_bytes 2) and the split planner prices 2 B per
/// element.  The slow F32 reference path keeps F32 keys.  The forward reads GlmStateLayout::ik_bytes, never the env.
bool glm_state_idx_fp16();
/// B14.11: STRATA_GLM_IDX_HADAMARD set to a non-zero value (read on every call): the glm-dsa fast-path carve (FP16 /
/// INT8 modes) marks the index keys rotated (GlmStateLayout::ik_had = 1).  dsa_prep / dsa_rope_q (token and prompt
/// paths, the NextN block too) then rotate keys and queries, and the slot fingerprint carries kGlmFpIdxHad in its
/// index-key word, so a slot (or a prompt-cache entry) written with one setting is refused by the other.  The slow F32
/// reference path never rotates.  New code reads GlmStateLayout::ik_had, never the env.
bool glm_state_idx_hadamard();
/// The fingerprint bit (word 10, OR-ed into the index-key element bytes) of a rotated key cache.
constexpr int32_t kGlmFpIdxHad = 0x100;
/// The fingerprint bit (word 0, OR-ed into the format) of a glm-dsa state computed with STRATA_GLM_RANDOM_ROUTING=1
/// (B15.7 bench mode): such slots and prompt-cache entries are refused by a normal load, and the reverse.
constexpr int32_t kGlmFpRandomRoute = 0x10000;
/// Word 0 of the fingerprint: its format, bumped on EVERY change of the fingerprint's words or of what a v3 slot holds,
/// so an older file is refused by its version and never by its bytes happening to differ.  History:
///   1  B4.2 to B14.5: words 0-11 (12 words);
///   2  B15.3: 14 words (12: the NextN block id, 13: the MTP1 carrier row bytes) and the MTP1 trailer after the caches.
constexpr int32_t kGlmFpFormat = 2;
using GlmStateFingerprint = std::array<int32_t, 14>;
GlmStateFingerprint glm_state_fingerprint(const Glm5Geometry& g, const GlmStateLayout& lay);

/// Slot file magics: "MAYALST2" (Flash, Maya's format, written unchanged) and "MAYALST3" (a fingerprint follows the
/// magic; glm-dsa writes it, both archs read it).
constexpr uint64_t kGlmSlotMagicV2 = 0x3254534C4159414Dull;
constexpr uint64_t kGlmSlotMagicV3 = 0x3354534C4159414Dull;
/// The magic this layout's slots are written with (Flash v2, glm-dsa v3).
inline uint64_t glm_slot_magic(const GlmStateLayout& lay) { return lay.glm_dsa() ? kGlmSlotMagicV3 : kGlmSlotMagicV2; }

}  // namespace strata::core
