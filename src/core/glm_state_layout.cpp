// src/core/glm_state_layout.cpp - the glm-dsa port B4.1 / B4.2: the per-layer state layout descriptor (see
// include/strata/core/glm_state_layout.hpp).  The Flash branch is Maya's carvers' arithmetic in their order; the
// glm-dsa branch is the latent + k_pe + full-layer index-key layout of BUILD-PLAN B4.1.
#include "strata/core/glm_env.hpp"
#include "strata/core/glm_state_layout.hpp"

#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glmd_random_route.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {

int64_t lat_floats_per_pos(const Glm5Geometry& g, GlmStateMode mode) {
    const bool dsa = g.arch == GlmArch::GlmDsa;
    switch (mode) {
    case GlmStateMode::Int8: return (int64_t) strata::kernels::glmf::lat8_rec_bytes(g.kv_lora) / 4;
    case GlmStateMode::Fp16: return (int64_t) g.kv_lora / 2;
    case GlmStateMode::SlowF32: break;
    }
    return dsa ? (int64_t) g.kv_row() : (int64_t) g.kv_lora;   // glm-dsa reference: one F32 [c_kv | k_pe] row
}

}  // namespace

bool GlmStateLayout::is_dsa(int il) const { return holds(il) && kind[(size_t) il] != kKda; }

bool GlmStateLayout::has_index_cache(int il) const {
    return is_dsa(il) && (!glm_dsa() || kind[(size_t) il] == kDsaFull);
}

std::vector<int> GlmStateLayout::dsa_layers() const {
    std::vector<int> r;
    if (mtp_il >= 0 && holds(mtp_il)) r.push_back(mtp_il);
    for (int il = l0; il < l1; ++il)
        if (is_dsa(il)) r.push_back(il);
    return r;
}

uint64_t GlmStateLayout::bytes_per_layer(int il) const {
    if (il < 0 || il > n_layers) return 0;
    if (kind.empty()) return 0;
    const uint8_t k = kind[(size_t) il];
    if (k == kKda) return (uint64_t) kda_run() * 4u;
    int64_t f = lat_pp * max_ctx + kpe_pp * max_ctx;
    if (glm_dsa()) {
        if (k == kDsaFull) f += ik_pp * max_ctx;
    } else {
        f += 2 * ik_pp * ik_ring + ik_pp * max_pools;
    }
    return (uint64_t) f * 4u;
}

std::vector<std::pair<int64_t, int64_t>> GlmStateLayout::slot_runs(int64_t n) const {
    std::vector<std::pair<int64_t, int64_t>> r;
    const std::vector<int> ls = dsa_layers();
    if (!glm_dsa()) {
        // Maya's MAYALST2 runs: the latents and the completed pools (the ik / ig rings' open pool goes via snap_pool_)
        const int64_t pools = (n + idx_kpool - 1) / idx_kpool;
        for (int il : ls) {
            r.push_back({dsa_lat[(size_t) il], n * lat_pp});
            r.push_back({dsa_pool[(size_t) il], pools * idx_key});
        }
        return r;
    }
    for (int il : ls) {
        r.push_back({dsa_lat[(size_t) il], n * lat_pp});
        if (has_kpe(il)) r.push_back({dsa_kpe[(size_t) il], n * kpe_pp});
    }
    for (int il : ls)
        if (has_index_cache(il)) r.push_back({dsa_ik[(size_t) il], n * ik_pp});
    return r;
}

std::string GlmStateLayout::describe() const {
    char buf[512];
    if (!glm_dsa()) {
        std::snprintf(buf, sizeof(buf), "state %llu bytes (%d DSA layers, ctx %lld)", (unsigned long long) bytes,
                      (int) dsa_layers().size(), (long long) max_ctx);
        return buf;
    }
    int n_full = 0;
    const std::vector<int> ls = dsa_layers();
    for (int il : ls) n_full += has_index_cache(il);
    std::snprintf(buf, sizeof(buf),
                  "state %llu bytes = residual %lld + caches %llu (%d layers x %lld B x %lld positions (latent%s) + "
                  "%d index-key layers x %lld B x %lld positions%s)",
                  (unsigned long long) bytes, (long long) residual_floats * 4, (unsigned long long) cache_bytes(),
                  (int) ls.size(),
                  (long long) (lat_pp + kpe_pp) * 4, (long long) (lat_rows > 0 ? lat_rows : max_ctx),
                  mode == GlmStateMode::SlowF32 ? " F32 [c_kv|k_pe]" : mode == GlmStateMode::Int8 ? " INT8 + k_pe FP16"
                                                                                                  : " FP16 + k_pe FP16",
                  n_full, (long long) ik_pp * 4, (long long) max_ctx, ik_had ? ", Hadamard-rotated" : "");
    return buf;
}

bool glm_state_carve(const Glm5Geometry& g, int l0, int l1, int mtp_il, int64_t max_ctx, GlmStateMode mode,
                     int ik_ring, GlmStateLayout& out, std::string& err, int64_t lat_rows) {
    GlmStateLayout L;
    const bool dsa = g.arch == GlmArch::GlmDsa;
    // KV streaming (--kv-resident N, Maya): the arena holds lat_rows positions of latent slots a layer, the rest
    // lives in pinned RAM (kv_stream_setup).  L1-6 glm-dsa: the latent AND k_pe hold lat_rows slots (one residency map
    // for both), the index keys stay whole (the indexer scores every position)
    if (lat_rows <= 0 || lat_rows > max_ctx) lat_rows = max_ctx;
    if (dsa && lat_rows != max_ctx && mode == GlmStateMode::SlowF32) {
        err = "glm state layout: KV streaming (--kv-resident) needs the fast path on glm-dsa";
        return false;
    }
    L.arch = g.arch;
    L.mode = mode;
    L.n_layers = g.n_layers;
    L.l0 = l0;
    L.l1 = l1;
    L.mtp_il = mtp_il;
    L.max_ctx = max_ctx;
    L.lat_rows = lat_rows;
    L.idx_key = g.idx_key;
    L.idx_kpool = g.idx_kpool;
    if (max_ctx <= 0 || l0 < 0 || l1 > g.n_layers || l0 > l1 || mtp_il > g.n_layers ||
        (int) g.n_head_kv.size() < g.n_layers) {
        err = "glm state layout: bad range [" + std::to_string(l0) + ", " + std::to_string(l1) + ") / ctx " +
              std::to_string(max_ctx);
        return false;
    }
    if (dsa) {
        if (g.qk_rope <= 0 || g.qk_rope % 2 != 0) {
            err = "glm state layout: glm-dsa needs an even rope tail (qk_rope " + std::to_string(g.qk_rope) + ")";
            return false;
        }
        if ((int) g.indexer_full.size() < g.n_layers) {
            err = "glm state layout: glm-dsa needs the indexer schedule (" + std::to_string(g.indexer_full.size()) +
                  " of " + std::to_string(g.n_layers) + " layers)";
            return false;
        }
        if (mode == GlmStateMode::Int8 && strata::kernels::glmf::lat8_rec_bytes(g.kv_lora) % 16 != 0) {
            err = "glm state layout: the INT8 latent record (" +
                  std::to_string(strata::kernels::glmf::lat8_rec_bytes(g.kv_lora)) + " B) is not 16-byte aligned";
            return false;
        }
        L.ik_ring = (int) max_ctx;   // no ring: cell s holds position s
        L.max_pools = 0;
        L.kpe_pp = mode == GlmStateMode::SlowF32 ? 0 : g.qk_rope / 2;   // FP16; inline in the F32 reference row
    } else {
        L.ik_ring = ik_ring;
        L.max_pools = max_ctx / g.idx_kpool;
        L.kpe_pp = 0;
    }
    L.lat_pp = lat_floats_per_pos(g, mode);
    L.ik_pp = g.idx_key;
    if (dsa && mode != GlmStateMode::SlowF32 && glm_state_idx_fp16()) {   // B10.8: FP16 index keys
        if (g.idx_key % 8 != 0) {
            err = "glm state layout: FP16 index keys need idx_key % 8 == 0 (16-byte rows)";
            return false;
        }
        L.ik_pp = g.idx_key / 2;
        L.ik_bytes = 2;
    }
    if (dsa && mode != GlmStateMode::SlowF32 && glm_state_idx_hadamard()) {   // B14.11: rotated index keys
        if (g.idx_key < 2 || g.idx_key > 512 || (g.idx_key & (g.idx_key - 1)) != 0) {
            err = "glm state layout: STRATA_GLM_IDX_HADAMARD needs idx_key a power of two <= 512 (have " +
                  std::to_string(g.idx_key) + ")";
            return false;
        }
        L.ik_had = 1;
    }
    L.kda_s_floats = (int64_t) g.d_inner() * g.kda_head_dim;
    L.kda_conv_floats = (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
    L.residual_floats = 2 * (int64_t) g.hc * g.n_embd;   // R and its hc_post ping-pong (glm-dsa: hc 1)

    const size_t nv = (size_t) g.n_layers + 1;
    const int64_t absent = dsa ? GlmStateLayout::kNone : 0;
    L.kda_S.assign(nv, 0);
    L.kda_conv.assign(nv, 0);
    L.dsa_lat.assign(nv, absent);
    L.dsa_kpe.assign(nv, absent);
    L.dsa_ik.assign(nv, absent);
    L.dsa_ig.assign(nv, absent);
    L.dsa_pool.assign(nv, absent);
    L.carried.assign(nv, 0);
    L.kind.assign(nv, GlmStateLayout::kDsa);
    for (int il = 0; il <= g.n_layers; ++il) {
        if (il < g.n_layers && !dsa && g.is_recr(il)) L.kind[(size_t) il] = GlmStateLayout::kKda;
        else if (dsa && g.idx_full(il)) L.kind[(size_t) il] = GlmStateLayout::kDsaFull;
    }

    int64_t floats = L.residual_floats;
    bool aligned = true;
    const auto dsa_layer = [&](int il) {
        const size_t i = (size_t) il;
        L.carried[i] = 1;
        L.dsa_lat[i] = floats;
        if (floats % 4 != 0) aligned = false;   // 16 B: the INT8 record's and the FP16 rows' vector loads
        floats += L.lat_pp * lat_rows;
        if (dsa) {
            if (L.kpe_pp > 0) {
                L.dsa_kpe[i] = floats;
                floats += L.kpe_pp * lat_rows;
            }
            if (L.kind[i] == GlmStateLayout::kDsaFull) {
                if (floats % 4 != 0) aligned = false;
                L.dsa_ik[i] = floats;
                floats += L.ik_pp * max_ctx;
            }
        } else {
            L.dsa_ik[i] = floats;
            floats += (int64_t) g.idx_key * L.ik_ring;
            L.dsa_ig[i] = floats;
            floats += (int64_t) g.idx_key * L.ik_ring;
            L.dsa_pool[i] = floats;
            floats += (int64_t) g.idx_key * L.max_pools;
        }
    };
    if (mtp_il >= 0) dsa_layer(mtp_il);
    for (int il = l0; il < l1; ++il) {
        if (L.kind[(size_t) il] == GlmStateLayout::kKda) {
            L.carried[(size_t) il] = 1;
            L.kda_S[(size_t) il] = floats;
            floats += L.kda_s_floats;
            L.kda_conv[(size_t) il] = floats;
            floats += L.kda_conv_floats;
        } else {
            dsa_layer(il);
        }
    }
    if (dsa && !aligned) {
        err = "glm state layout: a glm-dsa cache region is not 16-byte aligned";
        return false;
    }
    L.floats = floats;
    L.bytes = (uint64_t) floats * sizeof(float);
    out = std::move(L);
    return true;
}

bool glm_state_idx_fp16() {
    return strata::glmenv::env_flag("STRATA_GLM_IDX_FP16", false);
}

bool glm_state_idx_hadamard() {
    return strata::glmenv::env_flag("STRATA_GLM_IDX_HADAMARD", false);
}

GlmStateFingerprint glm_state_fingerprint(const Glm5Geometry& g, const GlmStateLayout& lay) {
    int n_full = 0;
    for (int il = 0; il < g.n_layers; ++il) n_full += g.arch == GlmArch::GlmDsa ? (int) g.idx_full(il) : 0;
    // B15.7: a state computed under random routing is marked, so a normal load refuses it (and the bench run never
    // restores a real conversation's state)
    const int32_t format = kGlmFpFormat | (lay.glm_dsa() && glmd::random_routing_seed() != 0 ? kGlmFpRandomRoute : 0);
    return GlmStateFingerprint{format,
                               (int32_t) g.arch,
                               g.n_layers,
                               g.n_embd,
                               g.kv_lora,
                               g.qk_rope,
                               g.idx_key,
                               n_full,
                               (int32_t) lay.mode,
                               (int32_t) (lay.kpe_pp * 4),
                               lay.glm_dsa() ? lay.ik_bytes | (lay.ik_had ? kGlmFpIdxHad : 0) : 0,   // index-key element
                               // bytes: 4 F32, 2 FP16 (B10.8); | kGlmFpIdxHad for rotated keys (B14.11)
                               lay.glm_dsa() ? (int32_t) (lay.lat_pp * 4) : 0,   // B14.5: latent row bytes
                               0, 0};   // B15.3: NextN block id, carrier bytes (Glm5Model::state_fp fills them)
}

}  // namespace strata::core
