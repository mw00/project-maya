// tests/glm_dsa/glm_state_layout_test.cpp - glm-dsa B4.1 / B4.2: the state layout descriptor (CPU only).
//
// Checked:
//   1. Flash (glm5-next): every offset vector, the ring, the byte totals and the slot runs equal Maya's hand carvers,
//      re-implemented here from the pre-descriptor code (load(), load_pack() in F32 / FP16 / INT8, seq_layout(),
//      the split search's per-layer state price, slot_dsa_runs()), for one part, both halves of a split and with the
//      NextN block;
//   2. glm-dsa (GLM-5.3 744B geometry): per layer latent + FP16 k_pe, F32 index keys only on the 21 full layers + the
//      NextN block, no ring / gate / pools; the 32K total equals 79 x 1152 x 32768 + 22 x 512 x 32768 (+ the
//      residual pair); INT8 records 544 B and every region 16-byte aligned; open_rows() == 0 (no snap_pool_);
//      slot runs cover exactly the first n positions of each cache; the fingerprint differs from Flash's and between
//      FP16 and INT8; a 3-part split's arenas sum to the single-part caches.
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using strata::core::GlmArch;
using strata::core::GlmIdxKind;
using strata::core::GlmStateLayout;
using strata::core::GlmStateMode;
using strata::core::Glm5Geometry;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- the pre-descriptor carvers (Maya v1.0.34)
struct OldLayout {
    std::vector<int64_t> kda_S, kda_conv, dsa_lat, dsa_ik, dsa_ig, dsa_pool;
    int ik_ring = 0;
    uint64_t bytes = 0;
};

// load(): F32 everything, no NextN block, vectors of n_layers
OldLayout old_load(const Glm5Geometry& g, int64_t max_ctx) {
    OldLayout o;
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = max_ctx / g.idx_kpool;
    int64_t floats = 2 * hc_dim;
    for (auto* v : {&o.kda_S, &o.kda_conv, &o.dsa_lat, &o.dsa_ik, &o.dsa_ig, &o.dsa_pool}) v->assign(g.n_layers, 0);
    o.ik_ring = (int) max_ctx;
    for (int il = 0; il < g.n_layers; ++il) {
        if (g.is_recr(il)) {
            o.kda_S[il] = floats;
            floats += (int64_t) g.d_inner() * g.kda_head_dim;
            o.kda_conv[il] = floats;
            floats += (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        } else {
            o.dsa_lat[il] = floats;
            floats += (int64_t) g.kv_lora * max_ctx;
            o.dsa_ik[il] = floats;
            floats += (int64_t) g.idx_key * max_ctx;
            o.dsa_ig[il] = floats;
            floats += (int64_t) g.idx_key * max_ctx;
            o.dsa_pool[il] = floats;
            floats += (int64_t) g.idx_key * max_pools;
        }
    }
    o.bytes = (uint64_t) floats * 4;
    return o;
}

// load_pack() / seq_layout(): 0 F32 (slow), 1 FP16, 2 INT8
OldLayout old_pack(const Glm5Geometry& g, int64_t max_ctx, int l0, int l1, int mtp_il, int mode) {
    OldLayout o;
    const int64_t hc_dim = (int64_t) g.hc * g.n_embd;
    const int64_t max_pools = max_ctx / g.idx_kpool;
    int64_t floats = 2 * hc_dim;
    const int64_t rec8 = g.kv_lora + g.kv_lora / 16;   // the INT8 record: codes + an FP16 scale per 32
    const int64_t lat_floats = mode == 2 ? rec8 / 4 * max_ctx
                               : mode == 1 ? (int64_t) g.kv_lora * max_ctx / 2
                                           : (int64_t) g.kv_lora * max_ctx;
    for (auto* v : {&o.kda_S, &o.kda_conv, &o.dsa_lat, &o.dsa_ik, &o.dsa_ig, &o.dsa_pool})
        v->assign((size_t) g.n_layers + 1, 0);
    o.ik_ring = mode != 0 ? (int) std::min<int64_t>(max_ctx, 8192 + 64) : (int) max_ctx;
    const auto dsa = [&](int il) {
        o.dsa_lat[il] = floats;
        floats += lat_floats;
        o.dsa_ik[il] = floats;
        floats += (int64_t) g.idx_key * o.ik_ring;
        o.dsa_ig[il] = floats;
        floats += (int64_t) g.idx_key * o.ik_ring;
        o.dsa_pool[il] = floats;
        floats += (int64_t) g.idx_key * max_pools;
    };
    if (mtp_il >= 0) dsa(mtp_il);
    for (int il = 0; il < g.n_layers; ++il) {
        if (il < l0 || il >= l1) continue;
        if (g.is_recr(il)) {
            o.kda_S[il] = floats;
            floats += (int64_t) g.d_inner() * g.kda_head_dim;
            o.kda_conv[il] = floats;
            floats += (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        } else {
            dsa(il);
        }
    }
    o.bytes = (uint64_t) floats * 4;
    return o;
}

// the split search's per-layer state price (doubles, as glm_search_bounds had them)
double old_price(const Glm5Geometry& g, int64_t max_ctx, bool fast, int l) {
    const double max_pools = (double) (max_ctx / g.idx_kpool);
    const double lat = fast ? (double) g.kv_lora * (double) max_ctx / 2 : (double) g.kv_lora * (double) max_ctx;
    const double dsa_state = 4.0 * (lat + 2.0 * g.idx_key * (double) max_ctx + (double) g.idx_key * max_pools);
    const double kda_state = 4.0 * ((double) g.d_inner() * g.kda_head_dim + 3.0 * g.d_inner() * (g.d_conv - 1));
    return l < g.n_layers && g.is_recr(l) ? kda_state : dsa_state;
}

// slot_dsa_runs (MAYALST2)
std::vector<std::pair<int64_t, int64_t>> old_runs(const Glm5Geometry& g, int mode, int l0, int l1, int mtp_il,
                                                  const OldLayout& o, int64_t n) {
    std::vector<std::pair<int64_t, int64_t>> r;
    const int64_t lat_pp = mode == 2 ? (g.kv_lora + g.kv_lora / 16) / 4 : mode == 1 ? g.kv_lora / 2 : g.kv_lora;
    const int64_t pools = (n + g.idx_kpool - 1) / g.idx_kpool;
    const auto dsa = [&](size_t il) {
        r.push_back({o.dsa_lat[il], n * lat_pp});
        r.push_back({o.dsa_pool[il], pools * g.idx_key});
    };
    if (mtp_il >= 0) dsa((size_t) mtp_il);
    for (int il = l0; il < l1; ++il)
        if (!g.is_recr(il)) dsa((size_t) il);
    return r;
}

bool same(const OldLayout& o, const GlmStateLayout& L, bool sized_n_layers) {
    auto eq = [&](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        if (sized_n_layers) return a.size() + 1 == b.size() && std::equal(a.begin(), a.end(), b.begin()) && b.back() == 0;
        return a == b;
    };
    return eq(o.kda_S, L.kda_S) && eq(o.kda_conv, L.kda_conv) && eq(o.dsa_lat, L.dsa_lat) && eq(o.dsa_ik, L.dsa_ik) &&
           eq(o.dsa_ig, L.dsa_ig) && eq(o.dsa_pool, L.dsa_pool) && o.ik_ring == L.ik_ring && o.bytes == L.bytes &&
           std::all_of(L.dsa_kpe.begin(), L.dsa_kpe.end(), [](int64_t v) { return v == 0; });
}

Glm5Geometry flash() {
    Glm5Geometry g;   // Maya's defaults: GLM-5.3-Flash
    g.n_head_kv.assign((size_t) g.n_layers, 0);
    for (int il = 0; il < g.n_layers; ++il)
        if (il % 4 == 3 || il == g.n_layers - 1) g.n_head_kv[(size_t) il] = 1;   // a KDA / DSA interleave
    g.indexer_full.assign((size_t) g.n_layers, 1);
    return g;
}

Glm5Geometry glm53() {
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_embd = 6144;
    g.n_layers = 78;
    g.nextn = 1;
    g.n_head = 64;
    g.n_head_kv.assign(78, 1);
    g.q_lora = 2048;
    g.kv_lora = 512;
    g.qk_nope = 192;
    g.qk_rope = 64;
    g.v_head = 256;
    g.idx_heads = 32;
    g.idx_key = 128;
    g.idx_top_k = 2048;
    g.idx_kpool = 1;
    g.idx_select_tail = 0;
    g.idx_rope = 64;
    g.idx_kind = GlmIdxKind::Token;
    g.hc = 1;
    g.indexer_full.assign(79, 0);
    for (int il = 0; il < 79; ++il) g.indexer_full[(size_t) il] = il < 2 || (il - 2) % 4 == 0 || il == 78;
    g.n_expert = 256;
    g.dense_lead = 3;
    return g;
}

}  // namespace

int main() {
    std::string err;
    // ---------------------------------------------------------------- 1. Flash, against the old carvers
    {
        const Glm5Geometry g = flash();
        for (int64_t ctx : {(int64_t) 4096, (int64_t) 8192 + 64, (int64_t) 131072}) {
            const std::string at = " (ctx " + std::to_string(ctx) + ")";
            GlmStateLayout L;
            bool ok = strata::core::glm_state_carve(g, 0, g.n_layers, -1, ctx, GlmStateMode::SlowF32, (int) ctx, L, err);
            check(ok && same(old_load(g, ctx), L, true), "Flash load(): offsets, ring and bytes equal Maya's" + at);
            const int split = 22;
            for (int mode = 0; mode < 3; ++mode) {
                const GlmStateMode m = (GlmStateMode) mode;
                const int ring = mode ? strata::core::glm_state_fast_ik_ring(ctx) : (int) ctx;
                const struct { int l0, l1, mtp; } parts[] = {{0, g.n_layers, -1}, {0, g.n_layers, g.n_layers},
                                                             {0, split, -1}, {split, g.n_layers, g.n_layers}};
                for (const auto& p : parts) {
                    const std::string what = "Flash load_pack mode " + std::to_string(mode) + " [" +
                                             std::to_string(p.l0) + "," + std::to_string(p.l1) + ") mtp " +
                                             std::to_string(p.mtp) + at;
                    ok = strata::core::glm_state_carve(g, p.l0, p.l1, p.mtp, ctx, m, ring, L, err);
                    const OldLayout o = old_pack(g, ctx, p.l0, p.l1, p.mtp, mode);
                    check(ok && same(o, L, false), what + ": offsets, ring and bytes");
                    bool runs_ok = true;
                    for (int64_t n : {(int64_t) 1, (int64_t) 7, (int64_t) 4097, ctx - 1})
                        runs_ok = runs_ok && L.slot_runs(n) == old_runs(g, mode, p.l0, p.l1, p.mtp, o, n);
                    check(runs_ok, what + ": slot runs equal slot_dsa_runs");
                    int n_dsa = p.mtp >= 0;
                    for (int il = p.l0; il < p.l1; ++il) n_dsa += !g.is_recr(il);
                    check(L.open_rows() == (int64_t) (g.idx_kpool - 1) * g.idx_key &&
                              L.snap_pool_floats() == (int64_t) n_dsa * 2 * (g.idx_kpool - 1) * g.idx_key &&
                              L.kda_run() == (int64_t) g.d_inner() * g.kda_head_dim +
                                                 (int64_t) 3 * g.d_inner() * (g.d_conv - 1),
                          what + ": snapshot sizes");
                }
            }
            // the split search's price: whole-context rings, FP16 on the fast path, F32 on the slow one
            for (bool fast : {false, true}) {
                ok = strata::core::glm_state_carve(g, 0, g.n_layers, g.n_layers, ctx,
                                                   fast ? GlmStateMode::Fp16 : GlmStateMode::SlowF32, (int) ctx, L, err);
                bool price_ok = ok;
                for (int l = 0; l <= g.n_layers; ++l)
                    price_ok = price_ok && (double) L.bytes_per_layer(l) == old_price(g, ctx, fast, l);
                check(price_ok, std::string("Flash split price (") + (fast ? "fast" : "slow") + ") equals the old one" + at);
            }
        }
        const GlmStateLayout L0;
        check(strata::core::glm_slot_magic(L0) == 0x3254534C4159414Dull, "Flash slots stay MAYALST2");
    }

    // ---------------------------------------------------------------- 2. glm-dsa (GLM-5.3 744B)
    {
        const Glm5Geometry g = glm53();
        const int64_t ctx = 32768;
        GlmStateLayout L;
        bool ok = strata::core::glm_state_carve(g, 0, 78, 78, ctx, GlmStateMode::Fp16, 0, L, err);
        check(ok, "glm-dsa FP16 carve: " + err);
        const uint64_t formula = 79ull * 1152 * 32768 + 22ull * 512 * 32768;
        check(formula == 3351248896ull, "the 32K formula is 3.35 GB");
        check(L.cache_bytes() == formula, "glm-dsa 32K caches = 79 x 1152 x 32768 + 22 x 512 x 32768 (" +
                                              std::to_string(L.cache_bytes()) + ")");
        check(L.bytes == formula + 2ull * 6144 * 4, "glm-dsa arena = the caches + the plain residual pair");
        check(L.lat_pp * 4 == 1024 && L.kpe_pp * 4 == 128 && L.ik_pp * 4 == 512, "FP16 latent 1024 B, k_pe 128 B, key 512 B");
        int n_ik = 0, n_kpe = 0;
        bool none_ok = true, align_ok = true, disjoint = true;
        std::vector<std::pair<int64_t, int64_t>> regions;
        for (int il = 0; il <= 78; ++il) {
            n_ik += L.has_index_cache(il);
            n_kpe += L.has_kpe(il);
            none_ok = none_ok && L.dsa_ig[il] == GlmStateLayout::kNone && L.dsa_pool[il] == GlmStateLayout::kNone &&
                      (L.has_index_cache(il) == (bool) g.idx_full(il)) &&
                      (g.idx_full(il) ? L.dsa_ik[il] >= 0 : L.dsa_ik[il] == GlmStateLayout::kNone) &&
                      L.kda_S[il] == 0 && L.kda_conv[il] == 0;
            for (int64_t off : {L.dsa_lat[il], L.dsa_kpe[il], L.dsa_ik[il]})
                if (off != GlmStateLayout::kNone) align_ok = align_ok && (off * 4) % 16 == 0;
            regions.push_back({L.dsa_lat[il], L.lat_pp * ctx});
            regions.push_back({L.dsa_kpe[il], L.kpe_pp * ctx});
            if (L.has_index_cache(il)) regions.push_back({L.dsa_ik[il], L.ik_pp * ctx});
        }
        std::sort(regions.begin(), regions.end());
        int64_t end = L.residual_floats;
        for (const auto& r : regions) {
            disjoint = disjoint && r.first == end;   // packed, no gap, no overlap
            end = r.first + r.second;
        }
        disjoint = disjoint && end == L.floats;
        check(n_ik == 22 && n_kpe == 79, "index keys on 21 full layers + NextN (22), k_pe on all 79");
        check(none_ok, "no ring, gate cache, pools or KDA state; absent regions are kNone");
        check(align_ok, "every glm-dsa region 16-byte aligned");
        check(disjoint, "the regions tile the arena after the residual");
        check(L.dsa_layers().size() == 79 && L.dsa_layers().front() == 78, "79 DSA layers, NextN block first");
        check(L.open_rows() == 0 && L.snap_pool_floats() == 0, "pool-less: open_rows 0, no snap_pool_ needed");
        check(L.ik_ring == ctx && L.max_pools == 0, "index keys by position (no ring)");
        uint64_t sum = 0;
        for (int il = 0; il <= 78; ++il) sum += L.bytes_per_layer(il);
        check(sum == formula && L.bytes_per_layer(6) == 32768ull * (1152 + 512) && L.bytes_per_layer(7) == 32768ull * 1152,
              "bytes_per_layer: full 1664 B, shared 1152 B a position, summing to the formula");

        // slot runs: latent + k_pe per layer, then index keys of the full layers
        const int64_t n = 1000;
        const auto runs = L.slot_runs(n);
        bool runs_ok = runs.size() == 79 * 2 + 22;
        for (size_t i = 0; i < 79 && runs_ok; ++i) {
            const int il = L.dsa_layers()[i];
            runs_ok = runs[2 * i] == std::make_pair(L.dsa_lat[il], n * L.lat_pp) &&
                      runs[2 * i + 1] == std::make_pair(L.dsa_kpe[il], n * L.kpe_pp);
        }
        size_t k = 158;
        for (int il : L.dsa_layers())
            if (g.idx_full(il)) runs_ok = runs_ok && runs[k++] == std::make_pair(L.dsa_ik[il], n * L.ik_pp);
        check(runs_ok, "glm-dsa slot runs: latent + k_pe x 79, then 22 index-key runs");
        check(strata::core::glm_slot_magic(L) == strata::core::kGlmSlotMagicV3, "glm-dsa slots are MAYALST3");
        std::printf("     %s\n", L.describe().c_str());

        // INT8: 544 B records, 16-byte aligned; 672 B a position with k_pe
        GlmStateLayout L8;
        ok = strata::core::glm_state_carve(g, 0, 78, 78, ctx, GlmStateMode::Int8, 0, L8, err);
        bool a8 = ok && L8.lat_pp * 4 == 544 && (L8.lat_pp * 4) % 16 == 0;
        for (int il = 0; il <= 78 && a8; ++il) a8 = (L8.dsa_lat[il] * 4) % 16 == 0 && (L8.dsa_kpe[il] * 4) % 16 == 0;
        check(a8, "INT8 record 544 B, every record 16-byte aligned");
        check(ok && L8.cache_bytes() == 79ull * 672 * 32768 + 22ull * 512 * 32768, "INT8 32K caches = 79 x 672 + 22 x 512");

        // reference path: one F32 [c_kv | k_pe] row a position, k_pe inline
        GlmStateLayout Ls;
        ok = strata::core::glm_state_carve(g, 0, 78, -1, 4096, GlmStateMode::SlowF32, 4096, Ls, err);
        check(ok && Ls.lat_pp == 576 && Ls.kpe_pp == 0 && !Ls.has_kpe(5) && Ls.dsa_kpe[5] == GlmStateLayout::kNone &&
                  Ls.cache_bytes() == 78ull * 2304 * 4096 + 21ull * 512 * 4096,
              "reference path: F32 576-float rows, k_pe inline, 21 key caches without the NextN block");

        // fingerprints
        const auto fp16 = strata::core::glm_state_fingerprint(g, L), fp8 = strata::core::glm_state_fingerprint(g, L8);
        GlmStateLayout Lf;
        const Glm5Geometry gf = flash();
        strata::core::glm_state_carve(gf, 0, gf.n_layers, -1, ctx, GlmStateMode::Fp16, 8256, Lf, err);
        const auto ffp = strata::core::glm_state_fingerprint(gf, Lf);
        check(fp16 != fp8 && fp16 != ffp && fp16[1] == 1 && fp16[7] == 21 && fp16[9] == 128 && fp16[10] == 4 &&
                  fp16[11] == 1024 && fp8[11] == 544 && ffp[11] == 0,
              "fingerprint: arch, 21 full layers, k_pe 128 B, F32 keys, latent row 1024 / 544 B; FP16 != INT8 != Flash");
        check(fp16[0] == strata::core::kGlmFpFormat && strata::core::kGlmFpFormat == 2 && fp16.size() == 14,
              "fingerprint format word 2: the 14-word fingerprint of B15.3 (bumped from 1 with the NextN words)");

        // a 3-part split at full layers: the parts' caches sum to the whole
        uint64_t parts = 0;
        const int b[4] = {0, 26, 54, 78};
        for (int p = 0; p < 3; ++p) {
            GlmStateLayout P;
            ok = strata::core::glm_state_carve(g, b[p], b[p + 1], p == 2 ? 78 : -1, ctx, GlmStateMode::Fp16, 0, P, err);
            parts += ok ? P.cache_bytes() : 0;
        }
        check(parts == formula, "3 parts [0,26) [26,54) [54,78)+NextN sum to the formula");

        // refusals
        Glm5Geometry bad = g;
        bad.qk_rope = 63;
        check(!strata::core::glm_state_carve(bad, 0, 78, -1, ctx, GlmStateMode::Fp16, 0, L, err), "odd rope tail refused");
        bad = g;
        bad.indexer_full.resize(10);
        check(!strata::core::glm_state_carve(bad, 0, 78, -1, ctx, GlmStateMode::Fp16, 0, L, err),
              "a missing indexer schedule refused");
        bad = g;
        bad.kv_lora = 520;   // record 552 B: not 16-byte aligned
        check(!strata::core::glm_state_carve(bad, 0, 78, -1, ctx, GlmStateMode::Int8, 0, L, err),
              "an INT8 record off 16 bytes refused");
    }
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
