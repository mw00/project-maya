// tests/glm_dsa/glm_dsa_split_test.cpp - glm-dsa B6.1-B6.3 on the CPU (no GPU): the glm-dsa split's pure rules
// (include/strata/core/glm_dsa_split.hpp) on a GLM-5.3 744B geometry and on Maya's Flash geometry.
//
//   1. B6.2 state: per position 1152 B on every attention layer + 512 B of index keys on the 21 full layers and the
//      NextN block (79 x 1152 + 22 x 512 = 102,272 B), the BUILD-PLAN section 2 per-GPU split at {26, 54}
//      (34,048 / 35,840 / 32,384 B), INT8 672 B rows, the 32K total 3.35 GB; Flash: 0 (Maya's formula prices it);
//   2. B6.1 boundaries: {26, 54} valid, 27 not, its nearest 26 / 30, glm_split_bounds_ok's refusal naming them; the
//      default bounds snapped (2 GPUs: 41 -> 42; 3 GPUs: rising, least distance, ties low); Flash unchanged;
//   3. B6.2 planner terms: the NextN block priced for n >= 2 on glm-dsa (n == 2 only on Flash), the CPU-lane pool rule,
//      the miss price over one shared pool (host_bps) against Maya's host_bps / n;
//   4. B6.3 part groups: 3 parts on 2 NUMA nodes by their GPU's node (2+1), round robin for 3+0, Maya's rule for
//      parts <= nodes.
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_state_layout.hpp"
#include "strata/core/glm_model.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using strata::core::Glm5Geometry;
using strata::core::GlmArch;
namespace sc = strata::core;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// GLM-5.3 744B as glm_dsa_geometry.cpp reads it: 78 layers + 1 NextN, the GLM-5.2 full-indexer schedule
Glm5Geometry glm53() {
    Glm5Geometry g;
    g.arch = GlmArch::GlmDsa;
    g.n_layers = 78;
    g.nextn = 1;
    g.n_embd = 6144;
    g.kv_lora = 512;
    g.qk_rope = 64;
    g.qk_nope = 192;
    g.idx_key = 128;
    g.idx_kpool = 1;
    g.idx_select_tail = 0;
    g.hc = 1;
    g.n_expert = 256;
    g.n_exp_used = 8;
    g.dense_lead = 3;
    g.w_scale = 2.5f;
    g.n_head_kv.assign(78, 1);
    g.indexer_full.assign(79, 0);
    for (int il = 0; il < 79; ++il) g.indexer_full[(size_t) il] = il >= 78 || il < 2 || (il - 2) % 4 == 0 ? 1 : 0;
    return g;
}

double part_row(const Glm5Geometry& g, int lb, int le, bool last) {
    double s = 0;
    for (int l = lb; l < le; ++l) s += (double) sc::glm_dsa_state_row_bytes(g, l, false);
    if (last) s += (double) sc::glm_dsa_state_row_bytes(g, g.n_layers, false);
    return s;
}

std::string str(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
}

}  // namespace

int main() {
    std::printf("glm_dsa_split_test\n");
    const Glm5Geometry g = glm53();
    const Glm5Geometry flash;   // Maya's defaults: glm5-next

    // 1. state
    {
        check(sc::glm_dsa_state_row_bytes(g, 0, false) == 1152 + 512, "full layer 0: 1152 B latent + 512 B index keys");
        check(sc::glm_dsa_state_row_bytes(g, 3, false) == 1152, "shared layer 3: 1152 B latent only");
        check(sc::glm_dsa_state_row_bytes(g, 78, false) == 1664, "the NextN block (blk.78): latent + index keys");
        check(sc::glm_dsa_state_row_bytes(g, 79, false) == 0 && sc::glm_dsa_state_row_bytes(g, -1, false) == 0,
              "outside the layers: 0");
        check(sc::glm_dsa_state_row_bytes(g, 3, true) == 672 && sc::glm_dsa_state_row_bytes(g, 2, true) == 1184,
              "INT8 latent: 544 B record + 128 B k_pe = 672 B (+ 512 B on a full layer)");
        check(sc::glm_dsa_state_row_bytes(g, 3, false, true) == 2304, "the slow path's F32 latent row: 2304 B");
        double total = 0;
        int full = 0;
        for (int l = 0; l <= 78; ++l) {
            total += (double) sc::glm_dsa_state_row_bytes(g, l, false);
            full += g.idx_full(l) ? 1 : 0;
        }
        check(full == 22 && total == 79.0 * 1152 + 22.0 * 512 && total == 102272.0, "79 x 1152 + 22 x 512 = 102,272 B a token");
        check(part_row(g, 0, 26, false) == 34048 && part_row(g, 26, 54, false) == 35840 && part_row(g, 54, 78, true) == 32384,
              "split {26, 54}: 34,048 / 35,840 / 32,384 B a token (BUILD-PLAN section 2)");
        double t32 = 0;
        for (int l = 0; l <= 78; ++l) t32 += sc::glm_dsa_state_bytes(g, l, 32768, false);
        check(std::fabs(t32 - 3.35e9) < 0.01e9, "32K context: 3.35 GB of state");
        // review fix: the planner's row formula and the B4.1 descriptor (GlmStateLayout::bytes_per_layer) must agree
        bool same = true;
        for (const bool int8 : {false, true}) {
            sc::GlmStateLayout lay;
            std::string e;
            const bool carved = sc::glm_state_carve(g, 0, 78, 78, 4096, int8 ? sc::GlmStateMode::Int8 : sc::GlmStateMode::Fp16,
                                                    0, lay, e);
            same = same && carved;
            for (int l = 0; carved && l <= 78; ++l)
                same = same && lay.bytes_per_layer(l) == (uint64_t) sc::glm_dsa_state_row_bytes(g, l, int8) * 4096u;
        }
        check(same, "glm_dsa_state_row_bytes x ctx == GlmStateLayout::bytes_per_layer for every layer (FP16 and INT8)");
        check(sc::glm_dsa_state_row_bytes(flash, 0, false) == 0 && sc::glm_dsa_state_bytes(flash, 5, 4096, true) == 0,
              "Flash: 0 (Maya's own formula prices it)");
    }

    // 2. boundaries
    {
        check(sc::glm_split_bounds_full(g, {26, 54}) && !sc::glm_split_bounds_full(g, {27, 54}),
              "{26, 54} are full-indexer boundaries, 27 is not");
        int lo = 0, hi = 0;
        sc::glm_nearest_boundaries(g, 27, lo, hi);
        check(lo == 26 && hi == 30 && sc::glm_nearest_boundaries_text(g, 27) == "26 or 30", "27's nearest: 26 or 30");
        check(sc::glm_nearest_boundaries_text(g, 26) == "26" && sc::glm_nearest_boundaries_text(g, 77) == "74",
              "a valid layer names itself; past the last full layer: 74");
        std::string err;
        const bool ok = sc::glm_split_bounds_ok(g, {27, 54}, err);
        check(!ok && err.find("layer 27") != std::string::npos && err.find("26 or 30") != std::string::npos,
              "MAYA_GLM_SPLIT=27,54 refused naming 26 or 30: " + err.substr(0, 60) + "...");
        check(sc::glm_split_bounds_ok(g, {26, 54}, err), "MAYA_GLM_SPLIT=26,54 accepted");
        // the default bounds: 2 GPUs n_layers / 2 + 2 = 41, 3 GPUs the proportional split's
        check(sc::glm_snap_bounds(g, {41}) == std::vector<int>{42}, "2 GPUs: default 41 snaps to 42");
        check(sc::glm_snap_bounds(g, {27, 53}) == std::vector<int>{26, 54}, "3 GPUs: {27, 53} -> {26, 54}");
        check(sc::glm_snap_bounds(g, {28, 52}) == std::vector<int>{26, 50}, "ties go to the lower layers: {28, 52} -> {26, 50}");
        check(sc::glm_snap_bounds(g, {75, 76}) == std::vector<int>{70, 74}, "near the end: {75, 76} -> {70, 74} (rising)");
        check(sc::glm_snap_bounds(g, {3, 4}) == std::vector<int>{2, 6}, "near the start: {3, 4} -> {2, 6}");
        check(sc::glm_snap_bounds(g, {26, 54}) == std::vector<int>{26, 54}, "valid bounds unchanged");
        const auto s3 = sc::glm_snap_bounds(g, {25, 26, 27});
        check(sc::glm_split_bounds_full(g, s3) && s3.size() == 3 && s3[0] < s3[1] && s3[1] < s3[2],
              "4 GPUs: {25, 26, 27} -> rising full layers " + str(s3));
        check(sc::glm_snap_bounds(flash, {23}) == std::vector<int>{23} && sc::glm_split_bounds_full(flash, {23, 31}),
              "Flash: any layer, unchanged");
    }

    // 3. planner terms
    {
        check(sc::glm_split_prices_mtp(flash, 2) && !sc::glm_split_prices_mtp(flash, 3), "Flash: NextN priced for n == 2 only");
        check(sc::glm_split_prices_mtp(g, 2) && sc::glm_split_prices_mtp(g, 3) && sc::glm_split_prices_mtp(g, 4) &&
                  !sc::glm_split_prices_mtp(g, 1),
              "glm-dsa: NextN priced for n >= 2 (the last part loads it for any n)");
        using V = std::vector<int>;
        const V nodes01 = {0, 1};   // the test PC's two NUMA nodes
        const auto ids = [&](const sc::Glm5Geometry& gg, int n, const char* env, const V& nodes = {0, 1},
                             const V& gpu = {}) { return sc::glm_cpu_pool_ids(gg, n, env, nodes, gpu); };
        check(ids(g, 3, nullptr) == V({0, 0, 0}) && ids(g, 3, "0") == V({0, 1, 2}) && ids(g, 2, nullptr) == V({0, 1}) &&
                  ids(g, 2, "1") == V({0, 0}) && ids(g, 1, "1") == V({0}) && ids(g, 3, "") == V({0, 0, 0}),
              "pool rule: one shared pool for 3+ parts, MAYA_GLM_CPU_SHARED overrides");
        // review fix: the planner reads MAYA_GLM_CPU_SHARED as the runtime does (glmtier::lane_pool_mode)
        check(ids(g, 3, "socket", nodes01, {0, 0, 1}) == V({0, 0, 1}) && ids(g, 3, "2", nodes01, {0, 0, 1}) == V({0, 0, 1}) &&
                  sc::glm_cpu_pool_count(ids(g, 3, "socket", nodes01, {0, 0, 1})) == 2,
              "glm-dsa, 'socket' or '2' on two nodes: a pool per socket, by the GPUs' nodes (2 pools)");
        check(ids(g, 3, "socket", nodes01) == V({0, 1, 0}) && ids(g, 3, "socket", nodes01, {-1, 7, 1}) == V({0, 1, 1}),
              "... GPU node unknown or offline: round robin over the nodes, as lane_pools");
        check(ids(g, 3, "socket", V{0}) == V({0, 0, 0}), "... one node: one shared pool");
        check(ids(flash, 3, "2", nodes01) == V({0, 0, 0}) && ids(flash, 3, "socket", nodes01) == V({0, 1, 2}),
              "Flash keeps Maya's reading: '2' one shared pool, 'socket' (atoi 0) a pool per part");
        const double h = 80e9;
        check(sc::glm_split_miss_bps(flash, h, 3, 1, false) == h / 3, "Flash: Maya's host_bps / n");
        check(sc::glm_split_miss_bps(g, h, 3, 1, false) == h, "glm-dsa, 3 parts, one shared pool: the whole host_bps");
        check(sc::glm_split_miss_bps(g, h, 3, 3, false) == h / 3, "glm-dsa, a pool per part: host_bps / n");
        check(sc::glm_split_miss_bps(g, h, 3, 2, false) == h / 2, "glm-dsa, a pool per socket on two: host_bps / 2");
        check(sc::glm_split_miss_bps(g, h, 2, 1, true) == h / 2, "pipelined parts run at once: host_bps / n");
        check(sc::glm_split_miss_bps(g, 0, 3, 1, false) == 2e10 && sc::glm_split_miss_bps(flash, 0, 3, 1, false) == 2e10 / 3,
              "unmeasured host: 20 GB/s");
        check(sc::glm_split_miss_bps(flash, 1e9, 4, 4, false) == 1e9, "at least 1 GB/s");
        // the miss price at the test PC's measured ~80 GB/s for a 3-part token: 30 RAM-tier experts of 12 MB = 360 MB
        const double ms_shared = 1e3 * 360e6 / sc::glm_split_miss_bps(g, h, 3, 1, false);
        const double ms_maya = 1e3 * 360e6 / sc::glm_split_miss_bps(flash, h, 3, 1, false);
        check(std::fabs(ms_shared - 4.5) < 1e-9 && std::fabs(ms_maya - 13.5) < 1e-9,
              "360 MB of misses: 4.5 ms over the shared pool (Maya's term: 13.5 ms)");
    }

    // 4. part groups
    {
        const std::vector<int> two = {0, 1};
        check(sc::glm_part_groups(two, {0, 1}, 2) == std::vector<int>{0, 1}, "2 parts, GPUs on nodes 0 / 1: by node");
        check(sc::glm_part_groups(two, {1, 0}, 2) == std::vector<int>{1, 0}, "2 parts, GPUs on nodes 1 / 0: by node");
        check(sc::glm_part_groups(two, {0, 0}, 2) == std::vector<int>{0, 1}, "2 parts, both GPUs on node 0: Maya's i % G");
        check(sc::glm_part_groups(two, {0, 0, 1}, 3) == std::vector<int>{0, 0, 1}, "3 parts, 2+1: parts 0, 1 on node 0, 2 on 1");
        check(sc::glm_part_groups(two, {1, 0, 0}, 3) == std::vector<int>{1, 0, 0}, "3 parts, 1+2: by node");
        check(sc::glm_part_groups(two, {0, 0, 0}, 3) == std::vector<int>{0, 1, 0}, "3 parts, 3+0: round robin (no idle socket)");
        check(sc::glm_part_groups(two, {0, -1, 1}, 3) == std::vector<int>{0, 1, 0}, "3 parts, a node unknown: round robin");
        check(sc::glm_part_groups(two, {0, 0}, 3) == std::vector<int>{0, 1, 0}, "3 parts, node list short: round robin");
        check(sc::glm_part_groups({-1}, {-1, -1, -1}, 3) == std::vector<int>{0, 0, 0}, "no NUMA: one group");
        check(sc::glm_part_groups({}, {0}, 1).empty() && sc::glm_part_groups(two, {}, 0).empty(), "nothing to give: empty");
    }

    if (g_fail) {
        std::printf("glm_dsa_split_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_split_test: ok\n");
    return 0;
}
