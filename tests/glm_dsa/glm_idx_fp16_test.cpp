// tests/glm_dsa/glm_idx_fp16_test.cpp - glm-dsa B10.8 (CPU only): FP16 index keys in the state descriptor, the split
// planner and the slot fingerprint, GLM-5.3 744B geometry.
//   1. without MAYA_GLM_IDX_FP16: F32 keys (ik_pp 128 floats, ik_bytes 4, fingerprint 4; full layer row 1664 B);
//   2. with MAYA_GLM_IDX_FP16=1: FP16 / INT8 carves store 64 floats (128 halves) per key, every region stays 16-byte
//      aligned, the 32K arena shrinks by exactly 22 x 256 x 32768 B (~0.18 GB, the plan's B10.8 figure), the
//      planner prices 1152 + 256 B on full layers, the fingerprint records 2 bytes per element (a slot saved with
//      F32 keys is refused by a FP16 build and back), slot runs cover n * 64 floats per full layer;
//   3. the slow F32 reference carve keeps F32 keys either way.
#include "strata/core/glm_dsa_split.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

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

bool carve(const Glm5Geometry& g, GlmStateMode mode, int64_t ctx, GlmStateLayout& L) {
    std::string err;
    const bool ok = strata::core::glm_state_carve(g, 0, g.n_layers, g.n_layers, ctx, mode, 0, L, err);
    if (!ok) std::printf("carve: %s\n", err.c_str());
    return ok;
}

}  // namespace

int main() {
    const Glm5Geometry g = glm53();
    const int64_t ctx = 32768;
    unsetenv("MAYA_GLM_IDX_FP16");
    GlmStateLayout F, F8, S;
    check(!strata::core::glm_state_idx_fp16(), "default: F32 index keys");
    check(carve(g, GlmStateMode::Fp16, ctx, F) && carve(g, GlmStateMode::Int8, ctx, F8) &&
              carve(g, GlmStateMode::SlowF32, ctx, S),
          "default carves");
    check(F.ik_pp == 128 && F.ik_bytes == 4, "default: ik_pp 128 floats, 4 B per element");
    check(strata::core::glm_dsa_state_row_bytes(g, 0, false) == 1152 + 512, "default planner: full layer 1664 B");
    const auto fp_f32 = strata::core::glm_state_fingerprint(g, F);
    check(fp_f32[10] == 4, "default fingerprint: 4 B per key element");

    setenv("MAYA_GLM_IDX_FP16", "1", 1);
    GlmStateLayout H, H8, HS;
    check(strata::core::glm_state_idx_fp16(), "MAYA_GLM_IDX_FP16=1 read");
    check(carve(g, GlmStateMode::Fp16, ctx, H) && carve(g, GlmStateMode::Int8, ctx, H8) &&
              carve(g, GlmStateMode::SlowF32, ctx, HS),
          "FP16-key carves");
    check(H.ik_pp == 64 && H.ik_bytes == 2 && H8.ik_pp == 64 && H8.ik_bytes == 2, "FP16 keys: 64 floats (128 halves)");
    check(HS.ik_pp == 128 && HS.ik_bytes == 4 && HS.bytes == S.bytes, "slow F32 path keeps F32 keys");
    int n_full = 0;
    bool aligned = true;
    for (int il = 0; il <= g.n_layers; ++il)
        if (H.has_index_cache(il)) {
            ++n_full;
            aligned &= H.dsa_ik[(size_t) il] % 4 == 0 && H8.dsa_ik[(size_t) il] % 4 == 0;
        }
    check(n_full == 22, "22 index caches (21 full trunk layers + NextN)");
    check(aligned, "FP16 key regions 16-byte aligned (FP16 and INT8 carves)");
    const uint64_t saved = (uint64_t) 22 * 256 * (uint64_t) ctx;
    check(F.bytes - H.bytes == saved && F8.bytes - H8.bytes == saved, "32K arena: -22 x 256 B x 32768 per FP16 key cache");
    std::printf("     saved at 32K: %.3f GB\n", saved / 1e9);
    check(strata::core::glm_dsa_state_row_bytes(g, 0, false) == 1152 + 256 &&
              strata::core::glm_dsa_state_row_bytes(g, 3, false) == 1152 &&
              strata::core::glm_dsa_state_row_bytes(g, 0, false, true) == 2304 + 512,
          "planner: full layer 1408 B with FP16 keys, F32 slow path 2816 B");
    const auto fp_f16 = strata::core::glm_state_fingerprint(g, H);
    check(fp_f16[10] == 2 && fp_f16 != fp_f32, "fingerprint: 2 B per key element, differs from the F32-key build");
    bool runs = true;
    for (const auto& r : H.slot_runs(100)) runs &= r.second == 100 * H.lat_pp || r.second == 100 * H.kpe_pp ||
                                                  r.second == 100 * H.ik_pp;
    check(runs, "slot runs: n * ik_pp floats per index cache");
    unsetenv("MAYA_GLM_IDX_FP16");
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
