// tests/glm_dsa/glm_dsa_mixer_cpu_test.cpp - the glm-dsa decode mixer's orchestration on the CPU twins (glm-dsa B4.6,
// B4.7 named wrong readings, B14.3 dense switch).  Plain C++: runs on any build, no GPU.
//
// glmdsa::mixer_layer (src/core/glm_dsa_mixer.hpp, the code Glm5Model::fast_dsa_glm runs) is instantiated with the
// glmd::twin kernels and float GEMVs over a random layer stack in the SMALL geometry of tools/glm_dsa_synth_gguf.py
// (and the GPU-shaped MID geometry), full and shared layers interleaved, every position through every layer in
// order, and compared per (layer, position) with the reference layer (the B3.4 CPU reference functions, F32 caches,
// the reference forward's selection policy):
//   * correct, FP16 latents: max rel L2 < 5e-3, the top-k sets equal the reference's once p + 1 > top_k, and the
//     recorded n_sel_last = min(p + 1, top_k) on every full layer;
//   * correct, INT8 latents (Maya's record): max rel L2 < 3e-2;
//   * the dense switch (STRATA_GLM_DSA_DENSE) against a reference attending every cell: max rel L2 < 5e-3, and it
//     differs from the sparse reference once selection is active;
//   * each named wrong reading (NEOX pairing, q_pe rope at offset 0, indexer LayerNorm eps 0, shared layers run as
//     full, scale 1/sqrt(qk_nope)) must FAIL the correct gate by a wide margin;
//   * a shared layer run without its full layer at the same position is refused (stale selection).
#include "glm_dsa_mixer_fixture.hpp"

#include <cstdio>
#include <memory>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct CpuBe {
    std::vector<std::unique_ptr<uint8_t[]>> mem;
    void* alloc(size_t n) {
        mem.emplace_back(new uint8_t[std::max<size_t>(n, 4)]());
        return mem.back().get();
    }
    void put(void* d, const void* h, size_t n) { std::memcpy(d, h, n); }
    void get(void* h, const void* s, size_t n) { std::memcpy(h, s, n); }
    void sync() {}
    bool gemv(const float* W, const float* x, float* y, int in, int out, float alpha) {
        for (int o = 0; o < out; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < in; ++i) acc += W[(size_t) o * in + i] * x[i];
            y[o] = alpha * acc;
        }
        return true;
    }
    bool prep(const glmd::DsaPrep& a) { glmd::twin::dsa_prep(a); return true; }
    bool rope_q(const glmd::DsaRopeQ& a) { glmd::twin::dsa_rope_q(a); return true; }
    bool score(const glmd::IdxScore& a) { glmd::twin::idx_score(a); return true; }
    bool topk(const float* s, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        glmd::twin::topk(s, ld, n_kv, 1, pos, top_k, sel);
        return true;
    }
    bool mla2(const glmd::Mla2& a) { glmd::twin::mla2(a); return true; }
};

mixfx::RunStats run(const mixfx::Model& m, mixfx::Mut mut, bool dense, int lat_q8, bool dense_ref) {
    CpuBe be;
    mixfx::Harness<CpuBe> h(be, m, mut, dense, lat_q8);
    return mixfx::run_stack(h, m, dense_ref);
}

void suite(const char* name, const mixfx::Model& m, bool full_suite = true) {
    using mixfx::Mut;
    std::printf("[%s: n_embd %d, %d heads, kv_lora %d + rope %d, indexer %d x %d, top-%d, %zu layers, %d positions]\n",
                name, m.d.n_embd, m.d.n_head, m.d.kv_lora, m.d.qk_rope, m.d.idx_heads, m.d.idx_key, m.d.top_k,
                m.L.size(), m.T);
    constexpr double kTol16 = 5e-3, kTol8 = 3e-2, kWrong = 2e-2;
    char buf[256];
    const mixfx::RunStats ok16 = run(m, Mut::None, false, 0, false);
    std::snprintf(buf, sizeof buf, "correct, FP16 latents: max rel L2 %.2e (mean %.2e) over %d layer-positions%s%s",
                  ok16.max_rel, ok16.mean_rel, ok16.n, ok16.err.empty() ? "" : " - ", ok16.err.c_str());
    check(ok16.err.empty() && ok16.max_rel < kTol16, buf);
    std::snprintf(buf, sizeof buf, "top-k sets equal the reference's on %d / %d selecting full-layer positions",
                  ok16.sel_equal, ok16.sel_checked);
    check(ok16.sel_checked > 0 && ok16.sel_equal == ok16.sel_checked, buf);
    check(ok16.sel_count_bad == 0, "n_sel_last = min(p + 1, top_k) on every full layer");
    if (!full_suite) return;   // (WIDE: the correct reading only - its GEMVs dominate the CPU time)
    const mixfx::RunStats ok8 = run(m, Mut::None, false, 1, false);
    std::snprintf(buf, sizeof buf, "correct, INT8 latents: max rel L2 %.2e (mean %.2e)", ok8.max_rel, ok8.mean_rel);
    check(ok8.err.empty() && ok8.max_rel < kTol8, buf);
    const mixfx::RunStats dn = run(m, Mut::None, true, 0, true);
    std::snprintf(buf, sizeof buf, "dense switch vs dense reference: max rel L2 %.2e", dn.max_rel);
    check(dn.err.empty() && dn.max_rel < kTol16 && dn.sel_count_bad == 0, buf);
    const mixfx::RunStats dns = run(m, Mut::None, true, 0, false);
    std::snprintf(buf, sizeof buf, "dense switch vs the sparse reference differs once p + 1 > top_k: max rel L2 %.2e",
                  dns.max_rel);
    check(dns.err.empty() && dns.max_rel > kWrong, buf);
    for (Mut mu : {Mut::Neox, Mut::RopeOffset, Mut::IdxEps, Mut::SharedAsFull, Mut::Scale}) {
        const mixfx::RunStats w = run(m, mu, false, 0, false);
        std::snprintf(buf, sizeof buf, "wrong reading '%s' fails: max rel L2 %.2e%s%s", mixfx::mut_name(mu), w.max_rel,
                      w.err.empty() ? "" : " - ", w.err.c_str());
        check(w.err.empty() && w.max_rel > kWrong && w.max_rel > 4 * ok16.max_rel, buf);
    }
    {   // a shared layer at a position its full layer has not run: refused
        CpuBe be;
        mixfx::Harness<CpuBe> h(be, m, Mut::None, false, 0);
        std::vector<float> out;
        std::string err;
        const bool ran = h.step(1, 0, out, err);
        check(!ran && err.find("shared layer") != std::string::npos, "a stale selection is refused: " + err);
        std::string e2;
        const bool a = h.step(0, 0, out, e2) && h.step(1, 0, out, e2);
        const bool b = !h.step(1, 1, out, err);
        check(a && b, "the selection is per position: layer 1 at p = 1 needs layer 0 at p = 1 first");
    }
}

}  // namespace

int main() {
    std::printf("glm_dsa_mixer_cpu_test: the glm-dsa decode mixer (glmdsa::mixer_layer) on the CPU twins\n");
    check(strata::core::glmdsa::sel_count(0, 2048, false) == 1 && strata::core::glmdsa::sel_count(2047, 2048, false) == 2048 &&
              strata::core::glmdsa::sel_count(2048, 2048, false) == 2048 &&
              strata::core::glmdsa::sel_count(5000, 2048, true) == 5001,
          "sel_count: identity to 2048 cells, then 2048; dense p + 1");
    {
        const std::vector<bool> pat = {true, false, false, true, false, false, true};
        suite("SMALL", mixfx::make_model(mixfx::small_dims(64), pat, 40, 1));
    }
    {
        const std::vector<bool> pat = {true, false, true, false};
        suite("MID", mixfx::make_model(mixfx::mid_dims(48), pat, 36, 2));
    }
    {   // the GLM-5.3 widths of glmd_kernel_parity's WIDE (n_embd 6144, q_lora 2048, 32 heads, indexer 32 x 128)
        mixfx::MixerDims d = mixfx::mid_dims(48);
        d.n_embd = 6144, d.n_head = 32, d.q_lora = 2048, d.idx_heads = 32, d.idx_key = 128, d.idx_rope = 64, d.top_k = 32;
        suite("WIDE", mixfx::make_model(d, {true, false}, 36, 3), false);
    }
    if (g_fail) {
        std::printf("glm_dsa_mixer_cpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_mixer_cpu_test: ok\n");
    return 0;
}
