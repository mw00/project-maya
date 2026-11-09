// tests/glm_dsa/glmd_kernel_parity.cu - glm-dsa B4.7 kernel parity (label gpu): the glm-dsa decode mixer on the device.
//
// glmdsa::mixer_layer (src/core/glm_dsa_mixer.hpp: the very sequence Glm5Model::fast_dsa_glm runs) is instantiated
// with the device kernels (glmd::dsa_prep / dsa_rope_q / idx_score / topk / mla2, F32 GEMVs by glmd::ref_gemv) over
// the fixture of tests/glm_dsa/glm_dsa_mixer_fixture.hpp and compared per (layer, position) with
//   * the B3.4 reference layer (F32 caches, the reference forward's selection policy):
//       correct FP16 latents rel L2 < 5e-3 and the top-k sets equal; INT8 latents < 3e-2; the dense switch
//       (STRATA_GLM_DSA_DENSE) vs a reference attending every cell < 5e-3;
//   * the CPU twins running the same orchestration in lockstep: rel L2 < 1e-3 (the device and the twins agree, so the
//     CPU test glm_dsa_mixer_cpu_test speaks for the kernels);
//   * the named wrong readings (NEOX pairing, q_pe rope at offset 0, indexer LayerNorm eps 0, shared layers run as
//     full, scale 1/sqrt(qk_nope)): each must FAIL the correct gate.
// Geometries: MID (PACK's MLA head with 4 heads, top-16) and WIDE (n_embd 6144, q_lora 2048, 32 heads in two mla2
// head blocks, the GLM-5.3 indexer 32 x 128, top-32).
// Kernel checks:
//   * glmd::idx_score vs Maya's gf::dsa_score on token-level keys (n_pools = p + 1): equal at n 1, 33, 2049, 5000;
//   * glmd::mla2 with a device cell count (capacity 2048, count k) equals the plain launch with n_sel = k;
//   * B14.4 CUDA-graph replay: one full and one shared layer captured once into graphs (MixerBufs::n_grid at the key's
//     capacity, as the engine captures), replayed at every position with only the device position word and the input
//     changed: equal (bitwise) to the direct launches, through the identity selection (p + 1 <= top_k) and past it.
// Exits 77 (ctest SKIP_RETURN_CODE) without a device.
#include "glm_dsa_mixer_fixture.hpp"

#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glmd_graph.hpp"
#include "strata/kernels/glmd_ref.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#ifndef STRATA_USE_HIP   // (hip_compat maps no capture mode but thread-local: the guard keeps check_hip_compat exact)
#define GLMD_CAPTURE_RELAXED cudaStreamCaptureModeRelaxed
#else
#define GLMD_CAPTURE_RELAXED hipStreamCaptureModeRelaxed
#endif

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

// ---------------------------------------------------------------- backends
struct GpuBe {
    cudaStream_t s = nullptr;   // a blocking stream: the harness's synchronous copies order with it
    std::vector<void*> mem;
    GpuBe() { CK(cudaStreamCreate(&s)); }
    ~GpuBe() {
        cudaStreamSynchronize(s);
        for (void* p : mem) cudaFree(p);
        cudaStreamDestroy(s);
    }
    void* alloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, std::max<size_t>(n, 4)));
        CK(cudaMemset(p, 0, std::max<size_t>(n, 4)));
        mem.push_back(p);
        return p;
    }
    void put(void* d, const void* h, size_t n) {
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
    }
    void get(void* h, const void* d, size_t n) {
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
    }
    void sync() { CK(cudaStreamSynchronize(s)); }
    bool gemv(const float* W, const float* x, float* y, int in, int out, float alpha) {
        glmd::ref_gemv(W, x, y, in, out, 1, alpha, s);
        return true;
    }
    bool prep(const glmd::DsaPrep& a) { return glmd::dsa_prep(a, s); }
    bool rope_q(const glmd::DsaRopeQ& a) { return glmd::dsa_rope_q(a, s); }
    bool score(const glmd::IdxScore& a) { return glmd::idx_score(a, s); }
    bool topk(const float* sc, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        return glmd::topk(sc, ld, n_kv, 1, pos, top_k, sel, s);
    }
    bool mla2(const glmd::Mla2& a) { return glmd::mla2(a, s); }
};

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
        glmd::cpu::gemv(W, x, y, in, out, 1, alpha);
        return true;
    }
    bool prep(const glmd::DsaPrep& a) { glmd::twin::dsa_prep(a); return true; }
    bool rope_q(const glmd::DsaRopeQ& a) { glmd::twin::dsa_rope_q(a); return true; }
    bool score(const glmd::IdxScore& a) { glmd::twin::idx_score(a); return true; }
    bool topk(const float* sc, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        glmd::twin::topk(sc, ld, n_kv, 1, pos, top_k, sel);
        return true;
    }
    bool mla2(const glmd::Mla2& a) { glmd::twin::mla2(a); return true; }
};

mixfx::RunStats run_gpu(const mixfx::Model& m, mixfx::Mut mut, bool dense, int lat_q8, bool dense_ref) {
    GpuBe be;
    mixfx::Harness<GpuBe> h(be, m, mut, dense, lat_q8);
    return mixfx::run_stack(h, m, dense_ref);
}

/// A geometry with GLM-5.3's widths where they shape the kernels (n_embd 6144, q_lora 2048, the indexer 32 x 128,
/// the MLA head) and 32 heads (two mla2 head blocks).
mixfx::MixerDims wide_dims(int64_t ctx) {
    mixfx::MixerDims d = mixfx::mid_dims(ctx);
    d.n_embd = 6144, d.n_head = 32, d.q_lora = 2048, d.idx_heads = 32, d.idx_key = 128, d.idx_rope = 64, d.top_k = 32;
    return d;
}

// ---------------------------------------------------------------- suites
void suite(const char* name, const mixfx::Model& m, bool mutations) {
    using mixfx::Mut;
    std::printf("[%s: n_embd %d, %d heads, kv_lora %d + rope %d, indexer %d x %d, top-%d, %zu layers, %d positions]\n",
                name, m.d.n_embd, m.d.n_head, m.d.kv_lora, m.d.qk_rope, m.d.idx_heads, m.d.idx_key, m.d.top_k,
                m.L.size(), m.T);
    constexpr double kTol16 = 5e-3, kTol8 = 3e-2, kWrong = 2e-2, kTwin = 1e-3;
    char buf[320];
    const mixfx::RunStats ok16 = run_gpu(m, Mut::None, false, 0, false);
    std::snprintf(buf, sizeof buf, "device, FP16 latents vs the reference: max rel L2 %.2e (mean %.2e) over %d%s%s",
                  ok16.max_rel, ok16.mean_rel, ok16.n, ok16.err.empty() ? "" : " - ", ok16.err.c_str());
    check(ok16.err.empty() && ok16.max_rel < kTol16, buf);
    std::snprintf(buf, sizeof buf, "top-k sets equal the reference's on %d / %d selecting full-layer positions",
                  ok16.sel_equal, ok16.sel_checked);
    check(ok16.sel_checked > 0 && ok16.sel_equal == ok16.sel_checked && ok16.sel_count_bad == 0, buf);
    const mixfx::RunStats ok8 = run_gpu(m, Mut::None, false, 1, false);
    std::snprintf(buf, sizeof buf, "device, INT8 latents vs the reference: max rel L2 %.2e", ok8.max_rel);
    check(ok8.err.empty() && ok8.max_rel < kTol8, buf);
    const mixfx::RunStats dn = run_gpu(m, Mut::None, true, 0, true);
    std::snprintf(buf, sizeof buf, "device, dense switch vs the dense reference: max rel L2 %.2e", dn.max_rel);
    check(dn.err.empty() && dn.max_rel < kTol16, buf);
    {   // the device against the twins, in lockstep
        GpuBe gb;
        CpuBe cb;
        mixfx::Harness<GpuBe> hg(gb, m, Mut::None, false, 0);
        mixfx::Harness<CpuBe> hc(cb, m, Mut::None, false, 0);
        double worst = 0;
        int sel_diff = 0;
        std::string err;
        std::vector<float> a, b;
        for (int p = 0; p < m.T && err.empty(); ++p)
            for (int il = 0; il < (int) m.L.size(); ++il) {
                if (!hg.step(il, p, a, err) || !hc.step(il, p, b, err)) break;
                worst = std::max(worst, mixfx::rel_l2(a.data(), b));
                if (m.L[(size_t) il].full && p + 1 > m.d.top_k) {
                    std::set<int> sa(hg.cells_h.begin(), hg.cells_h.end()), sb(hc.cells_h.begin(), hc.cells_h.end());
                    sel_diff += sa == sb ? 0 : 1;
                }
            }
        std::snprintf(buf, sizeof buf, "device vs CPU twins (same orchestration): max rel L2 %.2e, %d selection "
                      "differences%s%s", worst, sel_diff, err.empty() ? "" : " - ", err.c_str());
        check(err.empty() && worst < kTwin && sel_diff == 0, buf);
    }
    if (!mutations) return;
    for (Mut mu : {Mut::Neox, Mut::RopeOffset, Mut::IdxEps, Mut::SharedAsFull, Mut::Scale}) {
        const mixfx::RunStats w = run_gpu(m, mu, false, 0, false);
        std::snprintf(buf, sizeof buf, "wrong reading '%s' fails on the device: max rel L2 %.2e%s%s", mixfx::mut_name(mu),
                      w.max_rel, w.err.empty() ? "" : " - ", w.err.c_str());
        check(w.err.empty() && w.max_rel > kWrong && w.max_rel > 4 * ok16.max_rel, buf);
    }
}

// glmd::idx_score vs Maya's dsa_score on token-level keys
void score_vs_maya() {
    std::printf("[idx_score vs Maya's dsa_score, token-level keys, 32 x 128]\n");
    const int H = 32, K = 128, cap = 6000;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> iq((size_t) H * K), ik((size_t) cap * K), w(H);
    for (auto& v : iq) v = nd(rng);
    for (auto& v : ik) v = nd(rng);
    for (auto& v : w) v = nd(rng) * 0.02f;
    GpuBe be;
    float *d_iq = (float*) be.alloc(iq.size() * 4), *d_ik = (float*) be.alloc(ik.size() * 4), *d_w = (float*) be.alloc(H * 4);
    float *s1 = (float*) be.alloc((size_t) cap * 4), *s2 = (float*) be.alloc((size_t) cap * 4);
    int* p_dev = (int*) be.alloc(4);
    int* n_sel = (int*) be.alloc(4);
    be.put(d_iq, iq.data(), iq.size() * 4);
    be.put(d_ik, ik.data(), ik.size() * 4);
    be.put(d_w, w.data(), H * 4);
    for (int n : {1, 33, 2049, 5000}) {
        const int p = n - 1;
        be.put(p_dev, &p, 4);
        glmd::IdxScore a;
        a.iq = d_iq, a.ik_cache = d_ik, a.w = d_w, a.idx_heads = H, a.idx_dim = K, a.n_cap = cap, a.p_dev = p_dev;
        a.score = s1, a.skip_le = 0, a.n_sel_out = n_sel, a.sel_cap = 2048;
        const bool ok = glmd::idx_score(a, be.s);
        strata::kernels::glmf::dsa_score(d_iq, d_ik, d_w, K, H, n, s2, be.s);
        std::vector<float> a1((size_t) n), a2((size_t) n);
        be.get(a1.data(), s1, (size_t) n * 4);
        be.get(a2.data(), s2, (size_t) n * 4);
        int cnt = 0;
        be.get(&cnt, n_sel, 4);
        const bool same = std::memcmp(a1.data(), a2.data(), (size_t) n * 4) == 0;
        check(ok && same && cnt == std::min(n, 2048),
              "n = " + std::to_string(n) + ": bitwise equal to dsa_score, count " + std::to_string(cnt));
    }
}

// mla2 with a device cell count == mla2 with that many cells
void mla2_count() {
    std::printf("[mla2: the device cell count (capacity 2048) vs n_sel]\n");
    const int H = 8, L = 512, R = 64, NOPE = 192, V = 256, ctx = 3000, cap = 2048;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    auto h16 = [&](size_t n, float sd) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = glmd::twin::half_bits(nd(rng) * sd);
        return v;
    };
    auto b16 = [&](size_t n, float sd) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = mixfx::bf16_rn(nd(rng) * sd);
        return v;
    };
    std::vector<float> q((size_t) H * (NOPE + R));
    for (auto& v : q) v = nd(rng);
    const auto lat = h16((size_t) ctx * L, 1.0f), kpe = h16((size_t) ctx * R, 1.0f);
    const auto wk = b16((size_t) H * L * NOPE, 0.07f), wv = b16((size_t) H * V * L, 0.04f);
    std::vector<int> cells(cap);
    for (int i = 0; i < cap; ++i) cells[(size_t) i] = (i * 7 + 3) % ctx;
    GpuBe be;
    auto up = [&](const void* h, size_t n) {
        void* d = be.alloc(n);
        be.put(d, h, n);
        return d;
    };
    glmd::Mla2 a;
    a.q = (const float*) up(q.data(), q.size() * 4);
    a.q_stride = NOPE + R;
    a.wk_b = (const uint16_t*) up(wk.data(), wk.size() * 2);
    a.wv_b = (const uint16_t*) up(wv.data(), wv.size() * 2);
    a.lat = up(lat.data(), lat.size() * 2);
    a.k_pe = (const uint16_t*) up(kpe.data(), kpe.size() * 2);
    a.cells = (const int*) up(cells.data(), cells.size() * 4);
    a.n_head = H, a.qk_nope = NOPE, a.qk_rope = R, a.kv_lora = L, a.v_head = V, a.scale = 0.0625f;
    a.work_bytes = glmd::mla2_work_bytes(H, L, cap);
    a.work = be.alloc(a.work_bytes);
    float* o1 = (float*) be.alloc((size_t) H * V * 4);
    float* o2 = (float*) be.alloc((size_t) H * V * 4);
    int* cnt = (int*) be.alloc(4);
    for (int k : {1, 31, 32, 33, 1000, 2048}) {
        glmd::Mla2 x = a;
        x.n_sel = k;
        x.out = o1;
        const bool ok1 = glmd::mla2(x, be.s);
        be.put(cnt, &k, 4);
        glmd::Mla2 y = a;
        y.n_sel = cap;
        y.n_sel_dev = cnt;
        y.out = o2;
        const bool ok2 = glmd::mla2(y, be.s);
        std::vector<float> r1((size_t) H * V), r2((size_t) H * V);
        be.get(r1.data(), o1, r1.size() * 4);
        be.get(r2.data(), o2, r2.size() * 4);
        const double e = mixfx::rel_l2(r2.data(), r1);
        check(ok1 && ok2 && e < 1e-6, "count " + std::to_string(k) + ": rel L2 " + std::to_string(e));
    }
}

// B14.4: graph capture once, replay at every position
void graph_replay(const mixfx::Model& m) {
    std::printf("[B14.4: CUDA-graph replay of a full + shared layer pair, %d positions, top-%d]\n", m.T, m.d.top_k);
    GpuBe bd, bg;
    mixfx::Harness<GpuBe> hd(bd, m, mixfx::Mut::None, false, 0), hg(bg, m, mixfx::Mut::None, false, 0);
    // capture both layers at position 0 (the device word holds 0)
    cudaGraphExec_t ex[2] = {nullptr, nullptr};
    std::string err;
    bool ok = true;
    {
        const int p0 = 0;
        bg.put(hg.p_word, &p0, 4);
        // the selector's extent is a kernel argument, frozen by the capture: a captured caller sets n_grid to the key's
        // capacity, as the engine does while a graph is live (glm_dsa_mixer.cu: b.n_grid = graph_n_cap); left at 0 the
        // mixer passes p0 + 1 = 1 and every replay selects cell 0 alone.  graph_n_cap(p, 48) = 48 for every p < 40: one
        // key.  hd keeps n_grid 0 (the direct path's p + 1), so a selector that ignored the device position fails.
        hg.b.n_grid = (int) glmd::graph_n_cap(p0, m.d.max_ctx);
        for (int il = 0; il < 2 && ok; ++il) {
            const mixfx::LayerData& Ld = m.L[(size_t) il];
            mixfx::MixerLayer w;
            w.full = Ld.full;
            w.idx_src = Ld.idx_src;
            const auto& lw = hg.lw[(size_t) il];
            w.q_a_norm = lw.qa_norm, w.kv_a_norm = lw.kva_norm, w.ik_ln_w = lw.ln_w, w.ik_ln_b = lw.ln_b;
            w.wk_b = lw.wkb, w.wv_b = lw.wvb;
            mixfx::MixerCache c = hg.cache[(size_t) il];
            if (!w.full) c.ik = nullptr;
            cudaGraph_t gr = nullptr;
            CK(cudaStreamBeginCapture(bg.s, GLMD_CAPTURE_RELAXED));
            ok = strata::core::glmdsa::mixer_layer(hg, hg.d, hg.b, c, w, il, 0, false, hg.st, err);
            CK(cudaStreamEndCapture(bg.s, &gr));
            ok = ok && cudaGraphInstantiate(&ex[il], gr, 0) == cudaSuccess;
            cudaGraphDestroy(gr);
        }
    }
    check(ok, "both layers captured" + (err.empty() ? std::string() : " - " + err));
    if (!ok) return;
    double worst = 0;
    int differ = 0;
    std::vector<float> a, b((size_t) m.d.n_embd);
    for (int p = 0; p < m.T && err.empty(); ++p)
        for (int il = 0; il < 2; ++il) {
            if (!hd.step(il, p, a, err)) break;
            bg.put(hg.p_word, &p, 4);
            bg.put(hg.x_buf, m.x(il, p), (size_t) m.d.n_embd * 4);
            CK(cudaGraphLaunch(ex[il], bg.s));
            bg.get(b.data(), hg.b.mixer, b.size() * 4);
            differ += std::memcmp(a.data(), b.data(), b.size() * 4) == 0 ? 0 : 1;
            worst = std::max(worst, mixfx::rel_l2(b.data(), a));
        }
    for (auto& e : ex) cudaGraphExecDestroy(e);
    char buf[200];
    std::snprintf(buf, sizeof buf, "replay == direct at every position: %d differing outputs, max rel L2 %.2e%s%s", differ,
                  worst, err.empty() ? "" : " - ", err.c_str());
    check(err.empty() && differ == 0, buf);
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("glmd_kernel_parity: no CUDA device - skipped\n");
        return 77;
    }
    std::printf("glmd_kernel_parity: the glm-dsa decode mixer on the device (B4.7)\n");
    {
        const std::vector<bool> pat = {true, false, true, false};
        suite("MID", mixfx::make_model(mixfx::mid_dims(48), pat, 36, 2), true);
    }
    {
        const std::vector<bool> pat = {true, false};
        suite("WIDE", mixfx::make_model(wide_dims(48), pat, 40, 3), false);
    }
    score_vs_maya();
    mla2_count();
    graph_replay(mixfx::make_model(mixfx::mid_dims(48), {true, false}, 40, 4));
    const int le = glmd::launch_errors() + glmd::mixer_launch_errors();
    check(le == 0, "no glmd launch errors or refused shapes (" + std::to_string(le) + ")");
    if (g_fail) {
        std::printf("glmd_kernel_parity: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glmd_kernel_parity: ok\n");
    return 0;
}
