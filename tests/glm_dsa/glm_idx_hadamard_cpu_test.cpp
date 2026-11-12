// tests/glm_dsa/glm_idx_hadamard_cpu_test.cpp - glm-dsa B14.11 (CPU only): the 128-point Walsh-Hadamard rotation of the
// glm-dsa indexer keys and queries (MAYA_GLM_IDX_HADAMARD, GlmStateLayout::ik_had).
//   1. the transform: orthonormal (norm kept, H H x = x), ik's order and scale, the kernels' serial bits;
//   2. scores: dsa_prep (keys) + dsa_rope_q (queries) + idx_score through the twins with and without the rotation are
//      equal within float rounding, for F32 keys, at GLM-5.3's 32 x 128 indexer;
//   3. FP16 keys on adversarial outlier keys (a few LayerNorm channels with gains that push the key past FP16's
//      65504, as an outlier channel would): the top-2048 overlap with the F32 unrotated selection is at least as high
//      with the rotation as without (and no NaN score), and on plain random keys it stays within a near-tie of it;
//   4. the decode mixer through the twins (tests/glm_dsa/glm_dsa_mixer_fixture.hpp) with rotated F32 / FP16 keys:
//      outputs and selections against the unrotated reference layer;
//   5. the state layout: the switch marks the FP16 / INT8 carves (ik_had 1) but not the slow F32 one, the slot
//      fingerprint differs by kGlmFpIdxHad (a slot from one setting is refused by the other), an idx_key that is not
//      a power of two is refused by name, Flash never rotates.
#include "glm_dsa_mixer_fixture.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"
#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_fast.hpp"
#include "strata/kernels/glmd_mixer.hpp"

#include "../../src/kernels/cuda/glmd_had.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- 1. the transform
void test_transform() {
    std::printf("[1. the transform]\n");
    mixfx::Rng r(11);
    for (int n : {2, 8, 64, 128, 256, 512}) {
        std::vector<float> x(n), y;
        double n2 = 0;
        for (auto& v : x) {
            v = r.normal();
            n2 += (double) v * v;
        }
        y = x;
        glmd::twin::hadamard(y.data(), n);
        double m2 = 0;
        for (float v : y) m2 += (double) v * v;
        std::vector<float> z = y;
        glmd::twin::hadamard(z.data(), n);
        double back = 0;
        for (int i = 0; i < n; ++i) back = std::max(back, (double) std::fabs(z[i] - x[i]));
        char buf[160];
        std::snprintf(buf, sizeof buf, "n %d: |Hx|^2 / |x|^2 - 1 = %.1e, max |HHx - x| = %.1e", n, m2 / n2 - 1, back);
        check(std::fabs(m2 / n2 - 1) < 1e-5 && back < 1e-5, buf);
    }
    // the first basis vector goes to the constant row 1/sqrt(n); H is symmetric (row 1 of H = column 1)
    std::vector<float> e(128, 0.0f);
    e[0] = 1.0f;
    glmd::twin::hadamard(e.data(), 128);
    bool flat = true;
    for (float v : e) flat = flat && std::fabs(v - 1.0f / std::sqrt(128.0f)) < 1e-6f;
    std::vector<float> e1(128, 0.0f);
    e1[1] = 1.0f;
    glmd::twin::hadamard(e1.data(), 128);
    bool alt = true;
    for (int i = 0; i < 128; ++i) alt = alt && std::fabs(e1[i] - (i % 2 ? -1.0f : 1.0f) / std::sqrt(128.0f)) < 1e-6f;
    check(flat && alt, "H e0 = 1/sqrt(128) everywhere, H e1 alternates (ik's fast_ht butterfly order)");
    check(glmd::had_len_ok(128) && glmd::had_len_ok(2) && glmd::had_len_ok(512) && !glmd::had_len_ok(96) &&
              !glmd::had_len_ok(1024) && !glmd::had_len_ok(0),
          "lengths: powers of two in [2, 512]");
}

// ---------------------------------------------------------------- 2. / 3. keys, queries and scores through the twins
struct IdxSet {
    int H = 32, D = 128, R = 64, n = 0;
    std::vector<float> ik_raw, ln_w, ln_b, iq_raw, w, inv;
};

IdxSet make_set(int n, uint64_t seed, int outliers, float gain) {
    IdxSet s;
    s.n = n;
    mixfx::Rng r(seed);
    s.ik_raw = mixfx::rnd(r, (size_t) n * s.D, 1.0f);
    s.ln_w = mixfx::rnd(r, s.D, 0.2f, 1.0f);
    s.ln_b = mixfx::rnd(r, s.D, 0.3f);
    for (int k = 0; k < outliers; ++k) s.ln_w[(size_t) (3 + 37 * k) % s.D] = gain * (k % 2 ? -1.0f : 1.0f);
    s.iq_raw = mixfx::rnd(r, (size_t) s.H * s.D, 1.0f);
    s.w = mixfx::rnd(r, s.H, 1.0f / std::sqrt((float) (s.H * s.D)));
    s.inv.resize((size_t) s.R / 2);
    glmd::rope_inv_freq(s.inv.data(), s.R, 8e6);
    return s;
}

// keys of cells 0..n-1 (cell = position) and the query at position n - 1, then the token scores over the n cells
std::vector<float> scores(const IdxSet& s, int ik_f16, int had) {
    const int H = s.H, D = s.D, n = s.n;
    std::vector<float> keys((size_t) n * D);   // F32, or FP16 rows in the first half (cast)
    std::vector<float> qr_raw(32, 1.0f), qnorm(32, 1.0f), kv_raw(32 + 2, 0.5f), kv_norm(32, 1.0f);
    std::vector<uint16_t> lat((size_t) n * 32), k_pe((size_t) n * 2);
    std::vector<float> kinv(1, 1.0f);
    for (int p = 0; p < n; ++p) {
        glmd::DsaPrep a;
        a.qr_raw = qr_raw.data();
        a.q_a_norm = qnorm.data();
        a.q_lora = 32;
        a.kv_raw = kv_raw.data();
        a.kv_norm = kv_norm.data();
        a.lat = lat.data();
        a.kv_lora = 32;
        a.k_pe = k_pe.data();
        a.qk_rope = 2;
        a.inv_freq = kinv.data();
        a.ik_raw = s.ik_raw.data() + (size_t) p * D;
        a.ik_ln_w = s.ln_w.data();
        a.ik_ln_b = s.ln_b.data();
        a.idx_ln_eps = 1e-6f;
        a.ik_cache = keys.data();
        a.ik_f16 = ik_f16;
        a.ik_had = had;
        a.idx_dim = D;
        a.idx_rope = s.R;
        a.idx_inv_freq = s.inv.data();
        a.p = p;
        glmd::twin::dsa_prep(a);
    }
    std::vector<float> iq = s.iq_raw, q(256, 0.0f);
    glmd::DsaRopeQ r;
    r.q = q.data();
    r.n_head = 1;
    r.q_stride = 256;
    r.rot_off = 192;
    r.n_rot = 2;
    r.inv_freq = kinv.data();
    r.iq = iq.data();
    r.idx_heads = H;
    r.idx_dim = D;
    r.idx_rope = s.R;
    r.idx_inv_freq = s.inv.data();
    r.iq_had = had;
    r.p = n - 1;
    glmd::twin::dsa_rope_q(r);
    std::vector<float> sc((size_t) n, 0.0f);
    glmd::IdxScore x;
    x.iq = iq.data();
    x.ik_cache = keys.data();
    x.w = s.w.data();
    x.idx_heads = H;
    x.idx_dim = D;
    x.n_cap = n;
    x.p = n - 1;
    x.score = sc.data();
    x.key_f16 = ik_f16;
    glmd::twin::idx_score(x);
    return sc;
}

std::vector<int> top_set(std::vector<float> sc, int top, int* n_nan) {
    int nan = 0;
    for (float& v : sc)
        if (std::isnan(v)) {   // the selectors' contract forbids NaN: what a NaN-guarded selector would see
            v = -INFINITY;
            ++nan;
        }
    if (n_nan) *n_nan = nan;
    std::vector<int> sel((size_t) top);
    glmd::twin::topk(sc.data(), (int) sc.size(), (int) sc.size(), 1, nullptr, top, sel.data());
    return sel;
}

int overlap(const std::vector<int>& a, const std::vector<int>& b) {
    std::set<int> sa(a.begin(), a.end());
    int o = 0;
    for (int c : b) o += sa.count(c) ? 1 : 0;
    return o;
}

void test_scores() {
    std::printf("[2. scores with and without the rotation (F32 keys, 32 x 128 indexer)]\n");
    const IdxSet s = make_set(3000, 21, 0, 0.0f);
    const std::vector<float> a = scores(s, 0, 0), b = scores(s, 0, 1);
    double scale = 0, md = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        scale = std::max(scale, (double) std::fabs(a[i]));
        md = std::max(md, (double) std::fabs(a[i] - b[i]));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "max |score(H) - score| = %.2e of max |score| %.3f (%.1e relative)", md, scale,
                  md / scale);
    check(md <= 2e-5 * scale, buf);
    int nz = 0;
    for (float v : a) nz += v != 0.0f;
    check(nz > (int) a.size() / 2, "the scores are not degenerate (" + std::to_string(nz) + " non-zero)");
}

void test_fp16_outliers() {
    std::printf("[3. FP16 keys: top-2048 overlap with the F32 unrotated selection]\n");
    const int top = glmd::kTopK;
    struct Case {
        const char* name;
        int outliers;
        float gain;
        bool gate;   // adversarial: the rotation must not lose
    };
    for (const Case& c : {Case{"adversarial: 2 outlier channels, gain 6e4 (keys past 65504)", 2, 6e4f, true},
                          Case{"adversarial: 4 outlier channels, gain 3e4", 4, 3e4f, true},
                          Case{"plain random keys (no outlier)", 0, 0.0f, false}}) {
        const IdxSet s = make_set(6000, 31 + (uint64_t) c.outliers, c.outliers, c.gain);
        const std::vector<int> ref = top_set(scores(s, 0, 0), top, nullptr);
        int nan_plain = 0, nan_had = 0;
        const std::vector<int> plain = top_set(scores(s, 1, 0), top, &nan_plain);
        const std::vector<int> had = top_set(scores(s, 1, 1), top, &nan_had);
        const int op = overlap(ref, plain), oh = overlap(ref, had);
        char buf[300];
        std::snprintf(buf, sizeof buf, "%s: overlap FP16 %d / %d (%d NaN scores), FP16 + Hadamard %d / %d (%d NaN)",
                      c.name, op, top, nan_plain, oh, top, nan_had);
        if (c.gate)
            check(oh >= op && nan_had == 0 && oh >= top - 8, buf);
        else
            check(nan_had == 0 && nan_plain == 0 && oh >= top - 8 && op >= top - 8, buf);   // near-ties only
    }
}

// ---------------------------------------------------------------- 4. the decode mixer through the twins
struct TwinBe {
    std::vector<std::unique_ptr<uint8_t[]>> mem;
    const int* cur_p = nullptr;
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

void test_mixer(const char* name, const mixfx::MixerDims& d, const std::vector<bool>& pattern, int T) {
    const mixfx::Model m = mixfx::make_model(d, pattern, T, 7);
    for (int ik16 = 0; ik16 < 2; ++ik16) {
        mixfx::RunStats rs[2];
        for (int had = 0; had < 2; ++had) {
            TwinBe be;
            mixfx::Harness<TwinBe> h(be, m, mixfx::Mut::None, false, 0);
            be.cur_p = &h.cur_p;
            for (auto& c : h.cache) {
                c.ik_f16 = ik16;
                c.ik_had = had;
            }
            rs[had] = mixfx::run_stack(h, m, false);
        }
        char buf[300];
        std::snprintf(buf, sizeof buf,
                      "%s, %s keys: rotated max rel L2 %.2e mean %.2e (unrotated %.2e), selections equal to the "
                      "reference %d / %d (unrotated %d / %d)",
                      name, ik16 ? "FP16" : "F32", rs[1].max_rel, rs[1].mean_rel, rs[0].max_rel, rs[1].sel_equal,
                      rs[1].sel_checked,
                      rs[0].sel_equal, rs[0].sel_checked);
        // F32 keys: the rotation changes nothing but float rounding.  FP16 keys: rounding the rotated key differs
        // from rounding the plain one, so a near-tie selection may flip (one set of the run at most); the mean error
        // stays at the unrotated level
        const bool sel_ok = rs[1].sel_equal >= rs[0].sel_equal - (ik16 ? 1 : 0);
        const bool err_ok = ik16 ? rs[1].mean_rel < 5e-3 : rs[1].max_rel < 5e-3;
        check(rs[0].err.empty() && rs[1].err.empty() && rs[1].sel_checked > 0 && sel_ok && err_ok &&
                  rs[1].sel_count_bad == 0,
              buf + (rs[1].err.empty() ? std::string() : " - " + rs[1].err));
    }
}

// ---------------------------------------------------------------- 5. the state layout
strata::core::Glm5Geometry glm53(int idx_key = 128) {
    using namespace strata::core;
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
    g.idx_key = idx_key;
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

void test_layout() {
    using namespace strata::core;
    std::printf("[5. the state layout and the slot fingerprint]\n");
    unsetenv("MAYA_GLM_IDX_HADAMARD");
    unsetenv("MAYA_GLM_IDX_FP16");
    const Glm5Geometry g = glm53();
    std::string err;
    GlmStateLayout off, on, on8, onS, on16;
    check(!glm_state_idx_hadamard() && glm_state_carve(g, 0, 26, 78, 4096, GlmStateMode::Fp16, 0, off, err) &&
              off.ik_had == 0,
          "default: no rotation");
    setenv("MAYA_GLM_IDX_HADAMARD", "1", 1);
    check(glm_state_idx_hadamard() && glm_state_carve(g, 0, 26, 78, 4096, GlmStateMode::Fp16, 0, on, err) &&
              glm_state_carve(g, 0, 26, 78, 4096, GlmStateMode::Int8, 0, on8, err) &&
              glm_state_carve(g, 0, 26, 78, 4096, GlmStateMode::SlowF32, 0, onS, err) && on.ik_had == 1 &&
              on8.ik_had == 1 && onS.ik_had == 0,
          "MAYA_GLM_IDX_HADAMARD=1: FP16 / INT8 carves rotate, the slow F32 reference does not");
    check(on.bytes == off.bytes && on.ik_pp == off.ik_pp, "the rotation costs no state bytes");
    const GlmStateFingerprint f0 = glm_state_fingerprint(g, off), f1 = glm_state_fingerprint(g, on);
    check(f0 != f1 && f1[10] == (4 | kGlmFpIdxHad) && f0[10] == 4,
          "fingerprint word 10: 4 unrotated, 4 | 0x100 rotated (a slot of one setting is refused by the other)");
    setenv("MAYA_GLM_IDX_FP16", "1", 1);
    check(glm_state_carve(g, 0, 26, 78, 4096, GlmStateMode::Fp16, 0, on16, err) && on16.ik_had == 1 &&
              glm_state_fingerprint(g, on16)[10] == (2 | kGlmFpIdxHad) &&
              on16.describe().find("Hadamard-rotated") != std::string::npos,
          "with FP16 keys: word 10 = 2 | 0x100, the load line names the rotation");
    unsetenv("MAYA_GLM_IDX_FP16");
    GlmStateLayout bad;
    err.clear();
    const bool refused = !glm_state_carve(glm53(96), 0, 26, 78, 4096, GlmStateMode::Fp16, 0, bad, err);
    check(refused && err.find("MAYA_GLM_IDX_HADAMARD") != std::string::npos, "idx_key 96 is refused by name: " + err);
    Glm5Geometry flash = g;
    flash.arch = GlmArch::Glm5Next;
    GlmStateLayout fl;
    std::string ferr;
    const bool fok = glm_state_carve(flash, 0, 1, -1, 1024, GlmStateMode::Fp16, 1024, fl, ferr);
    check(!fok || fl.ik_had == 0, "Flash never rotates" + (fok ? std::string() : " (its carve refused: " + ferr + ")"));
    unsetenv("MAYA_GLM_IDX_HADAMARD");
}

}  // namespace

int main() {
    test_transform();
    test_scores();
    test_fp16_outliers();
    std::printf("[4. the decode mixer through the twins, rotated keys and queries]\n");
    test_mixer("small (idx 16 x 64, top-8)", mixfx::small_dims(64), {true, false, true, false}, 24);
    test_mixer("mid (idx 8 x 128, top-16)", mixfx::mid_dims(64), {true, false, false, true}, 28);
    test_layout();
    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
