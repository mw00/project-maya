// tests/glm_dsa/glm_dsa_wire_cpu_test.cpp - glm-dsa wave 5: the Volta kernels wired into the glm-dsa decode mixer, on the
// CPU twins (plain C++: runs on any build, no GPU).
//
//   1. the dispatch policy (include/strata/kernels/glmd_decode_policy.hpp): switch parsing, the documented thresholds
//      (WMMA scorer above 16K cells, radix top-k above 32K, both forced by 1 and disabled by 0, fused MLA only on), and
//      the graph rule: within one policy_regime() every decision is the same (a captured part never straddles a kernel
//      switch);
//   2. the reconciled indexer structs: wmma_args() / fused_args() carry every field of the mixer's IdxScore / Mla2;
//   3. the decode mixer (glmdsa::mixer_layer, the code fast_dsa_glm runs) through the policy twins
//      (glmd::twin::decode_score / decode_topk / decode_mla) with every switch forced on, FP16 index keys and native
//      Q8_0 W_kb / W_vb, against the B3.4 reference: the same gates as the wave-4 mixer test, looser where the
//      dtypes round (FP16 keys, Q8_0 weights, the fused kernel's FP16 Q / P);
//   4. a decode graph's fixed extents (MixerBufs::n_grid = graph_n_cap, the device cell count preset as glmd::DevPos
//      does) give bit-identical outputs and selections to the per-token extents;
//   5. the prompt path and the decode path agree on the caches (the cache-consistency half of B9.2 / B10.8): rows
//      written by twin::dsa_prep_rows (glm_prefill_dsa.cu's call) and by twin::dsa_prep one token at a time
//      (fast_dsa_glm's call) are byte-identical for FP16 / INT8 latents, k_pe and F32 / FP16 index keys; and the
//      prompt path's tiled selection over those keys (index_score_tile_keys + topk_merge_tile, as dsa_layer runs it)
//      matches the decode path's (decode_score + decode_topk) row by row outside near ties.
#include "glm_dsa_mixer_fixture.hpp"

#include "strata/kernels/glmd_batch.hpp"
#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_graph.hpp"

#include <cstdio>
#include <memory>
#include <set>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- 1. the policy
void test_policy() {
    std::printf("[the dispatch policy]\n");
    using namespace glmd;
    check(parse_switch(nullptr, kSwitchAuto) == kSwitchAuto && parse_switch("", kSwitchOff) == kSwitchOff &&
              parse_switch("auto", kSwitchOff) == kSwitchAuto && parse_switch("0", kSwitchOn) == kSwitchOff &&
              parse_switch("1", kSwitchOff) == kSwitchOn && parse_switch("yes", kSwitchOff) == kSwitchOn,
          "parse_switch: unset -> default, auto, 0 off, anything else on");
    unsetenv("STRATA_GLMD_FUSED_MLA");
    unsetenv("STRATA_GLMD_IDX_WMMA");
    unsetenv("STRATA_GLMD_TOPK_MB");
    unsetenv("STRATA_GLMD_FUSED_SPLIT");
    DecodePolicy e = decode_policy_env();
    check(!e.fused_mla && e.idx_wmma == kSwitchAuto && e.topk_mb == kSwitchAuto && !e.idx_fp16 &&
              e.fused_split == kFusedSplitDefault,
          "engine defaults: mla2, WMMA scorer auto, radix top-k auto, F32 keys (the layout sets FP16)");
    setenv("STRATA_GLMD_FUSED_MLA", "1", 1);
    setenv("STRATA_GLMD_IDX_WMMA", "0", 1);
    setenv("STRATA_GLMD_TOPK_MB", "1", 1);
    setenv("STRATA_GLMD_FUSED_SPLIT", "8", 1);
    e = decode_policy_env();
    check(e.fused_mla && e.idx_wmma == kSwitchOff && e.topk_mb == kSwitchOn && e.fused_split == 8,
          "the switches as set: fused on, WMMA off, radix forced, split 8");
    unsetenv("STRATA_GLMD_FUSED_MLA");
    unsetenv("STRATA_GLMD_IDX_WMMA");
    unsetenv("STRATA_GLMD_TOPK_MB");
    unsetenv("STRATA_GLMD_FUSED_SPLIT");
    DecodePolicy a;   // the engine defaults
    a.idx_wmma = kSwitchAuto;
    a.topk_mb = kSwitchAuto;
    const int K = 2048;
    check(!use_idx_wmma(a, 16384, K, 32, 128) && use_idx_wmma(a, 16385, K, 32, 128) &&
              !use_idx_wmma(a, 100000, K, 8, 128) && !use_idx_wmma(a, 100000, K, 32, 512),
          "auto WMMA scorer: n > 16384 and the WMMA shape (idx_heads % 16, <= 32; idx_dim % 16, <= 256)");
    check(!use_idx_wmma(a, 100000, K, 32, 32) && !use_idx_wmma(a, 100000, K, 16, 16) &&
              use_idx_wmma(a, 100000, K, 32, 64) && use_idx_wmma(a, 100000, K, 16, 32),
          "auto WMMA scorer: the f32 scratch must fit the key tile (2 * idx_heads <= idx_dim + 8)");
    check(!use_topk_mb(a, 32768, K) && use_topk_mb(a, 32769, K), "auto radix top-k: n > 32768");
    DecodePolicy f = a;
    f.idx_wmma = kSwitchOn;
    f.topk_mb = kSwitchOn;
    check(!use_idx_wmma(f, 2048, K, 32, 128) && use_idx_wmma(f, 2049, K, 32, 128) && !use_topk_mb(f, 2048, K) &&
              use_topk_mb(f, 2049, K),
          "forced (1): as soon as the selection is active (n > top_k)");
    DecodePolicy o;
    o.idx_wmma = kSwitchOff;
    o.topk_mb = kSwitchOff;
    check(!use_idx_wmma(o, 1 << 20, K, 32, 128) && !use_topk_mb(o, 1 << 20, K), "off (0): never");
    check(!use_fused_mla(a, 64, 192, 64, 512, 256, kWTypeBF16), "fused MLA off by default");
    DecodePolicy fu = a;
    fu.fused_mla = true;
    check(use_fused_mla(fu, 64, 192, 64, 512, 256, kWTypeBF16) && use_fused_mla(fu, 64, 192, 64, 512, 256, kWTypeQ8_0) &&
              !use_fused_mla(fu, 64, 192, 64, 256, 256, kWTypeBF16) && !use_fused_mla(fu, 128, 192, 64, 512, 256, kWTypeBF16),
          "fused MLA on: GLM-5.3's shape in BF16 and Q8_0; kv_lora 256 / 128 heads stay on mla2");
    check(decode_mla_work_bytes(fu, 64, 512, 64, 2048) >= mla_fused_work_bytes(64, 512, 64, 2048, fu.fused_split) &&
              decode_mla_work_bytes(fu, 64, 512, 64, 2048) >= mla2_work_bytes(64, 512, 2048) &&
              decode_mla_work_bytes(a, 64, 512, 64, 2048) == mla2_work_bytes(64, 512, 2048),
          "the attention scratch covers the kernel the policy runs");
    // the graph rule: one regime, one set of decisions (and the bits see every switch)
    bool same = true;
    for (const DecodePolicy* P : {&a, &f, &o}) {
        int last_regime = -1;
        bool w0 = false, m0 = false;
        for (int64_t p = 0; p < 140000; p += (p < 40000 ? 1 : 97)) {
            const int r = policy_regime(*P, p, K);
            const bool w = use_idx_wmma(*P, p + 1, K, 32, 128), m = use_topk_mb(*P, p + 1, K);
            if (r == last_regime && (w != w0 || m != m0)) same = false;
            last_regime = r, w0 = w, m0 = m;
        }
    }
    check(same, "within one graph regime the scorer and selector choices never change (auto, forced, off)");
    check(policy_bits(a, false, false, false) != policy_bits(fu, false, false, false) &&
              policy_bits(a, false, false, false) != policy_bits(o, false, false, false),
          "the graph kernel bits change with the switches");
    DecodePolicy a16 = a;
    a16.idx_fp16 = true;
    check(policy_bits(a, false, false, false) != policy_bits(a16, false, false, false) &&
              policy_bits(a, true, false, false) != policy_bits(a, false, false, false),
          "... and with the key dtype and the latent dtype");
}

// ---------------------------------------------------------------- 2. the struct reconciliation
void test_args() {
    std::printf("[IdxScore -> IdxScoreWmma, Mla2 -> FusedMla]\n");
    float iq[4], w[2], score[8];
    uint16_t keys[8];
    int pos = 5, cnt = 0;
    glmd::IdxScore s;
    s.iq = iq;
    s.w = w;
    s.ik_cache = (const float*) keys;
    s.key_f16 = 1;
    s.idx_heads = 16;
    s.idx_dim = 64;
    s.n_cap = 4096;
    s.p = 5;
    s.p_dev = &pos;
    s.score = score;
    s.skip_le = 2048;
    s.n_sel_out = &cnt;
    glmd::IdxScoreWmma x = glmd::wmma_args(s);
    check(x.iq == iq && x.w == w && x.keys == (const void*) keys && x.key_f16 == 1 && x.idx_heads == 16 &&
              x.idx_dim == 64 && x.T == 1 && x.pos == &pos && x.score == score && x.ld == 4096 && x.n_kv == 6,
          "wmma_args: every field; extent p + 1 = 6, ld the cache capacity");
    s.n_grid = 8192;
    x = glmd::wmma_args(s);
    check(x.n_kv == 4096 && x.ld == 4096, "... a graph's n_grid capped at the capacity");
    s.n_cap = 1 << 20;
    x = glmd::wmma_args(s);
    check(x.n_kv == 8192 && x.ld == (1 << 20), "... the graph extent below the capacity");
    glmd::Mla2 m;
    int cells[4];
    m.q = iq;
    m.q_stride = 256;
    m.wk_b = keys;
    m.wv_b = keys + 1;
    m.w_type = glmd::kWTypeQ8_0;
    m.lat = keys + 2;
    m.lat_q8 = 1;
    m.k_pe = keys + 3;
    m.cells = cells;
    m.n_sel = 2048;
    m.n_head = 64;
    m.qk_nope = 192;
    m.qk_rope = 64;
    m.kv_lora = 512;
    m.v_head = 256;
    m.scale = 0.0625f;
    m.work = score;
    m.work_bytes = 123;
    m.out = w;
    m.out_q8_1 = iq + 1;
    const glmd::FusedMla fm = glmd::fused_args(m, 16);
    check(fm.q == m.q && fm.q_stride == 256 && fm.wk_b == m.wk_b && fm.wv_b == m.wv_b && fm.w_type == glmd::kWTypeQ8_0 &&
              fm.lat == m.lat && fm.lat_q8 == 1 && fm.k_pe == m.k_pe && fm.cells == cells && fm.n_sel == 2048 &&
              fm.n_head == 64 && fm.qk_nope == 192 && fm.qk_rope == 64 && fm.kv_lora == 512 && fm.v_head == 256 &&
              fm.scale == 0.0625f && fm.n_split == 16 && fm.work == m.work && fm.work_bytes == 123 && fm.out == m.out &&
              fm.out_q8_1 == m.out_q8_1,
          "fused_args: every field, the split from the policy");
}

// ---------------------------------------------------------------- 3. / 4. the mixer through the policy twins
struct PolicyBe {
    std::vector<std::unique_ptr<uint8_t[]>> mem;
    glmd::DecodePolicy pol;
    const int* cur_p = nullptr;   // the harness's position (the policy's n = p + 1)
    int n_wmma = 0, n_mb = 0, n_fused = 0, n_calls = 0;
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
    bool score(const glmd::IdxScore& a) {
        n_wmma += glmd::use_idx_wmma(pol, *cur_p + 1, a.skip_le, a.idx_heads, a.idx_dim) ? 1 : 0;
        glmd::twin::decode_score(pol, a, *cur_p);
        return true;
    }
    bool topk(const float* s, int ld, int n_kv, const int* pos, int top_k, int* sel) {
        n_mb += glmd::use_topk_mb(pol, *cur_p + 1, top_k) ? 1 : 0;
        glmd::twin::decode_topk(pol, s, ld, n_kv, pos, *cur_p, top_k, sel);
        return true;
    }
    bool mla2(const glmd::Mla2& a) {
        ++n_calls;
        n_fused += glmd::use_fused_mla(pol, a.n_head, a.qk_nope, a.qk_rope, a.kv_lora, a.v_head, a.w_type) ? 1 : 0;
        glmd::twin::decode_mla(pol, a);
        return true;
    }
};

struct WireRun {
    mixfx::RunStats rs;
    int n_wmma = 0, n_mb = 0, n_fused = 0, n_calls = 0;
};

WireRun run_wire(const mixfx::Model& m, const glmd::DecodePolicy& pol, int lat_q8, int ik_f16, int w_type) {
    PolicyBe be;
    be.pol = pol;
    mixfx::Model mm = m;
    mm.d.pol = pol;
    mixfx::Harness<PolicyBe> h(be, mm, mixfx::Mut::None, false, lat_q8, w_type);
    be.cur_p = &h.cur_p;
    for (auto& c : h.cache) c.ik_f16 = ik_f16;
    WireRun r;
    r.rs = mixfx::run_stack(h, mm, false);
    r.n_wmma = be.n_wmma, r.n_mb = be.n_mb, r.n_fused = be.n_fused, r.n_calls = be.n_calls;
    return r;
}

void test_mixer(const char* name, const mixfx::Model& m) {
    std::printf("[%s: the mixer through the policy twins: %d heads, kv_lora %d, indexer %d x %d, top-%d, %d positions]\n",
                name, m.d.n_head, m.d.kv_lora, m.d.idx_heads, m.d.idx_key, m.d.top_k, m.T);
    char buf[300];
    glmd::DecodePolicy base;   // all off: the wave-4 kernels
    const WireRun b0 = run_wire(m, base, 0, 0, glmd::kWTypeBF16);
    std::snprintf(buf, sizeof buf, "baseline (all switches off): max rel L2 %.2e, top-k sets %d / %d", b0.rs.max_rel,
                  b0.rs.sel_equal, b0.rs.sel_checked);
    check(b0.rs.err.empty() && b0.rs.max_rel < 5e-3 && b0.rs.sel_checked > 0 && b0.rs.sel_equal == b0.rs.sel_checked &&
              b0.n_wmma == 0 && b0.n_mb == 0 && b0.n_fused == 0,
          buf + (b0.rs.err.empty() ? std::string() : " - " + b0.rs.err));
    glmd::DecodePolicy on;
    on.fused_mla = true;
    on.fused_split = 4;
    on.idx_wmma = glmd::kSwitchOn;
    on.topk_mb = glmd::kSwitchOn;
    const bool wmma_shape = glmd::idx_wmma_shape(m.d.idx_heads, m.d.idx_key);
    const bool fused_shape = glmd::mla_fused_supported(m.d.n_head, m.d.qk_nope, m.d.qk_rope, m.d.kv_lora, m.d.v_head,
                                                       glmd::kWTypeBF16);
    for (int ik16 = 0; ik16 < 2; ++ik16) {
        on.idx_fp16 = ik16 != 0;
        const WireRun r = run_wire(m, on, 0, ik16, glmd::kWTypeBF16);
        const double sel = r.rs.sel_checked ? (double) r.rs.sel_equal / r.rs.sel_checked : 0.0;
        std::snprintf(buf, sizeof buf,
                      "every switch forced, %s keys: max rel L2 %.2e (mean %.2e), top-k sets equal %d / %d; %d WMMA "
                      "scores, %d radix selections, %d / %d fused attentions",
                      ik16 ? "FP16" : "F32", r.rs.max_rel, r.rs.mean_rel, r.rs.sel_equal, r.rs.sel_checked, r.n_wmma,
                      r.n_mb, r.n_fused, r.n_calls);
        check(r.rs.err.empty() && r.rs.max_rel < 2e-2 && r.rs.mean_rel < 5e-3 && sel >= 0.9 && r.rs.sel_count_bad == 0 &&
                  (r.n_wmma > 0) == wmma_shape && r.n_mb > 0 && (r.n_fused == r.n_calls) == fused_shape,
              buf + (r.rs.err.empty() ? std::string() : " - " + r.rs.err));
    }
    if (m.d.qk_nope % 32 == 0 && m.d.kv_lora % 32 == 0) {   // B10.7: native Q8_0 W_kb / W_vb through both attentions
        glmd::DecodePolicy q = base;
        const WireRun r2 = run_wire(m, q, 0, 0, glmd::kWTypeQ8_0);
        q.fused_mla = true;
        q.fused_split = 4;
        const WireRun rf = run_wire(m, q, 0, 0, glmd::kWTypeQ8_0);
        std::snprintf(buf, sizeof buf, "native Q8_0 W_kb / W_vb: mla2 max rel L2 %.2e, %s max rel L2 %.2e",
                      r2.rs.max_rel, fused_shape ? "mla_fused" : "mla2 (fused refuses the shape)", rf.rs.max_rel);
        check(r2.rs.err.empty() && rf.rs.err.empty() && r2.rs.max_rel < 3e-2 && rf.rs.max_rel < 3e-2 &&
                  (rf.n_fused == rf.n_calls) == fused_shape,
              buf + (r2.rs.err + rf.rs.err).substr(0, 120));
    }
    {   // INT8 latents with everything on
        on.idx_fp16 = true;
        const WireRun r = run_wire(m, on, 1, 1, glmd::kWTypeBF16);
        std::snprintf(buf, sizeof buf, "INT8 latents + FP16 keys + every switch: max rel L2 %.2e", r.rs.max_rel);
        check(r.rs.err.empty() && r.rs.max_rel < 4e-2, buf + (r.rs.err.empty() ? std::string() : " - " + r.rs.err));
    }
    // 4. a graph's fixed extents vs the per-token ones: bit-identical
    for (const glmd::DecodePolicy* P : {&base, &on}) {
        PolicyBe b1, b2;
        b1.pol = b2.pol = *P;
        mixfx::Model mm = m;
        mm.d.pol = *P;
        mixfx::Harness<PolicyBe> h1(b1, mm, mixfx::Mut::None, false, 0), h2(b2, mm, mixfx::Mut::None, false, 0);
        b1.cur_p = &h1.cur_p;
        b2.cur_p = &h2.cur_p;
        const int ik16 = P->idx_fp16 ? 1 : 0;
        for (auto& c : h1.cache) c.ik_f16 = ik16;
        for (auto& c : h2.cache) c.ik_f16 = ik16;
        h2.b.n_sel_preset = true;   // the count is DevPos::n_sel, written before the token (glmd::set_dev_pos)
        bool same = true, ok = true;
        std::string err;
        std::vector<float> o1, o2;
        for (int p = 0; p < mm.T && ok; ++p) {
            const int n_sel = std::min(p + 1, mm.d.top_k);
            h2.be.put(h2.b.n_sel_dev, &n_sel, 4);
            h2.b.n_grid = (int) glmd::graph_n_cap(p, mm.d.max_ctx);
            for (int il = 0; il < (int) mm.L.size() && ok; ++il) {
                ok = h1.step(il, p, o1, err) && h2.step(il, p, o2, err);
                if (ok && (o1 != o2 || (mm.L[(size_t) il].full && h1.cells_h != h2.cells_h))) same = false;
            }
        }
        check(ok && same, std::string("decode graph extents (n_grid = graph_n_cap, preset DevPos count) are "
                                      "bit-identical to the per-token ones, switches ") +
                              (P == &base ? "off" : "forced") + (err.empty() ? "" : " - " + err));
    }
}

// ---------------------------------------------------------------- 5. the prompt path vs the decode path
void test_prompt_vs_decode() {
    std::printf("[the prompt path's caches and selection vs the decode path's]\n");
    mixfx::Rng r(77);
    const int QL = 64, KV = 512, R = 64, ID = 128, IH = 16, IR = 64, TOP = 32, N = 300, P0 = 0, KB = 64;
    std::vector<float> qa_norm(QL), kv_norm(KV), ln_w(ID), ln_b(ID), inv(R / 2), inv_i(IR / 2);
    for (auto& v : qa_norm) v = 1.0f + 0.1f * r.normal();
    for (auto& v : kv_norm) v = 1.0f + 0.1f * r.normal();
    for (auto& v : ln_w) v = 1.0f + 0.2f * r.normal();
    for (auto& v : ln_b) v = 0.3f * r.normal();
    glmd::rope_inv_freq(inv.data(), R, 8e6);
    glmd::rope_inv_freq(inv_i.data(), IR, 8e6);
    std::vector<float> qr_raw((size_t) N * QL), kv_raw((size_t) N * (KV + R)), ik_raw((size_t) N * ID);
    for (auto& v : qr_raw) v = r.normal();
    for (auto& v : kv_raw) v = r.normal();
    for (auto& v : ik_raw) v = r.normal();
    bool bytes_equal = true;
    for (int lat_q8 = 0; lat_q8 < 2; ++lat_q8)
        for (int ik16 = 0; ik16 < 2; ++ik16) {
            const size_t lat_b = lat_q8 ? (size_t) glmd::lat8_rec_bytes(KV) : (size_t) KV * 2;
            const size_t ik_b = (size_t) ID * (ik16 ? 2 : 4);
            std::vector<uint8_t> latA(lat_b * N), latB(lat_b * N), ikA(ik_b * N), ikB(ik_b * N);
            std::vector<uint16_t> kpA((size_t) R * N), kpB((size_t) R * N);
            std::vector<float> qrA((size_t) N * QL), qrB(QL);
            glmd::DsaPrepRows pr;
            pr.a.qr_raw = qr_raw.data();
            pr.a.q_a_norm = qa_norm.data();
            pr.a.qr = qrA.data();
            pr.a.q_lora = QL;
            pr.a.kv_raw = kv_raw.data();
            pr.a.kv_norm = kv_norm.data();
            pr.a.lat = latA.data();
            pr.a.lat_q8 = lat_q8;
            pr.a.kv_lora = KV;
            pr.a.k_pe = kpA.data();
            pr.a.qk_rope = R;
            pr.a.inv_freq = inv.data();
            pr.a.ik_raw = ik_raw.data();
            pr.a.ik_ln_w = ln_w.data();
            pr.a.ik_ln_b = ln_b.data();
            pr.a.ik_cache = (float*) ikA.data();
            pr.a.ik_f16 = ik16;
            pr.a.idx_dim = ID;
            pr.a.idx_rope = IR;
            pr.a.idx_inv_freq = inv_i.data();
            pr.a.p = P0;
            pr.T = N;
            pr.kv_ld = KV + R;
            glmd::twin::dsa_prep_rows(pr);
            for (int t = 0; t < N; ++t) {   // the decode path: one token at a time at p = t, its position from p_dev
                glmd::DsaPrep a = pr.a;
                const int p = P0 + t;
                a.qr_raw = qr_raw.data() + (size_t) t * QL;
                a.qr = qrB.data();
                a.kv_raw = kv_raw.data() + (size_t) t * (KV + R);
                a.ik_raw = ik_raw.data() + (size_t) t * ID;
                a.lat = latB.data();
                a.k_pe = kpB.data();
                a.ik_cache = (float*) ikB.data();
                a.p = 0;
                a.p_dev = &p;
                glmd::twin::dsa_prep(a);
            }
            const bool eq = latA == latB && kpA == kpB && ikA == ikB;
            if (!eq) bytes_equal = false;
            check(eq, std::string("caches byte-identical: ") + (lat_q8 ? "INT8" : "FP16") + " latents, k_pe, " +
                          (ik16 ? "FP16" : "F32") + " index keys over " + std::to_string(N) + " positions");
            if (lat_q8) continue;
            // the selection: rows at positions > top_k, the prompt's tiled walk vs the decode scorer + selector
            std::vector<float> iq((size_t) N * IH * ID), iw((size_t) N * IH);
            for (auto& v : iq) v = r.normal();
            for (auto& v : iw) v = std::fabs(r.normal()) / 45.0f;
            const int T = N - (TOP + 1), pq = TOP + 1;
            std::vector<float> tile((size_t) T * KB), rs[2] = {std::vector<float>((size_t) T * TOP), std::vector<float>((size_t) T * TOP)};
            std::vector<int> rc[2] = {std::vector<int>((size_t) T * TOP), std::vector<int>((size_t) T * TOP)},
                             rn[2] = {std::vector<int>(T), std::vector<int>(T)}, sel((size_t) T * TOP);
            const int64_t n_kv = pq + T;
            const int tiles = (int) ((n_kv + KB - 1) / KB);
            const int* ci = nullptr;
            const float* si = nullptr;
            const int* ni = nullptr;
            for (int k = 0; k < tiles; ++k) {
                const int64_t k0 = (int64_t) k * KB;
                const int nk = (int) std::min<int64_t>(KB, n_kv - k0);
                glmd::twin::index_score_tile_keys(iq.data() + (size_t) pq * IH * ID, iw.data() + (size_t) pq * IH,
                                                  ikA.data(), ik16, IH, ID, T, k0, nk, tile.data(), KB);
                glmd::TopkMerge mg;
                mg.score = tile.data();
                mg.ld = KB;
                mg.T = T;
                mg.p0 = pq;
                mg.k0 = k0;
                mg.nk = nk;
                mg.top_k = TOP;
                mg.in_cell = ci;
                mg.in_score = si;
                mg.in_n = ni;
                mg.out_cell = k == tiles - 1 ? sel.data() : rc[k & 1].data();
                mg.out_score = rs[k & 1].data();
                mg.out_n = rn[k & 1].data();
                glmd::twin::topk_merge_tile(mg);
                ci = mg.out_cell, si = mg.out_score, ni = mg.out_n;
            }
            for (int mode = 0; mode < 2; ++mode) {   // decode: the token scorer, then the forced WMMA scorer
                glmd::DecodePolicy pol;
                pol.idx_fp16 = ik16 != 0;
                pol.idx_wmma = mode ? glmd::kSwitchOn : glmd::kSwitchOff;
                pol.topk_mb = mode ? glmd::kSwitchOn : glmd::kSwitchOff;
                int rows_equal = 0;
                double overlap = 0;
                std::vector<float> score(N);
                std::vector<int> cells(TOP);
                int cnt = 0;
                for (int t = 0; t < T; ++t) {
                    const int p = pq + t;
                    glmd::IdxScore s;
                    s.iq = iq.data() + (size_t) p * IH * ID;
                    s.w = iw.data() + (size_t) p * IH;
                    s.ik_cache = (const float*) ikB.data();
                    s.key_f16 = ik16;
                    s.idx_heads = IH;
                    s.idx_dim = ID;
                    s.n_cap = N;
                    s.p = p;   // (the host position: the WMMA extent p + 1; the kernels read p_dev)
                    s.p_dev = &p;
                    s.score = score.data();
                    s.skip_le = TOP;
                    s.n_sel_out = &cnt;
                    s.sel_cap = TOP;
                    glmd::twin::decode_score(pol, s, p);
                    glmd::twin::decode_topk(pol, score.data(), N, p + 1, &p, p, TOP, cells.data());
                    const std::set<int> a(cells.begin(), cells.end()), b(sel.begin() + (size_t) t * TOP,
                                                                         sel.begin() + (size_t) (t + 1) * TOP);
                    rows_equal += a == b ? 1 : 0;
                    int both = 0;
                    for (int c : a) both += b.count(c) ? 1 : 0;
                    overlap += (double) both / TOP;
                    if (cnt != TOP) rows_equal = -1000000;
                }
                overlap /= T;
                char buf[200];
                std::snprintf(buf, sizeof buf, "%s keys, %s scorer: prompt vs decode top-%d sets equal on %d / %d rows "
                                               "(mean overlap %.4f)",
                              ik16 ? "FP16" : "F32", mode ? "WMMA" : "token", TOP, rows_equal, T, overlap);
                check(rows_equal >= 0 && overlap >= (mode ? 0.97 : 0.995), buf);
            }
        }
    (void) bytes_equal;
}

}  // namespace

int main() {
    std::printf("glm_dsa_wire_cpu_test: the Volta kernels in the glm-dsa decode mixer (CPU twins)\n");
    test_policy();
    test_args();
    {   // MID with a WMMA-shaped indexer (16 x 128) and the fused shape (kv_lora 512, rope 64)
        mixfx::MixerDims d = mixfx::mid_dims(64);
        d.idx_heads = 16;
        test_mixer("MID16", mixfx::make_model(d, {true, false, true, false}, 44, 5));
    }
    {   // SMALL: the WMMA shape (16 x 64) but not the fused one (kv_lora 32): the policy keeps mla2
        test_mixer("SMALL", mixfx::make_model(mixfx::small_dims(48), {true, false, false, true}, 30, 6));
    }
    test_prompt_vs_decode();
    if (g_fail) {
        std::printf("glm_dsa_wire_cpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("glm_dsa_wire_cpu_test: ok\n");
    return 0;
}
