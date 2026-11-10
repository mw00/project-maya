// tests/glm_dsa/glm_dsa_spec_test.cpp - glm-dsa B11.2 / B11.3 host side of the 3-GPU speculative decode (CPU only).
//
// Checked:
//   1. pricing: hand-computed H / T / hit / miss / step for a 3-part split at head {0} and {0,1}; the shared CPU-lane
//      pool term (head + tail lane work serialised); the priced choice follows the part costs (a slow last part
//      picks head {0,1}, a slow part 0 picks head {0}, an even split prefers {0,1} at 80% acceptance); acceptance 1
//      makes the step the slower group, 0 the sum (no gain); with STRATA_GLM_SPEC_DRAFT_EARLY's pricing a miss is the
//      hit step plus the head's redo (2H once the head is slower; unchanged while H <= T), which turns the choice of
//      a 10 / 10 / 11 split from {0,1} to {0};
//   2. the head rule: Flash keeps Maya's n / 2 and atoi(env) (also for "auto"); glm-dsa takes the recorded price when
//      unset or "auto", a number otherwise, n / 2 without a price; always clamped to [1, n - 1];
//   3. acceptance accounting per depth and context bucket, the accepted-run length and the redo count;
//   4. the glm-dsa rollback rows on GLM-5.3's layout for the test PC head groups {[0,26)} and {[0,26), [26,54)}, FP16 and
//      INT8 latents and FP16 index keys: one latent + one k_pe row per DSA layer and an index-key row exactly on the
//      full-indexer layers, all disjoint across positions and inside the arena, none on the NextN block unless asked.
#include "strata/core/glm_dsa_spec.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/glm_state_layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

using strata::core::GlmArch;
using strata::core::GlmIdxKind;
using strata::core::GlmStateLayout;
using strata::core::GlmStateMode;
using strata::core::Glm5Geometry;
namespace gs = strata::core::glmspec;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

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

void test_price() {
    std::printf("[pricing]\n");
    // parts 10 / 10 / 12 ms (the last carries the draft), no lane, separate pools, hop 0
    std::vector<gs::PartCost> p = {{0, 10.0, 0.0, 0}, {1, 10.0, 0.0, 1}, {2, 12.0, 0.0, 2}};
    const gs::SpecPrice a = gs::spec_price(p, 1, 0.8, 0.0);
    check(near(a.head_ms, 10) && near(a.tail_ms, 22) && near(a.hit_ms, 22) && near(a.miss_ms, 32) &&
              near(a.step_ms, 0.8 * 22 + 0.2 * 32) && near(a.base_ms, 32) && !a.shared_lane,
          "head {0}: H 10, T 22, hit 22, miss 32, step 24, base 32");
    const gs::SpecPrice b = gs::spec_price(p, 2, 0.8, 0.0);
    check(near(b.head_ms, 20) && near(b.tail_ms, 12) && near(b.hit_ms, 20) && near(b.miss_ms, 32) &&
              near(b.step_ms, 0.8 * 20 + 0.2 * 32),
          "head {0,1}: H 20, T 12, hit 20, miss 32, step 22.4");
    int best = -1;
    auto v = gs::spec_prices(p, 0.8, 0.0, best);
    check(v.size() == 2 && best == 1 && v[(size_t) best].head == 2, "an even split at 80% prices head {0,1}");
    check(near(v[1].speedup, 32.0 / 22.4, 1e-9), "speedup = base / step");
    // a heavy tail (the draft and a big last part): head {0}
    std::vector<gs::PartCost> q = {{0, 8.0, 0.0, 0}, {1, 8.0, 0.0, 1}, {2, 20.0, 0.0, 2}};
    gs::spec_prices(q, 0.8, 0.0, best);
    check(best == 1, "head {0,1} when the last part is the slowest (H 16 vs T 20)");
    std::vector<gs::PartCost> r = {{0, 20.0, 0.0, 0}, {1, 6.0, 0.0, 1}, {2, 6.0, 0.0, 2}};
    gs::spec_prices(r, 0.8, 0.0, best);
    check(best == 0, "head {0} when part 0 is the slowest (H 20 vs T 12, against H 26 vs T 6)");
    // acceptance extremes
    check(near(gs::spec_price(p, 1, 1.0, 0.0).step_ms, 22) && near(gs::spec_price(p, 1, 0.0, 0.0).step_ms, 32),
          "acceptance 1: the slower group; 0: the sum (no gain)");
    // shared lane: lane 6 ms on each part, one pool
    std::vector<gs::PartCost> s = {{0, 10.0, 6.0, 0}, {1, 10.0, 6.0, 0}, {2, 12.0, 6.0, 0}};
    const gs::SpecPrice c = gs::spec_price(s, 2, 1.0, 0.0);
    check(c.shared_lane && near(c.lane_overlap_ms, 18) && near(c.hit_ms, 20),
          "one shared pool: lane overlap 18 ms (12 head + 6 tail) under H 20");
    std::vector<gs::PartCost> s2 = {{0, 10.0, 9.0, 0}, {1, 10.0, 9.0, 0}, {2, 12.0, 9.0, 0}};
    const gs::SpecPrice c2 = gs::spec_price(s2, 2, 1.0, 0.0);
    check(near(c2.hit_ms, 27), "a lane-bound shared pool: the hit step is the lanes' sum (27 > H 20)");
    std::vector<gs::PartCost> s3 = {{0, 10.0, 9.0, 0}, {1, 10.0, 9.0, 0}, {2, 12.0, 9.0, 1}};
    check(!gs::spec_price(s3, 2, 1.0, 0.0).shared_lane && gs::spec_price(s3, 1, 1.0, 0.0).shared_lane,
          "pools by id: head {0,1} on pool 0 vs tail on pool 1 do not contend; head {0} shares pool 0 with part 1");
    // hops
    const gs::SpecPrice h = gs::spec_price(p, 1, 0.8, 0.5);
    check(near(h.head_ms, 10.5) && near(h.tail_ms, 22.5) && near(h.base_ms, 33), "hops: head 1, tail 1, base 2");
    check(near(gs::lane_share(6, 8, 900e9, 20e9), (2 / 20e9) / (6 / 900e9 + 2 / 20e9)) &&
              near(gs::lane_share(8, 8, 900e9, 20e9), 0.0),
          "lane_share from hits / routes at card and lane bandwidth");
    check(gs::spec_price(p, 0, 0.8).step_ms == 0.0 && gs::spec_price(p, 3, 0.8).step_ms == 0.0,
          "head outside [1, n-1] prices nothing");
}

void test_price_redo() {
    std::printf("[pricing: a miss waits for the head's rejected speculation (STRATA_GLM_SPEC_DRAFT_EARLY)]\n");
    // parts 10 / 10 / 11, hop 0: head {0} is H 10 vs T 21, head {0,1} H 20 vs T 11
    std::vector<gs::PartCost> f = {{0, 10.0, 0.0, 0}, {1, 10.0, 0.0, 1}, {2, 11.0, 0.0, 2}};
    const gs::SpecPrice a0 = gs::spec_price(f, 1, 0.8, 0.0), a1 = gs::spec_price(f, 1, 0.8, 0.0, true);
    check(near(a0.miss_ms, 31) && near(a1.miss_ms, 31) && near(a1.step_ms, a0.step_ms),
          "head {0} (H 10 <= T 21): a miss is T + H = 31 either way");
    const gs::SpecPrice b0 = gs::spec_price(f, 2, 0.8, 0.0), b1 = gs::spec_price(f, 2, 0.8, 0.0, true);
    check(near(b0.miss_ms, 31) && near(b1.miss_ms, 40) && near(b1.step_ms, 0.8 * 20 + 0.2 * 40),
          "head {0,1} (H 20 > T 11): T + H = 31 by default, hit + H = 2H = 40 when the redo waits");
    int best0 = -1, best1 = -1;
    gs::spec_prices(f, 0.8, 0.0, best0);
    gs::spec_prices(f, 0.8, 0.0, best1, true);
    check(best0 == 1 && best1 == 0,
          "the default prices head {0,1} (22.2 vs 23.0 ms a token), the waiting redo head {0} (23.0 vs 24.0)");
    // a lane-bound shared pool: the miss is the overlapped step plus the redo
    std::vector<gs::PartCost> s2 = {{0, 10.0, 9.0, 0}, {1, 10.0, 9.0, 0}, {2, 12.0, 9.0, 0}};
    const gs::SpecPrice c0 = gs::spec_price(s2, 2, 0.8, 0.0), c1 = gs::spec_price(s2, 2, 0.8, 0.0, true);
    check(near(c0.miss_ms, 32) && near(c1.hit_ms, 27) && near(c1.miss_ms, 47),
          "a lane-bound shared pool: miss = hit 27 + H 20 = 47 (default T + H = 32, below the hit step)");
    // the switch reaches the split search's price (hop 0.05 there; spec_accept_assumed 0.8)
    gs::spec_clear_prices();
    unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");
    const bool off = !gs::spec_draft_early();
    const int h0 = gs::spec_record_prices(f, false);
    setenv("STRATA_GLM_SPEC_DRAFT_EARLY", "1", 1);
    const bool on = gs::spec_draft_early();
    const int h1 = gs::spec_record_prices(f, false);
    unsetenv("STRATA_GLM_SPEC_DRAFT_EARLY");
    gs::spec_clear_prices();
    check(off && on && h0 == 2 && h1 == 1,
          "STRATA_GLM_SPEC_DRAFT_EARLY unset: the priced head is {0,1}; =1: {0} (read at each call)");
}

void test_head_rule() {
    std::printf("[head rule]\n");
    gs::spec_clear_prices();
    check(gs::spec_head_parts(3, nullptr, false) == 1 && gs::spec_head_parts(4, nullptr, false) == 2 &&
              gs::spec_head_parts(2, nullptr, false) == 1,
          "Flash unset: n / 2 (3 -> 1, 4 -> 2, 2 -> 1)");
    check(gs::spec_head_parts(3, "2", false) == 2 && gs::spec_head_parts(3, "9", false) == 2 &&
              gs::spec_head_parts(3, "0", false) == 1 && gs::spec_head_parts(3, "auto", false) == 1,
          "Flash env: atoi, clamped to [1, n-1] ('auto' reads 0 -> 1, as Maya)");
    check(gs::spec_head_parts(3, nullptr, true) == 1 && gs::spec_head_parts(3, "auto", true) == 1,
          "glm-dsa without a price: n / 2");
    std::vector<gs::PartCost> p = {{0, 10.0, 0.0, 0}, {1, 10.0, 0.0, 1}, {2, 12.0, 0.0, 2}};
    const int ph = gs::spec_record_prices(p, false);
    check(ph == 2 && gs::spec_priced_head(3) == 2 && gs::spec_priced_head(2) == -1, "a recorded price for 3 parts");
    check(gs::spec_head_parts(3, nullptr, true) == 2 && gs::spec_head_parts(3, "auto", true) == 2,
          "glm-dsa unset / auto: the priced head {0,1}");
    check(gs::spec_head_parts(3, "1", true) == 1 && gs::spec_head_parts(3, "2", true) == 2,
          "glm-dsa STRATA_GLM_SPEC_HEAD=1 / 2: the A/B override");
    check(gs::spec_head_parts(3, nullptr, false) == 1, "Flash ignores the glm-dsa price");
    gs::spec_clear_prices();
}

void test_tuner() {
    std::printf("[head A/B tuner]\n");
    gs::HeadTuner t;
    t.n = 3;
    t.window = 100;
    check(t.pick(3, 2) == 2, "nothing measured: the priced head first");
    t.feed(2, 120, 120 * 20.0);
    check(t.pick(3, 2) == 1, "the priced head measured: the other one next");
    t.feed(1, 60, 60 * 25.0);
    check(t.pick(3, 2) == 1, "... until its window is full");
    t.feed(1, 60, 60 * 25.0);
    check(t.pick(3, 2) == 2 && near(t.ms_per_token(2), 20.0) && near(t.ms_per_token(1), 25.0),
          "both measured: the faster (20 vs 25 ms a token)");
    t.feed(1, 1000, 1000 * 10.0);
    check(t.pick(3, 2) == 1, "cumulative: it switches when the averages cross");
    check(t.pick(4, 2) == 2, "another split: nothing measured for it");
    gs::HeadTuner u;
    u.n = 2;
    check(u.pick(2, 1) == 1, "two parts: the only head");
    // the process tuner through spec_head_parts / tune_begin / tune_end (glm-dsa, unset env, no price: n / 2 first)
    gs::tuner_reset();
    gs::spec_clear_prices();
    const int w = gs::spec_tune_window();
    check(w == 128, "default window 128 tokens");
    gs::tune_begin();
    check(gs::spec_head_parts(3, nullptr, true) == 1, "first call: head 1 (n / 2)");
    gs::tune_end(w);
    gs::tune_begin();
    check(gs::spec_head_parts(3, "auto", true) == 2, "head 1 measured over the window: head 2 next");
    gs::tune_end(4);
    check(gs::tuner_state().tokens[2] == 0, "a call that emitted under 8 tokens feeds nothing");
    gs::tune_end(w);
    check(gs::tuner_state().tokens[2] == 0, "an end without a begin feeds nothing");
    gs::tune_begin();
    check(gs::spec_head_parts(3, nullptr, true) == 2, "still head 2 (its window is not full)");
    gs::tune_end(w);
    const gs::HeadTuner st = gs::tuner_state();
    check(st.tokens[1] == w && st.tokens[2] == w, "both heads measured over the window");
    check(gs::spec_head_parts(3, "1", true) == 1 && gs::spec_head_parts(3, "2", true) == 2,
          "STRATA_GLM_SPEC_HEAD=<n> pins a head over the tuner");
    check(gs::spec_head_parts(3, nullptr, false) == 1, "Flash never tunes");
    gs::tuner_reset();
}

// S3: "no drafting" as the tuner's candidate 0 (STRATA_GLM_SPEC_TUNE_PLAIN=1; ctest runs this file with and without)
void test_tuner_plain() {
    std::printf("[head tuner: no drafting as a candidate]\n");
    gs::HeadTuner t;
    t.n = 3;
    t.window = 100;
    t.feed(0, 100, 100 * 5.0);
    check(t.tokens[0] == 0 && t.ms_per_token(0) > 1e29, "plain off: head 0 is never fed nor measured");
    t.plain = true;
    check(t.pick(3, 2) == 2, "plain on, nothing measured: the priced head first");
    t.feed(2, 100, 100 * 20.0);
    t.feed(1, 100, 100 * 25.0);
    check(t.pick(3, 2) == 0, "every head measured: no drafting next");
    t.feed(0, 100, 100 * 19.8);
    check(t.pick(3, 2) == 2, "no drafting within 2% of the best head (19.8 vs 20 ms): drafting stays");
    t.feed(0, 100, 100 * 15.0);
    check(near(t.ms_per_token(0), 17.4) && t.pick(3, 2) == 0, "no drafting 13% faster: it is kept (0)");
    t.feed(2, 1000, 1000 * 10.0);
    check(t.pick(3, 2) == 2, "cumulative: drafting again when its average drops below");
    gs::HeadTuner u;
    u.n = 3;
    u.window = 100;
    check(u.pick(3, 1) == 1 && !u.plain, "a default tuner: no plain candidate");
    // the process path: tune_plain_turn only after the split's first speculative decode set the tuner up
    gs::tuner_reset();
    gs::spec_clear_prices();
    const int w = gs::spec_tune_window();
    check(!gs::tune_plain_turn(3, nullptr), "no tuner for 3 parts yet: drafting first");
    check(!gs::tune_plain_turn(2, nullptr), "two parts: never");
    if (!gs::spec_tune_plain()) {
        gs::tune_begin();
        (void) gs::spec_head_parts(3, nullptr, true);
        gs::tune_end(w);
        gs::tune_begin();
        (void) gs::spec_head_parts(3, nullptr, true);
        gs::tune_end(w);
        check(!gs::tune_plain_turn(3, nullptr) && !gs::tuner_state().plain, "STRATA_GLM_SPEC_TUNE_PLAIN unset: never plain");
        gs::tuner_reset();
        return;
    }
    gs::tune_begin();
    check(gs::spec_head_parts(3, nullptr, true) == 1, "first decode: head 1 (n / 2)");
    gs::tune_end(w);
    check(!gs::tune_plain_turn(3, nullptr), "head 2 is not measured yet: no plain turn");
    gs::tune_begin();
    check(gs::spec_head_parts(3, nullptr, true) == 2, "then head 2");
    gs::tune_end(w);
    check(!gs::tune_plain_turn(3, "1"), "a pinned head: no plain turn");
    check(gs::tune_plain_turn(3, "auto"), "both heads measured: the plain turn");
    gs::tune_end(w);
    const gs::HeadTuner st = gs::tuner_state();
    check(st.plain && st.tokens[0] == w && st.tokens[1] == w && st.tokens[2] == w, "the plain turn fed head 0");
    check(gs::spec_head_parts(3, nullptr, true) >= 1, "spec_head_parts never returns 0");
    gs::tuner_reset();
}

void test_accept() {
    std::printf("[acceptance]\n");
    gs::AcceptStats s;
    const bool pat[10] = {true, true, false, true, true, true, false, false, true, true};
    for (int i = 0; i < 10; ++i) s.note(1, 100 + i, pat[i]);
    s.note(1, 5000, true);
    s.note(1, 5001, false);
    s.note(2, 5002, true);
    check(s.tried[0] == 12 && s.hit[0] == 8 && near(s.rate(1), 8.0 / 12.0), "depth 1: 8 / 12");
    check(s.tried[1] == 1 && s.hit[1] == 1 && s.tried_ctx[0] == 10 && s.hit_ctx[0] == 7 && s.tried_ctx[2] == 2 &&
              s.hit_ctx[2] == 1,
          "depth 2 counted apart; by position: <1K 7/10, 4-16K 1/2");
    check(s.run_max == 3 && s.redo == 4, "longest accepted run 3, head redos 4");
    check(s.line().find("depth 1 8/12") != std::string::npos && s.line().find("<1K 70.0%") != std::string::npos,
          "the log line: " + s.line());
    const int key = 0;
    gs::note(&key, 1, 10, 5, 5);
    gs::note(&key, 1, 11, 5, 6);
    const gs::AcceptStats r = gs::accept_stats(&key);
    check(r.tried[0] == 2 && r.hit[0] == 1, "the per-model registry (draft == truth is a hit)");
    gs::forget(&key);
    check(gs::accept_stats(&key).tried[0] == 0, "forget");
    check(gs::ctx_bucket(1023) == 0 && gs::ctx_bucket(1024) == 1 && gs::ctx_bucket(65536) == 4, "context buckets");
}

void test_rows() {
    std::printf("[rollback rows, GLM-5.3]\n");
    const Glm5Geometry g = glm53();
    const int64_t ctx = 4096;
    struct Case {
        const char* name;
        int l0, l1;
        GlmStateMode mode;
        bool fp16_keys;
    } cases[] = {{"head {[0,26)} FP16", 0, 26, GlmStateMode::Fp16, false},
                 {"head part [26,54) FP16", 26, 54, GlmStateMode::Fp16, false},
                 {"head {[0,26)} INT8", 0, 26, GlmStateMode::Int8, false},
                 {"head {[0,26)} FP16 keys", 0, 26, GlmStateMode::Fp16, true},
                 {"tail with NextN [54,78)+78", 54, 78, GlmStateMode::Fp16, false}};
    for (const Case& c : cases) {
        if (c.fp16_keys) setenv("STRATA_GLM_IDX_FP16", "1", 1); else unsetenv("STRATA_GLM_IDX_FP16");
        GlmStateLayout lay;
        std::string err;
        const int mtp = c.l1 == 78 ? 78 : -1;
        if (!strata::core::glm_state_carve(g, c.l0, c.l1, mtp, ctx, c.mode, (int) ctx, lay, err)) {
            check(false, std::string(c.name) + ": carve: " + err);
            continue;
        }
        const int64_t p = 2049;
        const std::vector<gs::SpecRow> rows = gs::spec_rows(lay, p);
        int n_lat = 0, n_kpe = 0, n_ik = 0, n_full = 0, n_mtp = 0;
        bool sizes = true, offsets = true;
        for (const auto& w : rows) {
            n_mtp += w.il == 78;
            if (w.kind == gs::SpecRow::Latent) {
                ++n_lat;
                sizes = sizes && w.floats == lay.lat_pp && w.off == lay.dsa_lat[(size_t) w.il] + p * lay.lat_pp;
            } else if (w.kind == gs::SpecRow::Kpe) {
                ++n_kpe;
                sizes = sizes && w.floats * 4 == 2 * g.qk_rope;
                offsets = offsets && w.off == lay.dsa_kpe[(size_t) w.il] + p * lay.kpe_pp;
            } else {
                ++n_ik;
                sizes = sizes && w.floats * 4 == (int64_t) g.idx_key * (c.fp16_keys ? 2 : 4);
                offsets = offsets && g.idx_full(w.il) && w.off == lay.dsa_ik[(size_t) w.il] + p * lay.ik_pp;
            }
        }
        for (int il = c.l0; il < c.l1; ++il) n_full += g.idx_full(il);
        const int nl = c.l1 - c.l0;
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s: %d latent, %d k_pe, %d index-key rows (%d layers, %d full), none on NextN",
                      c.name, n_lat, n_kpe, n_ik, nl, n_full);
        check(n_lat == nl && n_kpe == nl && n_ik == n_full && n_mtp == 0 && sizes && offsets, buf);
        if (mtp >= 0) {
            const auto with = gs::spec_rows(lay, p, true);
            int m = 0;
            for (const auto& w : with) m += w.il == 78;
            check(m == 3, std::string(c.name) + ": with_mtp adds the NextN block's 3 rows");
        }
        std::string why;
        check(gs::spec_rows_check(lay, {0, 1, 2047, 2048, 2049, ctx - 1}, why),
              std::string(c.name) + ": rows disjoint across positions, inside the arena" + (why.empty() ? "" : ": " + why));
        if (c.mode == GlmStateMode::Int8)
            check(lay.lat_pp * 4 == 544, std::string(c.name) + ": an INT8 latent row is one 544 B record");
        check(gs::spec_rows(lay, -1).empty() && gs::spec_rows(lay, ctx).empty(), std::string(c.name) + ": outside the context: none");
    }
    unsetenv("STRATA_GLM_IDX_FP16");
}

}  // namespace

int main() {
    test_price();
    test_price_redo();
    test_head_rule();
    test_tuner();
    test_tuner_plain();
    test_accept();
    test_rows();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
