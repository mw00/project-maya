// src/metal/metal_mm3_divergence_test.cpp - why does DSpark-on diverge from plain greedy on
// prose at temperature 0 while counting stays identical?  Two hypotheses:
//   (A) benign batched numerics: the verify window's multi-row forward produces slightly
//       different logits than single-row decode, flipping near-tie argmax picks;
//   (B) a verify-loop logic bug: keep/rewind accounting corrupts the state on rejection.
// This test discriminates on the synthetic fixture (no real model):
//   phase B: identical state + identical tokens - logits of one forward_rows window vs the
//            same tokens fed one at a time.  Pure batching delta, with top-2 gaps wherever
//            the argmax disagrees.
//   phase A: a replica of mm3_serve.cpp's verify loop (same keep/rewind arithmetic) driven
//            by FABRICATED drafts: the true next tokens with one deliberately wrong token
//            planted at a cycling position, forcing kept = 0..k across rounds.  Emitted
//            tokens are compared against the plain greedy reference; the first divergence
//            is attributed to its decisive row (top-2 gap vs phase B's delta scale), and a
//            shadow instance re-feeds the committed prefix one token at a time to check
//            the post-rewind state's next-token logits directly.
//   phase C (optional, when the dspark fixture loads against the synth target): the same
//            replica with the real drafter's proposals.
// usage: metal_mm3_divergence_test <fixture.gguf> [dspark-dir]
#include "mm3_metal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_err;

int32_t argmax(const std::vector<float>& l) {
    size_t best = 0;
    for (size_t v = 1; v < l.size(); ++v)
        if (l[v] > l[best]) best = v;
    return (int32_t) best;
}

// top-1 value, top-2 value and the argmax of one row
void top2(const float* l, int n, int32_t& pick, float& t1, float& t2) {
    pick = 0;
    t1 = l[0];
    t2 = -INFINITY;
    for (int v = 1; v < n; ++v) {
        if (l[v] > t1) {
            t2 = t1;
            t1 = l[v];
            pick = v;
        } else if (l[v] > t2) {
            t2 = l[v];
        }
    }
}

bool prefill_serve_style(maya::metal::Mm3Metal& m, const std::vector<int32_t>& ids,
                         std::vector<float>& logits) {
    std::vector<int32_t> head(ids.begin(), ids.end() - 1);
    return m.forward(head, logits, g_err) && m.forward({ids.back()}, logits, g_err);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_mm3_divergence_test <fixture.gguf> [dspark-dir]\n");
        return 2;
    }
    const std::string gguf = argv[1];
    const int NREF = 400, NEMIT = 300, K = 4;
    maya::metal::Mm3Options o;
    if (const char* v = std::getenv("MAYA_MM3_CPU"); v && *v && *v != '0') o.use_gpu = false;
    maya::metal::Mm3Metal model;
    model.set_options(o);
    if (!model.load(gguf, 1200, g_err)) {
        std::fprintf(stderr, "FAIL: load: %s\n", g_err.c_str());
        return 1;
    }
    const int V = model.geometry().n_vocab;
    std::fprintf(stderr, "divergence test on: %s\n", model.describe().c_str());

    // deterministic pseudo-random prompt
    std::vector<int32_t> prompt;
    uint32_t s = 12345;
    for (int i = 0; i < 48; ++i) {
        s = s * 1664525u + 1013904223u;
        prompt.push_back((int32_t) ((s >> 16) % (uint32_t) V));
    }

    // ---- plain greedy reference (single-row forwards, serve-style prefill) ----
    std::vector<float> logits;
    if (!prefill_serve_style(model, prompt, logits)) {
        std::fprintf(stderr, "FAIL: prefill: %s\n", g_err.c_str());
        return 1;
    }
    std::vector<int32_t> ref;
    {
        int32_t y = argmax(logits);
        ref.push_back(y);
        while ((int) ref.size() < NREF) {
            if (!model.forward({y}, logits, g_err)) {
                std::fprintf(stderr, "FAIL: forward: %s\n", g_err.c_str());
                return 1;
            }
            y = argmax(logits);
            ref.push_back(y);
        }
    }
    std::printf("reference: %d plain greedy tokens; first 16:", NREF);
    for (int i = 0; i < 16; ++i) std::printf(" %d", ref[(size_t) i]);
    std::printf("\n");

    // ---- phase B: forward_rows vs one-at-a-time, identical state and tokens ----
    double b_max = 0.0, b_sum = 0.0;
    long b_cnt = 0;
    int b_dis = 0;
    float b_dis_gap_max = 0.0f;
    double b_rowmax[8] = {};
    {
        model.reset();
        if (!prefill_serve_style(model, prompt, logits)) return 1;
        const int W = 5;
        for (int w = 0; w + W <= 300; w += W) {
            const int64_t base = model.position();
            std::vector<int32_t> win(ref.begin() + w, ref.begin() + w + W);
            std::vector<float> rows;
            if (!model.forward_rows(win, rows, g_err)) {
                std::fprintf(stderr, "FAIL: forward_rows: %s\n", g_err.c_str());
                return 1;
            }
            model.rewind_to(base);
            for (int i = 0; i < W; ++i) {
                std::vector<float> one;
                if (!model.forward({win[(size_t) i]}, one, g_err)) return 1;
                const float* r = rows.data() + (size_t) i * V;
                double mx = 0.0;
                for (int v = 0; v < V; ++v) {
                    const double d = std::fabs((double) r[v] - one[(size_t) v]);
                    if (d > mx) mx = d;
                    b_sum += d;
                    ++b_cnt;
                }
                if (mx > b_max) b_max = mx;
                if (mx > b_rowmax[i]) b_rowmax[i] = mx;
                int32_t pm, ps;
                float t1m, t2m, t1s, t2s;
                top2(r, V, pm, t1m, t2m);
                top2(one.data(), V, ps, t1s, t2s);
                if (pm != ps) {
                    ++b_dis;
                    if (t1s - t2s > b_dis_gap_max) b_dis_gap_max = t1s - t2s;
                    std::printf("  B disagree w=%d row=%d: multi %d, single %d, single-row top2 gap %.5f, "
                                "row max|d| %.2e\n",
                                w, i, pm, ps, t1s - t2s, mx);
                }
            }
        }
    }
    std::printf("phase B: max|d| %.3e, mean|d| %.3e, per-row max|d| [%.2e %.2e %.2e %.2e %.2e], "
                "argmax disagreements %d (largest single-row top2 gap among them %.5f)\n",
                b_max, b_cnt ? b_sum / b_cnt : 0.0, b_rowmax[0], b_rowmax[1], b_rowmax[2],
                b_rowmax[3], b_rowmax[4], b_dis, b_dis_gap_max);

    // ---- phase A: serve verify-loop replica with fabricated drafts ----
    struct Event {
        int round, row, cand, want;
        float gap;
    };
    std::vector<Event> events;
    int rounds = 0, offered = 0, accepted = 0, kept_hist[8] = {};
    int diverged_at = -1;
    double shadow_max = 0.0;
    int shadow_probes = 0;
    bool probe_next = true;   // probe the first rounds, then rounds after a rejection
    int probes_left = 14;
    {
        maya::metal::Mm3Metal shadow;
        shadow.set_options(o);
        if (!shadow.load(gguf, 1200, g_err)) {
            std::fprintf(stderr, "FAIL: shadow load: %s\n", g_err.c_str());
            return 1;
        }
        model.reset();
        if (!prefill_serve_style(model, prompt, logits)) return 1;
        std::vector<int32_t> emitted;
        int32_t y = argmax(logits);
        emitted.push_back(y);
        int cyc = 0;
        while ((int) emitted.size() < NEMIT) {
            const int t = (int) emitted.size() - 1;   // y == ref[t] as long as we match
            const int64_t pos = model.position();
            if (probe_next && probes_left > 0 && t + 1 < NREF) {
                // shadow: plain state for the same committed prefix, then feed y
                shadow.reset();
                std::vector<float> sl;
                if (!prefill_serve_style(shadow, prompt, sl)) return 1;
                bool okf = true;
                for (int i = 0; i < t; ++i) okf = okf && shadow.forward({ref[(size_t) i]}, sl, g_err);
                if (!okf) return 1;
                std::vector<float> l_spec, l_shad;
                if (!model.forward({y}, l_spec, g_err)) return 1;
                model.rewind_to(pos);
                if (!shadow.forward({y}, l_shad, g_err)) return 1;
                double mx = 0.0;
                for (int v = 0; v < V; ++v)
                    mx = std::max(mx, (double) std::fabs(l_spec[(size_t) v] - l_shad[(size_t) v]));
                if (mx > shadow_max) shadow_max = mx;
                ++shadow_probes;
                --probes_left;
                probe_next = false;
                std::printf("  A probe round=%d t=%d: post-state forward max|d| vs fresh plain state %.3e\n",
                            rounds, t, mx);
            }
            int k = K;
            if ((int) emitted.size() + k + 1 >= NREF) break;
            const int m = cyc % (K + 1);   // intended accepts 0..K
            ++cyc;
            std::vector<int32_t> drafts;
            for (int j = 0; j < k; ++j) {
                if (j < m)
                    drafts.push_back(ref[(size_t) (t + 1 + j)]);
                else if (j == m)
                    drafts.push_back((ref[(size_t) (t + 1 + j)] + 137) % V);
                else
                    drafts.push_back((ref[(size_t) (t + 1 + j)] + 269) % V);
            }
            std::vector<int32_t> win(1, y);
            win.insert(win.end(), drafts.begin(), drafts.end());
            std::vector<float> rows;
            if (!model.forward_rows(win, rows, g_err)) {
                std::fprintf(stderr, "FAIL: forward_rows (replica): %s\n", g_err.c_str());
                return 1;
            }
            offered += k;
            ++rounds;
            int kept = 0;
            int32_t next = -1;
            bool true_prefix = true;
            for (int i = 0; i <= k; ++i) {
                const float* r = rows.data() + (size_t) i * V;
                int32_t cand, pick1;
                float t1, t2;
                top2(r, V, cand, t1, t2);
                pick1 = cand;
                if (true_prefix && t + 1 + i < NREF && cand != ref[(size_t) (t + 1 + i)] &&
                    diverged_at < 0) {
                    events.push_back({rounds - 1, i, cand, ref[(size_t) (t + 1 + i)], t1 - t2});
                    std::printf("  A decisive event round=%d row=%d: multi-row pick %d, plain pick %d, "
                                "row top2 gap %.5f (phase B max|d| %.2e)\n",
                                rounds - 1, i, cand, ref[(size_t) (t + 1 + i)], t1 - t2, b_max);
                }
                if (i == k || cand != drafts[(size_t) i]) {
                    next = cand;
                    break;
                }
                ++accepted;
                emitted.push_back(drafts[(size_t) i]);
                if (emitted.back() != ref[emitted.size() - 1] && diverged_at < 0)
                    diverged_at = (int) emitted.size() - 1;
                kept = i + 1;
                (void) pick1;
            }
            kept_hist[kept]++;
            if (kept < k) probe_next = true;
            model.rewind_to(pos + 1 + kept);
            y = next;
            emitted.push_back(y);
            if (y != ref[emitted.size() - 1] && diverged_at < 0) diverged_at = (int) emitted.size() - 1;
            if (diverged_at >= 0) {
                std::printf("  A replica diverged from plain at emitted index %d (round %d)\n",
                            diverged_at, rounds - 1);
                break;
            }
        }
        std::printf("phase A: %d rounds, drafts offered %d accepted %d, kept histogram "
                    "[0]=%d [1]=%d [2]=%d [3]=%d [4]=%d, emitted %zu tokens, %s\n",
                    rounds, offered, accepted, kept_hist[0], kept_hist[1], kept_hist[2],
                    kept_hist[3], kept_hist[4], emitted.size(),
                    diverged_at < 0 ? "token-IDENTICAL to plain greedy" : "DIVERGED");
        std::printf("phase A shadow probes: %d, max post-rewind state delta %.3e (phase B scale %.2e)\n",
                    shadow_probes, shadow_max, b_max);
    }

    // ---- phase C: the real drafter, if it loads against the synth target ----
    if (argc > 2) {
        model.reset();
        if (!model.load_draft(argv[2], g_err)) {
            std::printf("phase C: SKIP - load_draft failed: %s\n", g_err.c_str());
        } else {
            const int B = model.draft_block();
            if (!prefill_serve_style(model, prompt, logits)) return 1;
            std::vector<int32_t> emitted;
            int32_t y = argmax(logits);
            emitted.push_back(y);
            long off2 = 0, acc2 = 0;
            int div2 = -1, r2 = 0;
            while ((int) emitted.size() < 200) {
                const int64_t pos = model.position();
                std::vector<int32_t> drafts;
                std::vector<float> confs;
                if (!model.draft_propose(y, pos, B, drafts, confs, g_err)) {
                    std::fprintf(stderr, "FAIL: draft_propose: %s\n", g_err.c_str());
                    return 1;
                }
                const int k = (int) drafts.size();
                if (k == 0) {
                    if (!model.forward({y}, logits, g_err)) return 1;
                    y = argmax(logits);
                    emitted.push_back(y);
                    if (y != ref[emitted.size() - 1]) div2 = (int) emitted.size() - 1;
                    if (div2 >= 0) break;
                    continue;
                }
                std::vector<int32_t> win(1, y);
                win.insert(win.end(), drafts.begin(), drafts.end());
                std::vector<float> rows;
                if (!model.forward_rows(win, rows, g_err)) return 1;
                off2 += k;
                ++r2;
                int kept = 0;
                int32_t next = -1;
                for (int i = 0; i <= k; ++i) {
                    const int32_t cand = argmax(std::vector<float>(
                        rows.begin() + (size_t) i * V, rows.begin() + (size_t) (i + 1) * V));
                    if (i == k || cand != drafts[(size_t) i]) {
                        next = cand;
                        break;
                    }
                    ++acc2;
                    emitted.push_back(drafts[(size_t) i]);
                    kept = i + 1;
                }
                model.rewind_to(pos + 1 + kept);
                y = next;
                emitted.push_back(y);
                if (emitted.back() != ref[emitted.size() - 1]) div2 = (int) emitted.size() - 1;
                if (div2 >= 0) break;
            }
            std::printf("phase C: real drafter, %d rounds, kept %ld of %ld drafts, emitted %zu, %s\n",
                        r2, acc2, off2, emitted.size(),
                        div2 < 0 ? "token-IDENTICAL to plain greedy" : "DIVERGED");
            if (div2 >= 0) std::printf("phase C: first divergence at emitted index %d\n", div2);
        }
    }

    // ---- verdict by the stated rule ----
    const double noise = 25.0 * b_max;
    bool a_evidence = events.empty() && diverged_at < 0;
    bool flips_within_noise = true;
    for (const Event& e : events)
        if (e.gap > noise && e.gap > 0.05f) flips_within_noise = false;
    const bool shadow_clean = shadow_max <= noise || shadow_max <= 0.01;
    std::printf("VERDICT-RULE: phaseB max|d|=%.2e (noise bound %.2e), decisive events=%zu all-within-noise=%d, "
                "shadow clean=%d\n",
                b_max, noise, events.size(), (int) flips_within_noise, (int) shadow_clean);
    if (a_evidence)
        std::printf("VERDICT: A-consistent - the verify loop is token-exact under forced rejections; "
                    "rows differ only at the measured batching scale\n");
    else if (flips_within_noise && shadow_clean)
        std::printf("VERDICT: A - every divergence sits at a near-tie within the measured batching "
                    "delta, and post-rewind state matches fresh state at the same scale\n");
    else
        std::printf("VERDICT: B-INDICATED - a divergence exceeds the batching-noise scale or the "
                    "post-rewind state is polluted; inspect the events above\n");
    return 0;
}
