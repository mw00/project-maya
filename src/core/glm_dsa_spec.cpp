// src/core/glm_dsa_spec.cpp - the glm-dsa port B11.2 / B11.3: head-group pricing, acceptance accounting and the
// glm-dsa rollback rows of the pipelined speculative decode (see include/strata/core/glm_dsa_spec.hpp).  Plain C++.
#include "strata/core/glm_env.hpp"
#include "strata/kernels/glmd_random_route.hpp"
#include "strata/core/glm_dsa_spec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace strata::core::glmspec {

// ---------------------------------------------------------------- pricing

SpecPrice spec_price(const std::vector<PartCost>& parts, int head, double accept, double hop_ms, bool redo_waits) {
    SpecPrice r;
    const int n = (int) parts.size();
    r.head = head;
    if (n < 2 || head < 1 || head >= n) return r;
    accept = std::min(1.0, std::max(0.0, accept));
    hop_ms = std::max(0.0, hop_ms);
    double lane_h = 0.0, lane_t = 0.0;
    for (int i = 0; i < n; ++i) {
        (i < head ? r.head_ms : r.tail_ms) += parts[(size_t) i].ms;
        (i < head ? lane_h : lane_t) += std::max(0.0, std::min(parts[(size_t) i].lane_ms, parts[(size_t) i].ms));
        r.base_ms += parts[(size_t) i].ms;
    }
    // hops: inside the head group, the head's hand-off to the tail (its pinned slots), inside the tail group
    r.head_ms += (double) head * hop_ms;
    r.tail_ms += (double) (n - head - 1) * hop_ms;
    r.base_ms += (double) (n - 1) * hop_ms;
    for (int i = 0; i < head && !r.shared_lane; ++i)
        for (int j = head; j < n; ++j)
            if (parts[(size_t) i].pool == parts[(size_t) j].pool && parts[(size_t) i].lane_ms > 0.0 &&
                parts[(size_t) j].lane_ms > 0.0) {
                r.shared_lane = true;
                break;
            }
    // a shared pool runs the head's and the tail's lane work one after the other (glmtier / Maya serialise a pool's
    // calls): the overlapped step cannot be shorter than both
    r.lane_overlap_ms = r.shared_lane ? lane_h + lane_t : 0.0;
    r.hit_ms = std::max(std::max(r.head_ms, r.tail_ms), r.lane_overlap_ms);
    // a rejected draft: the head's redo starts once the tail's token (and the next draft) is known, and the tail's
    // next position waits for the redo's hop
    r.miss_ms = r.tail_ms + r.head_ms;
    // ... and only once the head has finished the rejected speculation (decode_spec restores the head's states on
    // its stream and spec_head synchronises it before the redo): a miss is the overlapped step plus the redo, which
    // runs alone (the tail waits for its hop).  Equal to T + H while H <= T without a lane-bound pool; 2H once the
    // head is the slower group, where T + H under-prices it.  STRATA_GLM_SPEC_DRAFT_EARLY's pricing
    if (redo_waits) r.miss_ms = r.hit_ms + r.head_ms;
    r.step_ms = accept * r.hit_ms + (1.0 - accept) * r.miss_ms;
    r.speedup = r.step_ms > 0.0 ? r.base_ms / r.step_ms : 0.0;
    return r;
}

std::vector<SpecPrice> spec_prices(const std::vector<PartCost>& parts, double accept, double hop_ms, int& best,
                                   bool redo_waits) {
    std::vector<SpecPrice> v;
    best = -1;
    const int n = (int) parts.size();
    for (int h = 1; h < n; ++h) {
        v.push_back(spec_price(parts, h, accept, hop_ms, redo_waits));
        if (best < 0 || v.back().step_ms < v[(size_t) best].step_ms - 1e-9) best = (int) v.size() - 1;
    }
    return v;
}

bool spec_draft_early() { return strata::glmenv::env_flag("STRATA_GLM_SPEC_DRAFT_EARLY", false); }

double spec_accept_assumed() {
    const double a = strata::glmenv::env_num("STRATA_GLM_SPEC_ACCEPT", 0.8);
    return a >= 0.05 && a <= 1.0 ? a : 0.8;
}

double lane_share(double hits, double routes, double card_bps, double miss_bps) {
    const double miss = std::max(0.0, routes - hits);
    if (miss <= 0.0 || miss_bps <= 0.0) return 0.0;
    const double t_hit = card_bps > 0.0 ? std::max(0.0, hits) / card_bps : 0.0;
    const double t_miss = miss / miss_bps;
    return t_miss / (t_hit + t_miss);
}

namespace {

std::mutex g_mu;
std::map<int, int> g_priced;   // n_parts -> priced head

}  // namespace

int spec_record_prices(const std::vector<PartCost>& parts, bool log) {
    const int n = (int) parts.size();
    if (n < 2) return -1;
    const double a = spec_accept_assumed();
    int best = -1;
    const std::vector<SpecPrice> v = spec_prices(parts, a, 0.05, best, spec_draft_early());
    if (best < 0) return -1;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_priced[n] = v[(size_t) best].head;
    }
    if (log) {
        for (const SpecPrice& p : v) {
            std::string grp;
            for (int i = 0; i < p.head; ++i) grp += (i ? "," : "") + std::string("CUDA") + std::to_string(parts[(size_t) i].dev);
            std::fprintf(stderr,
                         "glm spec price: head {%s}: H %.1f ms, T %.1f ms (draft included)%s -> %.1f ms a token at %.0f%% "
                         "accepted (hit %.1f, miss %.1f), x%.2f against %.1f ms token by token%s\n",
                         grp.c_str(), p.head_ms, p.tail_ms,
                         p.shared_lane ? (", shared CPU-lane pool " + std::to_string((int) std::lround(p.lane_overlap_ms)) +
                                          " ms").c_str()
                                       : "",
                         p.step_ms, 100.0 * a, p.hit_ms, p.miss_ms, p.speedup, p.base_ms,
                         &p == &v[(size_t) best] ? "  <- priced" : "");
        }
        if (v[(size_t) best].speedup < 1.05)
            std::fprintf(stderr, "glm spec price: the best head is x%.2f (< 1.05): STRATA_GLM_NO_SPEC=1 may be faster "
                                 "(STRATA_GLM_SPEC_TUNE_PLAIN=1 measures it on 3+ GPUs)\n",
                         v[(size_t) best].speedup);
    }
    return v[(size_t) best].head;
}

int spec_priced_head(int n_parts) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_priced.find(n_parts);
    return it == g_priced.end() ? -1 : it->second;
}

void spec_clear_prices() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_priced.clear();
}

namespace {
HeadTuner g_tuner;
int g_tune_head = -1;                                     // the head the current decode_spec call runs
bool g_tune_live = false;
std::chrono::steady_clock::time_point g_tune_t0;
}  // namespace

bool spec_bench_off() { return glmd::random_routing_seed() != 0; }

int spec_tune_window() {
    static const int w = [] {
        return (int) std::clamp<long long>(strata::glmenv::env_int("STRATA_GLM_SPEC_TUNE", 128), 0, 1 << 30);
    }();
    return w;
}

int HeadTuner::pick(int n_parts, int base) const {
    const int hi = std::min(n_parts - 1, (int) ms.size() - 1);
    if (hi < 1) return 1;
    base = std::max(1, std::min(hi, base));
    if (n_parts != n) return base;   // (nothing measured for this split yet)
    if (tokens[(size_t) base] < window) return base;
    for (int h = 1; h <= hi; ++h)
        if (tokens[(size_t) h] < window) return h;
    if (plain && tokens[0] < window) return 0;
    int best = base;
    for (int h = 1; h <= hi; ++h)
        if (ms_per_token(h) < ms_per_token(best) - 1e-9) best = h;
    if (plain && ms_per_token(0) < ms_per_token(best) * (1.0 - kPlainMargin)) best = 0;
    return best;
}

void HeadTuner::feed(int head, int64_t n_tokens, double wall_ms) {
    if (head < 0 || (head == 0 && !plain) || head >= (int) ms.size() || n_tokens <= 0 || !(wall_ms > 0.0)) return;
    tokens[(size_t) head] += n_tokens;
    ms[(size_t) head] += wall_ms;
}

double HeadTuner::ms_per_token(int head) const {
    if (head < 0 || (head == 0 && !plain) || head >= (int) ms.size() || tokens[(size_t) head] <= 0) return 1e30;
    return ms[(size_t) head] / (double) tokens[(size_t) head];
}

bool spec_tune_plain() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_SPEC_TUNE_PLAIN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

int spec_head_parts(int n, const char* env, bool dsa) {
    int h = n / 2;
    const bool is_auto = env != nullptr && std::strcmp(env, "auto") == 0;
    if (dsa && (env == nullptr || is_auto)) {
        const int p = spec_priced_head(n);
        if (p > 0) h = p;
        const int w = spec_tune_window();
        if (n >= 3 && w > 0) {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_tuner.n != n) {
                g_tuner = HeadTuner{};
                g_tuner.n = n;
                g_tuner.plain = spec_tune_plain();
            }
            g_tuner.window = w;
            const int pk = g_tuner.pick(n, h);
            if (pk >= 1) h = pk;   // (0, no drafting: tune_plain_turn's; a speculative decode that runs anyway is
                                   // measured as the base head it then takes)
            g_tune_head = std::max(1, std::min(n - 1, h));
        }
    } else if (env != nullptr) {
        h = std::atoi(env);   // Maya's reading (on Flash "auto" too: atoi makes it 0, clamped to 1 below)
    }
    return std::max(1, std::min(n - 1, h));
}

void tune_begin() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_tune_live = true;
    g_tune_head = -1;
    g_tune_t0 = std::chrono::steady_clock::now();
}

bool tune_plain_turn(int n, const char* env) {
    if (n < 3 || !spec_tune_plain() || spec_tune_window() <= 0) return false;
    if (env != nullptr && std::strcmp(env, "auto") != 0) return false;   // (a pinned head: no tuner)
    const int p = spec_priced_head(n);   // (spec_head_parts' base; takes g_mu itself)
    const int base = p > 0 ? p : n / 2;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_tuner.n != n || !g_tuner.plain) return false;   // (the split's first decode drafts: it sets the tuner up)
    if (g_tuner.pick(n, base) != 0) return false;
    g_tune_live = true;
    g_tune_head = 0;
    g_tune_t0 = std::chrono::steady_clock::now();
    return true;
}

void tune_end(int64_t produced) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_tune_live) return;
    g_tune_live = false;
    if (g_tune_head < 0 || (g_tune_head == 0 && !g_tuner.plain) || produced < 8) return;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_tune_t0).count();
    g_tuner.feed(g_tune_head, produced, ms);
    const int n = g_tuner.n;
    bool all = n >= 3;
    for (int h = 1; h < n && all; ++h) all = g_tuner.tokens[(size_t) h] >= g_tuner.window;
    if (all && g_tuner.plain) all = g_tuner.tokens[0] >= g_tuner.window;
    if (all && !g_tuner.settled_logged) {
        g_tuner.settled_logged = true;
        std::string s;
        int best = 1;
        for (int h = 1; h < n; ++h) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "%shead of %d part%s %.1f ms a token (%lld tokens)", h > 1 ? ", " : "", h,
                          h > 1 ? "s" : "", g_tuner.ms_per_token(h), (long long) g_tuner.tokens[(size_t) h]);
            s += buf;
            if (g_tuner.ms_per_token(h) < g_tuner.ms_per_token(best)) best = h;
        }
        if (g_tuner.plain) {
            char buf[96];
            std::snprintf(buf, sizeof buf, ", no drafting %.1f ms a token (%lld tokens)", g_tuner.ms_per_token(0),
                          (long long) g_tuner.tokens[0]);
            s += buf;
            if (g_tuner.ms_per_token(0) < g_tuner.ms_per_token(best) * (1.0 - HeadTuner::kPlainMargin)) best = 0;
        }
        if (best == 0)
            std::fprintf(stderr, "glm spec tune: %s -> no drafting (STRATA_GLM_SPEC_HEAD pins a head)\n", s.c_str());
        else
            std::fprintf(stderr, "glm spec tune: %s -> head of %d part%s (STRATA_GLM_SPEC_HEAD pins one)\n", s.c_str(),
                         best, best > 1 ? "s" : "");
    }
}

HeadTuner tuner_state() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_tuner;
}

void tuner_reset() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_tuner = HeadTuner{};
    g_tune_head = -1;
    g_tune_live = false;
}

// ---------------------------------------------------------------- acceptance

int ctx_bucket(int64_t pos) {
    if (pos < 1024) return 0;
    if (pos < 4096) return 1;
    if (pos < 16384) return 2;
    if (pos < 65536) return 3;
    return 4;
}

const char* ctx_bucket_name(int b) {
    static const char* const k[kCtxBuckets] = {"<1K", "1-4K", "4-16K", "16-64K", ">=64K"};
    return b >= 0 && b < kCtxBuckets ? k[b] : "?";
}

void AcceptStats::note(int depth, int64_t pos, bool accepted) {
    if (depth < 1 || depth > kMaxDepth) return;
    ++tried[(size_t) depth - 1];
    if (accepted) ++hit[(size_t) depth - 1];
    if (depth != 1) return;
    const int b = ctx_bucket(pos);
    ++tried_ctx[(size_t) b];
    if (accepted) {
        ++hit_ctx[(size_t) b];
        run_max = std::max(run_max, ++run);
    } else {
        run = 0;
        ++redo;
    }
}

double AcceptStats::rate(int depth) const {
    if (depth < 1 || depth > kMaxDepth || tried[(size_t) depth - 1] == 0) return 0.0;
    return (double) hit[(size_t) depth - 1] / (double) tried[(size_t) depth - 1];
}

double AcceptStats::rate_ctx(int b) const {
    if (b < 0 || b >= kCtxBuckets || tried_ctx[(size_t) b] == 0) return 0.0;
    return (double) hit_ctx[(size_t) b] / (double) tried_ctx[(size_t) b];
}

std::string AcceptStats::line() const {
    std::string s;
    char buf[160];
    for (int d = 1; d <= kMaxDepth; ++d) {
        if (tried[(size_t) d - 1] == 0) continue;
        std::snprintf(buf, sizeof buf, "%sdepth %d %llu/%llu (%.1f%%)", s.empty() ? "" : " | ", d,
                      (unsigned long long) hit[(size_t) d - 1], (unsigned long long) tried[(size_t) d - 1],
                      100.0 * rate(d));
        s += buf;
    }
    std::string ctx;
    for (int b = 0; b < kCtxBuckets; ++b) {
        if (tried_ctx[(size_t) b] == 0) continue;
        std::snprintf(buf, sizeof buf, "%s%s %.1f%% (%llu)", ctx.empty() ? "" : ", ", ctx_bucket_name(b),
                      100.0 * rate_ctx(b), (unsigned long long) tried_ctx[(size_t) b]);
        ctx += buf;
    }
    if (!ctx.empty()) s += " | by position: " + ctx;
    std::snprintf(buf, sizeof buf, " | longest accepted run %llu | head redos %llu", (unsigned long long) run_max,
                  (unsigned long long) redo);
    return s + buf;
}

namespace {

std::map<const void*, AcceptStats> g_stats;
FILE* g_trace = nullptr;
bool g_trace_tried = false;

}  // namespace

AcceptStats accept_stats(const void* model) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_stats.find(model);
    return it == g_stats.end() ? AcceptStats{} : it->second;
}

void note(const void* model, int depth, int64_t pos, int draft, int truth) {
    static const long long every = [] {
        const char* v = getenv("STRATA_GLM_SPEC_LOG");   // (set: a line every N first-position drafts, N >= 1)
        return v != nullptr ? std::max(1LL, strata::glmenv::int_value("STRATA_GLM_SPEC_LOG", v, 1)) : 0LL;
    }();
    std::lock_guard<std::mutex> lk(g_mu);
    AcceptStats& s = g_stats[model];
    s.note(depth, pos, draft == truth);
    if (!g_trace_tried) {
        g_trace_tried = true;
        if (const char* path = getenv("STRATA_GLM_SPEC_TRACE"))
            if ((g_trace = std::fopen(path, "a")) != nullptr) std::fprintf(g_trace, "# pos depth draft truth accepted\n");
    }
    if (g_trace != nullptr) {
        std::fprintf(g_trace, "%lld %d %d %d %d\n", (long long) pos, depth, draft, truth, draft == truth ? 1 : 0);
        std::fflush(g_trace);
    }
    if (every > 0 && depth == 1 && s.tried[0] % (uint64_t) every == 0)
        std::fprintf(stderr, "glm spec acceptance: %s\n", s.line().c_str());
}

void report(const void* model, bool final) {
    AcceptStats s;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_stats.find(model);
        if (it == g_stats.end()) return;
        s = it->second;
    }
    static const bool each = getenv("STRATA_GLM_SPEC_LOG") != nullptr;
    if (s.tried[0] == 0 || (!final && !each)) return;
    std::fprintf(stderr, "glm spec acceptance%s: %s\n", final ? " (this load)" : "", s.line().c_str());
}

void forget(const void* model) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_stats.erase(model);
}

// ---------------------------------------------------------------- rollback rows

std::vector<SpecRow> spec_rows(const GlmStateLayout& lay, int64_t p, bool with_mtp) {
    std::vector<SpecRow> r;
    if (!lay.glm_dsa() || p < 0 || p >= lay.max_ctx) return r;
    for (int il : lay.dsa_layers()) {
        if (il == lay.mtp_il && !with_mtp) continue;
        const size_t i = (size_t) il;
        if (lay.dsa_lat[i] != GlmStateLayout::kNone && lay.lat_pp > 0)
            r.push_back(SpecRow{il, SpecRow::Latent, lay.dsa_lat[i] + p * lay.lat_pp, lay.lat_pp});
        if (lay.has_kpe(il)) r.push_back(SpecRow{il, SpecRow::Kpe, lay.dsa_kpe[i] + p * lay.kpe_pp, lay.kpe_pp});
        if (lay.has_index_cache(il) && lay.dsa_ik[i] != GlmStateLayout::kNone && lay.ik_pp > 0)
            r.push_back(SpecRow{il, SpecRow::IndexKey, lay.dsa_ik[i] + p * lay.ik_pp, lay.ik_pp});
    }
    return r;
}

bool spec_rows_check(const GlmStateLayout& lay, const std::vector<int64_t>& positions, std::string& why) {
    if (!lay.glm_dsa()) {
        why = "not a glm-dsa layout";
        return false;
    }
    std::vector<std::pair<int64_t, int64_t>> all;   // [begin, end) floats
    for (int64_t p : positions) {
        const std::vector<SpecRow> rows = spec_rows(lay, p, true);
        for (int il : lay.dsa_layers()) {
            bool lat = false, kpe = false, ik = false;
            for (const SpecRow& w : rows)
                if (w.il == il) (w.kind == SpecRow::Latent ? lat : w.kind == SpecRow::Kpe ? kpe : ik) = true;
            if (!lat || (lay.has_kpe(il) && !kpe) || (lay.has_index_cache(il) && !ik)) {
                why = "layer " + std::to_string(il) + " at position " + std::to_string(p) + " is missing a cache row";
                return false;
            }
        }
        for (const SpecRow& w : rows) {
            if (w.off < lay.residual_floats || w.off + w.floats > lay.floats || w.floats <= 0) {
                why = "layer " + std::to_string(w.il) + " row outside the arena at position " + std::to_string(p);
                return false;
            }
            all.emplace_back(w.off, w.off + w.floats);
        }
    }
    std::sort(all.begin(), all.end());
    for (size_t i = 1; i < all.size(); ++i)
        if (all[i].first < all[i - 1].second) {
            why = "two rows overlap at float " + std::to_string(all[i].first);
            return false;
        }
    return true;
}

}  // namespace strata::core::glmspec
