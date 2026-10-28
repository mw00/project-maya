// src/spec/draft_cost_test.cpp - DraftPolicy's cost-aware window (maya-ds4's DS4_DRAFT_COST=1) on the CPU, with
// synthetic round times and draft acceptance:
//   1. rows that cost little: the whole block is verified;
//   2. rows that cost a lot: fewer drafts - the size that commits the most tokens a millisecond, checked against every
//      size;
//   3. an outlier round (a stall, a cold read) moves neither the cost nor the window, a first round included;
//   4. context bands keep their own costs, and a band with none starts from the nearest one;
//   5. a size priced out in a slow stretch is measured again once it is stale, and taken back when it pays;
//   6. several windows in one step: the step's cost by its rows in all decides; a confident window takes more rows
//      than a doubtful one; the rows in all stay within the limit; no window can move to a better step alone; the
//      step is never worse than each request's share cut by the product of the confidences;
//   7. the acceptance table: drafts that hold less often than their confidence says pull it down, by position;
//   8. the lookup's choices are those of the policy before the cost-aware window, bit for bit (random rounds).
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using strata::spec::DraftPolicy;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the policy before the cost-aware window (src/spec/draft_policy.cpp as it was): the lookup's reference
namespace before {
constexpr int kMaxT = 16, kBuckets = 4;
constexpr double kShape[kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6,
                                      3.95, 4.3, 4.65, 5.0, 5.35, 5.7, 6.05, 6.4};
constexpr double kCostAlpha = 0.1, kTokAlpha = 0.05, kDecay = 0.97, kPriorN = 4.0;
constexpr double kPriorQ[kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr int kProbes = 3;
struct Policy {
    int max_t;
    double margin;
    std::array<double, kMaxT + 1> cost{}, cost_n{}, mtp_tok{}, mtp_n{};
    std::array<double, kBuckets> ok{}, bad{};
    Policy(int t, double m = 0.03) : max_t(std::clamp(t, 1, kMaxT)), margin(m) {}
    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }
    double lookup_rate(int match) const {
        const int b = bucket(match);
        return (ok[b] + kPriorN * kPriorQ[b]) / (ok[b] + bad[b] + kPriorN);
    }
    double cost_ms(int t) const {
        t = std::clamp(t, 1, kMaxT);
        if (cost_n[t] > 0) return cost[t];
        double num = 0.0, den = 0.0;
        for (int u = 1; u <= kMaxT; ++u)
            if (cost_n[u] > 0) {
                const double w = std::min(cost_n[u], 20.0);
                num += w * cost[u] * kShape[t] / kShape[u];
                den += w;
            }
        return den > 0 ? num / den : kShape[t];
    }
    double mtp_tokens(int t) const { return mtp_n[t] > 0 ? mtp_tok[t] : 1.0 + 0.7 * (t - 1); }
    DraftPolicy::Pick choose(int t_mtp, int lookup_k, int match) const {
        DraftPolicy::Pick p;
        p.t = std::clamp(t_mtp, 1, max_t);
        if (lookup_k <= 0) return p;
        const double base = mtp_tokens(p.t) / cost_ms(p.t);
        const double q = lookup_rate(match);
        double e = 1.0, qi = 1.0, best = 0.0;
        int best_t = 0;
        for (int k = 1; k <= std::min(lookup_k, max_t - 1); ++k) {
            qi *= q;
            e += qi;
            const double r = e / cost_ms(k + 1);
            if (r > best) { best = r; best_t = k + 1; }
        }
        if (best_t > 0 && best > base * (1.0 + margin)) {
            p.lookup = true;
            p.t = best_t;
            return p;
        }
        const int t_full = std::min(lookup_k, max_t - 1) + 1;
        if (t_full > p.t && cost_n[t_full] < kProbes && q >= 0.85) {
            p.lookup = true;
            p.t = t_full;
        }
        return p;
    }
    void observe(bool lookup, int t, int accepted, int match, double round_ms) {
        t = std::clamp(t, 1, kMaxT);
        if (round_ms > 0) {
            cost[t] = cost_n[t] > 0 ? (1.0 - kCostAlpha) * cost[t] + kCostAlpha * round_ms : round_ms;
            cost_n[t] += 1.0;
        }
        if (lookup) {
            const int b = bucket(match);
            ok[b] = kDecay * ok[b] + accepted;
            bad[b] = kDecay * bad[b] + (accepted < t - 1 ? 1.0 : 0.0);
        } else {
            const double got = accepted + 1.0;
            mtp_tok[t] = mtp_n[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok[t] + kTokAlpha * got : got;
            mtp_n[t] += 1.0;
        }
    }
};
}  // namespace before

// rounds of every size from `lo` to `hi` rows, `n` of each, at cost(rows)
template <class F>
void teach(DraftPolicy& p, int lo, int hi, int n, F cost) {
    for (int i = 0; i < n; ++i)
        for (int r = lo; r <= hi; ++r) p.observe_cost(r, cost(r));
}

// the size with the most tokens a ms by brute force (the smaller on a tie)
int best_rows(const DraftPolicy& p, const double* q, int n, int cap) {
    int best = 1;
    double bv = 1.0 / p.cost_ms(1);
    for (int k = 1; k <= std::min(n, cap - 1); ++k) {
        const double v = DraftPolicy::tokens(q, k) / p.cost_ms(k + 1);
        if (v > bv) bv = v, best = k + 1;
    }
    return best;
}

// a step's tokens a ms at its cost by its rows in all
double step_value(const DraftPolicy& p, const std::vector<std::vector<double>>& q, const std::vector<int>& rows) {
    int r = 0;
    double e = 0.0;
    for (size_t i = 0; i < q.size(); ++i) {
        r += rows[i];
        e += DraftPolicy::tokens(q[i].data(), rows[i] - 1);
    }
    return e / p.cost_ms(r);
}

}  // namespace

int main() {
    std::printf("draft_cost_test\n");
    const std::vector<float> c90(5, 0.9f);   // a DSpark block of 5 drafts, each 0.9 sure
    double q90[5];
    {
        DraftPolicy p(16);
        p.chain(c90.data(), 5, q90);
        const double c = c90[0];   // (0.9f)
        check(std::fabs(q90[0] - c) < 1e-12 && std::fabs(q90[4] - std::pow(c, 5)) < 1e-12,
              "before any draft is seen, the chain is the product of the confidences");
    }
    {   // 1. cheap rows
        DraftPolicy p(16);
        teach(p, 1, 6, 5, [](int r) { return 50.0 + 0.5 * (r - 1); });
        check(p.window(q90, 5, 16) == 6, "rows that cost little: the whole block of 5 drafts is verified");
        check(p.window(q90, 5, 4) == 4, "... never past the room it is given");
    }
    {   // 2. expensive rows
        DraftPolicy p(16);
        teach(p, 1, 6, 5, [](int r) { return 20.0 + 15.0 * (r - 1); });
        const int w = p.window(q90, 5, 16);
        check(w < 6 && w > 1, "rows that cost a lot: fewer drafts than the block holds");
        check(w == best_rows(p, q90, 5, 16), "... the size with the most tokens a millisecond (every size checked)");
        DraftPolicy p2(16);
        teach(p2, 1, 6, 5, [](int r) { return 20.0 + 60.0 * (r - 1); });
        check(p2.window(q90, 5, 16) == 1, "a row as dear as four plain steps: no draft at all");
    }
    {   // 3. outliers
        DraftPolicy p(16);
        const auto cost = [](int r) { return 40.0 + 2.0 * (r - 1); };
        teach(p, 1, 6, 20, cost);
        const int w0 = p.window(q90, 5, 16);
        p.observe_cost(4, 4000.0);
        check(p.cost_ms(4) < 1.05 * cost(4), "a stalled round (100x) moves a size's cost by under 5%");
        check(p.window(q90, 5, 16) == w0, "... and not the window");
        DraftPolicy f(16);
        teach(f, 1, 3, 5, cost);
        f.observe_cost(6, 10.0 * cost(6));   // a size's first round is a cold one
        check(f.cost_ms(6) <= 2.0 * cost(3) * 2.85 / 1.7 + 1e-9, "a first round far above the known sizes is capped");
        for (int i = 0; i < 3; ++i) f.observe_cost(6, cost(6));
        check(f.cost_ms(6) < 1.2 * cost(6), "... and three ordinary rounds bring it within 20%");
    }
    {   // 4. context bands
        DraftPolicy p(16);
        const auto cheap = [](int r) { return 40.0 + 1.0 * (r - 1); };
        const auto dear = [](int r) { return 40.0 + 20.0 * (r - 1); };
        p.set_context(1000);
        teach(p, 1, 6, 10, cheap);
        const int w_short = p.window(q90, 5, 16);
        p.set_context(100000);
        check(std::fabs(p.cost_ms(3) - cheap(3)) < 1e-9, "a band with nothing measured starts from the nearest one");
        teach(p, 1, 6, 30, dear);
        const int w_long = p.window(q90, 5, 16);
        check(std::fabs(p.cost_ms(3) - dear(3)) < 0.05 * dear(3), "... and learns its own costs");
        check(w_long < w_short, "rows that cost more at a long context: a shorter window there");
        p.set_context(2000);
        check(std::fabs(p.cost_ms(3) - cheap(3)) < 1e-9 && p.window(q90, 5, 16) == w_short,
              "back at a short context: its own costs and window, untouched");
        p.set_context(20000);
        check(std::fabs(p.cost_ms(3) - cheap(3)) < 1e-9, "the middle band starts from the shorter one");
        check(p.cost_text().find("ctx<8K:") != std::string::npos && p.cost_text().find("ctx>=64K:") != std::string::npos,
              "the log line names every band measured");
    }
    {   // 5. stale sizes
        DraftPolicy p(16);
        const auto cost = [](int r) { return 30.0 + 3.0 * (r - 1); };
        teach(p, 1, 6, 5, cost);
        check(p.window(q90, 5, 16) == 6, "before the slow stretch: the whole block");
        for (int i = 0; i < 40; ++i) p.observe_cost(6, 120.0);   // a slow stretch (a prompt's cold experts, say)
        check(p.window(q90, 5, 16) == 5, "a slow stretch prices the whole block out");
        for (int i = 0; i < 100; ++i) p.observe_cost(5, cost(5));
        check(p.window(q90, 5, 16) == 5, "100 rounds later: not tried yet");
        for (int i = 0; i < 200; ++i) p.observe_cost(5, cost(5));
        check(p.window(q90, 5, 16) == 6, "300 rounds later: the stale size is tried again");
        p.observe_cost(6, cost(6));
        check(std::fabs(p.cost_ms(6) - cost(6)) < 1e-9, "... its new round replaces the old figure");
        check(p.window(q90, 5, 16) == 6, "... and it is taken again, now that it pays");
        const std::vector<float> c60(5, 0.6f);
        double q60[5];
        p.chain(c60.data(), 5, q60);
        DraftPolicy u(16);
        teach(u, 1, 3, 5, cost);
        check(u.window(q90, 5, 16) == 4, "a size not measured yet is tried while its drafts likely hold");
        check(u.window(q60, 5, 16) == 3 && best_rows(u, q60, 5, 16) == 3, "... but not for drafts that likely fail");
        for (int i = 0; i < 3; ++i) u.observe_cost(4, 80.0);
        check(u.window(q90, 5, 16) != 4, "... and once measured a few times, its cost decides");
    }
    {   // 6. several windows in a step
        DraftPolicy p(16);
        std::vector<std::vector<double>> q(2, std::vector<double>(q90, q90 + 5));
        std::vector<const double*> qp{q[0].data(), q[1].data()};
        const int len[2] = {5, 5}, cap[2] = {16, 16};
        int rows[2];
        teach(p, 2, 16, 5, [](int r) { return 60.0 + 0.5 * (r - 2); });
        p.plan(qp.data(), len, cap, 2, 16, 0, 0.0, rows);
        check(rows[0] == 6 && rows[1] == 6, "two windows, rows that cost little: both whole blocks");
        DraftPolicy d(16);
        teach(d, 2, 16, 5, [](int r) { return 40.0 + 12.0 * (r - 2); });
        d.plan(qp.data(), len, cap, 2, 16, 0, 0.0, rows);
        check(rows[0] + rows[1] < 12, "two windows, rows that cost a lot: fewer drafts in all");
        bool local = true;
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k <= 5; ++k) {
                std::vector<int> alt{rows[0], rows[1]};
                alt[(size_t) i] = k + 1;
                local = local && step_value(d, q, alt) <= step_value(d, q, {rows[0], rows[1]}) + 1e-12;
            }
        check(local, "... no window can change its own size to a better step");
        bool best = true;   // every pair of sizes: none better than the plan
        for (int a = 1; a <= 6; ++a)
            for (int b = 1; b <= 6; ++b) best = best && step_value(d, q, {a, b}) <= step_value(d, q, {rows[0], rows[1]}) + 1e-12;
        check(best, "... nor any pair of sizes");
        // the fixed rule: each its share (8 rows), cut by the product of the confidences (0.5)
        int cut = 0;
        for (double all = 1.0; cut < 5 && (all *= 0.9) >= 0.5;) ++cut;
        check(step_value(d, q, {rows[0], rows[1]}) >= step_value(d, q, {cut + 1, cut + 1}),
              "... and the step is worth at least the fixed cut's");
        const std::vector<float> c95(5, 0.95f), c55(5, 0.55f);
        std::vector<std::vector<double>> qa(2, std::vector<double>(5));
        d.chain(c95.data(), 5, qa[0].data());
        d.chain(c55.data(), 5, qa[1].data());
        std::vector<const double*> qap{qa[0].data(), qa[1].data()};
        DraftPolicy m(16);
        teach(m, 2, 16, 5, [](int r) { return 40.0 + 6.0 * (r - 2); });
        m.plan(qap.data(), len, cap, 2, 16, 0, 0.0, rows);
        check(rows[0] > rows[1], "a confident window takes more rows than a doubtful one");
        DraftPolicy c(16);
        teach(c, 4, 16, 5, [](int r) { return 60.0 + 0.2 * (r - 4); });
        std::vector<std::vector<double>> q4(4, std::vector<double>(5));
        std::vector<const double*> q4p;
        for (auto& v : q4) {
            c.chain(c95.data(), 5, v.data());
            q4p.push_back(v.data());
        }
        const int len4[4] = {5, 5, 5, 5}, cap4[4] = {16, 16, 16, 16};
        int rows4[4];
        c.plan(q4p.data(), len4, cap4, 4, 16, 0, 0.0, rows4);
        check(rows4[0] + rows4[1] + rows4[2] + rows4[3] == 16, "four confident windows: the rows in all fill the limit");
        check(rows4[0] == 4 && rows4[1] == 4 && rows4[2] == 4 && rows4[3] == 4, "... shared evenly when they are as sure");
        c.plan(q4p.data(), len4, cap4, 3, 16, 3, 3.0, rows4);
        check(rows4[0] + rows4[1] + rows4[2] + 3 <= 16, "... and stay within it beside rows that do not draft");
        const int cap1[4] = {1, 16, 16, 16};
        c.plan(q4p.data(), len4, cap1, 4, 16, 0, 0.0, rows4);
        check(rows4[0] == 1, "a window with room for one row takes no draft");
    }
    {   // 7. the acceptance table
        DraftPolicy p(16);
        for (int i = 0; i < 400; ++i) p.observe_drafts(c90.data(), 5, i % 2 == 0 ? 0 : 5);
        const double c = c90[0];
        check(p.accept_rate(0, c) < 0.6, "first drafts 0.9 sure that hold half the time: their rate falls near 0.5");
        check(p.accept_rate(1, c) > 0.95, "... the second ones, always kept when reached, rise near 1");
        check(std::fabs(p.accept_rate(0, 0.62) - 0.62) < 1e-12, "... and drafts of another confidence keep their own");
        DraftPolicy t(16);
        teach(t, 1, 6, 5, [](int r) { return 20.0 + 9.0 * (r - 1); });
        double qb[5];
        t.chain(c90.data(), 5, qb);
        const int w_trust = t.window(qb, 5, 16);
        std::mt19937 rng(7);   // each draft holds 6 times in 10, whatever its 0.9 says
        for (int i = 0; i < 600; ++i) {
            int acc = 0;
            while (acc < 5 && rng() % 10 < 6) ++acc;
            t.observe_drafts(c90.data(), 5, acc);
        }
        t.chain(c90.data(), 5, qb);
        check(std::fabs(qb[0] - 0.6) < 0.06 && std::fabs(qb[1] - 0.36) < 0.06, "... and drafts that hold 6 in 10 chain as such");
        check(t.window(qb, 5, 16) < w_trust, "a drafter less right than it says gets a shorter window");
    }
    {   // 8. the lookup, bit for bit
        std::mt19937 rng(20261028);
        long long picks = 0, diffs = 0;
        for (int seq = 0; seq < 200; ++seq) {
            const int max_t = 2 + (int) (rng() % 15);
            DraftPolicy now(max_t);
            before::Policy was(max_t);
            for (int step = 0; step < 2000; ++step) {
                const int t = 1 + (int) (rng() % 16), match = (int) (rng() % 40);
                if (rng() % 3 == 0) {
                    const int k = (int) (rng() % 16);
                    const DraftPolicy::Pick a = now.choose(t, k, match), b = was.choose(t, k, match);
                    ++picks;
                    diffs += a.lookup != b.lookup || a.t != b.t;
                } else {
                    const bool lookup = rng() % 2 == 0;
                    const int acc = (int) (rng() % (unsigned) t);
                    const double ms = rng() % 50 == 0 ? 0.0 : 5.0 + (double) (rng() % 100000) / 997.0;
                    now.observe(lookup, t, acc, match, ms);
                    was.observe(lookup, t, acc, match, ms);
                }
            }
            for (int t = 1; t <= 16; ++t) diffs += now.cost_ms(t) != was.cost_ms(t);
            for (int m = 0; m < 40; m += 5) diffs += now.lookup_rate(m) != was.lookup_rate(m);
        }
        char what[128];
        std::snprintf(what, sizeof what, "the lookup's %lld picks, costs and rates as before the change (%lld differ)",
                      picks, diffs);
        check(diffs == 0, what);
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
