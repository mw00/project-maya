// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split; past 8 tokens the
// last slope, extrapolated.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0,  1.0,  1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6,
                                                   3.95, 4.3,  4.65, 5.0, 5.35, 5.7,  6.05, 6.4};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it

// The cost-aware window
constexpr long long kBandEdge[DraftPolicy::kBands - 1] = {8192, 65536};   // (a band from each edge on)
constexpr double kOutlier = 2.0;      // a round time counts as at most this x what its size is known to cost,
constexpr double kOutlierW = 0.25;    // and then as this much of a round
constexpr double kStale = 256.0;      // rounds without a measurement after which a size's cost is taken again
constexpr double kTryOdds = 0.5;      // a size not known well enough is tried while its drafts all hold this often
constexpr double kOddsDecay = 0.995;  // the block drafts' acceptance table: older drafts fade (per cell)

double unit(double c) { return c > 0.0 ? std::min(c, 1.0) : 0.0; }   // (NaN: 0)

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    const auto& c = cost_[(size_t) band_];
    const auto& cn = cost_n_[(size_t) band_];
    if (cn[t] > 0) return c[t];
    // scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cn[u] > 0) {
            const double w = std::min(cn[u], 20.0);
            num += w * c[u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return 1.0 + 0.7 * (t - 1);       // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && cost_n_[(size_t) band_][t_full] < kProbes && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) {
        double& c = cost_[(size_t) band_][t];
        double& cn = cost_n_[(size_t) band_][t];
        c = cn > 0 ? (1.0 - kCostAlpha) * c + kCostAlpha * round_ms : round_ms;
        cn += 1.0;
    }
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

// ---- the cost-aware window

void DraftPolicy::set_context(long long pos) {
    int b = 0;
    while (b < kBands - 1 && pos >= kBandEdge[b]) ++b;
    if (b == band_) return;
    const auto measured = [&](int k) {
        for (int t = 1; t <= kMaxT; ++t)
            if (cost_n_[(size_t) k][t] > 0) return true;
        return false;
    };
    if (!measured(b))   // (the nearest band that has costs, the shorter first: each size as one round of it)
        for (int d = 1; d < kBands; ++d) {
            const int src = b - d >= 0 && measured(b - d) ? b - d : b + d < kBands && measured(b + d) ? b + d : -1;
            if (src < 0) continue;
            for (int t = 1; t <= kMaxT; ++t)
                if (cost_n_[(size_t) src][t] > 0) {
                    cost_[(size_t) b][t] = cost_[(size_t) src][t];
                    cost_n_[(size_t) b][t] = 1.0;
                    last_[(size_t) b][t] = rounds_;
                }
            break;
        }
    band_ = b;
}

double DraftPolicy::accept_rate(int i, double conf) const {
    conf = unit(conf);
    const int p = std::clamp(i, 0, kOddsPos - 1), b = std::min((int) (conf * kOddsBins), kOddsBins - 1);
    return (held_[(size_t) p][(size_t) b] + kPriorN * conf) / (seen_[(size_t) p][(size_t) b] + kPriorN);
}

void DraftPolicy::chain(const float* conf, int n, double* q) const {
    double all = 1.0;
    for (int i = 0; i < n; ++i) q[i] = all *= accept_rate(i, conf[i]);
}

double DraftPolicy::tokens(const double* q, int k) {
    double e = 1.0;
    for (int i = 0; i < k; ++i) e += q[i];
    return e;
}

bool DraftPolicy::stale(int rows) const {
    return cost_n_[(size_t) band_][rows] > 0 && rounds_ - last_[(size_t) band_][rows] > kStale;
}

bool DraftPolicy::unsure(int rows) const { return cost_n_[(size_t) band_][rows] < kProbes || stale(rows); }

double DraftPolicy::row_ms(int rows) const {
    rows = std::clamp(rows, 1, kMaxT);
    double c = cost_ms(rows);
    // (fewer rows never cost more than a larger size measured lately: an old or unlucky estimate of a small window
    //  does not keep it out)
    for (int u = rows + 1; u <= kMaxT; ++u)
        if (cost_n_[(size_t) band_][u] > 0 && !stale(u)) c = std::min(c, cost_[(size_t) band_][u]);
    return c;
}

double DraftPolicy::sample_cap(int rows) const {
    const auto& c = cost_[(size_t) band_];
    const auto& cn = cost_n_[(size_t) band_];
    if (cn[rows] > 0) return kOutlier * c[rows];
    for (int u = rows + 1; u <= kMaxT; ++u)   // (a size not measured yet: twice a larger one's, else the shape's step up)
        if (cn[u] > 0) return kOutlier * c[u];
    for (int u = rows - 1; u >= 1; --u)
        if (cn[u] > 0) return kOutlier * c[u] * kShape[rows] / kShape[u];
    return std::numeric_limits<double>::infinity();
}

void DraftPolicy::plan(const double* const* q, const int* len, const int* cap, int n, int max_rows, int fixed_rows,
                       double fixed_tokens, int* rows) const {
    max_rows = std::min(max_rows, max_t_);
    std::vector<int> top((size_t) std::max(n, 0)), at((size_t) std::max(n, 0), 1);
    for (int i = 0; i < n; ++i) {
        top[(size_t) i] = std::max(0, std::min(len[i], cap[i] - 1));   // (the most drafts window i can take)
        rows[i] = 1;
    }
    const int base = fixed_rows + n;
    if (base >= max_rows) return;   // (no row left for a draft)
    // the drafts in the order they are likeliest to hold, across the windows (a window's chain only falls, so its
    // own order is kept): for every size of the step, its first drafts are the most tokens that size can commit
    std::vector<int> order;
    double tok = fixed_tokens + n, best_v = tok / row_ms(base);
    int best_r = base;
    for (int r = base; r < max_rows;) {
        int bi = -1;
        double bq = -1.0;
        for (int i = 0; i < n; ++i)
            if (at[(size_t) i] - 1 < top[(size_t) i] && q[i][at[(size_t) i] - 1] > bq)
                bq = q[i][at[(size_t) i] - 1], bi = i;
        if (bi < 0) break;
        order.push_back(bi);
        ++at[(size_t) bi];
        tok += bq;
        const double v = tok / row_ms(++r);
        if (v > best_v) best_v = v, best_r = r;   // (the cost is not convex: every size is weighed)
    }
    const int planned = best_r - base;
    for (int j = 0; j < planned; ++j) ++rows[order[(size_t) j]];
    // a size not known well enough (fewer than kProbes rounds, or none for kStale): the smallest one past the plan is
    // tried, its rows the next drafts in that order, while each of them holds at least kTryOdds of the time - so a
    // cost measured once in a slow stretch, or only guessed, cannot keep a size out for good (the drafts are verified
    // either way: only this round's speed is at stake)
    std::vector<int> probe(rows, rows + n);
    for (int j = planned, r = best_r; j < (int) order.size(); ++j) {
        const int i = order[(size_t) j];
        if (q[i][probe[(size_t) i] - 1] < kTryOdds) break;
        ++probe[(size_t) i];
        if (unsure(++r)) {
            std::copy(probe.begin(), probe.end(), rows);
            return;
        }
    }
}

int DraftPolicy::window(const double* q, int n, int cap) const {
    int rows = 1;
    plan(&q, &n, &cap, 1, kMaxT, 0, 0.0, &rows);
    return rows;
}

void DraftPolicy::observe_cost(int rows, double ms) {
    rows = std::clamp(rows, 1, kMaxT);
    rounds_ += 1.0;
    if (!(ms > 0)) return;
    double& c = cost_[(size_t) band_][rows];
    double& cn = cost_n_[(size_t) band_][rows];
    // a stall, a cold read: twice what the size is known to cost at most, as a quarter of a round
    const double cap = sample_cap(rows), w = ms > cap ? kOutlierW : 1.0;
    ms = std::min(ms, cap);
    if (stale(rows)) {   // measured again after a long stretch: this round, not the old figure
        c = ms;
        cn = kProbes;
    } else {
        const double a = w * std::max(kCostAlpha, 1.0 / (cn + w));   // (a running mean over the first rounds)
        c = cn > 0 ? (1.0 - a) * c + a * ms : ms;
        cn += w;
    }
    last_[(size_t) band_][rows] = rounds_;
}

void DraftPolicy::observe_drafts(const float* conf, int offered, int accepted) {
    // draft i was verified with every draft before it kept (i <= accepted); it held when i < accepted
    for (int i = 0; i < offered && i <= accepted; ++i) {
        const double c = unit(conf[i]);
        const size_t p = (size_t) std::min(i, kOddsPos - 1), b = (size_t) std::min((int) (c * kOddsBins), kOddsBins - 1);
        held_[p][b] = kOddsDecay * held_[p][b] + (i < accepted ? 1.0 : 0.0);
        seen_[p][b] = kOddsDecay * seen_[p][b] + 1.0;
    }
}

std::string DraftPolicy::cost_text() const {
    static const char* const kName[kBands] = {"ctx<8K:", "ctx<64K:", "ctx>=64K:"};
    std::string s;
    char b[48];
    for (int k = 0; k < kBands; ++k) {
        std::string row;
        for (int t = 1; t <= kMaxT; ++t)
            if (cost_n_[(size_t) k][t] > 0) {
                std::snprintf(b, sizeof b, " %d:%.1f(%.0f)", t, cost_[(size_t) k][t], cost_n_[(size_t) k][t]);
                row += b;
            }
        if (row.empty()) continue;
        s += (s.empty() ? "" : " | ") + std::string(kName[k]) + row;
    }
    return s;
}

}  // namespace strata::spec
