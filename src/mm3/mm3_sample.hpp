// src/mm3/mm3_sample.hpp - maya-mm3's sampler (mm3_main.cpp's serve loop): a request's sampling settings, the next
// token's distribution from a row of logits or from its candidates (Model::forward_cands), draws, and the residual of
// speculative sampling.  Header-only, so tests/mm3_sampler_test.cpp checks it against itself.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mm3_model.hpp"

namespace mm3::sample {

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    long long think_budget = 0;
    int think_end = -1;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    std::string file;       // GENI: the pictures
    std::vector<int> ids;
    int rid = -1;           // rid=N: the request's id (--parallel: every line it causes carries it)
};

inline bool penalised(const Req& r) {
    return r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f);
}

// the next token's distribution from a row's candidate logits - (id, logit) pairs holding every id the distribution
// can hold: the whole row, or Model::forward_cands' kc largest with their ties - and the logits of the extra ids (the
// penalised ones not among them): penalties over the history's last pen_n tokens (llama.cpp's), then greedy
// (temperature <= 0: the argmax alone, ties to the lower id) or temperature, top_k, top_p, min_p - (token, probability)
// pairs, the most likely first (ties: the lower id first), summing to 1.  Over the candidates it is the whole row's
// distribution whenever kc >= top_k + extra.size() and extra holds every penalised id.
inline std::vector<std::pair<int, double>> dist_c(std::vector<std::pair<int, float>> c, const std::vector<int>& extra,
                                                  const std::vector<float>& ex, const Req& r,
                                                  const std::vector<int>& hist) {
    if (penalised(r)) {
        std::unordered_map<int, int> cnt;
        const size_t n = r.pen_n < 0 ? hist.size() : std::min(hist.size(), (size_t) r.pen_n);
        for (size_t i = hist.size() - n; i < hist.size(); ++i) ++cnt[hist[i]];
        std::unordered_map<int, size_t> at;
        for (size_t i = 0; i < c.size(); ++i) at[c[i].first] = i;
        for (const auto& [t, k] : cnt) {
            auto it = at.find(t);
            size_t i = 0;
            if (it != at.end()) {
                i = it->second;
            } else {   // (a penalised id below the candidates: its logit from extra, a candidate now)
                const auto e = std::find(extra.begin(), extra.end(), t);
                if (e == extra.end()) continue;   // (outside the vocabulary)
                c.push_back({t, ex[(size_t) (e - extra.begin())]});
                i = c.size() - 1;
            }
            float& x = c[i].second;
            if (r.pen_repeat > 0.0f && r.pen_repeat != 1.0f) x = x > 0 ? x / r.pen_repeat : x * r.pen_repeat;
            x -= r.pen_freq * (float) k + r.pen_present;
        }
    }
    if (!(r.temperature > 0.0f)) {
        size_t b = 0;
        for (size_t i = 1; i < c.size(); ++i)
            if (c[i].second > c[b].second || (c[i].second == c[b].second && c[i].first < c[b].first)) b = i;
        return {{c[b].first, 1.0}};
    }
    std::vector<std::pair<float, int>> v;
    v.reserve(c.size());
    for (const auto& [id, x] : c) v.push_back({x / r.temperature, id});
    const size_t k = r.top_k > 0 ? std::min((size_t) r.top_k, v.size()) : v.size();
    std::partial_sort(v.begin(), v.begin() + (long) k, v.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    const auto& cs = v;
    double sum = 0;
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) sum += (p[i] = std::exp((double) (cs[i].first - cs[0].first)));
    size_t keep = k;
    double cum = 0;
    for (size_t i = 0; i < k; ++i) {
        cum += p[i] / sum;
        if (r.min_p > 0.0f && p[i] < r.min_p * p[0]) {
            keep = std::max<size_t>(1, i);
            break;
        }
        if (r.top_p < 1.0f && cum >= r.top_p) {
            keep = i + 1;
            break;
        }
    }
    double s2 = 0;
    for (size_t i = 0; i < keep; ++i) s2 += p[i];
    std::vector<std::pair<int, double>> out(keep);
    for (size_t i = 0; i < keep; ++i) out[i] = {cs[i].second, p[i] / s2};
    return out;
}

// ... from a whole row of logits
inline std::vector<std::pair<int, double>> dist(const float* row, int V, const Req& r, const std::vector<int>& hist) {
    std::vector<std::pair<int, float>> c((size_t) V);
    for (int i = 0; i < V; ++i) c[(size_t) i] = {i, row[i]};
    return dist_c(std::move(c), {}, {}, r, hist);
}

// a draw from d, without `skip` (renormalised; -1: none)
inline int draw(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double tot = 0;
    for (const auto& [t, q] : d)
        if (t != skip) tot += q;
    if (!(tot > 0)) return d[0].first;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    int last = d[0].first;
    for (const auto& [t, q] : d) {
        if (t == skip) continue;
        last = t;
        u -= q;
        if (u <= 0) return t;
    }
    return last;
}

// a draw from max(0, p - q) renormalised (the token after a sampled draft the target turned down: speculative
// sampling's residual, which leaves the token the target's own sample); p itself when that is empty
inline int draw_residual(const std::vector<std::pair<int, double>>& p, const std::vector<std::pair<int, float>>& q,
                         std::mt19937_64& rng) {
    std::vector<std::pair<int, double>> d;
    for (const auto& [t, pt] : p) {
        double qt = 0;
        for (const auto& [u, qu] : q)
            if (u == t) qt = qu;
        if (pt > qt) d.push_back({t, pt - qt});
    }
    return d.empty() ? draw(p, rng) : draw(d, rng);
}

// a forward's candidates for these rows (mm3::Model::forward_cands): kc = top_k + the penalised ids (the last pen_n
// tokens of the history and the window's - every row's own window within them); false: the whole rows instead (no
// top_k, or the whole history penalised)
inline bool cands_for(const Req& r, const std::vector<int>& hist, const std::vector<int>& more, mm3::Cands& c) {
    c.extra.clear();
    if (penalised(r)) {
        if (r.pen_n < 0) return false;
        std::vector<int> all(hist.end() - (long) std::min(hist.size(), (size_t) r.pen_n), hist.end());
        all.insert(all.end(), more.begin(), more.end());
        for (int t : all)
            if (std::find(c.extra.begin(), c.extra.end(), t) == c.extra.end()) c.extra.push_back(t);
    }
    if (r.temperature > 0.0f && !(r.top_k > 0)) return false;
    c.kc = (r.temperature > 0.0f ? r.top_k : 1) + (int) c.extra.size();
    return true;
}

}  // namespace mm3::sample
