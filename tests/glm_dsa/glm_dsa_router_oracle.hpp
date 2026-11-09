// tests/glm_dsa/glm_dsa_router_oracle.hpp - glm-dsa B5.1: the F64 HF-style router oracle, the random router rows and
// the comparison rule shared by glm_dsa_router_cpu_test (the CPU twin) and glm_dsa_router_test (the device kernel).
//
// The rule (BUILD-PLAN B5.1): top-8 ids equal the oracle's, in order, outside near-ties (two of the oracle's 9 best
// biased scores closer than 1e-6: a 1-ulp difference in the sigmoid may legitimately swap them); the weights' relative
// error below 1e-6.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace router_oracle {

constexpr int kE = 256, kK = 8;
constexpr float kScale = 2.5f;
constexpr double kTieGap = 1e-6, kRelTol = 1e-6;

struct Pick {
    int ids[kK];
    double w[kK];
    bool near_tie;
};

// HF modeling: sigmoid in F64, biased selection, top-k, w = p / (sum + 1e-20) * scale
inline Pick oracle(const float* logits, const float* bias, int E = kE, int k = kK, double scale = kScale) {
    std::vector<double> p((size_t) E), sel((size_t) E);
    std::vector<int> ord((size_t) E);
    for (int e = 0; e < E; ++e) {
        p[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) logits[e]));
        sel[(size_t) e] = p[(size_t) e] + (bias ? (double) bias[e] : 0.0);
        ord[(size_t) e] = e;
    }
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    Pick r{};
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        r.ids[i] = ord[(size_t) i];
        sum += p[(size_t) ord[(size_t) i]];
    }
    for (int i = 0; i < k; ++i) r.w[i] = p[(size_t) r.ids[i]] / (sum + 1e-20) * scale;
    r.near_tie = false;
    for (int i = 0; i < k && i + 1 < E; ++i)
        r.near_tie = r.near_tie || sel[(size_t) ord[(size_t) i]] - sel[(size_t) ord[(size_t) i + 1]] < kTieGap;
    return r;
}

// router rows like a trained router's: logits ~ N(0, 2), a few rows shifted far down (tiny sigmoids), and per-row bias
// vectors ~ U(-0.5, 0.5) (exp_probs_b, one per layer: `n_bias` of them)
struct Rows {
    std::vector<float> logits, bias;
    int n = 0, n_bias = 0;
};

inline Rows make_rows(int n, int n_bias, uint32_t seed) {
    Rows R;
    R.n = n;
    R.n_bias = n_bias;
    R.logits.resize((size_t) n * kE);
    R.bias.resize((size_t) n_bias * kE);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    for (auto& b : R.bias) b = ud(rng);
    for (int r = 0; r < n; ++r) {
        const float shift = r % 97 == 5 ? -8.0f : 0.0f;
        for (int e = 0; e < kE; ++e) R.logits[(size_t) r * kE + e] = nd(rng) + shift;
    }
    return R;
}

struct Tally {
    long rows = 0, near_ties = 0, id_fail = 0, w_fail = 0;
    double worst_rel = 0;
};

// one route (ids, w of k) against the oracle
inline void compare(const Pick& o, const int* ids, const float* w, Tally& t) {
    ++t.rows;
    if (o.near_tie) {
        ++t.near_ties;
        return;
    }
    bool same = true;
    for (int i = 0; i < kK; ++i) same = same && ids[i] == o.ids[i];
    if (!same) {
        ++t.id_fail;
        return;
    }
    for (int i = 0; i < kK; ++i) {
        const double rel = std::fabs((double) w[i] - o.w[i]) / std::fabs(o.w[i]);
        t.worst_rel = std::max(t.worst_rel, rel);
        if (!(rel < kRelTol)) {
            ++t.w_fail;
            return;
        }
    }
}

}  // namespace router_oracle
