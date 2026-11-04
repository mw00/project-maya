// src/metal/metal_sampler_test.cpp - maya-metal's token pick (sampler.cpp) against the CUDA engine's own reference
// for its sampler kernel: sampler_parity.cpp's host transcription (Philox and the sampled chain), copied verbatim
// below.  Random logits with ties, every key the server sends (temperature, top_k incl. 0 and > 64, top_p, min_p,
// seeds and counters, penalties with a history), the greedy path and logit bias - the same token every time.
#include "sampler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct RefParams {   // the fields of strata::kernels::SamplerParams the reference reads
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float temperature = 1.0f;
    int min_keep = 1;
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    uint64_t seed = 0, counter = 0;
};

// ---- verbatim from src/kernels/sampler_parity.cpp
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};

float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// The full SAMPLED chain, host side - the kernel's `sampler_kernel` in serial form, in llama.cpp's order:
// penalties on the raw logits during the top_k selection (ties to the lowest index), top_p's cut in double over
// the top_k list, the min_p prefix cut on its survivors, the temperature, and one Philox draw at
// (seed, counter + row).  One penalties stage (issue #53: this reference used to repeat the kernel's second one).
int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist,
                      const RefParams& p, int row) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;                       // 1..64 as given; 0 (off) and wider keep the widest list, 64
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best); sel_logit.push_back(bv);
    }
    // llama.cpp's order (issue #53): top_p over the whole top_k list, then min_p on its survivors
    const int n_sel = (int) sel_ids.size();
    int n_keep = n_sel;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_sel; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0.0;
        int cut = n_sel;
        for (int i = 0; i < n_sel; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_sel ? p.min_keep : n_sel;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };   // one penalties stage, before (#53)
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    int pick = sel_ids[(size_t) (n_keep - 1)];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}


// ---- end of the copy

}  // namespace

int main() {
    std::mt19937_64 rng(1234);
    int bad = 0, n = 0;
    for (int it = 0; it < 4000; ++it) {
        const int nv = 50 + (int) (rng() % 400);
        std::vector<float> l((size_t) nv);
        std::normal_distribution<float> nd(0.0f, 1.0f + (float) (rng() % 6));
        for (float& x : l) x = (rng() % 7 == 0) ? std::round(nd(rng)) : nd(rng);   // ties too
        RefParams r;
        maya::metal::Sampling s;
        const int kk[] = {0, 1, 5, 20, 40, 64, 100};
        r.top_k = s.top_k = kk[rng() % 7];
        r.top_p = s.top_p = (rng() % 3 == 0) ? 1.0f : 0.5f + 0.5f * (float) (rng() % 100) / 100.0f;
        r.min_p = s.min_p = (rng() % 3 == 0) ? 0.0f : (float) (rng() % 30) / 100.0f;
        r.temperature = s.temperature = 0.2f + 1.6f * (float) (rng() % 100) / 100.0f;
        r.seed = s.seed = rng() % 5;
        r.counter = s.counter = rng() % 1000;
        s.greedy = false;
        std::vector<int> hist;
        std::vector<int32_t> h32;
        if (rng() % 2) {   // penalties over a window of the history
            const int hl = 1 + (int) (rng() % 64);
            for (int i = 0; i < hl; ++i) {
                const int t = (int) (rng() % (uint64_t) nv);
                hist.push_back(t);
                h32.push_back(t);
            }
            r.penalty_last_n = s.pen_n = hl;
            r.penalty_repeat = s.pen_repeat = 1.0f + (float) (rng() % 50) / 100.0f;
            r.penalty_freq = s.pen_freq = (float) (rng() % 20) / 100.0f;
            r.penalty_present = s.pen_present = (float) (rng() % 20) / 100.0f;
        }
        const int want = sampled_reference(l, hist, r, 0);
        const int got = maya::metal::pick(l, s, h32);
        ++n;
        if (want != got && ++bad <= 5)
            std::fprintf(stderr, "case %d: reference %d, maya-metal %d (top_k %d top_p %.2f min_p %.2f T %.2f)\n", it, want,
                         got, r.top_k, r.top_p, r.min_p, r.temperature);
    }
    // greedy: the argmax, ties to the lower id; logit bias -100 bans
    {
        std::vector<float> l = {0.5f, 2.0f, 2.0f, -1.0f};
        maya::metal::Sampling s;
        if (maya::metal::pick(l, s, {}) != 1) ++bad, std::fprintf(stderr, "greedy tie\n");
        s.bias = {{1, -INFINITY}};
        if (maya::metal::pick(l, s, {}) != 2) ++bad, std::fprintf(stderr, "greedy bias\n");
        n += 2;
    }
    if (bad) {
        std::fprintf(stderr, "metal_sampler_test: %d of %d picks differ from the CUDA engine's reference\n", bad, n);
        return 1;
    }
    std::printf("metal_sampler_test: PASS (%d picks equal the CUDA engine's sampler reference)\n", n);
    return 0;
}
