// src/metal/sampler.cpp - see sampler.hpp.
#include "sampler.hpp"

#include <algorithm>
#include <cmath>

namespace maya::metal {

namespace {

constexpr int kTopKMax = 64;
constexpr int kPenCap = 4096;       // the penalty window at most (glm_model.hpp kPenCap)

}  // namespace


// sampler_parity.cpp's host transcription of the kernel's philox_uniform
float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

int32_t pick(const std::vector<float>& logits, const Sampling& p, const std::vector<int32_t>& hist) {
    const int nv = (int) logits.size();
    std::vector<float> s(logits);
    for (const auto& [id, b] : p.bias) s[(size_t) id] += b;
    if (p.pen_n > 0 && !hist.empty()) {   // llama.cpp's penalties on the window's counts
        const size_t h = std::min<size_t>((size_t) std::min(p.pen_n, kPenCap), hist.size());
        std::vector<std::pair<int32_t, int>> counts;
        std::vector<int32_t> win(hist.end() - (long) h, hist.end());
        std::sort(win.begin(), win.end());
        for (size_t i = 0; i < win.size();) {
            size_t j = i;
            while (j < win.size() && win[j] == win[i]) ++j;
            if (win[i] >= 0 && win[i] < nv) counts.push_back({win[i], (int) (j - i)});
            i = j;
        }
        for (const auto& [v, c] : counts) {
            float l = s[(size_t) v];
            if (l <= 0.0f) l *= p.pen_repeat;
            else l /= p.pen_repeat;
            l -= (float) c * p.pen_freq + 1.0f * p.pen_present;
            s[(size_t) v] = l;
        }
    }
    if (p.greedy || p.temperature <= 0.0f) {   // the argmax, ties to the lower id
        int best = 0;
        for (int v = 1; v < nv; ++v)
            if (s[(size_t) v] > s[(size_t) best]) best = v;
        return best;
    }
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < kTopKMax) ? p.top_k : kTopKMax);
    std::vector<int32_t> ids((size_t) nv);
    for (int v = 0; v < nv; ++v) ids[(size_t) v] = v;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int32_t a, int32_t b) {
        return s[(size_t) a] > s[(size_t) b] || (s[(size_t) a] == s[(size_t) b] && a < b);
    });
    ids.resize((size_t) k);
    int n_keep = k;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) s[(size_t) ids[(size_t) i]] - (double) s[(size_t) ids[0]]);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) s[(size_t) ids[(size_t) i]] - (double) s[(size_t) ids[0]]) / sum;
            if (cum >= (double) p.top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut < p.min_keep) cut = std::min(p.min_keep, k);
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = s[(size_t) ids[0]] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (s[(size_t) ids[(size_t) i]] < thresh) {
                n_keep = i;
                break;
            }
    }
    n_keep = std::max(n_keep, 1);
    const float inv_t = 1.0f / p.temperature;
    float mx = s[(size_t) ids[0]] * inv_t;
    for (int i = 1; i < n_keep; ++i) mx = std::fmax(mx, s[(size_t) ids[(size_t) i]] * inv_t);
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (s[(size_t) ids[(size_t) i]] * inv_t) - (double) mx);
    const float u = philox_uniform(p.seed, p.counter);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (s[(size_t) ids[(size_t) i]] * inv_t) - (double) mx) / sum;
        if ((double) u < cum) return ids[(size_t) i];
    }
    return ids[(size_t) (n_keep - 1)];
}


}  // namespace maya::metal
