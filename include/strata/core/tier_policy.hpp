#pragma once
// The decode expert tier's aged counts, opt-in (each unset = the rule as before):
//
//   STRATA_GLM_COUNT_HALF=<events>   the host's aged counts (what a boundary evicts, refills and swaps by) are halved
//                                    every this many count events, 8 a decode route (default 32768: 4096 routes of a
//                                    part, a few seconds of a batched decode).  Several conversations take turns and
//                                    pause for their tools; a longer memory keeps the experts all of them use.
//   STRATA_GLM_PROMPT_COUNTS=<w>     a prompt chunk's routing enters those counts with weight w (default 1).  A chunk
//                                    adds T x 8 events a layer - one read of a few hundred tokens halves every count
//                                    several times, a long one wipes them - so after a read the evictions follow that
//                                    one prompt.  0 = prompts neither add to the counts nor age them.
//   STRATA_GLM_LEND_SPREAD=1         the pool's tail (the slots a prompt borrows) laid out slot by slot across the
//                                    layers instead of layer by layer. A prompt borrows the region's front - a short
//                                    read a few hundred MB of it - so with the layer-major layout every read takes the
//                                    whole tails of the first MoE layers, the same few layers every time, and their
//                                    refill after the read goes by counts that read has just aged. Spread, it takes
//                                    a slot or two from every layer.
//
// Pure functions, CPU-tested (tests/core/tier_policy_test.cpp).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::core {

struct TierPolicy {
    uint64_t half = 32768;    // count events between two halvings of the host's aged counts
    double prompt_w = 1.0;    // a prompt route's weight in those counts (and in their aging)
    bool lend_spread = false; // the tail slot-major across the layers
};

/// The two settings' texts (null: unset) -> the policy.  Out-of-range values keep the default.
inline TierPolicy tier_policy_parse(const char* half, const char* prompt_w) {
    TierPolicy p;
    if (half != nullptr && half[0] != '\0') {
        const double v = std::atof(half);
        if (v >= 1024.0 && v <= 1e12) p.half = (uint64_t) v;
    }
    if (prompt_w != nullptr && prompt_w[0] != '\0') {
        const double v = std::atof(prompt_w);
        if (v >= 0.0 && v <= 1.0) p.prompt_w = v;
    }
    return p;
}

/// The engine's policy, read once.
inline const TierPolicy& tier_policy() {
    static const TierPolicy p = [] {
        TierPolicy q = tier_policy_parse(std::getenv("STRATA_GLM_COUNT_HALF"), std::getenv("STRATA_GLM_PROMPT_COUNTS"));
        const char* s = std::getenv("STRATA_GLM_LEND_SPREAD");
        q.lend_spread = s != nullptr && std::atoi(s) != 0;
        return q;
    }();
    return p;
}

/// A prompt chunk's count of an expert as it enters the aged counts (rounded to nearest; 0 when the weight is 0).
inline uint64_t tier_prompt_count(uint64_t cnt, double w) {
    if (w >= 1.0) return cnt;
    if (w <= 0.0) return 0;
    return (uint64_t) std::llround((double) cnt * w);
}

/// `add` more count events: how many halvings of every count are due now (events keeps the remainder).
/// The decode's own aging (one route = 8 events) and the prompt's take this one rule.
inline int tier_halvings(uint64_t& events, uint64_t add, uint64_t half) {
    if (half == 0) return 0;
    events += add;
    int n = 0;
    while (events >= half) {
        events -= half;
        ++n;
    }
    return n;
}

/// One count after n halvings (a count that halved 32 times or more is 0).
inline uint32_t tier_halved(uint32_t c, int n) { return n >= 32 ? 0u : c >> n; }


/// Where each layer's tail slots sit in the tail region: layer k's slot t at off[k] + t * step (step 0: its own
/// stride, the layer's tail contiguous).  Layer-major (spread false, or tails of different lengths): each layer's tail
/// after the one before.  Spread: off[k] = the strides of the layers before it, step = one slot of every layer - tail
/// slot t of all the layers, then t + 1; the region's first X bytes then hold whole columns (a slot of every layer)
/// and at most one partial one.  Both fill exactly sum(tails[k] * strides[k]) bytes, no two slots overlapping.
struct TailLayout {
    std::vector<size_t> off;
    size_t step = 0;
    bool spread = false;
};
inline TailLayout tier_tail_layout(const std::vector<size_t>& strides, const std::vector<int>& tails, bool spread) {
    TailLayout r;
    r.off.assign(strides.size(), 0);
    bool equal = !tails.empty();
    for (size_t k = 1; k < tails.size(); ++k) equal = equal && tails[k] == tails[0];
    size_t b = 0;
    if (spread && equal) {
        for (size_t k = 0; k < strides.size(); ++k) {
            r.off[k] = b;
            b += strides[k];
        }
        r.step = b;
        r.spread = true;
        return r;
    }
    for (size_t k = 0; k < strides.size(); ++k) {
        r.off[k] = b;
        b += (size_t) (tails[k] > 0 ? tails[k] : 0) * strides[k];
    }
    return r;
}

}  // namespace strata::core
