// tier_policy (STRATA_GLM_COUNT_HALF, STRATA_GLM_PROMPT_COUNTS, STRATA_GLM_LEND_SPREAD): the parse, the defaults
// that keep the rule as before, the prompt's weighted counts, the aging both paths share, and the tail layouts
// (both tile the region; spread, a loan takes every layer's slots evenly).  CPU only.
#include "strata/core/tier_policy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
}  // namespace

int main() {
    using namespace strata::core;
    // unset: the rule as before
    {
        const TierPolicy p = tier_policy_parse(nullptr, nullptr);
        expect(p.half == 32768 && p.prompt_w == 1.0, "defaults");
        const TierPolicy q = tier_policy_parse("", "");
        expect(q.half == 32768 && q.prompt_w == 1.0, "empty texts are unset");
    }
    // the parse and its ranges
    {
        const TierPolicy p = tier_policy_parse("262144", "0.25");
        expect(p.half == 262144 && p.prompt_w == 0.25, "values");
        expect(tier_policy_parse("1e6", nullptr).half == 1000000, "an exponent");
        expect(tier_policy_parse("100", nullptr).half == 32768, "too short a period keeps the default");
        expect(tier_policy_parse(nullptr, "1.5").prompt_w == 1.0, "a weight above 1 keeps the default");
        expect(tier_policy_parse(nullptr, "-1").prompt_w == 1.0, "a negative weight keeps the default");
        expect(tier_policy_parse(nullptr, "0").prompt_w == 0.0, "weight 0");
    }
    // the prompt's counts
    {
        expect(tier_prompt_count(37, 1.0) == 37, "weight 1 is the count");
        expect(tier_prompt_count(37, 0.0) == 0, "weight 0 adds nothing");
        expect(tier_prompt_count(37, 0.1) == 4, "weight 0.1 rounds");
        expect(tier_prompt_count(4, 0.1) == 0, "a small count at a small weight is 0");
    }
    // the aging: the same halvings as the engine's old loops at the default period
    {
        // the prompt's old loop: events += ev; while (events >= 32768) { events -= 32768; halve all }
        for (uint64_t start : {0ull, 100ull, 32760ull}) {
            for (uint64_t add : {0ull, 8ull, 4728ull, 32768ull, 99300ull, 1333000ull}) {
                uint64_t old_ev = start + add;
                int old_n = 0;
                while (old_ev >= 32768) {
                    old_ev -= 32768;
                    ++old_n;
                }
                uint64_t ev = start;
                const int n = tier_halvings(ev, add, 32768);
                expect(n == old_n && ev == old_ev, "halvings match the old loop");
            }
        }
        uint32_t c = 1000;
        for (int k = 0; k < 3; ++k) c >>= 1;
        expect(tier_halved(1000, 3) == c, "n halvings at once");
        expect(tier_halved(0xFFFFFFFFu, 32) == 0 && tier_halved(5, 40) == 0, "32+ halvings are 0");
        expect(tier_halved(7, 0) == 7, "no halving");
        uint64_t ev = 0;
        expect(tier_halvings(ev, 1000000, 262144) == 3 && ev == 1000000 - 3 * 262144, "a longer period");
        uint64_t z = 5;
        expect(tier_halvings(z, 100, 0) == 0, "period 0 never halves");
        // decode: 8 events a route; one halving per 4096 routes at the default
        uint64_t d = 0;
        int nh = 0;
        for (int r = 0; r < 4096 * 3; ++r) nh += tier_halvings(d, 8, 32768);
        expect(nh == 3 && d == 0, "4096 routes a halving");
    }
    // the tail layout: both layouts tile the region exactly; spread, a prefix of X bytes holds whole columns
    {
        const std::vector<size_t> strides = {12976128, 12976128, 12976128, 12976128, 11665408, 11665408, 11665408};
        const std::vector<int> tails(strides.size(), 10);
        size_t total = 0;
        for (size_t k = 0; k < strides.size(); ++k) total += (size_t) tails[k] * strides[k];
        for (bool spread : {false, true}) {
            const TailLayout L = tier_tail_layout(strides, tails, spread);
            expect(L.spread == spread, "spread when asked and the tails are equal");
            // every slot's byte range, sorted: they tile [0, total) with no gap and no overlap
            std::vector<std::pair<size_t, size_t>> iv;
            for (size_t k = 0; k < strides.size(); ++k)
                for (int t = 0; t < tails[k]; ++t) {
                    const size_t a = L.off[k] + (size_t) t * (L.step ? L.step : strides[k]);
                    iv.push_back({a, a + strides[k]});
                }
            std::sort(iv.begin(), iv.end());
            bool tiled = iv.front().first == 0 && iv.back().second == total;
            for (size_t i = 1; i < iv.size(); ++i) tiled = tiled && iv[i].first == iv[i - 1].second;
            expect(tiled, spread ? "spread: the slots tile the region" : "layer-major: the slots tile the region");
            // a loan of X bytes from the front: how many slots each layer loses
            for (size_t X : {(size_t) 415 << 20, (size_t) 1 << 30, total / 2, total}) {
                std::vector<int> lost(strides.size(), 0);
                for (size_t k = 0; k < strides.size(); ++k)
                    for (int t = 0; t < tails[k]; ++t)
                        if (L.off[k] + (size_t) t * (L.step ? L.step : strides[k]) < X) ++lost[k];
                const int mx = *std::max_element(lost.begin(), lost.end());
                const int mn = *std::min_element(lost.begin(), lost.end());
                if (spread)
                    expect(mx - mn <= 1, "spread: every layer loses the same slots, give or take one");
                else if (X < total)
                    expect(lost.front() == 10 && lost.back() < 10, "layer-major: the first layers lose their whole tail");
            }
        }
        const std::vector<int> uneven = {10, 9, 10};
        const TailLayout U = tier_tail_layout({1, 1, 1}, uneven, true);
        expect(!U.spread && U.step == 0 && U.off[1] == 10 && U.off[2] == 19, "uneven tails: layer-major");
        const TailLayout E = tier_tail_layout({}, {}, true);
        expect(!E.spread && E.off.empty(), "no layers");
    }
    if (fails == 0) std::printf("tier_policy_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
