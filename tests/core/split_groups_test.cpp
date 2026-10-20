// split_groups (STRATA_GLM_EVEN_GROUPS): the old rule's groups of k unchanged; the even rule's two groups at least,
// near-equal sizes of k at most, and in steady state - only the group that came back is ready - that group again,
// not cut smaller.  CPU only.
#include "strata/core/split_groups.hpp"

#include <cstdio>
#include <numeric>
#include <vector>

namespace {
int fails = 0;
void check(const std::vector<int>& got, const std::vector<int>& want, const char* what) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL: %s: got", what);
    for (const int g : got) std::fprintf(stderr, " %d", g);
    std::fprintf(stderr, ", want");
    for (const int w : want) std::fprintf(stderr, " %d", w);
    std::fprintf(stderr, "\n");
    ++fails;
}
}  // namespace

int main() {
    using strata::core::split_groups;
    // the old rule: k at a time, whatever runs
    check(split_groups(8, 8, 2, false), {2, 2, 2, 2}, "greedy 8 by 2");
    check(split_groups(4, 4, 4, false), {4}, "greedy 4 by 4 (one group)");
    check(split_groups(7, 7, 3, false), {3, 3, 1}, "greedy 7 by 3");
    check(split_groups(0, 3, 3, false), {}, "nothing ready");
    // the even rule from scratch (all ready)
    check(split_groups(1, 1, 4, true), {1}, "one request");
    check(split_groups(2, 2, 4, true), {1, 1}, "two requests, k 4");
    check(split_groups(3, 3, 4, true), {2, 1}, "three, k 4");
    check(split_groups(4, 4, 4, true), {2, 2}, "four, k 4");
    check(split_groups(5, 5, 4, true), {3, 2}, "five, k 4");
    check(split_groups(8, 8, 4, true), {4, 4}, "eight, k 4");
    check(split_groups(8, 8, 3, true), {3, 3, 2}, "eight, k 3");
    check(split_groups(7, 7, 3, true), {3, 2, 2}, "seven, k 3");
    check(split_groups(8, 8, 2, true), {2, 2, 2, 2}, "eight, k 2 (as greedy)");
    check(split_groups(6, 6, 1, true), {1, 1, 1, 1, 1, 1}, "k 1");
    // steady state: the group that came back is ready, the others in flight
    check(split_groups(2, 4, 4, true), {2}, "a returning 2 of 4, k 4");
    check(split_groups(4, 8, 4, true), {4}, "a returning 4 of 8, k 4");
    check(split_groups(3, 8, 3, true), {3}, "a returning 3 of 8, k 3");
    check(split_groups(2, 8, 3, true), {2}, "a returning 2 of 8, k 3");
    // ... and one that finished leaves a smaller group, never one above the size
    check(split_groups(1, 3, 4, true), {1}, "a returning 1 of 3");
    check(split_groups(4, 5, 4, true), {2, 2}, "4 ready of 5 (size 3): two groups");
    // every split covers the ready ones with groups of 1..k
    for (int k = 1; k <= 8; ++k)
        for (int active = 1; active <= 8; ++active)
            for (int ready = 0; ready <= active; ++ready)
                for (const bool even : {false, true}) {
                    const std::vector<int> g = split_groups(ready, active, k, even);
                    bool ok = std::accumulate(g.begin(), g.end(), 0) == ready;
                    for (const int x : g) ok = ok && x >= 1 && x <= k;
                    if (!ok) {
                        std::fprintf(stderr, "FAIL: ready %d active %d k %d even %d\n", ready, active, k, (int) even);
                        ++fails;
                    }
                }
    if (fails == 0) std::printf("split_groups_test: ok\n");
    return fails == 0 ? 0 : 1;
}
