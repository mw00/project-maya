#pragma once
// The split decode's groups (STRATA_GLM_SPLIT_ROWS = k): which of the requests ready for a step go through the parts
// together.  The groups pipeline across the split - while one group's token is on the second GPU, the next one's is
// on the first - so ONE group leaves every GPU but one idle at any moment.
//
// The old rule (greedy) fills groups of k in turn: at k or fewer requests that is one group, and the GPUs take turns
// (two GPUs, k 4, 4 requests: slower than k 2).  STRATA_GLM_EVEN_GROUPS=1 instead cuts the requests running in all
// (`active`) into G = max(2, ceil(active / k)) groups of near-equal size, s = ceil(active / G) at most: k 4 is then
// two groups of 2 at 4 requests (the pipeline full) and two of 4 at 8.  The size comes from all the running requests,
// not from the ones ready now: in steady state the ready ones are the group that just came back, and cutting those
// again (4 -> 2 + 2 -> 1 + 1 ...) would leave groups of one.
#include <algorithm>
#include <vector>

namespace strata::core {

/// The sizes of the groups `ready` requests go in, in order (each 1..k, summing to `ready`); `active` requests run in
/// all (ready and in flight), k >= 1.  even = false: groups of k and the rest (the old rule).
inline std::vector<int> split_groups(int ready, int active, int k, bool even) {
    std::vector<int> out;
    if (ready <= 0) return out;
    k = std::max(1, k);
    if (!even) {
        for (int left = ready; left > 0; left -= k) out.push_back(std::min(k, left));
        return out;
    }
    active = std::max(active, ready);
    const int groups = active <= 1 ? 1 : std::max(2, (active + k - 1) / k);
    const int size = std::min(k, (active + groups - 1) / groups);
    // the ready ones in ceil(ready / size) groups, the sizes differing by one at most (7 at size 3: 3 2 2)
    const int n = (ready + size - 1) / size;
    for (int i = 0; i < n; ++i) out.push_back(ready / n + (i < ready % n ? 1 : 0));
    return out;
}

}  // namespace strata::core
