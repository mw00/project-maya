// include/strata/core/glm_draft_adaptive.hpp - glm-dsa B15.9: the adaptive draft depth controller for multi-token
// drafts (the B14.13 n-gram proposals today, B11.4's chain drafts later), behind STRATA_GLM_DRAFT_ADAPTIVE.
//
// llama.cpp #27210 (common/speculative-adaptive.h, draft-mtp-adaptive): a deterministic hysteresis state machine
// instead of an epsilon-greedy tuner.  The depth climbs one step after climb_threshold(depth) consecutive verifies
// that accepted every drafted token (2, 4, then the hardened 3 -> 4 barrier of 10 that keeps prose pinned, 6, 3, 2)
// and drops one step when the accumulated miss pressure, the sum of (n_draft - n_accepted) over the rounds since the
// last full accept, reaches drop_pressure(depth) = max(5 * depth, 20).  It stays within [floor, cap] and resets per
// generation (one controller per decode_spec call: NgramSession).
//
// STRATA_GLM_DRAFT_ADAPTIVE=<floor> (default off; 1 = on with floor 1; 2..8 also sets the floor, which is the start
// depth): NgramSession proposes at most depth() tokens (the cap is STRATA_GLM_NGRAM_MAX).  Feedback without a
// multi-token verify: ProposalTracker checks each proposal against the tokens the decode then confirms, one round at
// a time (a proposal issued while another is unresolved is not tracked, as a k-token verify would not have issued
// it), so the controller sees exactly what a k-token verify would have accepted.  In today's depth-1 pipeline the
// depth sizes last_proposal() (the rows a B11.4 verify would load); the first token's verify is unchanged.
//
// Plain C++, header only.  tests/glm_dsa/glm_draft_adaptive_test.cpp.
#pragma once

#include "strata/core/glm_env.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::core::glmng {

/// STRATA_GLM_DRAFT_ADAPTIVE: 0 off, else the floor (1..8).  Read on every call.
inline int draft_adaptive_floor() {
    const long long f = strata::glmenv::env_int("STRATA_GLM_DRAFT_ADAPTIVE", 0);
    return f <= 0 ? 0 : (int) std::min<long long>(f, 8);
}

struct AdaptiveDepth {
    int n_cur = 0;     // the current depth
    int n_climb = 0;   // consecutive full accepts
    int n_drop = 0;    // accumulated miss pressure
    uint64_t climbs = 0, drops = 0, rounds = 0;

    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10;   // the hardened 3 -> 4 barrier
            case 4: return 6;
            case 5: return 3;
            default: return 2;   // 6 and deeper
        }
    }
    static int drop_pressure(int depth) { return std::max(depth * 5, 20); }

    void reset(int n_max, int floor) {
        n_cur = std::min(std::max(1, floor), std::max(1, n_max));
        n_climb = n_drop = 0;
    }
    /// One verify round: n_draft tokens drafted, n_accepted of them accepted.
    void update(int n_draft, int n_accepted, int n_max, int floor) {
        if (n_draft <= 0) return;
        ++rounds;
        const int cap = std::max(1, n_max), lo = std::max(1, floor);
        if (n_accepted >= n_draft) {
            n_drop = 0;
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                ++n_cur;
                ++climbs;
                n_climb = 0;
            }
        } else {
            n_climb = 0;
            if (n_cur > lo) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    --n_cur;
                    ++drops;
                    n_drop = 0;
                }
            }
        }
    }
    int depth() const { return n_cur; }
};

/// Resolves proposals against the confirmed tokens, one round at a time.
class ProposalTracker {
public:
    /// A proposal for the tokens that follow the last confirmed one.  Ignored while another is unresolved.
    bool issue(const int32_t* toks, int n) {
        if (!prop_.empty() || n <= 0) return false;
        prop_.assign(toks, toks + n);
        at_ = 0;
        return true;
    }
    /// The next confirmed token.  Returns true and sets (n_draft, n_accepted) when the open proposal resolves (its
    /// first miss, or its last token matched).
    bool push(int32_t tok, int& n_draft, int& n_accepted) {
        if (prop_.empty()) return false;
        if (prop_[(size_t) at_] == tok && at_ + 1 < (int) prop_.size()) {
            ++at_;
            return false;
        }
        n_draft = (int) prop_.size();
        n_accepted = prop_[(size_t) at_] == tok ? at_ + 1 : at_;
        prop_.clear();
        return true;
    }
    bool open() const { return !prop_.empty(); }

private:
    std::vector<int32_t> prop_;
    int at_ = 0;
};

}  // namespace strata::core::glmng
