// src/ling/ling_moe_plan.hpp - how a staged expert set of a long prompt block is run in token tiles.
//
// The prompt's MoE runs one macro block of B tokens (up to the prefill block, 8192) at a time, but its set scratch
// (the gate / up rows and the down's rows) holds P x n_used rows (P: the prompt chunk).  The staged sets are packed to
// that many rows, but one expert alone can have more once B > P x n_used (a token picks an expert once, so an expert
// has at most B rows): 4096 tokens with --prefill-chunk 256 and 8 experts a token, 1024 with 64.  Such a set used to
// write past the scratch (an illegal memory access in moe_combine, or other rows overwritten).  It is now run P tokens
// at a time: every tile has at most P x n_used rows.  Sets that fit run in one pass as before (the same launches).
//
// Host code only, no CUDA: tests/ling_unit_test.cpp runs it.
#pragma once

#include <algorithm>
#include <vector>

namespace ling::moe_plan {

struct Tile {
    int t0 = 0;         // first token of the tile (in the block)
    int T = 0;          // tokens
    int max_rows = 0;   // the busiest expert of the set: its rows in the tile (0: no token of the tile uses the set)
    int rows = 0;       // all rows of the set in the tile (<= T x n_used)
};

// in_set[e]: the expert's index in the set, or -1; ids: B x n_used router choices (host copy)
inline std::vector<Tile> tiles(const int* ids, int B, int n_used, const std::vector<int>& in_set, int P) {
    std::vector<Tile> out;
    if (B <= 0 || n_used <= 0 || P <= 0) return out;
    int n_set = 0;
    for (int v : in_set) n_set = std::max(n_set, v + 1);
    std::vector<int> cnt((size_t) std::max(1, n_set), 0);
    for (int t0 = 0; t0 < B; t0 += P) {
        Tile tl;
        tl.t0 = t0;
        tl.T = std::min(P, B - t0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int t = t0; t < t0 + tl.T; ++t)
            for (int j = 0; j < n_used; ++j) {
                const int e = ids[(size_t) t * (size_t) n_used + (size_t) j];
                if (e < 0 || e >= (int) in_set.size() || in_set[(size_t) e] < 0) continue;
                ++tl.rows;
                tl.max_rows = std::max(tl.max_rows, ++cnt[(size_t) in_set[(size_t) e]]);
            }
        out.push_back(tl);
    }
    return out;
}

// a set of `rows` rows in one pass over the block fits the scratch of max_set_rows rows
inline bool one_pass(long long rows, int max_set_rows) { return rows <= (long long) max_set_rows; }

}  // namespace ling::moe_plan
