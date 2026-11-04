#pragma once
// Several conversations drafting with the NextN (MTP) block at once (STRATA_GLM_MULTI_MTP, glm_mtp_rows.cu): the
// host-side rules of a batched verify step, kept apart so they can be tested without a GPU.
//
// A conversation's WINDOW in a step is the token it feeds (the last one it wrote) and its drafts: 1 + d rows at
// positions p .. p + d.  Row t's pick is the model's token for position p + t + 1; draft t + 1 stands while it equals
// what was written at row t, so the kept rows are the first one and every draft before the first miss.
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

/// A batched step's head takes the plain argmax for a row only for a greedy request (or temperature <= 0) without
/// penalties; else the sampler (pick_tokens), which also gives a greedy request with penalties its penalized pick.  The
/// plain rows' head (rows_head) and the windows' head take the same rule, so a request's picks do not change rule when
/// it moves between the two paths.
inline bool rows_pick_argmax(const strata::kernels::SamplerParams& sp) {
    return (sp.greedy || sp.temperature <= 0.0f) && sp.penalty_last_n <= 0;
}

/// The rows a conversation takes in a step: 1 + its drafts, no more than the tokens it may still write (each row
/// writes one) and the positions its context has left (`room`: max_ctx - 1 - p), at least 1.
inline int mtp_rows_count(int drafts, int64_t tokens_left, int64_t room) {
    int64_t n = 1 + (int64_t) std::max(0, drafts);
    n = std::min(n, tokens_left);
    n = std::min(n, room);
    return (int) std::max<int64_t>(1, n);
}

/// A group's windows cut to `cap` rows in all: the longest one gives up a draft first (the earlier one on a tie), and
/// every conversation keeps its first row.  Returns the rows in all (more than cap only when the group has more
/// conversations than cap).
inline int mtp_rows_fit(std::vector<int>& counts, int cap) {
    int total = 0;
    for (int& c : counts) {
        c = std::max(1, c);
        total += c;
    }
    while (total > cap) {
        int best = -1;
        for (int i = 0; i < (int) counts.size(); ++i)
            if (counts[(size_t) i] > 1 && (best < 0 || counts[(size_t) i] > counts[(size_t) best])) best = i;
        if (best < 0) break;
        --counts[(size_t) best];
        --total;
    }
    return total;
}

/// What the caller makes of the token a row wrote (its stop conditions, a forced next token).
enum class MtpTake { more, stop, forced };

/// The kept rows of one window: win[0 .. T) the tokens fed (win[0] the last written token, win[1 ..] the drafts),
/// picks[0 .. T) the model's picks after each.  `took(t, pick)` writes row t's token (it may replace the pick - a forced
/// token - and returns what it wrote through `written`) and says whether the conversation goes on; a row is looked at
/// only while every draft before it stood.  Returns the rows kept (1 .. T): row t is kept when it was looked at.
template <class Took>
int mtp_accept(const int32_t* win, const int32_t* picks, int T, Took&& took) {
    int keep = 0;
    for (int t = 0; t < T; ++t) {
        int32_t written = picks[t];
        const MtpTake r = took(t, picks[t], written);
        keep = t + 1;
        if (r != MtpTake::more || t + 1 >= T) break;
        if (win[t + 1] != written) break;   // the draft after it missed: the rows from there are not the model's
    }
    return keep;
}

}  // namespace strata::core
