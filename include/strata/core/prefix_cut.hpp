#pragma once
// The end of a prompt's system prompt as a cut of its read (generate.cpp's prep_prompt): the state there goes into a
// conversation slot of its own (slot_prefix, #179), so another conversation with the same system prompt and tools - a
// new conversation of the same client, another request of the same kind - restores it instead of reading it all
// again.  The one-conversation path cuts there through its checkpoint marks; without checkpoints (STRATA_GLM_CKPTS=0,
// and the several-conversation scheduler, which keeps none) the cut comes from here (STRATA_GLM_PREFIX_CUT=0: not).  CPU
// only, no state.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

/// The first user-turn marker at index 1 or later (the system prompt is [0, it)), 0: none.
inline size_t system_prompt_end(const int32_t* p, size_t n, int32_t user_tok) {
    for (size_t i = 1; i < n; ++i)
        if (p[i] == user_tok) return i;
    return 0;
}

/// Where a read of p[from, n - 1) (the prompt path's part: all but the last token) cuts to keep the system prompt's
/// state: its end when that is at least min_len tokens, at least 512 past `from` (a restored state of the system
/// prompt, or most of it, is not cut again) and leaves at least 64 tokens to the prompt path after it; else 0.
inline size_t prefix_cut(const int32_t* p, size_t n, size_t from, size_t min_len, int32_t user_tok) {
    const size_t e = system_prompt_end(p, n, user_tok);
    if (e == 0 || e < min_len || e < from + 512 || n < 1 || e + 64 > n - 1) return 0;
    return e;
}

/// `at` into the ascending call ends `marks` of a read from `from`, unless it is 0, already there, or closer than
/// min_gap to `from` or to a neighbouring mark (each call must reach the prompt path's minimum; the existing marks win).
inline void add_mark(std::vector<size_t>& marks, size_t at, size_t from, size_t min_gap = 64) {
    if (at == 0 || at < from + min_gap) return;
    const auto it = std::lower_bound(marks.begin(), marks.end(), at);
    if (it != marks.end() && *it < at + min_gap) return;            // (also: already there)
    if (it != marks.begin() && *(it - 1) + min_gap > at) return;
    marks.insert(it, at);
}

/// A kept slot's tokens are a system prompt's state (slot_prefix's) when they hold no user turn: a conversation's own
/// state always holds one.  (The slot files carry no such flag; this tells them apart when they are indexed again.)
inline bool is_system_prefix(const std::vector<int32_t>& toks, int32_t user_tok) {
    return !toks.empty() && std::find(toks.begin(), toks.end(), user_tok) == toks.end();
}

}  // namespace strata::core
