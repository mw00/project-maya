// src/ds4/ds4_marks.hpp - maya-ds4's restore points: which one a new prompt goes back to, which of the ones a request
// asks for its read keeps, and which one gives way to a new one.  A restore point is the state at a position (the
// window rings and the compressors' partial groups; the rest of the state is indexed by position).  Besides the three a
// conversation always has (the end of its last prompt, the position before it, the anchor), the server may ask for more
// on a GEN line (marks=<pos>,... and shared=<pos>): the start of the last user message, a point near the prompt's tail,
// the end of a start that other conversations share (a system prompt and its tools).  The read stops at each to keep
// it, so a prompt that differs from the conversation after one of them - its last message edited, a new question
// after the same document, a history rendered again - goes back there instead of further.  Pure logic, no GPU
// (tests/ds4_marks_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::ds4 {

// the restore points a conversation keeps besides its three, at most (MAYA_DS4_RESTORE_POINTS is clamped to it)
constexpr int kMarksMax = 16;

// the latest of the points `at` (positions; -1: none) at or before `limit` - where the state's tokens still agree with
// the new prompt - and at or before the state's own position `pos`; the first of equal ones; -1: none
inline int latest_point(const std::vector<int64_t>& at, int64_t limit, int64_t pos) {
    int b = -1;
    for (size_t i = 0; i < at.size(); ++i)
        if (at[i] >= 0 && at[i] <= limit && at[i] <= pos && (b < 0 || at[i] > at[(size_t) b])) b = (int) i;
    return b;
}

// a restore point a prompt's read keeps: where, and whether it is the shared start (kept before the others)
struct MarkAt {
    int64_t pos = 0;
    bool shared = false;
};

// The restore points the read of a prompt's tokens [from, total) keeps, in order, from the positions a request asks
// for (`asked`, and `shared`: -1 none): those the read starts at or goes past - from `from` (the state there, kept
// before the read) to short of the prompt's last token (checkpoint 1 keeps the state before it).  One inside a
// picture's span (`spans`: start, length) is moved to the span's start (a span is read whole); one fewer than `near`
// tokens past `from` is kept at `from` (no read of its own for a few tokens), and of two fewer than `near` apart the
// first is kept, with the second's role too.  At most `cap`: the shared start first, then the latest.
inline std::vector<MarkAt> mark_plan(const std::vector<int64_t>& asked, int64_t shared, int64_t from, int64_t total,
                                     int cap, int64_t near,
                                     const std::vector<std::pair<int64_t, int64_t>>& spans = {}) {
    std::vector<MarkAt> v;
    if (cap <= 0 || from < 0 || total - from < 2) return v;
    const auto add = [&](int64_t p, bool sh) {
        for (const auto& s : spans)
            if (p > s.first && p < s.first + s.second) p = s.first;
        if (p < from || p >= total - 1) return;
        if (p - from < near) p = from;
        v.push_back({p, sh});
    };
    for (int64_t p : asked) add(p, false);
    if (shared >= 0) add(shared, true);
    std::stable_sort(v.begin(), v.end(), [](const MarkAt& a, const MarkAt& b) { return a.pos < b.pos; });
    std::vector<MarkAt> out;
    for (const MarkAt& m : v) {
        if (!out.empty() && m.pos - out.back().pos < std::max<int64_t>(near, 1)) {
            out.back().shared = out.back().shared || m.shared;
            continue;
        }
        out.push_back(m);
    }
    while ((int) out.size() > cap) {   // (the earliest that is not the shared start goes first)
        auto it = std::find_if(out.begin(), out.end(), [](const MarkAt& m) { return !m.shared; });
        out.erase(it == out.end() ? out.begin() : it);
    }
    return out;
}

// a kept restore point as the choice of the one to give up sees it
struct MarkUse {
    int64_t pos = 0;
    bool shared = false;
    uint64_t used = 0;   // (its last use: kept or gone back to - a clock that only grows)
};

// the restore point that gives way to a new one when a conversation holds its cap: the least recently used of those
// that are not a shared start, else the least recently used; -1: none held
inline int mark_victim(const std::vector<MarkUse>& v) {
    for (int pass = 0; pass < 2; ++pass) {
        int b = -1;
        for (size_t i = 0; i < v.size(); ++i)
            if ((pass == 1 || !v[i].shared) && (b < 0 || v[i].used < v[(size_t) b].used)) b = (int) i;
        if (b >= 0) return b;
    }
    return -1;
}

}  // namespace strata::ds4
