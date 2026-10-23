// prefix_cut (STRATA_GLM_PREFIX_CUT): where a prompt's read ends a call to keep its system prompt's state - the first
// user turn, only where both calls reach the prompt path's minimum and the read does not start at or near it - the
// marks it goes into (sorted, spaced, the existing ones win), and kept slots told apart by their tokens.  With the
// marks, a read covers every token once, in order, and one call ends at the system prompt's end.  CPU only.
#include "strata/core/prefix_cut.hpp"

#include <cstdio>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
constexpr int32_t kUser = 154827, kObs = 154829;
// a prompt: [gMASK]<sop>, a system prompt up to `sys`, the user turn's marker there, then text to n tokens
std::vector<int32_t> prompt(size_t sys, size_t n) {
    std::vector<int32_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = (int32_t) (1000 + i % 5000);
    if (sys > 0 && sys < n) p[sys] = kUser;
    return p;
}
// the call ends a read of [from, end) takes: one call to each mark, then one to the end (as generate.cpp's prep_prompt
// walks them)
std::vector<size_t> walk(size_t from, size_t end, const std::vector<size_t>& marks) {
    std::vector<size_t> out;
    size_t at = from;
    for (size_t mi = 0; at < end; ++mi) {
        const size_t e = mi < marks.size() ? marks[mi] : end;
        out.push_back(e);
        at = e;
    }
    return out;
}
}  // namespace

int main() {
    using namespace strata::core;
    // system_prompt_end: the first user turn at index 1 or later
    {
        const std::vector<int32_t> none = prompt(0, 100);
        expect(system_prompt_end(none.data(), none.size(), kUser) == 0, "no user turn: 0");
        std::vector<int32_t> p = prompt(40, 100);
        p[0] = kUser;
        p[70] = kUser;
        expect(system_prompt_end(p.data(), p.size(), kUser) == 40, "the first user turn past index 0");
        p[30] = kObs;
        expect(system_prompt_end(p.data(), p.size(), kUser) == 40, "a tool result is not a user turn");
    }
    // prefix_cut
    {
        const std::vector<int32_t> p = prompt(30446, 33500);
        expect(prefix_cut(p.data(), p.size(), 0, 1024, kUser) == 30446, "a check's system prompt, read from 0");
        expect(prefix_cut(p.data(), p.size(), 30446, 1024, kUser) == 0, "restored at the system prompt: no cut");
        expect(prefix_cut(p.data(), p.size(), 29935, 1024, kUser) == 0, "restored within 512 of it: no cut");
        expect(prefix_cut(p.data(), p.size(), 29934, 1024, kUser) == 30446, "restored 512 before it: cut");
        expect(prefix_cut(p.data(), p.size(), 0, 40000, kUser) == 0, "shorter than the minimum: no cut");
        const std::vector<int32_t> s = prompt(800, 5000);
        expect(prefix_cut(s.data(), s.size(), 0, 1024, kUser) == 0, "a system prompt under SLOT_MIN: no cut");
        expect(prefix_cut(s.data(), s.size(), 0, 512, kUser) == 800, "... over a lower one: cut");
        // the last token goes through the token path: the read is [from, n - 1), at least 64 after the cut
        const std::vector<int32_t> t1 = prompt(3000, 3065), t2 = prompt(3000, 3064);
        expect(prefix_cut(t1.data(), t1.size(), 0, 1024, kUser) == 3000, "64 tokens to read after it: cut");
        expect(prefix_cut(t2.data(), t2.size(), 0, 1024, kUser) == 0, "63 tokens to read after it: no cut");
        const std::vector<int32_t> none = prompt(0, 40000);
        expect(prefix_cut(none.data(), none.size(), 0, 1024, kUser) == 0, "no user turn: no cut");
        expect(prefix_cut(none.data(), 0, 0, 1024, kUser) == 0, "an empty prompt: no cut");
    }
    // add_mark
    {
        std::vector<size_t> m;
        add_mark(m, 0, 0);
        expect(m.empty(), "0 is no mark");
        add_mark(m, 30446, 0);
        expect(m == std::vector<size_t>{30446}, "into no marks");
        add_mark(m, 30446, 0);
        expect(m == std::vector<size_t>{30446}, "already there: unchanged");
        std::vector<size_t> c{8000, 30446, 46830};   // ckpt_marks' (single path): the system prompt's end is there
        add_mark(c, 30446, 0);
        expect((c == std::vector<size_t>{8000, 30446, 46830}), "a checkpoint mark there already: unchanged");
        std::vector<size_t> d{8000, 46830};
        add_mark(d, 30446, 0);
        expect((d == std::vector<size_t>{8000, 30446, 46830}), "between two marks, sorted");
        std::vector<size_t> e{30400, 46830};
        add_mark(e, 30446, 0);
        expect((e == std::vector<size_t>{30400, 46830}), "46 after a mark: dropped, the mark stays");
        std::vector<size_t> f{30500};
        add_mark(f, 30446, 0);
        expect((f == std::vector<size_t>{30500}), "54 before a mark: dropped");
        std::vector<size_t> g;
        add_mark(g, 1050, 1000);
        expect(g.empty(), "within 64 of the read's start: dropped");
    }
    // the read with the cut: every token once, in order, a call ends at the system prompt's end - alone, and among
    // checkpoint-like marks before and after it
    {
        const std::vector<int32_t> p = prompt(30446, 33500);
        const size_t end = p.size() - 1;
        for (const std::vector<size_t>& base :
             {std::vector<size_t>{}, std::vector<size_t>{16384}, std::vector<size_t>{16384, 32768}}) {
            std::vector<size_t> marks = base;
            add_mark(marks, prefix_cut(p.data(), p.size(), 0, 1024, kUser), 0);
            const std::vector<size_t> calls = walk(0, end, marks);
            bool ordered = !calls.empty() && calls.back() == end, at_cut = false;
            size_t prev = 0;
            for (const size_t c : calls) {
                ordered = ordered && c > prev;
                at_cut = at_cut || c == 30446;
                prev = c;
            }
            expect(ordered, "the calls cover the read in order");
            expect(at_cut, "a call ends at the system prompt's end");
        }
        // restored from the prefix slot: no cut, the rest in one call - the same call the cold read made after the cut
        std::vector<size_t> marks;
        add_mark(marks, prefix_cut(p.data(), p.size(), 30446, 1024, kUser), 30446);
        const std::vector<size_t> rest = walk(30446, end, marks);
        expect(rest.size() == 1 && rest[0] == end, "restored: one call for the rest");
        std::vector<size_t> cold_marks;
        add_mark(cold_marks, prefix_cut(p.data(), p.size(), 0, 1024, kUser), 0);
        const std::vector<size_t> cold = walk(0, end, cold_marks);
        expect(cold.size() == 2 && cold[0] == 30446 && cold[1] == end, "cold: the system prompt, then that same call");
    }
    // is_system_prefix
    {
        const std::vector<int32_t> sys(prompt(0, 30446));
        expect(is_system_prefix(sys, kUser), "a system prompt's state");
        expect(!is_system_prefix(prompt(30446, 33500), kUser), "a conversation (a user turn in it)");
        expect(!is_system_prefix({}, kUser), "no tokens: no prefix");
    }
    if (fails == 0) std::printf("prefix_cut_test: ok\n");
    return fails == 0 ? 0 : 1;
}
