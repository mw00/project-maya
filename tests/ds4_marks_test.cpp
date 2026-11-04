// maya-ds4's restore points (src/ds4/ds4_marks.hpp): the one a new prompt goes back to, the ones a request's read
// keeps, the one that gives way to a new one - and, put together as the serve loop puts them (prompt_in: the restore,
// then the read in pieces), the cases they are for: the last user message edited, a new question after the same
// document, a history rendered again.  No GPU, no model: a conversation here is its tokens and its points' positions.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_marks.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using strata::ds4::latest_point;
using strata::ds4::mark_plan;
using strata::ds4::mark_victim;
using strata::ds4::MarkAt;
using strata::ds4::MarkUse;
using strata::ds4::TailCost;

namespace {

// A conversation as the engine keeps it, positions only: ck[0] the end of the last prompt, ck[1] the position before
// it, ck[2] the anchor, then the marks - and what the serve loop does with a prompt (prompt_in): go back to the latest
// point the prompt still agrees with, read the rest in pieces that stop at the marks the plan keeps.
struct Conv {
    std::vector<int> toks;
    int64_t ck[3] = {-1, -1, -1};
    std::vector<MarkUse> mk;
    uint64_t clock = 0;
    int cap = 4;
    std::vector<int64_t> pieces;   // the last read's pieces (their ends)

    int64_t pos() const { return (int64_t) toks.size(); }
    std::vector<int64_t> points() const {
        std::vector<int64_t> v(ck, ck + 3);
        for (const MarkUse& m : mk) v.push_back(m.pos);
        return v;
    }
    void mark(bool shared) {
        for (MarkUse& m : mk)
            if (m.pos == pos()) {
                m.shared = m.shared || shared;
                m.used = ++clock;
                return;
            }
        if ((int) mk.size() >= cap) mk.erase(mk.begin() + mark_victim(mk));
        mk.push_back({pos(), shared, ++clock});
    }
    bool restore(int64_t limit) {
        const std::vector<int64_t> at = points();
        const int b = latest_point(at, limit, pos());
        if (b < 0) return false;
        const int64_t p = at[(size_t) b];
        toks.resize((size_t) p);
        for (int64_t& c : ck)
            if (c > p) c = -1;
        for (size_t i = mk.size(); i-- > 0;)
            if (mk[i].pos > p) mk.erase(mk.begin() + (long) i);
        for (MarkUse& m : mk)
            if (m.pos == p) m.used = ++clock;
        return true;
    }
    // a request's prompt -> the tokens it reused (prompt_in's `from`)
    int64_t prompt(const std::vector<int>& ids, const std::vector<int64_t>& marks = {}, int64_t shared = -1) {
        size_t c = 0;
        while (c < toks.size() && c < ids.size() && toks[c] == ids[c]) ++c;
        int64_t from = 0;
        bool continued = false;
        if (!toks.empty() && c == toks.size() && c < ids.size()) {
            from = pos();
            continued = true;
        } else if (restore((int64_t) std::min(c, ids.size() - 1)))
            from = pos();
        else {
            toks.clear();
            ck[0] = ck[1] = ck[2] = -1;
            mk.clear();
        }
        const int64_t total = (int64_t) ids.size();
        const std::vector<MarkAt> cuts = mark_plan(marks, shared, from, total, cap, 16);
        pieces.clear();
        size_t k = 0;
        if (!cuts.empty() && cuts[0].pos == from) mark(cuts[k++].shared);
        int64_t done = from;
        while (done < total) {
            int64_t n = total - done;
            if (k < cuts.size() && cuts[k].pos < done + n) n = cuts[k].pos - done;
            toks.insert(toks.end(), ids.begin() + done, ids.begin() + done + n);
            if (done + n == total) ck[1] = total - 1;   // (feed_mark: the state before the last token)
            done += n;
            pieces.push_back(done);
            if (k < cuts.size() && done == cuts[k].pos) mark(cuts[k++].shared);
        }
        ck[0] = total;
        if (!continued && ck[1] > 0) ck[2] = ck[1];
        return from;
    }
    // the answer's tokens, decoded after the prompt
    void answer(const std::vector<int>& a) { toks.insert(toks.end(), a.begin(), a.end()); }
};

std::vector<int> run(int start, int n) {
    std::vector<int> v;
    for (int i = 0; i < n; ++i) v.push_back(start + i);
    return v;
}
std::vector<int> cat(std::vector<int> a, const std::vector<int>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

}  // namespace

int main() {
    // latest_point: the latest at or before the limit and the state's position; -1 entries and later ones never
    assert(latest_point({}, 100, 100) == -1);
    assert(latest_point({-1, -1, -1}, 100, 100) == -1);
    assert(latest_point({50, 49, 10}, 100, 100) == 0);
    assert(latest_point({50, 49, 10}, 49, 100) == 1);
    assert(latest_point({50, 49, 10}, 48, 100) == 2);
    assert(latest_point({50, 49, 10}, 9, 100) == -1);
    assert(latest_point({50, 49, 10, 30}, 45, 100) == 3);   // (a mark between the anchor and the end)
    assert(latest_point({50, 49, 10, 30}, 100, 40) == 3);   // (none past the state's own position)
    assert(latest_point({30, -1, 30, 30}, 100, 100) == 0);  // (equal ones: the first)
    assert(latest_point({0, -1, -1}, 0, 5) == 0);           // (a point at 0 counts, as checkpoint 1 of a 1-token prompt)

    // mark_plan: the asked points the read [from, total) reaches, in order
    {
        const auto v = mark_plan({300, 100, 200}, -1, 0, 1000, 4, 16);
        assert(v.size() == 3 && v[0].pos == 100 && v[1].pos == 200 && v[2].pos == 300);
        assert(!v[0].shared && !v[1].shared && !v[2].shared);
    }
    {   // before `from`, at or past the last token: not this read's (checkpoint 1 keeps the position before the end)
        const auto v = mark_plan({100, 500, 999, 1000, 1200}, -1, 400, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 500);
        assert(mark_plan({998}, -1, 0, 1000, 4, 16).size() == 1);
        assert(mark_plan({500}, -1, 999, 1000, 4, 16).empty());   // (one token to read: no room for a point)
    }
    {   // at `from`: kept as the state is; fewer than `near` past it: at `from` too (no read of a few tokens alone)
        auto v = mark_plan({400}, -1, 400, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 400);
        v = mark_plan({410}, -1, 400, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 400);
        v = mark_plan({416}, -1, 400, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 416);
    }
    {   // two close together: the first, with the shared role of either; the same position twice: once
        auto v = mark_plan({505}, 500, 0, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 500 && v[0].shared);
        v = mark_plan({500}, 505, 0, 1000, 4, 16);
        assert(v.size() == 1 && v[0].pos == 500 && v[0].shared);
        v = mark_plan({500, 500, 500}, -1, 0, 1000, 4, 0);
        assert(v.size() == 1 && v[0].pos == 500);
    }
    {   // over the cap: the shared start first, then the latest
        const auto v = mark_plan({100, 200, 300, 400}, 50, 0, 1000, 2, 16);
        assert(v.size() == 2 && v[0].pos == 50 && v[0].shared && v[1].pos == 400);
        assert(mark_plan({100, 200}, 50, 0, 1000, 0, 16).empty());   // (cap 0: none)
    }
    {   // a picture's span is read whole: a point inside it moves to its start (and then may be before `from`)
        auto v = mark_plan({150, 300}, -1, 0, 1000, 4, 16, {{100, 100}});
        assert(v.size() == 2 && v[0].pos == 100 && v[1].pos == 300);
        v = mark_plan({150}, -1, 120, 1000, 4, 16, {{100, 100}});
        assert(v.empty());
        v = mark_plan({100, 200}, -1, 0, 1000, 4, 16, {{100, 100}});   // (its start and its end are fine)
        assert(v.size() == 2 && v[0].pos == 100 && v[1].pos == 200);
    }

    // mark_victim: the least recently used that is not a shared start, else the least recently used
    assert(mark_victim({}) == -1);
    assert(mark_victim({{10, false, 5}, {20, false, 3}, {30, false, 9}}) == 1);
    assert(mark_victim({{10, true, 1}, {20, false, 3}, {30, false, 2}}) == 2);
    assert(mark_victim({{10, true, 4}, {20, true, 3}}) == 1);

    // ---- the cases, as the serve loop runs them
    const std::vector<int> sys = run(1000, 3000);   // a system prompt and its tools: 3,000 tokens
    {   // the last user message edited: back to its start (the boundary), not to token 0
        Conv cv;
        const std::vector<int> hist = cat(sys, run(5000, 6000));   // the conversation so far
        const std::vector<int> p1 = cat(hist, run(20000, 200));     // ... and the last user message
        assert(cv.prompt(p1, {(int64_t) hist.size()}, (int64_t) sys.size()) == 0);
        assert((cv.pieces == std::vector<int64_t>{3000, 9000, 9200}));   // (the read stopped at both points)
        cv.answer(run(30000, 500));
        const std::vector<int> p2 = cat(hist, run(21000, 180));   // the same history, the last message edited
        assert(cv.prompt(p2, {(int64_t) hist.size()}, (int64_t) sys.size()) == (int64_t) hist.size());
        assert((cv.pieces == std::vector<int64_t>{(int64_t) p2.size()}));   // (its point at `from`: no split)
        // without the points the same edit read from token 0 (the end, the position before it and the anchor are
        // all past where it differs)
        Conv old;
        old.prompt(p1);
        old.answer(run(30000, 500));
        assert(old.prompt(p2) == 0);
    }
    {   // a new question after the same long document, in one message: back to the point near the tail
        Conv cv;
        const std::vector<int> doc = cat(sys, run(40000, 64000));
        const std::vector<int> q1 = cat(doc, run(200000, 100)), q2 = cat(doc, run(210000, 120));
        const int64_t tail = 512;
        assert(cv.prompt(q1, {(int64_t) q1.size() - tail}, 3000) == 0);
        cv.answer(run(30000, 400));
        assert(cv.prompt(q2, {(int64_t) q2.size() - tail}, 3000) == (int64_t) q1.size() - tail);   // (412 re-read)
        cv.answer(run(31000, 400));
        // a third question: back to the second's tail point, or the first's - the latest one before its question
        const std::vector<int> q3 = cat(doc, run(220000, 90));
        const int64_t back = cv.prompt(q3, {(int64_t) q3.size() - tail}, 3000);
        assert(back >= (int64_t) doc.size() - tail && back <= (int64_t) doc.size());
    }
    {   // another agent behind the same system prompt and tools: back to the shared start
        Conv cv;
        const std::vector<int> a = cat(sys, run(50000, 8000)), b = cat(sys, run(60000, 9000));
        cv.prompt(a, {(int64_t) a.size() - 200}, 3000);
        cv.answer(run(30000, 300));
        assert(cv.prompt(b, {(int64_t) b.size() - 200}, 3000) == 3000);
        // ... and the shared start outlives the other points: many turns later it is still there
        std::vector<int> conv = b;
        for (int t = 0; t < 10; ++t) {
            cv.answer(run(70000 + 1000 * t, 300));
            conv = cat(cat(conv, run(70000 + 1000 * t, 300)), run(90000 + 1000 * t, 50));
            cv.prompt(conv, {(int64_t) conv.size() - 50}, 3000);
        }
        assert((int) cv.mk.size() <= cv.cap);
        bool has_shared = false;
        for (const MarkUse& m : cv.mk) has_shared = has_shared || (m.pos == 3000 && m.shared);
        assert(has_shared);
        assert(cv.prompt(cat(sys, run(80000, 100))) == 3000);
    }
    {   // a turn that continues in place: its boundary a token past the state's end is kept at the end (no split)
        Conv cv;
        const std::vector<int> p1 = cat(sys, run(5000, 2000));
        cv.prompt(p1, {3000}, 3000);
        cv.answer(run(30000, 300));
        std::vector<int> p2 = cat(p1, run(30000, 300));
        const int64_t boundary = (int64_t) p2.size() + 1;   // (the end of the answer, then the user's turn)
        p2 = cat(p2, run(40000, 100));
        assert(cv.prompt(p2, {boundary}, 3000) == (int64_t) p1.size() + 300);
        assert((cv.pieces == std::vector<int64_t>{(int64_t) p2.size()}));
        bool at_end = false;
        for (const MarkUse& m : cv.mk) at_end = at_end || m.pos == (int64_t) p1.size() + 300;
        assert(at_end);
    }
    {   // a restore drops the points past it (they saw tokens the state no longer has); the cap holds
        Conv cv;
        cv.cap = 2;
        const std::vector<int> p = cat(sys, run(5000, 9000));
        cv.prompt(p, {5000, 8000, 11000}, 3000);
        assert(cv.mk.size() == 2);   // (the cap: the shared start and the latest)
        assert(cv.mk[0].pos == 3000 && cv.mk[1].pos == 11000);
        cv.answer(run(30000, 100));
        cv.prompt(cat(sys, run(7000, 50)));   // (differs right after the shared start)
        for (const MarkUse& m : cv.mk) assert(m.pos <= 3000);
        for (int64_t c : cv.ck) assert(c < 0 || c <= cv.pos());
    }
    {   // the tail point's cost (#668): kept until measured; left out where its run's wall time a token is more than
        // twice the long read's (most experts on the CPU lane, or streamed again through a split's stages); probed again
        TailCost tc;
        assert(tc.keep() && !tc.measured());
        assert(!tc.add(2000.0, 2000, 50.0, 512));        // (too short before the point to be a long run)
        assert(!tc.add(0.0, 19000, 600.0, 512));         // (nothing timed)
        assert(!tc.add(19000.0, 19000, 600.0, 0));       // (no tail run)
        assert(tc.add(19000.0, 19000, 640.0, 512));      // 1.25x: a GPU that holds the experts
        assert(tc.keep() && !tc.costly());
        assert(tc.add(19000.0, 19000, 768.0, 512));      // 1.5x
        assert(tc.measured() && !tc.costly());
        assert(tc.last_tail_tokens == 512 && tc.last_tail_ms == 768.0 && tc.last_main_ms_token == 1.0);
        for (int i = 0; i < 100; ++i) assert(tc.keep());
        TailCost cpu;
        assert(cpu.add(18000.0, 19000, 4000.0, 512));    // 8.2x: the CPU lane's pass
        assert(cpu.keep());                              // (one read: not decided yet)
        assert(cpu.add(18000.0, 19000, 3800.0, 512));
        assert(cpu.costly());
        int kept = 0;
        for (int i = 0; i < 64; ++i) kept += cpu.keep(32);
        assert(kept == 2);                               // (one in 32: measured again)
        // the PC changed (more experts in VRAM): the probes' cheap runs bring it back
        assert(cpu.add(19000.0, 19000, 600.0, 512) && cpu.add(19000.0, 19000, 600.0, 512));
        assert(cpu.add(19000.0, 19000, 600.0, 512) && cpu.costly());   // (a running mean: 8x -> 2.03x)
        assert(cpu.add(19000.0, 19000, 600.0, 512) && !cpu.costly() && cpu.keep());
        // #668's 4-GPU layer split, experts streamed: the tail run a near-constant ~2.5 s, 4.2x and 6.3x a token (the
        // time a pick over the 4 stages read ~1x, and kept it)
        TailCost split;
        assert(split.add(7768.0, 6899, 2456.0, 512) && split.add(11205.0, 14328, 2521.0, 512));
        assert(split.costly() && split.ratio > 4.0);
        // a split whose tail run costs a little more a token than the long read (its stages one after another): kept
        TailCost cheap;
        assert(cheap.add(10000.0, 20000, 450.0, 512) && cheap.add(10000.0, 20000, 460.0, 512));
        assert(cheap.measured() && !cheap.costly());
    }
    std::printf("ds4_marks_test: ok\n");
    return 0;
}
