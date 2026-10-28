// maya-ds4 --serve --parallel: a prompt read beside running requests (src/ds4/ds4_interleave.hpp) - its parts, the
// running requests' time between them and when a long read gives way.  A simulated clock, no GPU, no model.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_interleave.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

using strata::ds4::ReadPace;

namespace {

// A prompt of `chunks` chunks of chunk_ms each read beside one request that decodes a step every step_ms (none:
// decoding false), the server's loop as ds4_main.cpp runs it: a part (part_chunks), then the running request's steps
// (decode_more).  The parts' chunk ranges, the request's longest wait between two steps while the prompt is read, and
// when the prompt is read.
struct Sim {
    std::vector<std::pair<int, int>> parts;
    double longest = 0, read_at = 0;
    int steps = 0;
};
Sim simulate(const ReadPace& p, int chunks, double chunk_ms, bool decoding, double step_ms) {
    Sim s;
    double t = 0, last_step = 0;
    if (!p.parts(decoding)) {   // one call, as without the pace
        s.parts.push_back({0, chunks});
        t = chunks * chunk_ms;
        s.longest = decoding ? t : 0;
        s.read_at = t;
        return s;
    }
    int k = 0;
    while (k < chunks) {
        const int c = p.part_chunks(chunks - k, decoding);
        assert(c >= 1 && k + c <= chunks);
        s.parts.push_back({k, k + c});
        const double part_ms = c * chunk_ms;
        t += part_ms;
        k += c;
        if (decoding) s.longest = std::max(s.longest, t - last_step);
        if (k == chunks) break;
        const double budget = p.budget_ms(part_ms), t0 = t;
        int n = 0;
        while (p.decode_more(n, t - t0, budget, decoding, false)) {
            t += step_ms;
            last_step = t;
            ++n;
        }
        s.steps += n;
    }
    s.read_at = t;
    return s;
}

// the parts are the chunks in order, each once (the same chunks as one call)
void check_cover(const Sim& s, int chunks) {
    int k = 0;
    for (const auto& pr : s.parts) {
        assert(pr.first == k && pr.second > pr.first);
        k = pr.second;
    }
    assert(k == chunks);
}

}  // namespace

int main() {
    // off (the default): one call, nothing between, nobody gives way - as before
    {
        const ReadPace p;
        assert(!p.on());
        assert(!p.parts(false) && !p.parts(true));
        assert(p.part_chunks(12, true) == 12 && p.part_chunks(12, false) == 12);
        assert(p.budget_ms(5000) == 0);
        assert(!p.decode_more(0, 0, 0, true, false));
        assert(!p.give_way(10, 100000, 20, 0, true, false));
        const Sim s = simulate(p, 16, 4000, true, 50);
        assert(s.parts.size() == 1 && s.longest == 64000 && s.steps == 0);
    }
    // a share: parts of one chunk beside a running request, the rest in one run with none running (alone: as before)
    {
        ReadPace p;
        p.share = 0.5;
        assert(p.on() && p.parts(true) && !p.parts(false));
        assert(p.part_chunks(12, true) == 1 && p.part_chunks(12, false) == 12);
        assert(p.part_chunks(1, true) == 1 && p.part_chunks(0, true) == 0);
        assert(p.budget_ms(4000) == 2000 && p.budget_ms(-5) == 0);
        // at least one step, then until the budget is spent; nobody running: none
        assert(p.decode_more(0, 0, 0, true, false));
        assert(p.decode_more(3, 1999, 2000, true, false) && !p.decode_more(4, 2000, 2000, true, false));
        assert(!p.decode_more(0, 0, 2000, false, false));
        const Sim alone = simulate(p, 16, 4000, false, 50);
        assert(alone.parts.size() == 1 && alone.read_at == 64000);
    }
    // the running request's longest wait: the whole prompt without a share, a chunk with one; the prompt pays the
    // share of its time (all but the last part's)
    {
        ReadPace off, half, two;
        half.share = 0.5;
        two.share = 2.0;
        const Sim a = simulate(off, 16, 4000, true, 50), b = simulate(half, 16, 4000, true, 50),
                  c = simulate(two, 16, 4000, true, 50);
        check_cover(a, 16);
        check_cover(b, 16);
        check_cover(c, 16);
        assert(b.parts.size() == 16 && c.parts.size() == 16);
        assert(a.longest == 64000 && b.longest == 4000 && c.longest == 4000);
        assert(b.read_at == 64000 + 15 * 2000 && c.read_at == 64000 + 15 * 8000);
        assert(b.steps == 15 * 40 && c.steps == 15 * 160);
        // a step longer than the budget: still one a part
        const Sim d = simulate(half, 4, 100, true, 500);
        assert(d.steps == 3 && d.longest == 100);
        std::printf("interleave: a 16-chunk prompt beside a decode: the longest wait %.0f / %.0f / %.0f ms at share "
                    "0 / 0.5 / 2, the prompt read at %.0f / %.0f / %.0f ms\n",
                    a.longest, b.longest, c.longest, a.read_at, b.read_at, c.read_at);
    }
    // the yield's parts: n chunks where nothing runs, one at a time beside a running request with a share
    {
        ReadPace p;
        p.yield = 4;
        assert(p.on() && p.parts(false) && p.parts(true));
        assert(p.part_chunks(10, false) == 4 && p.part_chunks(3, false) == 3 && p.part_chunks(10, true) == 4);
        const Sim s = simulate(p, 10, 1000, false, 50);
        check_cover(s, 10);
        assert(s.parts.size() == 3 && s.parts[2].second - s.parts[2].first == 2);
        // no share: nothing decodes between the parts - except right after a request was let in (its first token)
        assert(!p.decode_more(0, 0, 0, true, false));
        assert(p.decode_more(0, 0, 0, true, true) && !p.decode_more(1, 0, 0, true, true));
        p.share = 1.0;
        assert(p.part_chunks(10, true) == 1 && p.part_chunks(10, false) == 4);
    }
    // giving way: to a request that needs at most half of what the read has left, with a conversation free, at most
    // twice a read, not with one chunk left, never from a read that was itself let in
    {
        ReadPace p;
        p.yield = 2;
        assert(p.give_way(1000, 100000, 20, 0, true, false));
        assert(p.give_way(50000, 100000, 20, 0, true, false));
        assert(!p.give_way(50001, 100000, 20, 0, true, false));
        assert(!p.give_way(1000, 100000, 20, 0, false, false));   // (no conversation free)
        assert(!p.give_way(1000, 100000, 20, 0, true, true));     // (nested)
        assert(p.give_way(1000, 100000, 20, 1, true, false) && !p.give_way(1000, 100000, 20, 2, true, false));
        assert(!p.give_way(1000, 100000, 1, 0, true, false));     // (one chunk left: it ends sooner than it would wait)
        assert(!p.give_way(0, 100000, 20, 0, true, false));       // (nothing to read: no request)
        ReadPace q = p;
        q.yield = 0;
        assert(!q.give_way(1000, 100000, 20, 0, true, false));
        q = p;
        q.ratio = 0.25;
        assert(q.give_way(25000, 100000, 20, 0, true, false) && !q.give_way(25001, 100000, 20, 0, true, false));
    }
    // a 185K read in 8K chunks, nothing else running, a short question 3 s in: it is let in at the first part's end
    // (yield 2: two chunks), the long read goes on from chunk 2 and every chunk is read once, in order
    {
        ReadPace p;
        p.yield = 2;
        const int chunks = 23;
        const double chunk_ms = 8800, arrive = 3000;
        const int64_t total = 185000, per = total / chunks;
        std::vector<std::pair<int, int>> parts;
        double t = 0, let_in = -1;
        int k = 0, yields = 0;
        while (k < chunks) {
            const int c = p.part_chunks(chunks - k, false);
            parts.push_back({k, k + c});
            t += c * chunk_ms;
            k += c;
            if (k < chunks && t >= arrive && let_in < 0 &&
                p.give_way(500, total - k * per, chunks - k, yields, true, false)) {
                let_in = t;
                ++yields;
            }
        }
        assert(let_in == 2 * chunk_ms && yields == 1);
        int at = 0;
        for (const auto& pr : parts) {
            assert(pr.first == at);
            at = pr.second;
        }
        assert(at == chunks);
    }
    std::printf("ds4_interleave_test: all checks passed\n");
    return 0;
}
