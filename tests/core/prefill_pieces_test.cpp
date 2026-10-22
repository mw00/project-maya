// prefill_pieces (STRATA_GLM_PREFILL_INTERLEAVE) and prefill_gaps (STRATA_GLM_PREFILL_GAPS): the pieces a prompt is
// read in - every token once, in order, no piece but a whole segment shorter than the minimum - and the decode gaps of
// a pipelined read - every part idle in a gap, every chunk read once in order by every part, no deadlock; prefill_chunks
// (STRATA_GLM_PREFILL_MIN_CHUNKS): the chunks a call is cut into - as before when off, the floor's count when on.  CPU
// only.
#include "strata/core/prefill_chunks.hpp"
#include "strata/core/prefill_gaps.hpp"
#include "strata/core/prefill_pieces.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
int fails = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
// the call ends a prompt [from, end) is read in, marks kept (as generate.cpp's prep_prompt walks them)
std::vector<size_t> walk(size_t from, size_t end, const std::vector<size_t>& marks, size_t piece) {
    std::vector<size_t> out;
    size_t at = from, mi = 0;
    while (at < end) {
        const size_t seg = mi < marks.size() ? marks[mi] : end;
        const size_t e = strata::core::prefill_piece_end(at, seg, piece);
        out.push_back(e);
        if (e == seg && mi < marks.size()) ++mi;
        at = e;
    }
    return out;
}
// a pipelined read of nc chunks over np parts with prefill_run's protocol (hand-off through two slots, the gaps'
// gate): true when every part read every chunk in order and no part was at work during a gap; gaps: the boundaries
// that get one (a bit per boundary, all when -1)
bool pipeline(int np, int nc, long long gaps, int* n_gaps) {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int> produced((size_t) np, 0), consumed((size_t) np, 0);
    strata::core::PipelineGaps G(np, nc);
    std::atomic<int> busy{0};
    std::atomic<bool> bad{false};
    std::vector<std::vector<int>> order((size_t) np);
    const auto work = [&](int k, int c) {
        busy.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::microseconds(200 + 150 * ((k + c) % 3)));
        order[(size_t) k].push_back(c);
        busy.fetch_sub(1);
    };
    const auto hand_on = [&](int k, int c) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return consumed[(size_t) k] >= c - 1; });
        produced[(size_t) k] = c + 1;
        cv.notify_all();
    };
    std::vector<std::thread> later;
    for (int k = 1; k < np; ++k)
        later.emplace_back([&, k] {
            for (int c = 0; c < nc; ++c) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return produced[(size_t) k - 1] > c && G.may_start(k, c); });
                    consumed[(size_t) k - 1] = c + 1;
                    cv.notify_all();
                }
                work(k, c);
                if (k + 1 < np) hand_on(k, c);
                std::lock_guard<std::mutex> lk(mu);
                G.finished[(size_t) k] = c + 1;
                cv.notify_all();
            }
        });
    *n_gaps = 0;
    for (int c = 0; c < nc; ++c) {
        work(0, c);
        hand_on(0, c);
        if (c + 1 < nc) {
            if (gaps < 0 || ((gaps >> c) & 1)) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return G.all_idle(c); });
                }
                // the gap: nobody at work, and nobody starts meanwhile
                for (int i = 0; i < 3; ++i) {
                    if (busy.load() != 0) bad = true;
                    std::this_thread::sleep_for(std::chrono::microseconds(300));
                }
                ++*n_gaps;
            }
            std::lock_guard<std::mutex> lk(mu);
            G.released = c;
            cv.notify_all();
        }
    }
    for (auto& t : later) t.join();
    bool ok = !bad.load();
    for (int k = 0; k < np; ++k) {
        ok = ok && (int) order[(size_t) k].size() == nc;
        for (int c = 0; ok && c < nc; ++c) ok = order[(size_t) k][(size_t) c] == c;
    }
    return ok;
}
}  // namespace

int main() {
    using strata::core::prefill_piece_end;
    // off: the whole segment
    expect(prefill_piece_end(0, 10000, 0) == 10000, "off: one call");
    expect(prefill_piece_end(100, 1000, 2048) == 1000, "shorter than a piece: one call");
    // pieces
    expect(prefill_piece_end(0, 10000, 2048) == 2048, "first piece");
    expect(prefill_piece_end(8192, 10000, 2048) == 10000, "last piece takes the rest (1808)");
    expect(prefill_piece_end(0, 2100, 2048) == 2100, "a 52-token rest goes with the piece");
    expect(prefill_piece_end(0, 2112, 2048) == 2048, "a 64-token rest is its own piece");
    // a walk: every call ends past the last one, the marks are call ends, the last end is the prompt's
    {
        const std::vector<size_t> marks = {3000, 20000};
        const auto ends = walk(17, 30001, marks, 2048);
        bool ok = !ends.empty() && ends.back() == 30001;
        size_t prev = 17;
        for (const size_t e : ends) {
            ok = ok && e > prev && (e - prev >= 64 || e == 30001);
            prev = e;
        }
        for (const size_t m : marks) {
            bool hit = false;
            for (const size_t e : ends) hit = hit || e == m;
            ok = ok && hit;
        }
        expect(ok, "walk with marks: ordered, marks kept, pieces >= 64");
        const auto off = walk(17, 30001, marks, 0);
        expect(off == std::vector<size_t>({3000, 20000, 30001}), "walk off: the marks and the end, as before");
    }
    // the setting
    using strata::core::prefill_interleave_parse;
    expect(prefill_interleave_parse(nullptr).tokens == 0, "unset: off");
    expect(prefill_interleave_parse("0").tokens == 0, "0: off");
    expect(prefill_interleave_parse("2048").tokens == 2048 && prefill_interleave_parse("2048").rounds == 1,
           "2048: one round");
    expect(prefill_interleave_parse("1024,4").rounds == 4, "1024,4: four rounds");
    expect(prefill_interleave_parse("16").tokens == 256, "a piece is 256 at least");
    expect(prefill_interleave_parse("2048").exact && !prefill_interleave_parse("2048,1,cut").exact, "exact unless ,cut");
    using strata::core::prefill_aligned_piece;
    expect(prefill_aligned_piece(2048, 8128) == 8128, "exact: one whole chunk at least");
    expect(prefill_aligned_piece(16384, 8128) == 16256, "exact: two chunks");
    expect(prefill_aligned_piece(2048, 0) == 2048, "no chunk: as asked");
    // the decode gaps' setting
    using strata::core::prefill_gaps_parse;
    expect(prefill_gaps_parse(nullptr).percent == 0 && prefill_gaps_parse("0").percent == 0, "gaps unset / 0: off");
    expect(prefill_gaps_parse("25").percent == 25 && prefill_gaps_parse("25").max_ms == 4000.0, "gaps 25");
    expect(prefill_gaps_parse("95").percent == 90, "gaps: 90% at most");
    expect(prefill_gaps_parse("20,1500").max_ms == 1500.0, "gaps: the longest gap");
    using strata::core::prefill_gap_budget_ms;
    expect(prefill_gap_budget_ms(prefill_gaps_parse("20"), 2800.0) == 700.0, "a 20% gap after a 2.8 s stage: 0.7 s");
    expect(prefill_gap_budget_ms(prefill_gaps_parse("50,1000"), 2800.0) == 1000.0, "the cap");
    expect(prefill_gap_budget_ms(prefill_gaps_parse("0"), 2800.0) == 0.0, "off: none");
    // the gaps' gate: part k starts chunk c after boundary c + k - 1, none past the last
    {
        strata::core::PipelineGaps G(2, 4);
        expect(G.may_start(1, 0) == false && G.all_idle(0), "boundary 0: part 1 waits, nothing to wait for");
        G.released = 0;
        expect(G.may_start(1, 0) && !G.may_start(1, 1) && !G.all_idle(1), "after boundary 0: chunk 0 only");
        G.finished[1] = 1;
        expect(G.all_idle(1), "boundary 1: part 1 has read chunk 0");
        expect(G.may_start(1, 3), "the last chunk: no boundary after part 0's last");
    }
    {
        int n = 0;
        bool ok = true;
        for (int np = 2; np <= 3; ++np)
            for (int nc = 1; nc <= 7; ++nc) {
                ok = ok && pipeline(np, nc, -1, &n) && n == (nc > 0 ? nc - 1 : 0);
                ok = ok && pipeline(np, nc, 0, &n) && n == 0;
                ok = ok && pipeline(np, nc, 0x5, &n);
            }
        expect(ok, "pipelined reads with gaps: every chunk in order, every part idle in a gap");
    }
    // the chunks a call is cut into (STRATA_GLM_PREFILL_MIN_CHUNKS)
    {
        using namespace strata::core;
        expect(prefill_min_chunks_parse(nullptr).n == 1 && prefill_min_chunks_parse("1").n == 1 &&
                   prefill_min_chunks_parse("0").n == 1,
               "min chunks unset / 0 / 1: off");
        expect(prefill_min_chunks_parse("2").n == 2 && prefill_min_chunks_parse("2").min_tokens == 512, "2: 512 at least");
        expect(prefill_min_chunks_parse("3,1024").n == 3 && prefill_min_chunks_parse("3,1024").min_tokens == 1024,
               "3,1024");
        expect(prefill_min_chunks_parse("40,10").n == 8 && prefill_min_chunks_parse("40,10").min_tokens == 256,
               "min chunks: 8 at most, 256 tokens at least");
        // off: what prefill() took before (one GPU: full chunks; a split: equal ones, prefill_min_chunks of 256+)
        const auto before = [](int64_t nn, int tmax, bool split, int call_min) {
            const int64_t k = std::max<int64_t>((nn + tmax - 1) / tmax,
                                                split ? std::min<int64_t>(call_min, nn / 256) : 1);
            return !split ? (int) std::min<int64_t>(tmax, (nn + 63) / 64 * 64)
                          : (int) std::min<int64_t>(tmax, ((nn + k - 1) / k + 63) / 64 * 64);
        };
        bool same = true, exact = true, sane = true;
        for (const int tmax : {8192, 7616, 4096})
            for (int64_t nn = 1; nn < 40000; nn += 13) {
                for (const bool split : {false, true})
                    for (const int cm : {1, 2, 3})
                        same = same && prefill_chunk_len(nn, tmax, split, split ? prefill_floor_chunks(nn, cm, 256) : 1) ==
                                           before(nn, tmax, split, cm);
                for (const int n : {2, 3, 4, 6, 8})
                    for (const int mt : {256, 512, 1024, 2048}) {
                        const int64_t fl = prefill_floor_chunks(nn, n, mt);
                        const int len = prefill_chunk_len(nn, tmax, true, fl);
                        const int64_t c = prefill_chunk_count(nn, len), want = std::max<int64_t>((nn + tmax - 1) / tmax, fl);
                        sane = sane && len > 0 && len % 64 == 0 && len <= tmax && c >= 1 && c <= want &&
                               (fl == 1 || (nn - (c - 1) * (int64_t) len) > 0) && (fl <= 1 || nn / fl >= mt);
                        if (n <= 4) exact = exact && c == want;
                    }
            }
        expect(same, "min chunks off: the chunks as before");
        expect(exact, "min chunks 2..4: exactly that many chunks (or what the largest chunk needs)");
        expect(sane, "min chunks: 64-token multiples, no chunk past the largest, none under the minimum");
        // the cases the setting is for: a turn on a restored conversation, one chunk before
        const auto cut = [](int64_t nn, int n, int mt) {
            const int len = prefill_chunk_len(nn, 8192, true, prefill_floor_chunks(nn, n, mt));
            return prefill_chunk_count(nn, len);
        };
        expect(cut(1000, 2, 512) == 1 && cut(1024, 2, 512) == 2 && cut(4000, 4, 512) == 4 && cut(8000, 3, 512) == 3,
               "min chunks: 1000 tokens stay whole at 512, 1024 in two, 4000 in four, 8000 in three");
        expect(cut(20000, 2, 512) == 3 && cut(20000, 4, 512) == 4, "min chunks: a long read keeps its own count");
        expect(cut(1500, 4, 512) == 2 && cut(1500, 4, 256) == 4, "min chunks: as many as the minimum lets");
        // a boundary's gap is due once its budget is half a round
        const PrefillGaps g = prefill_gaps_parse("20");
        expect(prefill_gap_due(g, 100.0, 0.0), "gap due: no round known yet");
        expect(prefill_gap_due(g, 400.0, 200.0) && !prefill_gap_due(g, 300.0, 200.0),
               "gap due: a 20% budget of a 400 ms stage is half a 200 ms round, 300 ms is not");
        expect(!prefill_gap_due(prefill_gaps_parse("0"), 1e6, 0.0), "gap due: never with gaps off");
    }
    if (fails == 0) std::printf("prefill_pieces_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
