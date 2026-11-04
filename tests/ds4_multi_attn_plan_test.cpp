// DS4_MULTI_ATTN's layout (src/ds4/ds4_multi_attn.hpp): a multi step's runs' window buffers side by side in one buffer,
// and the sparse attention's launches over them.  Every row must see in the shared buffer the positions attn_core's
// own window buffer gives it, the runs' writes must not overlap, and the launches must cover every run once.  The
// buffers are simulated with position labels, no GPU, no model.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_multi_attn.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::ds4;

namespace {

constexpr long long kZero = -1;   // a window row before position 0 (ring_to_ext's zeros)

// the label of window row r read from the ring as ring_to_ext reads it for a window from p0: the position p0 - win + r
// (its ring row holds the newest position of its slot), zeros before position 0
long long ring_label(long long p0, int win, int r) {
    const long long p = p0 - win + r;
    return p >= 0 ? p : kZero;
}

// attn_core's window buffer for one run (p0, n rows; a drafter block: mtp) and what each of its rows attends to there:
// the labels of rows win0 .. win0 + nwin - 1, then (mtp) its block's rows through the selection
std::vector<std::vector<long long>> per_run(long long p0, int n, int win, bool mtp) {
    std::vector<long long> ext((size_t) win + n);
    for (int r = 0; r < win; ++r) ext[(size_t) r] = mtp ? 1000000 + r : ring_label(p0, win, r);   // (mtp: as they lie)
    for (int t = 0; t < n; ++t) ext[(size_t) win + t] = 2000000 + p0 + t;                         // (the KV finish's)
    std::vector<std::vector<long long>> seen((size_t) n);
    for (int t = 0; t < n; ++t) {
        int w0, nw;
        if (mtp) {
            w0 = 0;
            nw = (int) std::min<long long>(win, p0);
        } else {
            const int first = (int) std::max<long long>(win - p0, t + 1);
            w0 = first;
            nw = win + t - first + 1;
        }
        for (int j = 0; j < nw; ++j) seen[(size_t) t].push_back(ext[(size_t) w0 + j]);
        if (mtp)
            for (int i = 0; i < n; ++i) seen[(size_t) t].push_back(ext[(size_t) win + i]);   // (iota_rows: the block)
    }
    return seen;
}

// the plan's shared buffer, written as ring_to_ext_runs and finish_rows write it (each run's ring rows at e0 + r, each
// row's KV at dst), read as the shared launch reads it (win0 / nwin, then mtp: iota_rows_at's base e0 + win)
std::vector<std::vector<long long>> shared(const MaPlan& p, int win, bool mtp) {
    std::vector<long long> buf((size_t) ma_rows_max(win), -7);
    std::vector<int> owner((size_t) ma_rows_max(win), -1);
    auto put = [&](int row, long long v, int run) {
        assert(row >= 0 && row < ma_rows_max(win) && row < p.rows);
        assert(owner[(size_t) row] == -1);   // (no two writes land on one row)
        owner[(size_t) row] = run;
        buf[(size_t) row] = v;
    };
    for (int k = 0; k < p.nrun; ++k) {   // (attn_multi's RingRuns: p0 = win for a block's ring, its rows as they lie)
        const MaRun& u = p.run[k];
        for (int r = 0; r < win; ++r) put(u.e0 + r, mtp ? 1000000 + r : ring_label(u.p0, win, r), k);
    }
    for (int t = 0; t < p.T; ++t) put(p.dst[t], 2000000 + p.pos[t], p.row_run[t]);
    std::vector<std::vector<long long>> seen((size_t) p.T);
    for (int t = 0; t < p.T; ++t) {
        const MaRun& u = p.run[p.row_run[t]];
        for (int j = 0; j < p.nwin[t]; ++j) {
            const int row = p.win0[t] + j;
            assert(owner[(size_t) row] == p.row_run[t]);   // (only its own run's block)
            seen[(size_t) t].push_back(buf[(size_t) row]);
        }
        if (mtp)
            for (int i = 0; i < u.n; ++i) {
                const int row = u.e0 + win + i;
                assert(owner[(size_t) row] == p.row_run[t]);
                seen[(size_t) t].push_back(buf[(size_t) row]);
            }
    }
    return seen;
}

void check_plan(const std::vector<int>& lens, const std::vector<long long>& p0s, int win, bool mtp) {
    int rseq[kMaRows];
    int64_t rpos[kMaRows];
    int T = 0;
    for (size_t k = 0; k < lens.size(); ++k)
        for (int i = 0; i < lens[k]; ++i, ++T) {
            rseq[T] = 10 + 3 * (int) k;   // (distinct conversations, not 0 ..)
            rpos[T] = p0s[k] + i;
        }
    MaPlan p;
    assert(ma_plan(rseq, rpos, T, win, mtp, p));
    assert(p.T == T && p.nrun == (int) lens.size() && p.rows <= ma_rows_max(win));
    int rows = 0;
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        assert(u.seq == rseq[u.r0] && u.n == lens[(size_t) k] && u.p0 == p0s[(size_t) k] && u.e0 == rows);
        rows += win + u.n;
        for (int i = 0; i < u.n; ++i) {
            assert(p.row_run[u.r0 + i] == k);
            assert(p.pos[u.r0 + i] == (int) (u.p0 + i));
            assert(p.dst[u.r0 + i] == u.e0 + win + i);
        }
    }
    assert(rows == p.rows);
    const std::vector<std::vector<long long>> got = shared(p, win, mtp);
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        const std::vector<std::vector<long long>> want = per_run(u.p0, u.n, win, mtp);
        for (int i = 0; i < u.n; ++i) assert(got[(size_t) u.r0 + i] == want[(size_t) i]);
    }
}

void check_spans(const MaPlan& p, const int* cls, int max_rows) {
    MaSpan sp[kMaRows];
    const int ns = ma_spans(p, cls, max_rows, sp);
    assert(ns >= 1 && ns <= p.nrun);
    int run = 0, row = 0;
    for (int i = 0; i < ns; ++i) {
        const MaSpan& b = sp[i];
        assert(b.run0 == run && b.r0 == row && b.nrun >= 1);
        int n = 0;
        for (int k = b.run0; k < b.run0 + b.nrun; ++k) {
            n += p.run[k].n;
            if (b.cls != 0) assert(cls[k] == b.cls);
        }
        assert(n == b.n);
        if (b.cls == 0) assert(b.nrun == 1);
        else assert(b.n <= max_rows);
        if (b.cls == 0 && cls[b.run0] != 0) assert(b.n > max_rows);   // (only a run too long for the shared launch)
        if (i > 0) {   // (as few launches as the rule allows: the span before could not take this one's first run)
            const MaSpan& a = sp[i - 1];
            assert(a.cls == 0 || a.cls != cls[b.run0] || a.n + p.run[b.run0].n > max_rows);
        }
        run += b.nrun;
        row += b.n;
    }
    assert(run == p.nrun && row == p.T);
}

}  // namespace

int main() {
    const int win = 128;
    // a decode step: one row a conversation, at the start, inside the first window and well past it
    check_plan({1}, {0}, win, false);
    check_plan({1, 1, 1, 1}, {0, 5, 127, 128}, win, false);
    check_plan(std::vector<int>(16, 1), std::vector<long long>(16, 70000), win, false);
    // verify windows (spec rows), a long one, a window that crosses position 128
    check_plan({3, 3, 3, 3}, {10, 200, 125, 4096}, win, false);
    check_plan({9, 1}, {120, 7}, win, false);
    check_plan({16}, {3}, win, false);
    // drafter blocks
    check_plan({4, 4, 4}, {0, 60, 3000}, win, true);
    check_plan({5, 5, 5}, {127, 128, 129}, win, true);

    std::mt19937 rng(1234);
    for (int it = 0; it < 20000; ++it) {
        const bool mtp = rng() % 4 == 0;
        std::vector<int> lens;
        std::vector<long long> p0s;
        int T = 0;
        while (T < kMaRows) {
            const int n = 1 + (int) (rng() % (rng() % 3 == 0 ? 9 : 3));
            if (T + n > kMaRows) break;
            lens.push_back(n);
            const int kind = (int) (rng() % 3);
            p0s.push_back(kind == 0 ? (long long) (rng() % 8) : kind == 1 ? 100 + (long long) (rng() % 60)
                                                                        : (long long) (rng() % 1000000));
            T += n;
            if (rng() % 5 == 0) break;
        }
        check_plan(lens, p0s, win, mtp);
        // the spans over random classes
        int rseq[kMaRows];
        int64_t rpos[kMaRows];
        int t = 0;
        for (size_t k = 0; k < lens.size(); ++k)
            for (int i = 0; i < lens[k]; ++i, ++t) {
                rseq[t] = (int) k;
                rpos[t] = p0s[k] + i;
            }
        MaPlan p;
        assert(ma_plan(rseq, rpos, t, win, mtp, p));
        int cls[kMaRows];
        for (int k = 0; k < p.nrun; ++k) cls[k] = (int) (rng() % 3);
        check_spans(p, cls, 16);
        check_spans(p, cls, 7);
        check_spans(p, cls, 1 + (int) (rng() % 16));
    }

    // what the plan refuses: no rows, too many, a conversation whose rows are apart (two runs)
    {
        MaPlan p;
        int rseq[kMaRows + 1] = {};
        int64_t rpos[kMaRows + 1] = {};
        assert(!ma_plan(rseq, rpos, 0, win, false, p));
        assert(!ma_plan(rseq, rpos, kMaRows + 1, win, false, p));
        assert(ma_plan(rseq, rpos, kMaRows, win, false, p) && p.nrun == 1);   // (all one conversation: one run)
        const int apart[3] = {4, 5, 4};
        const int64_t pos3[3] = {10, 20, 11};
        assert(!ma_plan(apart, pos3, 3, win, false, p));
        assert(!ma_plan(apart, pos3, 1, 0, false, p));
        const int together[3] = {4, 4, 5};
        assert(ma_plan(together, pos3, 3, win, false, p) && p.nrun == 2 && p.run[0].n == 2 && p.run[1].n == 1);
        // the classes: same-class runs share a launch up to max_rows, class 0 a launch a run
        MaSpan sp[kMaRows];
        const int same[2] = {2, 2}, none[2] = {0, 0}, mixed[2] = {2, 1};
        assert(ma_spans(p, same, 16, sp) == 1 && sp[0].n == 3 && sp[0].nrun == 2 && sp[0].cls == 2);
        assert(ma_spans(p, same, 2, sp) == 2 && sp[1].r0 == 2 && sp[1].cls == 2);
        assert(ma_spans(p, same, 1, sp) == 2 && sp[0].cls == 0 && sp[1].cls == 2);   // (run 0 too long: alone)
        assert(ma_spans(p, none, 16, sp) == 2 && sp[0].cls == 0 && sp[1].cls == 0);
        assert(ma_spans(p, mixed, 16, sp) == 2);
    }
    std::printf("ds4_multi_attn_plan_test: ok\n");
    return 0;
}
