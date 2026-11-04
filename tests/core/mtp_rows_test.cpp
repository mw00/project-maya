// The host-side rules of several conversations drafting at once (STRATA_GLM_MULTI_MTP, strata/core/mtp_rows.hpp): a
// window's rows from its drafts, budget and room; a group cut to its row cap; the kept rows of a window, with stops,
// forced tokens and the last row; the head's argmax-or-sampler rule.  CPU only.
#include "strata/core/mtp_rows.hpp"

#include <cstdio>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++fails;
}
void check_v(const std::vector<int>& got, const std::vector<int>& want, const char* what) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL: %s: got", what);
    for (const int g : got) std::fprintf(stderr, " %d", g);
    std::fprintf(stderr, ", want");
    for (const int w : want) std::fprintf(stderr, " %d", w);
    std::fprintf(stderr, "\n");
    ++fails;
}
}  // namespace

int main() {
    using namespace strata::core;
    // ---- a window's rows
    check(mtp_rows_count(3, 100, 100) == 4, "3 drafts: 4 rows");
    check(mtp_rows_count(0, 100, 100) == 1, "no drafts: 1 row");
    check(mtp_rows_count(3, 2, 100) == 2, "2 tokens left: 2 rows");
    check(mtp_rows_count(3, 100, 3) == 3, "3 positions left: 3 rows");
    check(mtp_rows_count(3, 0, 0) == 1, "nothing left: still the first row");
    check(mtp_rows_count(-1, 5, 5) == 1, "a negative draft count is none");

    // ---- a group cut to its cap
    {
        std::vector<int> c = {4, 4, 4, 4};
        check(mtp_rows_fit(c, 16) == 16, "16 rows fit 16");
        check_v(c, {4, 4, 4, 4}, "nothing cut at the cap");
    }
    {
        std::vector<int> c = {4, 4, 4, 4};
        check(mtp_rows_fit(c, 12) == 12, "16 rows cut to 12");
        check_v(c, {3, 3, 3, 3}, "every window one draft shorter");
    }
    {
        std::vector<int> c = {4, 2, 4, 1};
        check(mtp_rows_fit(c, 8) == 8, "11 rows cut to 8");
        check_v(c, {2, 2, 3, 1}, "the longest first, the earlier on a tie");
    }
    {
        std::vector<int> c = {2, 2, 2};
        check(mtp_rows_fit(c, 2) == 3, "more conversations than the cap: one row each");
        check_v(c, {1, 1, 1}, "every first row kept");
    }
    {
        std::vector<int> c = {0, 3};
        check(mtp_rows_fit(c, 16) == 4, "a zero count is one row");
        check_v(c, {1, 3}, "zero raised to one");
    }

    // ---- the kept rows
    const auto plain = [](int, int32_t, int32_t&) { return MtpTake::more; };
    {
        const int32_t win[4] = {10, 11, 12, 13}, picks[4] = {11, 12, 13, 14};
        check(mtp_accept(win, picks, 4, plain) == 4, "every draft stands: all 4 rows (the last one's pick a bonus)");
    }
    {
        const int32_t win[4] = {10, 11, 99, 13}, picks[4] = {11, 12, 13, 14};
        std::vector<int> seen;
        const int keep = mtp_accept(win, picks, 4, [&](int t, int32_t p, int32_t&) {
            seen.push_back(t * 1000 + p);
            return MtpTake::more;
        });
        check(keep == 2, "the second draft misses: 2 rows");
        check_v(seen, {11, 1012}, "rows 0 and 1 written (row 1's pick replaces the wrong draft)");
    }
    {
        const int32_t win[3] = {10, 50, 51}, picks[3] = {11, 12, 13};
        check(mtp_accept(win, picks, 3, plain) == 1, "the first draft misses: 1 row");
    }
    {
        const int32_t win[1] = {10}, picks[1] = {11};
        check(mtp_accept(win, picks, 1, plain) == 1, "no drafts: 1 row");
    }
    {
        const int32_t win[4] = {10, 11, 12, 13}, picks[4] = {11, 12, 13, 14};
        const int keep = mtp_accept(win, picks, 4, [](int t, int32_t, int32_t&) {
            return t == 1 ? MtpTake::stop : MtpTake::more;
        });
        check(keep == 2, "a stop at row 1: 2 rows kept, nothing after");
    }
    {
        const int32_t win[4] = {10, 11, 12, 13}, picks[4] = {11, 12, 13, 14};
        const int keep = mtp_accept(win, picks, 4, [](int t, int32_t, int32_t&) {
            return t == 0 ? MtpTake::forced : MtpTake::more;
        });
        check(keep == 1, "a forced next token after row 0: the drafts after it are not looked at");
    }
    {
        // the written token is what the next draft is compared with, not the pick
        const int32_t win[3] = {10, 77, 12}, picks[3] = {11, 12, 13};
        const int keep = mtp_accept(win, picks, 3, [](int t, int32_t, int32_t& w) {
            if (t == 0) w = 77;
            return MtpTake::more;
        });
        check(keep == 3, "row 0 wrote 77 (not its pick 11): the draft 77 stands");
    }

    // ---- the head's rule (rows_head and the windows' head alike): argmax only for greedy without penalties
    {
        strata::kernels::SamplerParams sp;
        sp.greedy = true;
        check(rows_pick_argmax(sp), "greedy: argmax");
        sp.penalty_last_n = 64;
        check(!rows_pick_argmax(sp), "greedy with penalties: the sampler (penalized greedy)");
        sp = strata::kernels::SamplerParams();
        sp.temperature = 0.0f;
        check(rows_pick_argmax(sp), "temperature 0: argmax");
        sp.penalty_last_n = 1;
        check(!rows_pick_argmax(sp), "temperature 0 with penalties: the sampler");
        sp = strata::kernels::SamplerParams();
        sp.temperature = 0.7f;
        check(!rows_pick_argmax(sp), "sampled: the sampler");
    }
    if (fails == 0) std::printf("mtp_rows_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
