// src/ds4/ds4_interleave.hpp - maya-ds4 --serve --parallel: a prompt read while other requests decode.  The prompt's
// run of chunks is read in parts of whole chunks of its one-call plan (Model::feed_plan / feed_part: the same chunks,
// so the same arithmetic) - each part as many chunks as its measured time fits in DS4_PROMPT_GAP_MS - and between two
// parts the running requests decode for DS4_PROMPT_SHARE x the time the part took: a long prompt slows them down
// instead of stopping them for its whole read.  With DS4_PROMPT_YIELD a long read
// also gives way, at a part's end, to a waiting request that needs much less reading: that one is read and starts
// decoding, then the long read goes on where it stopped.  The rules alone, no model and no GPU
// (tests/ds4_interleave_test.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::ds4 {

// A read's parts as measured: a part of n chunks takes about a + b n (a: what a part costs whatever its size - a run's
// start and end -, b: a chunk), fitted from the latest part and the latest one of another size; with one size seen, b
// is its time a chunk (a part's cost in it: an estimate on the safe side)
struct PartTimes {
    int n1 = 0, n2 = 0;          // the latest part's chunks, and the latest other size's (0: none)
    double ms1 = 0.0, ms2 = 0.0;
    void add(int n, double ms) {
        if (n <= 0 || !(ms >= 0.0)) return;
        if (n1 > 0 && n != n1) n2 = n1, ms2 = ms1;
        n1 = n;
        ms1 = ms;
    }
    // the most chunks a part can take in `gap` ms (at least 1): by the fit where it has one (a chunk's time above 0),
    // else by the latest part's time a chunk - and one more than the latest when that one ended under the gap, so a
    // part's fixed cost shows (the next fit has two sizes)
    int fit(double gap) const {
        if (n1 <= 0 || !(gap > 0.0)) return 1;
        if (n2 > 0) {
            const double b = (ms1 - ms2) / (double) (n1 - n2), a = ms1 - b * (double) n1;
            if (b > 0.0) return (int) std::max(1.0, std::floor((gap - std::max(0.0, a)) / b));
        }
        int n = (int) std::max(1.0, std::floor(gap * (double) n1 / std::max(ms1, 1e-9)));
        if (ms1 < gap) n = std::max(n, n1 + 1);
        return n;
    }
};

struct ReadPace {
    double share = 0.0;   // the running requests' decode time per unit of prompt time between two parts (0: off)
    double gap_ms = 2500.0;   // beside running requests, a part's chunks: as many as fit this long (at least one)
    // a long read gives way to a much shorter waiting one (0: off); n: where nothing else decodes the read goes in
    // parts of n chunks (so it can stop at their ends), beside decoding requests in the share's parts
    int yield = 0;
    double ratio = 0.5;   // ... to a request that needs at most this share of the tokens the read has left
    int yields_max = 2;   // ... at most this often a read

    bool on() const { return share > 0.0 || yield > 0; }
    // a read in parts at all (else one call, as without the pace): beside running requests with a share, or to give
    // way when a shorter one comes
    bool parts(bool decoding) const { return (share > 0.0 && decoding) || yield > 0; }
    // the next part's chunks of the `left` still to read: beside running requests (with a share) as many as the
    // read's parts so far say fit gap_ms (the first part: one), the yield's part where none runs, else the rest in
    // one run
    int part_chunks(int left, bool decoding, const PartTimes& times = PartTimes{}) const {
        if (left <= 1) return std::max(left, 0);
        if (share > 0.0 && decoding) return std::min(left, times.fit(gap_ms));
        if (yield > 0) return std::min(left, yield);
        return left;
    }
    // the running requests' time after a part that took part_ms
    double budget_ms(double part_ms) const { return share > 0.0 ? share * std::max(0.0, part_ms) : 0.0; }
    // one more step of the running requests, `steps` done in `spent_ms` of `budget`?  At least one with a share, or
    // right after a request was let in (its first token goes out before the long read goes on); then until the
    // budget is spent
    bool decode_more(int steps, double spent_ms, double budget, bool decoding, bool admitted) const {
        if (!decoding) return false;
        if (steps == 0) return share > 0.0 || admitted;
        return share > 0.0 && spent_ms < budget;
    }
    // give way here to a waiting request that needs `need` tokens read?  The read has `left_tokens` in `left_chunks`
    // still to read and gave way `yields` times; a conversation must be free for the other, and a read that was let
    // in through a yield does not give way itself (nested)
    bool give_way(int64_t need, int64_t left_tokens, int left_chunks, int yields, bool free_seq, bool nested) const {
        return yield > 0 && !nested && free_seq && yields < yields_max && left_chunks > 1 && need > 0 &&
               (double) need <= ratio * (double) left_tokens;
    }
};

}  // namespace strata::ds4
