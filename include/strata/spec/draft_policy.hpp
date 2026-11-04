// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
//
// THE COST-AWARE WINDOW (maya-ds4's DS4_DRAFT_COST=1) sizes a block drafter's window (DSpark: its drafts all at
// once, each with a confidence) the same way. Verified up to draft k, a window of k + 1 rows commits
//
//   E(k) = 1 + q_0 + q_1 + ... + q_(k-1),   q_i = a_0 a_1 ... a_i   (the chance drafts 0..i all hold)
//
// tokens on average, a_i = draft i's confidence moved towards what drafts of about that confidence at that position
// did (accept_rate). The window is the k with the best E(k) / cost(k + 1); in a step that verifies other windows
// beside it, the step's tokens in all over the step's cost by its rows in all (plan), so one request's extra rows are
// charged for the time they add to every other request in the step. Its costs are kept apart by context length
// (set_context: a row's attention costs more at 50K than at 1K); a round time far above what its size is known to
// cost (a stall, a cold read) counts as twice that at most, and as a quarter of a round; a size not measured enough,
// or not for a long stretch, is tried when its drafts are likely to hold. The lookup's calls above never touch any of it (band 0, as before).
#pragma once

#include <array>
#include <string>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 16;   // (maya-ds4's verify windows; the GLM paths pass 8 at most)
    static constexpr int kBuckets = 4;
    static constexpr int kBands = 3;   // the cost-aware window's context bands (< 8K, < 64K, longer)

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens (this band's)

    // ---- the cost-aware window (above)
    /// The next rounds' context length (the longest of a step's conversations): its band's costs from here. A band
    /// with nothing measured yet starts from the nearest one that has (each size worth one round).
    void set_context(long long pos);
    /// The chance draft i (0-based), of confidence `conf`, holds when the drafts before it did: the confidence, moved
    /// towards what drafts at that position and of about that confidence did (the confidence counts as 4 of them).
    double accept_rate(int i, double conf) const;
    /// q[i] = the chance drafts 0..i all hold, for n drafts of these confidences.
    void chain(const float* conf, int n, double* q) const;
    /// E(k): the tokens a window of k drafts commits on average (1 + q[0] + .. + q[k - 1]).
    static double tokens(const double* q, int k);
    /// One window: its rows (1 + drafts), at most `cap` rows and n + 1, for drafts whose chain is q[0..n).
    int window(const double* q, int n, int cap) const;
    /// Several windows in one step (--parallel): rows[i] = 1 + window i's drafts (at most cap[i] rows and len[i]
    /// drafts), for the most tokens a millisecond in all at this policy's cost by the step's rows; `fixed_rows` rows
    /// that commit `fixed_tokens` go beside them (requests that do not draft, or keep their own cut); at most
    /// `max_rows` in all. The drafts go in the order they are likeliest to hold, across the windows, up to the step
    /// size that pays best. q[i]: window i's chain.
    void plan(const double* const* q, const int* len, const int* cap, int n, int max_rows, int fixed_rows,
              double fixed_tokens, int* rows) const;
    /// After a round or step of the cost-aware window: its rows in all and its time (the drafting included).
    void observe_cost(int rows, double ms);
    /// After a block drafter's window: its drafts' confidences, how many were verified, how many held.
    void observe_drafts(const float* conf, int offered, int accepted);
    /// The measured cost of every size, by band: "ctx<8K: 1:41.8(12) 2:44.0(30) ..." (the log's line; empty: none).
    std::string cost_text() const;

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;
    double row_ms(int rows) const;     // the cost-aware window's cost of a size (never above a larger fresh one's)
    double sample_cap(int rows) const; // a round time counts as this at most (twice what the size is known to cost)
    bool stale(int rows) const;        // measured, but not for kStale rounds
    bool unsure(int rows) const;       // fewer than kProbes rounds, or stale

    static constexpr int kOddsPos = 8, kOddsBins = 10;   // accept_rate's table: draft position x confidence tenth

    int max_t_;
    double margin_;
    int band_ = 0;                                         // (0 unless set_context moved it)
    double rounds_ = 0;                                    // observe_cost() calls so far
    std::array<std::array<double, kMaxT + 1>, kBands> cost_{}, cost_n_{};   // round ms by band and window size
    std::array<std::array<double, kMaxT + 1>, kBands> last_{};              // the round each size was last measured
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<std::array<double, kOddsBins>, kOddsPos> held_{}, seen_{};   // block drafts held / verified, decayed
};

}  // namespace strata::spec
