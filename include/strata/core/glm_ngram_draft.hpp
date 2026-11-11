// include/strata/core/glm_ngram_draft.hpp - glm-dsa B14.13: n-gram (prompt-lookup / suffix-match) speculative drafts for
// the glm-dsa pipelined speculative decode, with the NextN (MTP) block's draft as the fallback.
//
// MAYA_GLM_NGRAM_DRAFT=1 (default off).  When the conversation's current suffix (the prompt plus every confirmed
// token) occurred earlier with at least MAYA_GLM_NGRAM_MIN tokens of agreement (default 3), the tokens that
// followed that earlier occurrence are the draft; otherwise the MTP draft stands.  Edits, refactors, quoted code and
// tool-call arguments repeat their input, so the earlier continuation is often exact (ik_llama.cpp measured 57 to 76%
// acceptance with its ngram-mod drafts, docs/speculative.md; BUILD-PLAN B14.13).
//
// How it joins Maya's decode_spec (src/core/glm_fast_path.cu, a depth-1 pipeline: the head part runs position q+2 on
// a draft while the tail finishes q+1): the MTP block STILL runs every step - its call writes the NextN block's own
// latent / k_pe / index-key rows, which later MTP drafts attend over, so skipping it would corrupt the fallback - and
// NgramSession::pick() replaces its token by the n-gram's when the matcher has a match and the chooser prefers it.
// The gain is therefore acceptance, not draft latency.  A proposal holds up to MAYA_GLM_NGRAM_MAX tokens (default 4,
// capped at 8: a multi-token verify (B11.4) loads the union of the routed experts of every drafted position, so long
// drafts cost RAM-tier reads); the depth-1 pipeline verifies its first token, last_proposal() keeps the rest for a
// multi-token verify.
//
// The chooser (DraftBook): an exponential average of each source's acceptance (alpha 1/16, prior 0.75 for both);
// the n-gram draft is used while its average stays within 0.05 of the MTP draft's (or during its first 8 drafts), so a
// prose turn where suffix matches do not continue falls back to MTP by itself, and an edit turn keeps the n-gram.
//
// Plain C++ (no CUDA).  tests/glm_dsa/glm_ngram_draft_test.cpp covers the matcher and the bookkeeping.
#pragma once

#include "strata/core/glm_draft_adaptive.hpp"   // B15.9

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core::glmng {

/// The switches (read on every call).
struct NgramConfig {
    bool on = false;      // MAYA_GLM_NGRAM_DRAFT
    int n_max = 4;        // MAYA_GLM_NGRAM_MAX: proposal length, 1..kMaxDraft
    int min_match = 3;    // MAYA_GLM_NGRAM_MIN: shortest suffix match that proposes, 1..16
    int max_match = 32;   // the backward extension stops here (the longest-match tie-break needs no more)
    int candidates = 16;  // most recent earlier occurrences examined per proposal
};
constexpr int kMaxDraft = 8;
NgramConfig ngram_config_env();

/// The suffix matcher over a token history.  Every min_match-gram of the history indexes its end positions; a
/// proposal takes the current suffix's gram, examines its most recent earlier occurrences (up to `candidates`),
/// extends each match backwards up to max_match tokens and proposes the tokens that followed the longest one (the
/// most recent on equal length).  A match shorter than min_match, or one with nothing after it, proposes nothing.
class NgramMatcher {
public:
    explicit NgramMatcher(const NgramConfig& c = NgramConfig{});
    void reset();
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }
    /// Up to min(max_k, n_max) proposed next tokens into out; returns how many (0: no match).  max_k < 0: n_max.
    int propose(int32_t* out, int max_k = -1);
    /// The length of the match behind the last proposal (0: none) and where the proposal was copied from.
    int last_match() const { return last_match_; }
    int64_t last_source() const { return last_src_; }
    size_t size() const { return hist_.size(); }
    const NgramConfig& config() const { return cfg_; }

private:
    uint64_t gram_key(size_t end) const;   // the hash of hist_[end - min_match, end)
    NgramConfig cfg_;
    std::vector<int32_t> hist_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> ends_;   // gram -> end positions, in append order
    int last_match_ = 0;
    int64_t last_src_ = -1;
};

/// Where a verified draft came from.
enum class DraftSrc : int { Mtp = 0, Ngram = 1 };

/// The acceptance bookkeeping: outstanding drafts in issue order (the pipeline keeps two), per-source counts and
/// averages, and the chooser.
struct DraftBook {
    static constexpr double kAlpha = 1.0 / 16.0, kPrior = 0.75, kMargin = 0.05;
    static constexpr int kWarmup = 8;
    uint64_t drafts[2] = {0, 0}, hits[2] = {0, 0};
    uint64_t agree = 0;          // n-gram drafts equal to the MTP draft they replaced
    uint64_t no_match = 0;       // steps with no n-gram match (MTP stood)
    uint64_t declined = 0;       // steps with a match the chooser declined
    double ema[2] = {kPrior, kPrior};
    std::deque<DraftSrc> pending;
    /// A draft entered the pipeline.
    void issued(DraftSrc s) { pending.push_back(s); }
    /// The oldest outstanding draft was verified: hit = it equalled the truth.  false when nothing was outstanding.
    bool verdict(bool hit);
    /// Use the n-gram's draft now?
    bool prefer_ngram() const;
    double rate(DraftSrc s) const;   // hits / drafts (0 without drafts)
    std::string line() const;        // a one-line summary
};

/// The engine side: one per decode_spec call (hooks in Glm5Model::decode_spec).  Inactive (every call a no-op that
/// returns the MTP draft) unless MAYA_GLM_NGRAM_DRAFT=1.
class NgramSession {
public:
    /// Takes the history registered for `model` (set_history; the prompt of the request), if any.
    explicit NgramSession(const void* model);
    ~NgramSession();
    NgramSession(const NgramSession&) = delete;
    NgramSession& operator=(const NgramSession&) = delete;
    bool on() const { return on_; }
    /// A confirmed token (the sequence's next).
    void push(int32_t tok);
    /// The draft to verify for the next position: the n-gram's first token when it matches and the chooser prefers
    /// it, else mtp_draft.  Records the source as outstanding.
    int32_t pick(int32_t mtp_draft);
    /// The oldest outstanding draft against the truth for its position.
    void verdict(int32_t draft, int32_t truth);
    /// The MTP head's own draft for the oldest outstanding position (what pick() was given), or `used` when the
    /// session is off or nothing is outstanding.  B11.3's acceptance counter (glmspec::note) records this one, so the
    /// MTP acceptance figure (the M4 gate) never counts an n-gram draft.  Call it before verdict().
    int32_t mtp_draft(int32_t used) const { return mtp_.empty() ? used : mtp_.front(); }
    const DraftBook& book() const { return book_; }
    /// The whole last n-gram proposal (empty when MTP stood): the tokens a multi-token verify would check.  With
    /// MAYA_GLM_DRAFT_ADAPTIVE at most depth() tokens (B15.9).
    const std::vector<int32_t>& last_proposal() const { return prop_; }
    /// B15.9: the proposal length now (the adaptive controller's depth, else MAYA_GLM_NGRAM_MAX).
    int depth() const { return ad_floor_ > 0 ? ad_.depth() : m_.config().n_max; }
    const AdaptiveDepth& adaptive() const { return ad_; }
    bool adaptive_on() const { return ad_floor_ > 0; }
    /// B15.9: tokens drafted / accepted in the tracked rounds (what a depth() verify would have checked / kept).
    uint64_t tracked_drafted() const { return ad_rows_; }
    uint64_t tracked_accepted() const { return ad_acc_; }

private:
    int ad_floor_ = 0;          // B15.9: MAYA_GLM_DRAFT_ADAPTIVE (0 off)
    AdaptiveDepth ad_;
    ProposalTracker tr_;
    uint64_t ad_rows_ = 0, ad_acc_ = 0;
    bool on_ = false;
    NgramMatcher m_;
    DraftBook book_;
    std::vector<int32_t> prop_;
    std::deque<int32_t> mtp_;   // the MTP drafts of the outstanding positions, in issue order (with book_.pending)
};

/// Registers the token history (the prompt) the next NgramSession of `model` starts from (replaces any earlier one;
/// an empty vector clears it).  Thread-safe.
void set_history(const void* model, const std::vector<int32_t>& tokens);
/// The same with the history in two pieces (a request's prompt, then the tokens it has produced so far): the several-
/// conversation scheduler registers both before each speculative stint, which may resume an interrupted request.
void set_history(const void* model, const std::vector<int32_t>& prompt, const std::vector<int32_t>& emitted);
/// Takes (and clears) it.
std::vector<int32_t> take_history(const void* model);

}  // namespace strata::core::glmng
