// include/maya/core/glm_lookup.hpp - MAYA_GLM_LOOKUP: prompt lookup drafts beside the DFlash drafter.
//
// The suffix drafter (maya/spec/suffix_drafter.hpp, DeepSeek's prompt lookup) over the conversation's tokens - the
// prompt as read and every token written: where the context's tail repeats earlier text (an edit, quoted code, a tool
// call's arguments) the tokens that followed there are a round's drafts, up to the pipelined window's slots, past the
// drafter's block.  decode_dflash keeps a draft only when it is the token the sampler picks for its row (the window's
// own pick, drawn from the request's seed and the row's counter), so whichever source a draft came from every token
// written is the model's own pick: greedy output is the same tokens, sampled output the same draws.  Only the number
// of tokens a round commits moves.
//
// The choice between the two sources (choose) reads the two proposals and the match length, nothing drawn and nothing
// timed, so a request decodes the same rounds every time:
//   - a match of `trust` tokens or more: the lookup's run, whatever the drafter says (the drafter's block when it
//     repeats the whole run and goes on past it);
//   - a shorter match (from `min_match`): the drafter's block where the two differ; where the block is the run's start
//     (or the drafter made none) the run goes on past it, by at most as many drafts as the match is long - a draft
//     that fails still costs its enqueue (and, without MAYA_GLM_WIN_DROP, part of its pass);
//   - no match: the drafter's block, as without the switch.
// DeepSeek weighs the lookup by measured round times and acceptance (maya/spec/draft_policy.hpp) and takes it only
// where it starts with the drafter's first draft; its rule that the choice must not read the round's draws
// (DS4_LOOKUP_SAMPLED) guards the p/q acceptance of drawn drafts - here drafts are points checked against the pick, so
// no choice can move the distribution, and a kept draft costs a third of a round's fixed part: the source expected to
// keep more is the better one, and the match length says which.
//
// MAYA_GLM_LOOKUP=2: a trusted run is verified alone - the drafter is not started for that round (the round's drafts
// are pushed as soon as its anchor is queued instead of after the drafter's pass) - while the lookup's first drafts
// have been holding: every proposal's first token is scored against the row-0 pick whichever source the round took
// (first_seen / first_hit, decayed), and a run stands alone only from kAloneSeen scored proposals at kAloneRate.
// The drafter's ring is fed the kept positions every round either way (its next draft reads them).
//
// Pictures: an image's positions are rows of the vision encoder, not tokens (the prompt carries one placeholder id for
// each) - history() puts one negative mark a run of them, a different one each run, so nothing matches across or into
// a picture, and propose() ends a run at a mark.
//
// The index is kept across requests when the new prompt starts with everything it holds (SuffixDrafter::restart: the
// prompt + the answer in the ids the model wrote), rebuilt otherwise.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "maya/spec/suffix_drafter.hpp"

namespace maya::core {

struct GlmLookup {
    enum Source { kDrafter = 0, kLookup = 1, kExtend = 2 };
    struct Choice {
        Source src = kDrafter;
        int n = 0;          // the window's drafts
        int n_drafter = 0;  // ... the first of them the drafter's (kExtend: the block the run goes on from)
    };
    static constexpr int kMaxMatch = 64;
    static constexpr double kDecay = 0.98, kAloneSeen = 8.0, kAloneRate = 0.8;

    maya::spec::SuffixDrafter sfx;
    int mode = 1;        // MAYA_GLM_LOOKUP: 1 beside the drafter, 2 a trusted run alone too
    int min_match = 3;   // MAYA_GLM_LOOKUP_MIN: the shortest match that proposes (DeepSeek's --suffix-draft 3)
    int trust = 16;      // MAYA_GLM_LOOKUP_TRUST: the match from which the run is taken over the drafter's block
    int max_n = 1 << 20; // MAYA_GLM_LOOKUP_MAX: a lookup window's drafts at most (the window's slots cap it too)
    double first_seen = 0.0, first_hit = 0.0;   // the proposals' first tokens against the pick, decayed (engine's life)
    // a request's counts (begin() clears them)
    size_t kept_index = 0, index_at_start = 0;
    long long proposals = 0, first_right = 0;
    long long rounds[3] = {}, alone_rounds = 0;
    long long off_drafter = 0, kept_drafter = 0, off_lookup = 0, kept_lookup = 0;

    GlmLookup(int mode_, int min_match_, int trust_, int max_n_, size_t capacity)
        : sfx(std::max(3, min_match_), kMaxMatch, capacity), mode(mode_), min_match(std::max(3, min_match_)),
          trust(std::max(min_match, trust_)), max_n(std::max(1, max_n_)) {}

    /// MAYA_GLM_LOOKUP's index for a context of `max_context` tokens, or nullptr (unset / 0: off).
    static GlmLookup* from_env(size_t max_context) {
        const char* v = std::getenv("MAYA_GLM_LOOKUP");
        const int mode = v != nullptr ? std::atoi(v) : 0;
        if (mode <= 0) return nullptr;
        const auto num = [](const char* name, int dflt) {
            const char* e = std::getenv(name);
            return e != nullptr && e[0] ? std::atoi(e) : dflt;
        };
        return new GlmLookup(mode, num("MAYA_GLM_LOOKUP_MIN", 3), num("MAYA_GLM_LOOKUP_TRUST", 16),
                             num("MAYA_GLM_LOOKUP_MAX", 1 << 20), max_context + 4096);
    }

    /// The index's tokens for a prompt: its ids with each run of image positions (ascending `img_pos`) as one mark,
    /// -1 - the run's number.
    static void history(const std::vector<int32_t>& prompt, const std::vector<int64_t>& img_pos,
                        std::vector<int32_t>& out) {
        out.clear();
        out.reserve(prompt.size());
        size_t k = 0;
        int32_t run = 0;
        bool in_run = false;
        for (size_t i = 0; i < prompt.size(); ++i) {
            while (k < img_pos.size() && (size_t) img_pos[k] < i) ++k;
            if (k < img_pos.size() && (size_t) img_pos[k] == i) {
                if (!in_run) out.push_back(-1 - run++);
                in_run = true;
                continue;
            }
            in_run = false;
            out.push_back(prompt[i]);
        }
    }

    /// A request's start: the index holds `hist` (history() of its prompt) - kept where it extends what was held.
    void begin(const std::vector<int32_t>& hist) {
        kept_index = sfx.restart(hist.data(), hist.size());
        index_at_start = sfx.size();
        proposals = first_right = alone_rounds = 0;
        rounds[0] = rounds[1] = rounds[2] = 0;
        off_drafter = kept_drafter = off_lookup = kept_lookup = 0;
    }
    /// A token written (fed to the state next).
    void wrote(int32_t tok) { sfx.append(tok); }

    /// The run after the longest earlier match of the context's tail: up to `room` tokens to `out`, `match` its length.
    int propose(int room, int32_t* out, int& match) {
        match = 0;
        int n = sfx.propose(std::min(room, max_n), out);
        for (int i = 0; i < n; ++i)
            if (out[i] < 0) n = i;   // (a picture's mark ends the run)
        if (n > 0) match = sfx.last_match();
        return n;
    }

    /// MAYA_GLM_LOOKUP=2: this proposal verified alone, the drafter not started for the round.
    bool alone(int n, int match) const {
        return mode >= 2 && n > 0 && match >= trust && first_seen >= kAloneSeen && first_hit >= kAloneRate * first_seen;
    }

    /// The round's drafts from the drafter's block d[0, nd) and the lookup's run l[0, nl) of a `match`-token match
    /// (see the top of the file): written to out (room for max(nd, nl)), described by the return.
    static Choice choose(const int32_t* d, int nd, const int32_t* l, int nl, int match, int trust, int32_t* out) {
        Choice c;
        const auto take = [&](const int32_t* src, int n) {
            for (int i = 0; i < n; ++i) out[i] = src[i];
            c.n = n;
        };
        int j = 0;   // the drafts the two agree on
        while (j < nd && j < nl && d[j] == l[j]) ++j;
        if (nl <= 0 || (j == nl && nd >= nl)) {   // no run, or the block holds all of it
            take(d, nd);
            c.n_drafter = nd;
        } else if (match >= trust) {
            take(l, nl);
            c.src = kLookup;
        } else if (j == nd && std::min(nl, nd + match) > nd) {
            take(l, std::min(nl, nd + match));
            c.src = kExtend;
            c.n_drafter = nd;
        } else {
            take(d, nd);
            c.n_drafter = nd;
        }
        return c;
    }

    /// A proposal's first token against the token that went out of row 0 (every round with a proposal).
    void score_first(bool right) {
        first_seen = first_seen * kDecay + 1.0;
        first_hit = first_hit * kDecay + (right ? 1.0 : 0.0);
        ++proposals;
        first_right += right ? 1 : 0;
    }
    /// A verified round: its choice and how many of its drafts were kept; `was_alone`: no draft was made for it.
    void observe(const Choice& c, int kept, bool was_alone) {
        ++rounds[c.src];
        alone_rounds += was_alone ? 1 : 0;
        off_drafter += c.n_drafter;
        kept_drafter += std::min(kept, c.n_drafter);
        off_lookup += c.n - c.n_drafter;
        kept_lookup += std::max(0, kept - c.n_drafter);
    }
    /// The request's line (decode_dflash's end).
    void report(std::FILE* f) const {
        const long long r = rounds[0] + rounds[1] + rounds[2];
        std::fprintf(f, "glm lookup: index %zu tokens at the start (%zu kept from the last request), %zu at the end | "
                        "%lld rounds: %lld drafter, %lld lookup (%lld alone), %lld drafter + lookup | drafts offered / "
                        "kept: drafter %lld / %lld, lookup %lld / %lld | %lld proposals, first token right %lld\n",
                     index_at_start, kept_index, sfx.size(), r, rounds[kDrafter], rounds[kLookup], alone_rounds,
                     rounds[kExtend], off_drafter, kept_drafter, off_lookup, kept_lookup, proposals, first_right);
    }
};

}  // namespace maya::core
