// src/metal/spec.hpp - speculative decoding on maya-metal: the MTP drafter (the model's NextN block), as the CUDA
// engine runs it (src/core/glm_mtp.cu decode_mtp): a round drafts n tokens with chained greedy NextN steps, one
// verify window checks them, and every row is picked exactly as plain decoding picks it - so the tokens are plain
// decoding's, only more of them per pass over the weights.
#pragma once

#include "glm_metal.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace maya::metal {

// what the caller does with a token, the same calls plain decoding makes
struct DecodeHooks {
    // the pick of one row's logits (the sampler, the bias, the penalties, a forced token): called once per emitted
    // token, in order, after every earlier token of the round was emitted
    std::function<int32_t(const float* logits)> pick;
    // a token is out: false ends the decoding (EOS, STOP, max_new, the loop guard)
    std::function<bool(int32_t token)> emit;
    // the tokens still allowed (max_new - produced): a round drafts at most this many minus one
    std::function<int64_t()> left;
};

// the draft length a round uses, from what the drafts of each depth were worth (accepted share, the measured time
// of a round of that length) - glm_mtp.cu's mtp_choose
class DraftLength {
public:
    explicit DraftLength(int n_cap);
    int choose();
    void record(int n, int accepted, double ms);   // a round of n drafts kept `accepted` of them in ms

private:
    int cap_, start_, fixed_;
    bool adapt_;
    std::vector<double> hit_, reach_;   // per depth (1..cap): drafts accepted / drafts offered, decayed
    std::vector<double> ms_;            // per length (0..cap): ms of a round, measured
    std::vector<int> ms_n_;
    int last_n_ = -1, run_ = 0, rounds_ = 0, probe_at_ = 32, probe_gap_ = 32, probing_ = 0, probe_n_ = -1, best_ = 0;
};

// a drafter: up to n_max drafts after y (at position p0, not fed yet); false on an error
using Drafter = std::function<bool(int32_t y, int64_t p0, int n_max, std::vector<int32_t>& drafts, std::string& err)>;

// the decode loop from `y` (the token just picked, not fed yet) until a hook ends it: each round drafts, verifies the
// drafts in one window, keeps the rows plain decoding would have written.  `n_max` caps a round's drafts; `done`
// hears each round (drafts offered, accepted, ms)
bool decode_spec(GlmMetal& model, int32_t y, int n_max, const Drafter& draft,
                 const std::function<void(int, int, double)>& done, const DecodeHooks& hooks, std::string& err);

// MTP: the NextN block's chained greedy steps, the draft length chosen by `len`
bool decode_mtp(GlmMetal& model, int32_t y, DraftLength& len, const DecodeHooks& hooks, std::string& err);

}  // namespace maya::metal
