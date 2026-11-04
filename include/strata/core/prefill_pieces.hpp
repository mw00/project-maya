#pragma once
// STRATA_GLM_PREFILL_INTERLEAVE=<tokens>[,<rounds>[,cut]]: the several-conversation scheduler reads a new request's prompt
// in pieces of <tokens> while other requests decode, and between two pieces gives every running request <rounds>
// decode steps (1 by default) - instead of stopping them all until the whole prompt is read (Sarathi-Serve's chunked
// prefill, done as alternating calls here: the prompt path and the decode cannot share a step).  Each piece is one
// call of the prompt path; a call must reach its minimum (Glm5Model::prefill: 32 tokens), so a piece never leaves a
// shorter remainder behind it - that rest goes with it.
#include <cstddef>
#include <cstdlib>
#include <cstdio>

namespace strata::core {

struct PrefillInterleave {
    size_t tokens = 0;   // 0: off (the prompt is read in one go, as before)
    int rounds = 1;      // decode rounds between two pieces (every running request a token each)
    // true (the default): a piece is a whole number of the chunks the uncut read takes (prefill_aligned_piece), each
    // call cut in those chunks - the chunk boundaries are the uncut read's.  Not bit-exact for that: each piece is a
    // call of its own that lends and returns the pool's tail, and the decode steps between pieces fetch and promote
    // experts, so which experts are resident, staged or on the CPU lane for a chunk can differ from the uncut read,
    // and with it the order the experts' outputs are summed in.  `,cut` (false): pieces of exactly `tokens`, in
    // chunks of their own - shorter stalls; the chunk sizes change the prompt's numerics too
    bool exact = true;
};

/// STRATA_GLM_PREFILL_INTERLEAVE's text (null: unset) -> the setting; a piece is 256 tokens at least.
inline PrefillInterleave prefill_interleave_parse(const char* v) {
    PrefillInterleave r;
    if (v == nullptr || v[0] == '\0') return r;
    long long t = 0;
    int k = 1;
    char mode[16] = {0};
    const int got = std::sscanf(v, "%lld,%d,%15s", &t, &k, mode);
    if (got < 1 || t <= 0) return r;
    r.tokens = (size_t) (t < 256 ? 256 : t);
    r.rounds = got >= 2 ? (k < 1 ? 1 : k > 64 ? 64 : k) : 1;
    r.exact = !(got >= 3 && mode[0] == 'c');
    return r;
}

/// An exact piece: `tokens` rounded to a whole number (1 at least) of the uncut read's chunk `chunk`.
inline size_t prefill_aligned_piece(size_t tokens, size_t chunk) {
    if (chunk == 0) return tokens;
    const size_t m = (tokens + chunk / 2) / chunk;
    return (m < 1 ? 1 : m) * chunk;
}

/// The end of the next prompt-path call from `at` toward `seg_end` (a checkpoint mark or the prompt's last token, > at)
/// in pieces of `piece` tokens (0: none - the whole segment): at + piece, unless what would remain after it is shorter
/// than min_rest, which then goes in this piece.
inline size_t prefill_piece_end(size_t at, size_t seg_end, size_t piece, size_t min_rest = 64) {
    if (piece == 0 || seg_end <= at || seg_end - at <= piece) return seg_end;
    const size_t cut = at + piece;
    return seg_end - cut < min_rest ? seg_end : cut;
}

}  // namespace strata::core
