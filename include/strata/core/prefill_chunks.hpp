#pragma once
// The chunks a prompt-path call is cut into (Glm5Model::prefill, prefill_chunk_for), and
// STRATA_GLM_PREFILL_MIN_CHUNKS=<n>[,<min tokens>]: on a split, every call is cut into n chunks at least (none shorter
// than <min tokens>, 512 by default).  The parts of a split pipeline a call's chunks (part 1 on chunk c while part 0 is
// on c + 1), but a call that fits in one chunk - most reads of a long conversation: a turn appended to a restored one -
// runs on one GPU after the other, each idle half the time.  In n chunks each part works n / (n + parts - 1) of the
// call instead (two parts: 2 chunks 67%, 3 75%, 4 80%), against a fixed cost a chunk (every layer's routes synced to
// the host for its plan, the expert weights read once a chunk) and fewer rows a GEMM.  The chunk length changes the
// prompt's arithmetic, as --prefill N does.
#include <cstdint>
#include <cstdio>

namespace strata::core {

struct PrefillMinChunks {
    int n = 1;              // 1: off (as many chunks as the largest chunk needs)
    int min_tokens = 512;   // no chunk of a cut call shorter (a call under 2 x this stays whole)
};

/// STRATA_GLM_PREFILL_MIN_CHUNKS's text (null: unset) -> the setting; n is 1..8, a chunk 256 tokens at least.
inline PrefillMinChunks prefill_min_chunks_parse(const char* v) {
    PrefillMinChunks r;
    if (v == nullptr || v[0] == '\0') return r;
    int n = 0, t = 0;
    const int got = std::sscanf(v, "%d,%d", &n, &t);
    if (got < 1 || n <= 1) return r;
    r.n = n > 8 ? 8 : n;
    if (got >= 2 && t > 0) r.min_tokens = t < 256 ? 256 : t;
    return r;
}

/// How many chunks a split wants at least for a call of nn tokens: `want` of them, as long as each keeps min_tokens.
inline int64_t prefill_floor_chunks(int64_t nn, int want, int min_tokens) {
    if (want <= 1 || min_tokens <= 0) return 1;
    const int64_t fit = nn / min_tokens;
    return fit < want ? (fit < 1 ? 1 : fit) : want;
}

/// The chunk length a call of nn tokens takes when the largest chunk every part holds is tmax: one GPU (split false)
/// takes full chunks with the remainder last (no pipeline to balance; every chunk stages nearly every expert); a split
/// cuts it into EQUAL chunks - as many as tmax needs, and at least `floor_chunks` (prefill_floor_chunks) - each rounded
/// up to 64 tokens.  The call then reads ceil(nn / length) chunks (Glm5Model::prefill_run), the last one shorter -
/// exactly that floor for up to 4 chunks of 256 tokens or more; with more, shorter ones the rounding can leave one less.
inline int prefill_chunk_len(int64_t nn, int tmax, bool split, int64_t floor_chunks) {
    if (nn <= 0 || tmax <= 0) return 0;
    const auto up64 = [](int64_t x) { return (x + 63) / 64 * 64; };
    if (!split) return (int) (up64(nn) < tmax ? up64(nn) : tmax);
    int64_t k = (nn + tmax - 1) / tmax;
    if (floor_chunks > k) k = floor_chunks;
    const int64_t len = up64((nn + k - 1) / k);
    return (int) (len < tmax ? len : tmax);
}

/// The chunks a call of nn tokens in chunks of len reads (prefill_run's count)
inline int64_t prefill_chunk_count(int64_t nn, int len) { return len > 0 && nn > 0 ? (nn + len - 1) / len : 0; }

}  // namespace strata::core
