#pragma once
// The chunks a prompt-path call is cut into (Glm5Model::prefill, prefill_chunk_for), and
// MAYA_GLM_PREFILL_MIN_CHUNKS=<n>[,<min tokens>]: on a split, every call is cut into n chunks at least (none shorter
// than <min tokens>, 512 by default).  The parts of a split pipeline a call's chunks (part 1 on chunk c while part 0 is
// on c + 1), but a call that fits in one chunk - most reads of a long conversation: a turn appended to a restored one -
// runs on one GPU after the other, each idle half the time.  In n chunks each part works n / (n + parts - 1) of the
// call instead (two parts: 2 chunks 67%, 3 75%, 4 80%), against a fixed cost a chunk (every layer's routes synced to
// the host for its plan, the expert weights read once a chunk) and fewer rows a GEMM.  The chunk length changes the
// prompt's arithmetic, as --prefill N does.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace maya::core {

// The largest chunk on Auto's 256-token grid whose buffers fit. The test must be monotonic on [first, last].
template <class Fits>
inline int64_t prefill_grid_chunk(int64_t first, int64_t last, Fits fits) {
    int64_t lo = (first + 255) / 256, hi = last / 256, best = 0;
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (fits(mid * 256)) {
            best = mid * 256;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

inline constexpr int kPreMinT = 1024, kPreMinSlots = 16;   // the smallest chunk and ring used for prestaging

struct PrefillPoolFit {
    int chunk = 0, prestage = 0;
    size_t bytes = 0, loan = 0;
};

// MAYA_GLM_PREFILL_MB takes precedence over the percentage. Both still leave the pool its slot floor.
inline size_t prefill_pool_loan(int64_t per, int64_t keep, size_t row, int64_t pct, size_t fixed_bytes) {
    const int64_t rows = std::max<int64_t>(0, per - keep);
    if (fixed_bytes != SIZE_MAX) return std::min((size_t) rows * row, fixed_bytes);
    const int64_t share = std::max<int64_t>(0, pct * per / 100);
    return (size_t) std::min(rows, share) * row;
}

// Reconcile a prepared chunk with the measured pool loan. Trim its prestage ring first; only Auto may take a
// smaller chunk. Never grow the ring beyond the events allocated for it. nonring measures the complete aligned
// layout (including the whole KV staging area) without that ring.
template <class Nonring>
inline PrefillPoolFit prefill_pool_fit(int chunk, int prestage, size_t stride, size_t loan, bool automatic,
                                      Nonring nonring) {
    size_t base = nonring(chunk);
    if (base > loan) {
        if (!automatic) return {0, 0, base, loan};
        chunk = (int) prefill_grid_chunk(256, chunk - 1, [&](int64_t t) { return nonring(t) <= loan; });
        if (chunk == 0) return {0, 0, nonring(256), loan};
        base = nonring(chunk);
    }
    // The ring is carved in whole 256-byte blocks; an expert stride need only be 16-byte aligned.
    const size_t room = (loan - base) & ~size_t{255};
    const int slots = chunk >= kPreMinT && stride > 0 ? (int) std::min((size_t) prestage, room / stride) : 0;
    const int np = slots >= kPreMinSlots ? slots : 0;
    return {chunk, np, base + (((size_t) np * stride + 255) & ~size_t{255}), loan};
}

struct PrefillMinChunks {
    int n = 1;              // 1: off (as many chunks as the largest chunk needs)
    int min_tokens = 512;   // no chunk of a cut call shorter (a call under 2 x this stays whole)
};

/// MAYA_GLM_PREFILL_MIN_CHUNKS's text (null: unset) -> the setting; n is 1..8, a chunk 256 tokens at least.
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

}  // namespace maya::core
