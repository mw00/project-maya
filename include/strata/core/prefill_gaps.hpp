#pragma once
// STRATA_GLM_PREFILL_GAPS=<percent>[,<max gap ms>]: while a new prompt is read on a split (the parts pipeline its chunks:
// part 1 on chunk c while part 0 is on c + 1), the other requests decode at the chunk boundaries - every part idle, the
// expert pool's tail still lent, the pipeline not drained - for about <percent>% of the time (the gap after a stage
// lasts percent / (100 - percent) of that stage, a decode round at least, at most <max gap ms>).  The prompt is read
// in the same chunks as without, but not bit-exact: a gap's decode can move experts into this card's spare slots, so
// the next chunk can find other experts resident, staged or on the CPU lane than the uncut read does, and sum their
// outputs in another order.  A running request waits about one stage instead of the whole read (a gap whose round
// overruns <max gap ms> ends the gaps of that read: Glm5Model::prefill_gap_wanted).
// Unlike STRATA_GLM_PREFILL_INTERLEAVE, which reads the prompt in separate calls (each one fills and drains the
// pipeline), the read stays one call.
//
// The boundaries, counted by part 0's chunks: boundary b comes after part 0 has read chunk b (b = 0 .. nc - 2).  At
// boundary b part k (k >= 1) has read chunk b - k and waits to start chunk b - k + 1 until the gap there is over.
#include <cstdio>
#include <vector>

namespace strata::core {

struct PrefillGaps {
    int percent = 0;          // 0: off
    double max_ms = 4000.0;   // the longest gap (the read's own stall bound)
};

/// STRATA_GLM_PREFILL_GAPS's text (null: unset) -> the setting; the share is 1..90 percent.
inline PrefillGaps prefill_gaps_parse(const char* v) {
    PrefillGaps r;
    if (v == nullptr || v[0] == '\0') return r;
    int p = 0;
    double ms = 0;
    const int got = std::sscanf(v, "%d,%lf", &p, &ms);
    if (got < 1 || p <= 0) return r;
    r.percent = p > 90 ? 90 : p;
    if (got >= 2 && ms > 0) r.max_ms = ms;
    return r;
}

/// The decode time a gap after a stage of `stage_ms` gets: percent / (100 - percent) of it, at most max_ms.
inline double prefill_gap_budget_ms(const PrefillGaps& g, double stage_ms) {
    if (g.percent <= 0 || stage_ms <= 0) return 0.0;
    const double b = stage_ms * (double) g.percent / (double) (100 - g.percent);
    return b < g.max_ms ? b : g.max_ms;
}

/// Is a gap at this chunk boundary due: the budget the read earned since the last gap (prefill_gap_budget_ms of the
/// stage since then) is half a decode round (round_ms: the last gap's round; 0, not known yet: due) or more.  A gap is
/// one round at least, so with short chunks (STRATA_GLM_PREFILL_MIN_CHUNKS) a gap at every boundary would take more
/// than the share; a boundary that is not due adds its stage to the next one's.
inline bool prefill_gap_due(const PrefillGaps& g, double stage_ms, double round_ms) {
    if (g.percent <= 0) return false;
    return round_ms <= 0 || prefill_gap_budget_ms(g, stage_ms) >= 0.5 * round_ms;
}

/// The pipeline's gap bookkeeping (prefill_run holds its own mutex around every call): np parts, nc chunks.
struct PipelineGaps {
    int np = 0, nc = 0;
    std::vector<int> finished;   // chunks part k has read (its stream synchronized)
    int released = -1;           // the last boundary whose gap is over (or that had none)
    PipelineGaps(int parts, int chunks) : np(parts), nc(chunks), finished((size_t) (parts > 0 ? parts : 0), 0) {}
    /// the last boundary: none after the last chunk
    int last_boundary() const { return nc - 2; }
    /// part k (>= 1) may start chunk c: the boundary before it is over, or there is none
    bool may_start(int k, int c) const {
        const int b = c + k - 1;
        return b > last_boundary() || b < 0 || released >= b;
    }
    /// at boundary b every later part is idle: part k has read chunk b - k (when there is one)
    bool all_idle(int b) const {
        for (int k = 1; k < np; ++k) {
            const int need = b - k + 1;   // chunks 0 .. b - k
            if (need > 0 && finished[(size_t) k] < (need < nc ? need : nc)) return false;
        }
        return true;
    }
};

}  // namespace strata::core
