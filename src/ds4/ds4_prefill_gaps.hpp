#pragma once
// MAYA_DS4_PREFILL_GAPS=<percent>[,<max gap ms>]: while a new prompt is read on a layer split, the other
// conversations decode at the read's chunk boundaries - every device held at a boundary, the pipeline and the
// uncut read's chunks kept, the expert pool's tail still lent - for about <percent>% of the time (the gap after a
// stage lasts percent / (100 - percent) of that stage, one decode round at least, at most <max gap ms>).  The
// prompt is read in the same chunks as without, but not bit-exact: a gap's decode can move experts into spare
// slots, so a later chunk can find other experts resident, staged or on the CPU lane than the uncut read does.
// A running conversation waits about one stage instead of the whole read (a gap whose round overruns <max gap ms>
// ends the gaps of that read).  The boundaries are counted by the first group's chunks: boundary b comes after
// group 0 has read chunk b; group g has read chunk b - g and waits to start b - g + 1 until the gap there is over.
#include <cstdio>

namespace strata::ds4 {

struct PrefillGaps {
    int percent = 0;          // 0: off
    double max_ms = 4000.0;   // the longest gap (the read's own stall bound)

    bool on() const { return percent > 0; }
};

// MAYA_DS4_PREFILL_GAPS's text (null: unset) -> the setting; the share is 1..90 percent.  An unset share is off;
// a malformed one is off rather than a guess.
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

// The decode time a gap after a stage of `stage_ms` gets: percent / (100 - percent) of it, at most max_ms.
inline double prefill_gap_budget_ms(const PrefillGaps& g, double stage_ms) {
    if (g.percent <= 0 || stage_ms <= 0) return 0.0;
    const double b = stage_ms * (double) g.percent / (double) (100 - g.percent);
    return b < g.max_ms ? b : g.max_ms;
}

// Is a gap at this chunk boundary due: the budget the read earned since the last gap is half a decode round
// (the last gap's round; 0, not known yet: due) or more.  A gap is one round at least, so short chunks would
// otherwise take more than the share; a boundary that is not due adds its stage to the next one's.
inline bool prefill_gap_due(const PrefillGaps& g, double stage_ms, double round_ms) {
    if (g.percent <= 0) return false;
    return round_ms <= 0 || prefill_gap_budget_ms(g, stage_ms) >= 0.5 * round_ms;
}

}  // namespace strata::ds4
