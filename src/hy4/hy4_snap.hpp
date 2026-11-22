// src/hy4/hy4_snap.hpp - maya-hy4's restore points (Options::snap_every, `--snap-every N`).  The rings (hy4_ring.hpp)
// keep the last `rows` positions only, so a prompt that leaves the sequence further back than that (an agent client
// that rewrites an early message, or compacts its history) is read again from 0.  A restore point s keeps what a forward
// from s reads that the rings may have written over since: each sliding-window layer's rows of the n_swa positions
// before s, and the trunk's state at s - 1 (the MTP block's first input).  The indexed layers' and the MTP block's caches
// keep a row a position, and s is a multiple of the indexer's pool (its groups start there), so a forward from s needs
// nothing else of them.  The points are copied to pinned host memory as the sequence passes them; such a prompt is read
// from the latest point at or before the split instead of from 0.
// (No GPU: tests/hy4_snap_test.cpp drives these against a slot-by-slot model of the rings.)
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace hy4 {

struct SnapPoint {
    int64_t at = -1;     // the point (-1: none held)
    uint64_t hash = 0;   // snap_hash of the sequence's tokens [0, at)
};

// FNV-1a of ids [0, n), continued from h: a point's sequence, checked against a prompt's start before it is used
inline uint64_t snap_hash(const int* ids, int64_t n, uint64_t h = 1469598103934665603ull) {
    for (int64_t i = 0; i < n; ++i) {
        h ^= (uint64_t) (uint32_t) ids[i];
        h *= 1099511628211ull;
    }
    return h;
}

// the first point (a multiple of step) a window at q may take before it writes: its rows [s - n_swa, s) all still held
// (the rings hold [max(ring_lo, q - rows), q) then).  The points up to q from there are the window's to take.
inline int64_t snap_first(int64_t q, int64_t ring_lo, int64_t rows, int64_t n_swa, int64_t step) {
    const int64_t lo = std::max<int64_t>(ring_lo, q - rows);
    return (std::max<int64_t>(lo + n_swa, step) + step - 1) / step * step;
}

// the sequence cut at pos0: the points past it were another sequence's
inline void snap_drop(std::vector<SnapPoint>& pts, int64_t pos0) {
    for (SnapPoint& p : pts)
        if (p.at > pos0) p.at = -1;
}

// the latest point of pts (slot i: the point (i + 1) * step, so in the points' order) at or before upto and end whose
// sequence is ids' start: its slot, or -1
inline int64_t snap_pick(const std::vector<SnapPoint>& pts, const int* ids, int64_t upto, int64_t end) {
    int64_t best = -1, hn = 0;
    uint64_t h = snap_hash(nullptr, 0);
    for (size_t i = 0; i < pts.size(); ++i) {
        const SnapPoint& p = pts[i];
        if (p.at <= 0) continue;
        if (p.at > upto || p.at > end) break;
        h = snap_hash(ids + hn, p.at - hn, h);
        hn = p.at;
        if (h == p.hash) best = (int64_t) i;
    }
    return best;
}

// ring_lo once point s is put back: the rows below s - n_swa are not (a forward from s reads n_swa - 1 of them, and
// the indexer's pool, at most n_swa, starts at s)
inline int64_t snap_ring_lo(int64_t s, int64_t n_swa) { return s - n_swa; }

}  // namespace hy4
