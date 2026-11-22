// src/hy4/hy4_ring.hpp - the positions maya-hy4's rings still hold (#1091).  A ring (the sliding-window layers' caches,
// the indexed layers' value | gate rows, the trunk's states the MTP block reads) keeps a position's row in slot
// position % rows, so a read up to `end` leaves [end - rows, end).  A rewind moves `end` back but does not bring back
// the slots the longer read wrote over: Model::Impl::ring_lo keeps the lowest position still held across rewinds
// since the last read from 0, and a forward may start only where the rows it reads before it are at or above both.
// (No GPU: tests/hy4_ring_test.cpp drives these against a slot-by-slot model of the rings.)
#pragma once

#include <algorithm>
#include <cstdint>

namespace hy4 {

// ring_lo once a forward from pos0 starts, the rows so far ending at `end` (before it moves to pos0; a read from 0
// starts over)
inline int64_t ring_lo_at(int64_t pos0, int64_t ring_lo, int64_t end, int64_t rows) {
    return pos0 == 0 ? 0 : std::max<int64_t>(ring_lo, end - rows);
}

// whether a forward may start at pos: the `back` positions before it that it reads (its window, its indexer pool's
// first rows) all held - at or above ring_lo and end - rows (the window start clamped at 0: a prompt shorter than the
// window resumes at its next token)
inline bool ring_holds(int64_t pos, int64_t end, int64_t ring_lo, int64_t rows, int64_t back) {
    if (pos == 0) return true;
    if (pos < 0 || pos > end) return false;
    return std::max<int64_t>(0, pos - back) >= std::max<int64_t>(ring_lo, end - rows);
}

}  // namespace hy4
