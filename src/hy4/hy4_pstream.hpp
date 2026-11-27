// src/hy4/hy4_pstream.hpp - HY4_PSTREAM's bookkeeping: a layer's experts outside VRAM split into passes, the order the
// passes' copies and kernels may be queued in on two staging buffers, and which expert each pinned copy belongs to as
// HY4_HOT swaps experts.  (No GPU: tests/hy4_pstream_test.cpp.)
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hy4 {

// m experts outside VRAM in passes of at most `cap`: the fewest passes, each of at most `per` experts (a layer's passes
// are as even as they go - the buffers hold `cap`)
struct PsPlan {
    int passes = 0, per = 0;
    int count(int m, int p) const { return std::min(per, m - p * per); }   // pass p's experts
};
inline PsPlan ps_plan(int m, int cap) {
    PsPlan p;
    if (m <= 0 || cap <= 0) return p;
    p.passes = (m + cap - 1) / cap;
    p.per = (m + p.passes - 1) / p.passes;
    return p;
}

// whether pass j's copy may be queued with `run` passes' kernels queued so far: the two buffers alternate, so pass j's
// buffer is pass j - 2's, and that pass must have been run (its done event recorded) first
inline bool ps_may_copy(size_t j, size_t run) { return j < run + 2; }

// a swap of HY4_HOT: expert `out` leaves VRAM, expert `in` (in RAM so far) goes into it.  of[e] is the pinned slot expert
// e has (-1: in VRAM), ex[q] the expert slot q holds.  Returns the slot `in` gives up, which `out` takes now; -1: `in`
// has none (nothing changed)
inline int ps_swap(std::vector<int>& of, std::vector<int>& ex, int out, int in) {
    const int q = of[(size_t) in];
    if (q < 0) return -1;
    of[(size_t) in] = -1;
    of[(size_t) out] = q;
    ex[(size_t) q] = out;
    return q;
}

}  // namespace hy4
