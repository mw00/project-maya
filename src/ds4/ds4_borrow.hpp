// src/ds4/ds4_borrow.hpp - DS4_TP_BORROW's arithmetic: a tensor-parallel group's members lend the prompt path the room
// of their last k VRAM slots of every layer (Chunk::borrow, as one GPU does), each member its own part of a slot - pure
// functions, CPU only, shared by ds4_model.cu and tests/ds4_pool_test.cpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::ds4 {

// a group's tail slots a layer: member i's buffers of need[i] bytes (+ 256, the region's alignment) in its tail of k
// slots of step[i] bytes (its parts of one slot of each of its layers) - the most any member needs, at most max_k (the
// group's fewest VRAM slots a layer, less one: a layer keeps a slot of its own); -1 when a member has no step
inline int tp_tail_k(const std::vector<size_t>& need, const std::vector<size_t>& step, int max_k) {
    if (need.empty() || need.size() != step.size()) return -1;
    int k = 0;
    for (size_t i = 0; i < need.size(); ++i) {
        if (step[i] == 0) return -1;
        k = std::max(k, (int) ((need[i] + 256 + step[i] - 1) / step[i]));
    }
    return std::min(k, std::max(0, max_k));
}

// whether every member's buffers fit its tail of k slots
inline bool tp_tail_fits(const std::vector<size_t>& need, const std::vector<size_t>& step, int k) {
    if (k <= 0 || need.size() != step.size()) return false;
    for (size_t i = 0; i < need.size(); ++i)
        if (need[i] + 256 > (size_t) k * step[i]) return false;
    return true;
}

// a member's slots of one layer wholly before its buffers (laid out from b0 to its tail's end): the slot's part at off
// in the tail, u bytes, step apart, at most avail of them - the run keeps those in VRAM; a group keeps the fewest of
// its members' (both stream the rest, each its part)
inline int tail_keep(size_t off, size_t u, size_t step, size_t b0, int avail) {
    int keep = 0;
    while (keep < avail && off + (size_t) keep * step + u <= b0) ++keep;
    return keep;
}

}  // namespace strata::ds4
