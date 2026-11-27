// src/ds4/ds4_tp_kind.hpp - whether two GPUs are "of one kind" for the DeepSeek engine's tensor-parallel groups (a run
// of --tp devices whose memory is the same), as pure host arithmetic shared by ds4_model.cu and tests/ds4_tp_kind_test.cpp.
//
// The totals are compared within a small share of the larger one, not to the byte: the driver reserves a few MiB more on
// the boot / console GPU (#1276: 3 MiB on an RTX 3090 that holds the framebuffer), which left it and its neighbour out of
// any group.  Cards of different sizes differ by gigabytes, far beyond the share, so they still stay apart.  Each
// group's room is its tightest member's own free memory, so a member a few MiB smaller is planned for as it is.
#pragma once

#include <cstddef>

namespace maya::ds4 {

// The share of the larger total that two GPUs' totals may differ by and still count as one kind (0.5%: ~120 MiB on a
// 24 GiB card, ~40 MiB on an 8 GiB one).
constexpr double kTpKindShare = 0.005;

inline bool tp_same_kind(std::size_t total_a, std::size_t total_b) {
    const std::size_t hi = total_a > total_b ? total_a : total_b;
    const std::size_t lo = total_a > total_b ? total_b : total_a;
    return (double) (hi - lo) <= kTpKindShare * (double) hi;
}

}  // namespace maya::ds4
