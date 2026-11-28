#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace hy4 {

inline int parse_tp_stage_slots(const char* raw, int default_slots, int min_slots = 16) {
    if (!raw || !*raw) return default_slots;
    const int n = std::atoi(raw);
    if (n <= 0) return default_slots;
    return std::max(min_slots, std::min(default_slots, n));
}

inline int tp_stage_slots_for_group(size_t n_gpus, int tp_cap, int default_slots, int extra_single_gpu_slots = 0) {
    return n_gpus > 1 ? tp_cap : default_slots + extra_single_gpu_slots;
}

inline int tp_stage_pass_count(size_t n_experts, int cap) {
    if (n_experts == 0 || cap <= 0) return 0;
    return static_cast<int>((n_experts + static_cast<size_t>(cap) - 1) / static_cast<size_t>(cap));
}

}  // namespace hy4
