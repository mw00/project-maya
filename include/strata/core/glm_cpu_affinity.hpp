// include/strata/core/glm_cpu_affinity.hpp - glm-dsa B15.5: size CPU thread pools from the process's affinity mask.
//
// std::thread::hardware_concurrency() counts the machine's logical CPUs, not the ones this process may run on.  Under
// `numactl --cpunodebind=0` (or a cgroup / taskset mask) it still says 80 on the test PC, so a pool sized from it puts two
// threads or more on each allowed CPU (llama.cpp #29301: llama-bench read its default -t before numactl's mask
// applied; #30267: cgroup quotas).  The CPU-lane setup (glm_fast_path.cu fast_cpu_lane_setup) and the split search's
// RAM-bandwidth probe (glm_model.cu host_read_bps) size their threads from mask_cpus() instead; the physical-core
// count and the per-node CPU lists already came from sched_getaffinity.
#pragma once

#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace strata::core::glmaff {

/// CPUs in a sched_getaffinity mask (CPU_COUNT); 0 when the mask could not be read.
inline int mask_cpus_raw() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    return CPU_COUNT(&set);
#else
    return 0;
#endif
}

/// The logical CPUs this process may run on: the affinity mask's count, else hardware_concurrency(), at least 1.
inline int mask_cpus() {
    const int m = mask_cpus_raw();
    if (m > 0) return m;
    const int h = (int) std::thread::hardware_concurrency();
    return h > 0 ? h : 1;
}

}  // namespace strata::core::glmaff
