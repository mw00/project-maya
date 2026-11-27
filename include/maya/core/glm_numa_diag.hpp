// include/maya/core/glm_numa_diag.hpp - read-only multi-NUMA startup diagnostics for GLM models.
//
// Two misconfigurations silently hurt multi-GPU / dual-socket Linux hosts without failing startup:
//   1) MAYA_GLM_CPU_SHARED=socket on GLM-5.3-Flash (non-DSA): lane_pool_mode() treats "socket" via atoi()==0,
//      falling back to one small per-part CPU pool instead of a shared pool.
//   2) kernel.numa_balancing=1 while MAYA_GLM_ROW_HALVES=1 pins expert row halves across NUMA nodes via mbind:
//      Linux automatic NUMA migration can migrate faulted pages away from their preferred half.
// Pure read-only predicates and file reader; never modifies host sysctl or pool selection.
// Header-only, host C++, no CUDA or ggml dependency.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace maya::core::glmnuma {

/// True when MAYA_GLM_CPU_SHARED="socket" was explicitly requested on a multi-part non-DSA Flash load
/// (where lane_pool_mode() falls back to PerPart via atoi("socket") == 0).
inline bool shared_socket_fallback_on_flash(bool dsa, int n_parts, bool own_count, const char* shared_env) {
    if (dsa || n_parts < 2 || own_count || shared_env == nullptr) return false;
    return std::strcmp(shared_env, "socket") == 0;
}

/// True when the contents of `/proc/sys/kernel/numa_balancing` represent an enabled value (positive integer, e.g. "1").
inline bool parse_numa_balancing_enabled(const char* text) {
    if (text == nullptr) return false;
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text) return false;
    return v > 0;
}

/// Read-only check of `/proc/sys/kernel/numa_balancing` (or `path` for unit tests). Returns false if unreadable.
inline bool proc_numa_balancing_enabled(const char* path = "/proc/sys/kernel/numa_balancing") {
    if (path == nullptr) return false;
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) return false;
    char buf[64] = {};
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0) return false;
    buf[n] = '\0';
    return parse_numa_balancing_enabled(buf);
}

}  // namespace maya::core::glmnuma
