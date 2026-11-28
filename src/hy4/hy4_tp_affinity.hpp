#pragma once

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace hy4::tp_affinity {

inline std::vector<int> parse_cpulist(const std::string& spec) {
    std::vector<int> cpus;
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        if (a < 0 || b < a) continue;
        for (int c = a; c <= b; ++c) cpus.push_back(c);
    }
    return cpus;
}

inline bool respect_inherited_affinity() {
    static const bool respect = [] {
        const char* v = std::getenv("HY4_TP_RESPECT_AFFINITY");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return respect;
}

#if defined(__linux__)
inline bool inherited_affinity_mask(cpu_set_t& out) {
    if (!respect_inherited_affinity()) return false;
    static cpu_set_t saved;
    static const bool ok = [] {
        CPU_ZERO(&saved);
        return sched_getaffinity(0, sizeof saved, &saved) == 0 && CPU_COUNT(&saved) > 0;
    }();
    if (ok) out = saved;
    return ok;
}

inline std::vector<int> filter_cpus(const std::vector<int>& cpus, bool has_mask, const cpu_set_t& allowed) {
    std::vector<int> out;
    out.reserve(cpus.size());
    for (int c : cpus) {
        if (c >= 0 && (!has_mask || (c < CPU_SETSIZE && CPU_ISSET(c, &allowed)))) out.push_back(c);
    }
    return out;
}

inline bool make_affinity_set(const std::vector<int>& cpus, bool has_mask, const cpu_set_t& allowed, cpu_set_t& out) {
    CPU_ZERO(&out);
    for (int c : cpus) {
        if (c >= 0 && c < CPU_SETSIZE && (!has_mask || CPU_ISSET(c, &allowed))) CPU_SET(c, &out);
    }
    return CPU_COUNT(&out) > 0;
}
#endif

}  // namespace hy4::tp_affinity
