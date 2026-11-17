#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core::glm_cache_memory {

inline bool permits(uint64_t available, uint64_t reserve, uint64_t bytes) {
    return available >= reserve && bytes <= available - reserve;
}

inline std::optional<uint64_t> quantity(std::string_view text) {
    uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
    return value;
}

// cgroup v2 exposes limits at every ancestor. The namespace root is included.
// cgroup v1 or a missing controller fails closed for the candidate.
inline std::optional<uint64_t> linux_available(const std::filesystem::path& meminfo,
                                               const std::filesystem::path& membership,
                                               const std::filesystem::path& root) {
    std::ifstream mem(meminfo), groups(membership);
    std::string line;
    std::optional<uint64_t> available;
    while (std::getline(mem, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream fields(line.substr(13));
        uint64_t kb = 0;
        std::string unit;
        if (!(fields >> kb >> unit) || unit != "kB" || kb > UINT64_MAX / 1024) return std::nullopt;
        available = kb * 1024;
        break;
    }
    if (!available) return std::nullopt;
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) continue;
        const std::filesystem::path relative(line.substr(4));
        if (relative.is_absolute()) return std::nullopt;
        for (const auto& component : relative)
            if (component == ".." || component == ".") return std::nullopt;
        auto dir = relative.empty() ? root : root / relative;
        for (;;) {
            for (const char* setting : {"memory.max", "memory.high"}) {
                std::ifstream limit_file(dir / setting);
                std::string limit;
                if (!(limit_file >> limit)) {
                    // The v2 root has no limit files. Verify the v2 mount before accepting that case.
                    std::error_code ec;
                    if (dir == root && !std::filesystem::exists(dir / setting, ec) && !ec &&
                        std::filesystem::exists(root / "cgroup.controllers", ec) && !ec) continue;
                    return std::nullopt;
                }
                if (limit == "max") continue;
                const auto maximum = quantity(limit);
                std::ifstream usage_file(dir / "memory.current");
                std::string usage;
                if (!maximum || !(usage_file >> usage)) return std::nullopt;
                const auto used = quantity(usage);
                if (!used) return std::nullopt;
                *available = std::min(*available, *used < *maximum ? *maximum - *used : 0);
            }
            if (dir == root) return available;
            dir = dir.parent_path();
        }
    }
    return std::nullopt;
}
inline uint64_t available_bytes() {
#if defined(__linux__)
    return linux_available("/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup").value_or(0);
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile);
#else
    return 0;
#endif
}

}  // namespace strata::core::glm_cache_memory
