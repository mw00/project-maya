#include "peer_budget.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::peer {
namespace {

// The conservative cache-reclaim allowance: only clean inactive file pages come back (the same
// three counters the engine's copy reads; the structs are spelled out here to keep the daemon
// header-free of strata's includes).
struct CgroupMemoryStat {
    bool valid = false;
    uint64_t current = 0, inactive_file = 0, file_dirty = 0, file_writeback = 0;
};

bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             CgroupMemoryStat& stat) {
    std::ifstream in(path / "memory.stat");
    if (!in) return false;
    stat = {};
    stat.valid = true;
    stat.current = current;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream is(line);
        std::string key;
        uint64_t value = 0;
        if (!(is >> key >> value)) continue;
        if (key == "inactive_file") stat.inactive_file = value;
        else if (key == "file_dirty") stat.file_dirty = value;
        else if (key == "file_writeback") stat.file_writeback = value;
    }
    return true;
}

/// The additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

std::ifstream::pos_type file_size_or_zero(const std::filesystem::path& p) {
    std::error_code ec;
    const auto s = std::filesystem::file_size(p, ec);
    return ec ? 0 : s;
}

}   // namespace

uint64_t available_memory_bytes() {
    uint64_t bytes = 0;
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info("/proc/meminfo");
    if (info) {
        std::string line;
        while (std::getline(info, line)) {
            std::istringstream is(line);
            std::string key, unit;
            uint64_t value = 0;
            if (!(is >> key >> value >> unit) || key != "MemAvailable:" || unit != "kB") continue;
            bytes = value * 1024ULL;
            break;
        }
    }
    if (bytes == 0) return 0;
    // Account for the tightest cgroup-v2 ancestor limit when its normal mount is visible (a
    // container's own limit, not the host's figures).
    std::error_code ec;
    std::ifstream groups("/proc/self/cgroup");
    if (!groups) return bytes;
    std::string line;
    while (std::getline(groups, line)) {
        // "0::/path" (v2, one line in a container); the v1 lines "cpu::/..." are skipped
        const size_t colon = line.find("::");
        if (colon == std::string::npos) continue;
        std::filesystem::path rel = line.substr(colon + 2);
        if (rel.empty() || rel == "/") continue;
        std::filesystem::path root = std::filesystem::path("/sys/fs/cgroup") / rel;
        // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
        // Walking up stops at the first mountpoint boundary, the tightest ancestor first.
        for (std::filesystem::path p = root; p != "/sys/fs/cgroup" && p.has_parent_path(); p = p.parent_path()) {
            std::ifstream max_in(p / "memory.max");
            std::ifstream cur_in(p / "memory.current");
            if (!max_in || !cur_in) continue;               // not a memory-cgroup boundary
            std::string max_s, cur_s;
            if (!(max_in >> max_s) || !(cur_in >> cur_s)) continue;
            if (max_s == "max") break;                      // unbounded: the walk is done
            const uint64_t limit = std::strtoull(max_s.c_str(), nullptr, 10);
            const uint64_t current = std::strtoull(cur_s.c_str(), nullptr, 10);
            CgroupMemoryStat stat;
            if (!read_cgroup_memory_stat(p, current, stat)) break;
            uint64_t cgroup_available = 0;
            if (!cgroup_available_bytes(limit, stat, cgroup_available)) break;
            bytes = std::min(bytes, cgroup_available);
            break;                                          // the tightest visible ancestor decided
        }
        break;                                              // v2 has one line; the walk is done
    }
    return bytes;
}

uint64_t budget_probe(uint64_t cap) {
    const uint64_t avail = available_memory_bytes();
    const uint64_t quarter = avail / 4;
    return std::min(quarter, cap);
}

}   // namespace strata::peer
