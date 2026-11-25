// src/ling/ling_sys.hpp - OS hygiene, MADV_NOHUGEPAGE arenas, NVMe O_DIRECT expert I/O with fallback,
// physical-core affinity/NUMA discovery, binary expert profile ("LNGP"), and keyed per-PC calibration store.
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace ling::sys {

// ---- MADV_NOHUGEPAGE anonymous arena (never uses transparent huge pages) ----
inline void* alloc_nohuge(size_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(_WIN32)
    return std::malloc(bytes);
#else
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
#if defined(MADV_NOHUGEPAGE)
    ::madvise(p, bytes, MADV_NOHUGEPAGE);
#endif
    return p;
#endif
}

inline void free_nohuge(void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return;
#if defined(_WIN32)
    std::free(p);
#else
    ::munmap(p, bytes);
#endif
}

// ---- Current RSS in bytes (/proc/self/statm) ----
inline uint64_t process_rss_bytes() {
#if defined(_WIN32)
    return 0;
#else
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long sz = 0, rss = 0;
    const int got = std::fscanf(f, "%llu %llu", &sz, &rss);
    std::fclose(f);
    const long ps = ::sysconf(_SC_PAGESIZE);
    return (got == 2 && ps > 0) ? (uint64_t) rss * (uint64_t) ps : 0;
#endif
}

// ---- Linux systemd-oomd active check ----
inline bool systemd_oomd_active() {
#if defined(_WIN32)
    return false;
#else
    // Fast /proc scan for running systemd-oomd daemon (zero subprocess overhead)
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator("/proc", ec)) {
        const std::string name = entry.path().filename().string();
        if (name.empty() || !std::isdigit((unsigned char) name[0])) continue;
        std::ifstream cf(entry.path() / "comm");
        std::string comm;
        if (std::getline(cf, comm) && comm == "systemd-oomd") return true;
    }
    return false;
#endif
}

// ---- Rotational HDD check for a file path ----
inline bool is_rotational_disk(const std::string& path, std::string& dev_name) {
    dev_name.clear();
#if defined(_WIN32)
    (void) path;
    return false;
#else
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return false;
    const unsigned maj = major(st.st_dev);
    const unsigned min = minor(st.st_dev);
    char linkp[128], resolved[4096];
    std::snprintf(linkp, sizeof linkp, "/sys/dev/block/%u:%u", maj, min);
    const ssize_t n = ::readlink(linkp, resolved, sizeof resolved - 1);
    if (n <= 0) return false;
    resolved[n] = 0;
    // Walk up to parent block device if this is a partition (e.g. .../sda/sda1 -> sda)
    std::filesystem::path bp = std::filesystem::path("/sys/dev/block") / std::filesystem::path(resolved);
    bp = bp.lexically_normal();
    for (int depth = 0; depth < 3; ++depth) {
        std::filesystem::path rot = bp / "queue" / "rotational";
        std::ifstream rf(rot);
        int val = -1;
        if (rf >> val) {
            dev_name = bp.filename().string();
            return val == 1;
        }
        if (!bp.has_parent_path()) break;
        bp = bp.parent_path();
    }
    return false;
#endif
}

// ---- Physical cores honoring process CPU affinity mask + Linux NUMA topology & pinning ----
struct CpuTopology {
    int physical_cores = 1;
    int numa_nodes = 1;
    std::vector<int> node_ids;
    std::vector<int> node_core_counts;
    std::map<int, std::vector<int>> node_cpus;  // allowed logical CPUs per NUMA node
    std::string summary;
};

inline std::vector<int> parse_cpulist(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream ss(text);
    for (std::string part; std::getline(ss, part, ',');) {
        while (!part.empty() && std::isspace((unsigned char) part.back())) part.pop_back();
        if (part.empty()) continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = (dash == std::string::npos) ? a : std::atoi(part.c_str() + dash + 1);
        if (a < 0 || b < a || b - a > 4096) continue;
        for (int c = a; c <= b; ++c) cpus.push_back(c);
    }
    return cpus;
}

inline CpuTopology detect_cpu_topology() {
    CpuTopology top;
#if defined(_WIN32)
    top.physical_cores = std::max(1, (int) std::thread::hardware_concurrency() / 2);
    top.numa_nodes = 1;
    top.node_ids = {0};
    top.node_core_counts = {top.physical_cores};
    top.summary = "1 node, " + std::to_string(top.physical_cores) + " physical cores";
    return top;
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    const bool have_mask = (::sched_getaffinity(0, sizeof mask, &mask) == 0);
    std::set<std::pair<int, int>> seen_phys;
    std::map<int, std::set<std::pair<int, int>>> node_phys;
    std::map<int, std::vector<int>> node_cpus;
    const long ncpu = ::sysconf(_SC_NPROCESSORS_CONF);
    const int lim = (ncpu > 0 && ncpu < 4096) ? (int) ncpu : 256;

    std::map<int, int> cpu_to_node;
    for (int n = 0; n < 64; ++n) {
        std::ifstream nf("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        std::string line;
        if (!nf || !std::getline(nf, line)) continue;
        for (int c : parse_cpulist(line)) cpu_to_node[c] = n;
    }

    for (int cpu = 0; cpu < lim; ++cpu) {
        if (have_mask && !CPU_ISSET(cpu, &mask)) continue;
        char p_core[128], p_pkg[128];
        std::snprintf(p_core, sizeof p_core, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        std::snprintf(p_pkg, sizeof p_pkg, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        int core_id = cpu, pkg_id = 0;
        {
            std::ifstream f(p_core);
            if (!(f >> core_id)) continue;
        }
        {
            std::ifstream f(p_pkg);
            f >> pkg_id;
        }
        auto it = cpu_to_node.find(cpu);
        const int nid = (it != cpu_to_node.end()) ? it->second : 0;
        seen_phys.insert({pkg_id, core_id});
        node_phys[nid].insert({pkg_id, core_id});
        node_cpus[nid].push_back(cpu);
    }
    if (seen_phys.empty()) {
        int cnt = 0;
        if (have_mask) cnt = CPU_COUNT(&mask);
        top.physical_cores = std::max(1, cnt > 0 ? cnt : 1);
        top.numa_nodes = 1;
        top.node_ids = {0};
        top.node_core_counts = {top.physical_cores};
        for (int cpu = 0; cpu < lim; ++cpu)
            if (!have_mask || CPU_ISSET(cpu, &mask)) top.node_cpus[0].push_back(cpu);
    } else {
        top.physical_cores = (int) seen_phys.size();
        top.numa_nodes = std::max(1, (int) node_phys.size());
        top.node_cpus = std::move(node_cpus);
        for (const auto& kv : node_phys) {
            top.node_ids.push_back(kv.first);
            top.node_core_counts.push_back((int) kv.second.size());
        }
    }
    std::ostringstream oss;
    oss << top.numa_nodes << " node(s), " << top.physical_cores << " physical core(s) in affinity mask [";
    for (size_t i = 0; i < top.node_core_counts.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "n" << top.node_ids[i] << ":" << top.node_core_counts[i];
    }
    oss << "]";
    top.summary = oss.str();
    return top;
#endif
}

inline int gpu_numa_node(const char* pci_bus_id) {
#if defined(__linux__)
    if (pci_bus_id == nullptr || pci_bus_id[0] == '\0') return -1;
    std::string bus(pci_bus_id);
    for (char& ch : bus)
        if (ch >= 'A' && ch <= 'Z') ch = (char) (ch - 'A' + 'a');
    std::ifstream f("/sys/bus/pci/devices/" + bus + "/numa_node");
    int nid = -1;
    if (f >> nid && nid >= 0) return nid;
    return -1;
#else
    (void) pci_bus_id;
    return -1;
#endif
}

inline bool build_numa_cpuset(const CpuTopology& top, int node_id, std::vector<int>& out_cpus) {
    out_cpus.clear();
    if (top.numa_nodes <= 1 || node_id < 0) return false;
    auto it = top.node_cpus.find(node_id);
    if (it == top.node_cpus.end() || it->second.empty()) return false;
    out_cpus = it->second;
    return true;
}

inline bool pin_thread_to_numa_node(const CpuTopology& top, int node_id) {
#if defined(__linux__)
    std::vector<int> cpus;
    if (!build_numa_cpuset(top, node_id, cpus)) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    int added = 0;
    for (int c : cpus) {
        if (c >= 0 && c < CPU_SETSIZE) {
            CPU_SET(c, &set);
            ++added;
        }
    }
    if (added == 0) return false;
    return ::pthread_setaffinity_np(::pthread_self(), sizeof set, &set) == 0;
#else
    (void) top;
    (void) node_id;
    return false;
#endif
}

inline int default_cpu_threads(const CpuTopology& top) {
    // Physical cores minus 2 (leave headroom for launch thread + OS), minimum 1
    return std::max(1, top.physical_cores - 2);
}

// ---- NVMe O_DIRECT file reader with automatic buffered fallback & chunk calibration ----
class DirectGgufReader {
public:
    DirectGgufReader() = default;
    ~DirectGgufReader() { close(); }
    DirectGgufReader(const DirectGgufReader&) = delete;
    DirectGgufReader& operator=(const DirectGgufReader&) = delete;

    bool open(const std::string& path, bool want_direct, std::string& err) {
        close();
        path_ = path;
        fd_buf_ = maya::ds4::os::open_ro(path);
        if (fd_buf_ == maya::ds4::os::kBadFd) {
            err = "cannot open GGUF for expert I/O: " + path;
            return false;
        }
        if (want_direct) {
            fd_dir_ = maya::ds4::os::open_direct(path);
            if (fd_dir_ == maya::ds4::os::kBadFd) {
                std::fprintf(stderr,
                             "maya-ling: O_DIRECT open refused on %s (errno %d: %s) - using buffered page-cache reads\n",
                             path.c_str(), errno, std::strerror(errno));
            }
        }
        return true;
    }

    void close() {
        if (fd_dir_ != maya::ds4::os::kBadFd) {
#if !defined(_WIN32)
            maya::ds4::os::direct_forget((int) fd_dir_);
#endif
            maya::ds4::os::close_fd(fd_dir_);
            fd_dir_ = maya::ds4::os::kBadFd;
        }
        if (fd_buf_ != maya::ds4::os::kBadFd) {
            maya::ds4::os::close_fd(fd_buf_);
            fd_buf_ = maya::ds4::os::kBadFd;
        }
    }

    bool is_open() const { return fd_buf_ != maya::ds4::os::kBadFd; }
    bool direct_active() const {
#if defined(_WIN32)
        return false;
#else
        return fd_dir_ != maya::ds4::os::kBadFd && !maya::ds4::os::direct_is_refused((int) fd_dir_);
#endif
    }

    void set_chunk_bytes(size_t cb) {
        chunk_bytes_ = std::max<size_t>(64u << 10, (cb + 4095u) & ~size_t(4095u));
    }
    size_t chunk_bytes() const { return chunk_bytes_; }

    // Calibrate optimal O_DIRECT read chunk size (128 KiB / 256 KiB / 512 KiB / 1 MiB) on first use.
    size_t calibrate_chunk(uint64_t sample_off, size_t sample_len) {
        if (!is_open() || sample_len < (256u << 10)) return chunk_bytes_;
        const size_t cands[4] = {128u << 10, 256u << 10, 512u << 10, 1024u << 10};
        std::vector<uint8_t> dst(sample_len);
        size_t best_c = 512u << 10;
        double best_ms = 1e18;
        for (size_t c : cands) {
            set_chunk_bytes(c);
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = true;
            for (int r = 0; r < 3 && ok; ++r) ok = read_span(sample_off, sample_len, dst.data());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ok && ms < best_ms) {
                best_ms = ms;
                best_c = c;
            }
        }
        set_chunk_bytes(best_c);
        return best_c;
    }

    // Thread-safe exact read of [off, off + n) into dst. Uses O_DIRECT in calibrated chunks with automatic fallback.
    bool read_span(uint64_t off, size_t n, uint8_t* dst) const {
        if (n == 0) return true;
        if (!is_open()) return false;
#if !defined(_WIN32)
        if (fd_dir_ != maya::ds4::os::kBadFd && !maya::ds4::os::direct_is_refused((int) fd_dir_)) {
            thread_local std::vector<uint8_t> aligned_storage;
            size_t done = 0;
            bool ok = true;
            while (done < n) {
                const size_t step = std::min(chunk_bytes_, n - done);
                const uint64_t cur_off = off + done;
                const uint64_t a0 = cur_off & ~(maya::ds4::os::kDirectAlign - 1);
                const uint64_t span = maya::ds4::os::direct_span(cur_off, step);
                if (aligned_storage.size() < span + maya::ds4::os::kDirectAlign) {
                    aligned_storage.resize((size_t) span + maya::ds4::os::kDirectAlign);
                }
                const uintptr_t base = ((uintptr_t) aligned_storage.data() + (maya::ds4::os::kDirectAlign - 1)) &
                                       ~(uintptr_t) (maya::ds4::os::kDirectAlign - 1);
                uint8_t* abuf = (uint8_t*) base;
                if (!maya::ds4::os::pread_direct(fd_dir_, abuf, (uint64_t) step, cur_off)) {
                    ok = false;
                    break;
                }
                std::memcpy(dst + done, abuf + (cur_off - a0), step);
                done += step;
            }
            if (ok) return true;
        }
#endif
        return maya::ds4::os::pread_full(fd_buf_, dst, (uint64_t) n, off);
    }

private:
    std::string path_;
    maya::ds4::os::fd_t fd_buf_ = maya::ds4::os::kBadFd;
    maya::ds4::os::fd_t fd_dir_ = maya::ds4::os::kBadFd;
    size_t chunk_bytes_ = 512u << 10;
};

// ---- Binary expert placement profile ("LNGP") ----
struct ExpertProfile {
    uint32_t n_moe_layers = 0;
    uint32_t n_expert = 0;
    std::vector<uint32_t> counts;  // [n_moe_layers * n_expert]

    bool load(const std::string& path, int expected_moe_layers, int expected_experts, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            err = "cannot open profile: " + path;
            return false;
        }
        char magic[4] = {};
        uint32_t ver = 0, nl = 0, ne = 0;
        f.read(magic, 4);
        f.read((char*) &ver, 4);
        f.read((char*) &nl, 4);
        f.read((char*) &ne, 4);
        if (!f || std::memcmp(magic, "LNGP", 4) != 0 || ver != 1) {
            err = path + ": invalid Ling expert profile header (expected LNGP v1)";
            return false;
        }
        if ((int) ne != expected_experts) {
            err = path + ": profile expert count mismatch (file has " + std::to_string(ne) + ", model has " +
                  std::to_string(expected_experts) + ")";
            return false;
        }
        if ((int) nl < expected_moe_layers) {
            err = path + ": profile layer count mismatch (file has " + std::to_string(nl) + ", model needs " +
                  std::to_string(expected_moe_layers) + ")";
            return false;
        }
        counts.resize((size_t) nl * (size_t) ne);
        if (!f.read((char*) counts.data(), (std::streamsize) (counts.size() * sizeof(uint32_t)))) {
            err = path + ": truncated profile payload";
            return false;
        }
        n_moe_layers = nl;
        n_expert = ne;
        return true;
    }

    bool save(const std::string& path, std::string& err) const {
        if (n_moe_layers == 0 || n_expert == 0 || counts.size() != (size_t) n_moe_layers * n_expert) {
            err = "empty profile";
            return false;
        }
        const std::string tmp = path + ".tmp";
        std::ofstream f(tmp, std::ios::binary);
        if (!f) {
            err = "cannot write profile: " + tmp;
            return false;
        }
        const char magic[4] = {'L', 'N', 'G', 'P'};
        const uint32_t ver = 1;
        f.write(magic, 4);
        f.write((const char*) &ver, 4);
        f.write((const char*) &n_moe_layers, 4);
        f.write((const char*) &n_expert, 4);
        f.write((const char*) counts.data(), (std::streamsize) (counts.size() * sizeof(uint32_t)));
        f.close();
        if (!f) {
            err = "write failed: " + tmp;
            return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            err = "rename failed: " + ec.message();
            return false;
        }
        return true;
    }

    const uint32_t* layer_counts(int moe_idx) const {
        if (moe_idx < 0 || (uint32_t) moe_idx >= n_moe_layers) return nullptr;
        return counts.data() + (size_t) moe_idx * n_expert;
    }
};

// ---- Expert VRAM slot placement planner (shared with model-free regression tests) ----
inline void plan_expert_slots(int n_expert, int n_res, const uint32_t* profile_counts,
                              std::vector<int>& h_slot, std::vector<int>& h_owner,
                              std::vector<float>& h_score, std::vector<int>& ranked) {
    n_expert = std::max(0, n_expert);
    n_res = std::max(0, std::min(n_res, n_expert));
    h_slot.assign((size_t) n_expert, -1);
    h_owner.assign((size_t) std::max(1, n_res), -1);
    h_score.assign((size_t) n_expert, 0.0f);
    ranked.resize((size_t) n_expert);
    std::iota(ranked.begin(), ranked.end(), 0);
    if (profile_counts != nullptr && n_expert > 0) {
        uint32_t maxc = 1;
        for (int e = 0; e < n_expert; ++e) maxc = std::max(maxc, profile_counts[e]);
        for (int e = 0; e < n_expert; ++e) h_score[(size_t) e] = 4.0f * (float) profile_counts[e] / (float) maxc;
        if (n_res < n_expert) {
            std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
                return profile_counts[a] > profile_counts[b];
            });
        }
    }
    for (int sl = 0; sl < n_res; ++sl) {
        const int e = ranked[(size_t) sl];
        h_slot[(size_t) e] = sl;
        h_owner[(size_t) sl] = e;
    }
}

// ---- Keyed per-PC calibration store (key<TAB>k=v ...) ----
// Key includes: host, model, quant, ctx, gpus, driver, kernel_hash.
inline std::string cal_key(const std::string& host, const std::string& model, const std::string& quant, int ctx,
                           const std::string& gpus, const std::string& driver, const std::string& khash) {
    return host + "|" + model + "|" + quant + "|" + std::to_string(ctx) + "|" + gpus + "|" + driver + "|" + khash;
}

inline std::map<std::string, std::string> cal_load(const std::string& path, const std::string& key) {
    std::map<std::string, std::string> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        if (line.substr(0, tab) != key) continue;
        std::istringstream iss(line.substr(tab + 1));
        std::string kv;
        while (iss >> kv) {
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) out[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }
    return out;
}

inline void cal_save(const std::string& path, const std::string& key, const std::map<std::string, std::string>& vals) {
    std::vector<std::string> lines;
    {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            const size_t tab = line.find('\t');
            if (tab != std::string::npos && line.substr(0, tab) == key) continue;
            lines.push_back(line);
        }
    }
    std::ostringstream oss;
    oss << key << '\t';
    bool first = true;
    for (const auto& kv : vals) {
        if (!first) oss << ' ';
        first = false;
        oss << kv.first << '=' << kv.second;
    }
    lines.push_back(oss.str());
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp);
    if (!out) return;
    for (const auto& l : lines) out << l << '\n';
    out.close();
    std::filesystem::rename(tmp, path, ec);
}

inline std::string host_name() {
#if defined(_WIN32)
    return "windows";
#else
    char buf[128] = {};
    if (::gethostname(buf, sizeof buf - 1) == 0 && buf[0] != 0) return std::string(buf);
    return "linux";
#endif
}

}  // namespace ling::sys
