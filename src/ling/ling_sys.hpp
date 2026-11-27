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
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winioctl.h>   // (IOCTL_STORAGE_QUERY_PROPERTY: is_rotational_disk; windows.h itself comes with ds4_os.hpp)
#else
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
// (Windows: VirtualAlloc's committed pages, as maya-ds4's RAM copies - page-aligned, the alignment
//  cudaHostRegister pins whole pages of; Windows gives a process large pages only when asked)
inline void* alloc_nohuge(size_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(_WIN32)
    return maya::ds4::os::anon_alloc(bytes, -1, 1, false);
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
    maya::ds4::os::anon_free(p, bytes);
#else
    ::munmap(p, bytes);
#endif
}

// ---- Drop the whole pages of a mapped source range whose bytes were copied out ----
// Linux: MADV_DONTNEED (the private read-only file mapping's pages go; a later touch reads them from the file again).
// Windows: VirtualUnlock trims them from the working set, as maya-ds4's release_copied_cache - clean file-backed
// pages, recoverable by a later fault. Edge pages (shared with a neighbouring tensor) stay. Returns the bytes handed
// to the OS as a hint (0: none - a short range, or the OS refused).
inline size_t drop_copied_pages(const void* p, size_t len) {
    if (p == nullptr) return 0;
#if defined(_WIN32)
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const size_t ps = (size_t) si.dwPageSize;
#elif defined(__linux__)
    const size_t ps = (size_t) ::sysconf(_SC_PAGESIZE);
#else
    const size_t ps = 0;
#endif
    if (ps == 0 || len < ps * 2) return 0;
    const uintptr_t a0 = ((uintptr_t) p + ps - 1) & ~(uintptr_t) (ps - 1);
    const uintptr_t a1 = ((uintptr_t) p + len) & ~(uintptr_t) (ps - 1);
    if (a1 <= a0) return 0;
#if defined(_WIN32)
    SetLastError(ERROR_SUCCESS);
    const BOOL unlocked = VirtualUnlock((void*) a0, (SIZE_T) (a1 - a0));
    // (Microsoft's Working Set documentation: unlocked pages are removed even though the call reports NOT_LOCKED)
    return (unlocked || GetLastError() == ERROR_NOT_LOCKED) ? (size_t) (a1 - a0) : 0;
#elif defined(__linux__)
    return ::madvise((void*) a0, (size_t) (a1 - a0), MADV_DONTNEED) == 0 ? (size_t) (a1 - a0) : 0;
#else
    return 0;
#endif
}

// ---- Current RSS in bytes (/proc/self/statm; Windows: the working set) ----
inline uint64_t process_rss_bytes() {
#if defined(_WIN32)
    return maya::ds4::os::working_set_bytes();
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

#if defined(_WIN32)
// ---- Windows: whether a file's disk has a seek penalty (1: a hard drive, 0: no - an SSD, -1: it cannot be asked) ----
// The file's volume asked through IOCTL_STORAGE_QUERY_PROPERTY's StorageDeviceSeekPenaltyProperty (no administrator
// rights needed); -1 for a network share, a volume over several disks, a driver that does not answer. mount: the
// volume's mount path ("C:\").
inline int seek_penalty(const std::string& path, std::string& mount_name) {
    mount_name.clear();
    const std::wstring wp = maya::ds4::os::wide(path);
    wchar_t mount[MAX_PATH + 1] = {}, vol[MAX_PATH + 1] = {};
    if (!GetVolumePathNameW(wp.c_str(), mount, MAX_PATH)) return -1;
    const int n = WideCharToMultiByte(CP_UTF8, 0, mount, -1, nullptr, 0, nullptr, nullptr);
    if (n > 1) {
        mount_name.assign((size_t) n - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, mount, -1, &mount_name[0], n, nullptr, nullptr);
    }
    if (!GetVolumeNameForVolumeMountPointW(mount, vol, MAX_PATH)) return -1;   // "\\?\Volume{...}\"
    std::wstring dev(vol);
    if (!dev.empty() && dev.back() == L'\\') dev.pop_back();                   // (the volume itself, not its root)
    HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceSeekPenaltyProperty;
    q.QueryType = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR d{};
    DWORD got = 0;
    const BOOL asked = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof q, &d, sizeof d, &got, nullptr);
    CloseHandle(h);
    if (!asked || got < sizeof d) return -1;
    return d.IncursSeekPenalty ? 1 : 0;
}
#endif

// ---- Rotational HDD check for a file path ----
inline bool is_rotational_disk(const std::string& path, std::string& dev_name) {
    dev_name.clear();
#if defined(_WIN32)
    if (seek_penalty(path, dev_name) == 1) return true;
    dev_name.clear();
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
    // (the physical cores the process's affinity allows, as maya-ds4 counts them - GetLogicalProcessorInformationEx,
    //  processor group 0; a CPU without SMT is not halved. One node: Ling pins threads per NUMA node on Linux only,
    //  so a multi-node Windows PC is run as one node, as before)
    const auto nodes = maya::ds4::os::node_core_threads();
    int cores = 0;
    std::vector<int> cpus;
    for (const auto& nd : nodes)
        for (const auto& core : nd) {
            ++cores;
            cpus.insert(cpus.end(), core.cpus.begin(), core.cpus.end());
        }
    const int logical = std::max(1, (int) std::thread::hardware_concurrency());
    if (cores == 0) {   // (the OS did not say: the estimate from before)
        cores = std::max(1, logical / 2);
        cpus.resize((size_t) logical);
        std::iota(cpus.begin(), cpus.end(), 0);
    }
    std::sort(cpus.begin(), cpus.end());
    top.physical_cores = cores;
    top.numa_nodes = 1;
    top.node_ids = {0};
    top.node_core_counts = {cores};
    top.node_cpus[0] = std::move(cpus);
    top.summary = "1 node, " + std::to_string(cores) + " physical core(s)" +
                  (nodes.size() > 1 ? " (" + std::to_string(nodes.size()) + " NUMA nodes seen; not pinned per node on Windows)"
                                    : "");
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
// (Windows: buffered positional reads (ReadFile with an offset, thread-safe) always - maya-ds4's choice there too:
//  the GGUF is also mapped by this process, and Windows' unbuffered reads of a mapped file take a slow path;
//  direct_active() is false and the same bytes come through the system cache)
class DirectGgufReader {
public:
    DirectGgufReader() = default;
    ~DirectGgufReader() { close(); }
    DirectGgufReader(const DirectGgufReader&) = delete;
    DirectGgufReader& operator=(const DirectGgufReader&) = delete;

    bool open(const std::string& path, bool want_direct, std::string& err) {
        return open_impl(path, want_direct, 0, err);
    }

    // Test hooks to exercise open-time and read-time O_DIRECT -> buffered fallback deterministically.
    bool open_direct_refused_for_test(const std::string& path, int simulated_errno, std::string& err) {
        return open_impl(path, true, simulated_errno, err);
    }
#if !defined(_WIN32)
    bool force_direct_read_refusal_for_test() {
        if (!is_open()) return false;
        if (fd_dir_ != maya::ds4::os::kBadFd) {
            maya::ds4::os::direct_forget((int) fd_dir_);
            maya::ds4::os::close_fd(fd_dir_);
            fd_dir_ = maya::ds4::os::kBadFd;
        }
        const int wfd = ::open(path_.c_str(), O_WRONLY);
        if (wfd < 0) return false;
        fd_dir_ = (maya::ds4::os::fd_t) wfd;
        return true;
    }
#endif

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
    bool open_impl(const std::string& path, bool want_direct, int simulated_direct_errno, std::string& err) {
        close();
        path_ = path;
        fd_buf_ = maya::ds4::os::open_ro(path);
        if (fd_buf_ == maya::ds4::os::kBadFd) {
            err = "cannot open GGUF for expert I/O: " + path;
            return false;
        }
#if defined(_WIN32)
        // (buffered by design there - the class comment: no unbuffered handle, and no "refused" line in every log)
        (void) want_direct;
        (void) simulated_direct_errno;
#else
        if (want_direct) {
            if (simulated_direct_errno != 0) {
                errno = simulated_direct_errno;
                fd_dir_ = maya::ds4::os::kBadFd;
            } else {
                fd_dir_ = maya::ds4::os::open_direct(path);
            }
            if (fd_dir_ == maya::ds4::os::kBadFd) {
                std::fprintf(stderr,
                             "maya-ling: O_DIRECT open refused on %s (errno %d: %s) - using buffered page-cache reads\n",
                             path.c_str(), errno, std::strerror(errno));
            }
        }
#endif
        return true;
    }

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
    // (the PC's own name, so two Windows PCs sharing a calibration folder keep their own entries)
    char buf[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD n = (DWORD) sizeof buf;
    if (GetComputerNameA(buf, &n) && n > 0) return std::string(buf, (size_t) n);
    return "windows";
#else
    char buf[128] = {};
    if (::gethostname(buf, sizeof buf - 1) == 0 && buf[0] != 0) return std::string(buf);
    return "linux";
#endif
}

}  // namespace ling::sys
