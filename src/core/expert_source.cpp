// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/sysmacros.h>   // major(), minor(): the drive probe's cache key
#endif
#endif

namespace strata::core {

namespace detail {

int pick_reads_at_once(const std::vector<int>& counts, const std::vector<double>& mbps, double tol) {
    double best = 0;
    for (size_t i = 0; i < counts.size() && i < mbps.size(); ++i) best = std::max(best, mbps[i]);
    if (best <= 0) return 0;
    for (size_t i = 0; i < counts.size() && i < mbps.size(); ++i)
        if (mbps[i] >= (1.0 - tol) * best) return counts[i];
    return 0;
}

int cached_reads_at_once(const std::string& text, const std::string& key) {
    int n = 0;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        if (line.size() <= key.size() || line.compare(0, key.size(), key) != 0 || line[key.size()] != ' ') continue;
        const size_t at = line.find(" reads=", key.size());
        if (at == std::string::npos) continue;
        const int v = std::atoi(line.c_str() + at + 7);
        if (v > 0) n = v;
    }
    return n;
}

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's inactive_file can race memory.current, so bound it to charged usage first.
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

}  // namespace detail

namespace {
constexpr uint64_t kPinnedMemoryHeadroom = 8ull << 30;

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

bool available_memory_bytes(uint64_t& bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    bytes = (uint64_t) status.ullAvailPhys;
    return bytes > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info("/proc/meminfo");
    std::string line;
    bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup-v2 ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups("/proc/self/cgroup");
    if (!groups) return false;
    bool resolved_v2 = false;
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) continue;
        const std::filesystem::path root("/sys/fs/cgroup");
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        resolved_v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    detail::CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!detail::cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    return resolved_v2;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    bytes = (uint64_t) pages * (uint64_t) page_bytes;
    return bytes > 0;
#endif
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (st.st_size < 0 || (uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) (st.st_size < 0 ? 0 : st.st_size),
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    // The prefetch scratch: one stamp per blob (which epoch last touched it) and staging for one layer's
    // distinct offsets.  Allocated here, at open, because begin_layer runs on the token path (P2.T10).
    prefetch_offsets_.assign((size_t) n_expert, 0);
    prefetch_stamp_.assign((size_t) blob_count, -1);
    prefetch_epoch_ = 0;
    prefetched_ = 0;
    return true;
}

void FileExpertSource::close() {
    if (complement_arena_ != nullptr) {
        if (complement_pinned_) (void) cudaFreeHost(complement_arena_);
        else std::free(complement_arena_);
    }
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_ready_ = false;
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    prefetch_offsets_.clear();
    prefetch_stamp_.clear();
    prefetch_epoch_ = 0;
    prefetched_ = 0;
    reads_ = 0;
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || ids == nullptr || k <= 0) return;
    if ((size_t) layer >= layer_offsets_.size() || (size_t) layer >= layer_blob_bytes_.size()) return;
    const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
    if (blob_bytes == 0) return;
    const uint64_t layer_offset = layer_offsets_[(size_t) layer];

    // The distinct offsets this layer will read OFF THE MAPPING.  `k` counts entries, not distinct experts -
    // the single-token dispatch passes the router's top-k, the multi-token dispatch passes n_tok * k, and a
    // verify window can route every expert of the layer - so dedup is by expert, O(1) per entry through the
    // stamp table.  Complement-resident blobs are skipped: they are host memory, not page-cache pages, and
    // prefetching them would be a syscall against bytes that never fault.
    const int64_t epoch = ++prefetch_epoch_;
    int64_t n_uniq = 0;
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (prefetch_stamp_[index] == epoch) continue;
        prefetch_stamp_[index] = epoch;
        if (complement_ready_ && index < complement_offsets_.size() &&
            complement_offsets_[index] != kNoComplement) continue;
        const uint64_t off = layer_offset + (uint64_t) e * blob_bytes;
        if (off > mapped_bytes_ || blob_bytes > mapped_bytes_ - off) continue;   // mapped_blob's bounds, same rule
        prefetch_offsets_[(size_t) n_uniq++] = off;
    }
    if (n_uniq == 0) return;

#if defined(_WIN32)
    // Resolved at runtime rather than taken from the SDK headers: PrefetchVirtualMemory arrived in Windows 8,
    // and this way the call is simply skipped on whatever Windows cannot do it, whatever the build's
    // _WIN32_WINNT.  WIN32_MEMORY_RANGE_ENTRY is {void*, size_t} - this local struct is layout-identical.
    struct RangeEntry { void* address; size_t bytes; };
    using PrefetchFn = BOOL (WINAPI*)(HANDLE, ULONG_PTR, RangeEntry*, ULONG);
    static const PrefetchFn prefetch = []() -> PrefetchFn {
        const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        return kernel32 ? reinterpret_cast<PrefetchFn>(GetProcAddress(kernel32, "PrefetchVirtualMemory"))
                        : nullptr;
    }();
    if (prefetch == nullptr) return;
    RangeEntry batch[64];
    for (int64_t at = 0; at < n_uniq; at += 64) {
        const int n = (int) std::min<int64_t>(64, n_uniq - at);
        for (int i = 0; i < n; ++i) {
            batch[i].address = const_cast<uint8_t*>(base_ + prefetch_offsets_[(size_t) (at + i)]);
            batch[i].bytes = (size_t) blob_bytes;
        }
        // Fire and forget, best-effort by contract: the API queues the reads and returns, and FALSE means the
        // pages fault at first touch instead - the behaviour without this call, not an error to act on.  The
        // counter says what was QUEUED, which is the only thing deterministic about an advisory prefetch.
        (void) prefetch(GetCurrentProcess(), (ULONG_PTR) n, batch, 0);
    }
    prefetched_ += n_uniq;
#else
#ifdef POSIX_FADV_WILLNEED
    if (fd_ < 0) return;
    for (int64_t i = 0; i < n_uniq; ++i)
        (void) ::posix_fadvise(fd_, (off_t) prefetch_offsets_[(size_t) i], (off_t) blob_bytes, POSIX_FADV_WILLNEED);
    prefetched_ += n_uniq;
#endif
#endif
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs) {
    err.clear();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    primary_gpu_pairs.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            if (cache.slot_of(layer, expert) != kNotResident)
                primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;

    if (bytes > 0) {
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        if (physical <= kPinnedMemoryHeadroom || bytes > physical - kPinnedMemoryHeadroom) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM minus the 8 GiB safety headroom",
                          (double) bytes / 1073741824.0);
            err = message;
            return false;
        }
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "mapped pinned" : "pageable resident");
        std::fflush(stderr);
        if (pin) {
            const cudaError_t allocated = cudaHostAlloc(&arena, (size_t) bytes,
                                                         cudaHostAllocMapped | cudaHostAllocPortable);
            if (allocated != cudaSuccess) {
                err = std::string("FileExpertSource: mapped pinned complement allocation failed: ") +
                      cudaGetErrorString(allocated);
                std::fprintf(stderr, "%s\n", err.c_str());
                std::fflush(stderr);
                (void) cudaGetLastError();
                return false;
            }
            host = (const uint8_t*) arena;
            void* alias = nullptr;
            const cudaError_t aliased = cudaHostGetDevicePointer(&alias, arena, 0);
            if (aliased != cudaSuccess || alias == nullptr) {
                err = std::string("FileExpertSource: mapped pinned complement device alias failed: ") +
                      cudaGetErrorString(aliased);
                std::fprintf(stderr, "%s\n", err.c_str());
                std::fflush(stderr);
                (void) cudaGetLastError();
                (void) cudaFreeHost(arena);
                return false;
            }
            device = (const uint8_t*) alias;
        } else {
            arena = std::malloc((size_t) bytes);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            host = (const uint8_t*) arena;
        }
    }

    uint64_t copied = 0;
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
            const uint64_t offset = offsets[index];
            if (offset == kNoComplement) continue;
            const uint8_t* source = mapped_blob(layer, expert);
            if (source == nullptr || offset > bytes || blob_bytes > bytes - offset) {
                err = "FileExpertSource: invalid blob bounds while building the cache complement";
                if (arena != nullptr) {
                    if (pin) (void) cudaFreeHost(arena);
                    else std::free(arena);
                }
                return false;
            }
            std::memcpy((uint8_t*) host + (size_t) offset, source, (size_t) blob_bytes);
            copied += blob_bytes;
        }
#if !defined(_WIN32)
        const uint64_t layer_offset = layer_offsets_[(size_t) layer];
        const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
        const uint64_t layer_end = layer_offset + layer_bytes;
        const long page_bytes = sysconf(_SC_PAGESIZE);
        if (page_bytes <= 0) {
            err = "FileExpertSource: cannot determine page size for mapped-page release";
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
        const uint64_t page = (uint64_t) page_bytes;
        const uint64_t advice_start = layer_offset - layer_offset % page;
        const uint64_t end_remainder = layer_end % page;
        const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
        const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
        if (advice_end > advice_start &&
            madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
            err = "FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer);
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
        const int advice = posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED);
        if (advice != 0) {
            err = "FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer);
            if (arena != nullptr) {
                if (pin) (void) cudaFreeHost(arena);
                else std::free(arena);
            }
            return false;
        }
#endif
        if ((layer + 1) % 8 == 0 || layer + 1 == n_layers_) {
            std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB)\n",
                         (long long) (layer + 1), (long long) n_layers_, (double) copied / 1073741824.0);
            std::fflush(stderr);
        }
    }

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = pin && bytes > 0;
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB\n",
                 complement_pinned_ ? "mapped pinned" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0);
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* mapped_fallback = mapped_blob(layer, expert);
    const uint8_t* result = complement_ready_
        ? detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, mapped_fallback)
        : mapped_fallback;
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return complement_device_ + (size_t) complement_offsets_[index];
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[128];                     // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= 128 && n <= d.plan->cap) {
        int64_t distinct[128], first_of[128];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->device_alias(d.layers, 0) != nullptr;
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->blob(d.layers, e);
                        if (src != nullptr && d.src->pinned(d.layers, e)) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42 && q2_own_kernels())   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers);
                // (an expert staged from the model file: its ring slot is reused, so the copy is waited for - a
                //  queued one could read it after it was overwritten where the runtime defers queued copies)
                const bool staged = !d.src->stable(d.layers, e);
                if (b == nullptr || !(staged ? d.cache->fill_slot_blocking(cand, b, ferr, nb)
                                             : d.cache->fill_slot(cand, b, cs, ferr, nb))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    // a layer's experts may sit in another shard of the model (native_experts.txt v3): a name beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) l];
    };
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = file_of(l);
            if (name != open_name) {
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                if (!f) { bad = true; return; }
                open_name = name;
            }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, uint64_t max_pinned_bytes) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    PinnedArena* a = new PinnedArena(want + (uint64_t) blob, bounds, max_pinned_bytes);
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    const LoadStats st = from_gguf ? load_experts_gguf(gguf_, a->data(), lay, threads)
                                   : load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    return true;
}

void ArenaExpertSource::close() {
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

// ================================ THE MODEL FILE ITSELF (v1.1.32) ================================

namespace {
uint64_t total_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t) status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
#endif
}

#if defined(__linux__)
// How many unbuffered reads at once the drive under the model file serves fastest - the stagers' count when
// MAYA_EXPERT_STAGERS names none.  It belongs to the drive, not the CPU: an NVMe reads ~10% faster with 8 slices in
// flight than with 3, a USB disk gains nothing past 2.  Measured once (~1 s: 2, 4, 8 and 16 threads reading
// `piece`-sized slices at random places of the file, twice over, interleaved) and cached per drive in
// <XDG_CONFIG_HOME or ~/.config>/project-maya/drive-reads.txt; MAYA_EXPERT_DRIVE_PROBE=0 skips it.  0: not measured
// (the caller keeps its own count).
int drive_reads_at_once(int dfd, uint64_t file_bytes, uint64_t piece, std::string& said) {
    const char* off = std::getenv("MAYA_EXPERT_DRIVE_PROBE");
    if ((off != nullptr && std::atoi(off) == 0) || dfd < 0) return 0;
    struct stat st{};
    if (::fstat(dfd, &st) != 0) return 0;
    piece = std::max<uint64_t>(65536, (piece + 4095) & ~uint64_t{4095});
    if (file_bytes < 64 * piece) return 0;
    // the drive: its device numbers, the block device's name and size (a renumbered disk is measured again)
    const unsigned mj = major(st.st_dev), mn = minor(st.st_dev);
    const std::string sys = "/sys/dev/block/" + std::to_string(mj) + ":" + std::to_string(mn);
    std::error_code ec;
    std::string name = std::filesystem::canonical(sys, ec).filename().string(), size;
    if (ec || name.empty()) name = "dev";
    {
        std::ifstream f(sys + "/size");
        std::getline(f, size);
    }
    const std::string key = name + ":" + std::to_string(mj) + ":" + std::to_string(mn) + ":" + (size.empty() ? "0" : size);
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    const char* home = std::getenv("HOME");
    if ((xdg == nullptr || !*xdg) && (home == nullptr || !*home)) return 0;
    const std::filesystem::path dir = (xdg != nullptr && *xdg) ? std::filesystem::path(xdg) / "project-maya"
                                                                : std::filesystem::path(home) / ".config" / "project-maya";
    const std::filesystem::path cache = dir / "drive-reads.txt";
    std::string text;
    {
        std::ifstream f(cache);
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    if (const int n = detail::cached_reads_at_once(text, key); n > 0) {
        said = std::to_string(n) + " reads at once on " + name + " (cached in " + cache.string() + ")";
        return n;
    }
    const std::vector<int> counts = {2, 4, 8, 16};
    std::vector<double> mbps(counts.size(), 0.0);
    const uint64_t slots = (file_bytes - piece) / 4096;
    bool failed = false;
    for (int round = 0; round < 2 && !failed; ++round)
        for (size_t k = 0; k < counts.size() && !failed; ++k) {
            std::atomic<uint64_t> bytes{0};
            std::atomic<bool> bad{false};
            const auto t0 = std::chrono::steady_clock::now();
            const auto until = t0 + std::chrono::milliseconds(120);
            std::vector<std::thread> th;
            for (int t = 0; t < counts[k]; ++t)
                th.emplace_back([&, t] {
                    void* buf = nullptr;
                    if (posix_memalign(&buf, 4096, (size_t) piece) != 0) { bad = true; return; }
                    uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t) (1 + t + 64 * (round * 8 + (int) k));
                    do {   // (each thread's first read always runs: a slow disk still gives every count a reading)
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        const ssize_t r = ::pread(dfd, buf, (size_t) piece, (off_t) ((x % slots) * 4096));
                        if (r != (ssize_t) piece) { bad = true; break; }
                        bytes += (uint64_t) r;
                    } while (std::chrono::steady_clock::now() < until);
                    std::free(buf);
                });
            for (auto& x : th) x.join();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (bad) failed = true;
            else if (s > 0) mbps[k] = std::max(mbps[k], (double) bytes.load() / 1e6 / s);
        }
    const int n = failed ? 0 : detail::pick_reads_at_once(counts, mbps);
    if (n <= 0) return 0;
    char line[256];
    std::snprintf(line, sizeof line, "%s reads=%d mbps=%.0f/%.0f/%.0f/%.0f piece=%llu\n", key.c_str(), n, mbps[0], mbps[1],
                  mbps[2], mbps[3], (unsigned long long) piece);
    std::filesystem::create_directories(dir, ec);
    if (std::FILE* f = std::fopen(cache.string().c_str(), "a")) {
        std::fputs(line, f);
        std::fclose(f);
    }
    char b[200];
    std::snprintf(b, sizeof b, "%d reads at once on %s (measured %.0f/%.0f/%.0f/%.0f MB/s at 2/4/8/16)", n, name.c_str(),
                  mbps[0], mbps[1], mbps[2], mbps[3]);
    said = b;
    return n;
}
#endif
}  // namespace

uint64_t expert_ram_room(uint64_t* avail_out, uint64_t* headroom_out, bool unbuffered) {
    uint64_t avail = 0;
    const uint64_t total = total_memory_bytes();
    if (!available_memory_bytes(avail)) avail = total / 2;   // (no reading of it: assume half the RAM is free)
    // the file cache the other experts are read through needs this much; read unbuffered (O_DIRECT) they need none, and
    // the OS, the server and the engine's own buffers are what stays free
    uint64_t headroom = unbuffered ? std::max<uint64_t>(4ull << 30, total / 100 * 8)
                                   : std::max<uint64_t>(6ull << 30, total / 100 * 12);
    if (const char* v = std::getenv("MAYA_EXPERT_HEADROOM_GB")) {   // (a measurement knob)
        const double gb = std::atof(v);
        if (gb >= 0.0) headroom = (uint64_t) (gb * 1073741824.0);
    }
    if (avail_out) *avail_out = avail;
    if (headroom_out) *headroom_out = headroom;
    return avail > headroom ? avail - headroom : 0;
}

GgufExpertSource::~GgufExpertSource() { close(); }

struct GgufExpertSource::Stage {
    struct Task { int64_t slot, layer, expert; int part; };   ///< part: 0..2 one slice, -1 the whole expert
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Task> queue;
    std::vector<std::thread> threads;
    std::unique_ptr<std::atomic<int>[]> busy;   ///< per ring slot: the slices (1: whole) a stager still has to write
    bool quit = false;
};

void GgufExpertSource::stage_worker() {
    for (;;) {
        Stage::Task t{};
        {
            std::unique_lock<std::mutex> lk(st_->mu);
            st_->cv.wait(lk, [&] { return st_->quit || !st_->queue.empty(); });
            if (st_->quit && st_->queue.empty()) return;
            t = st_->queue.front();
            st_->queue.pop_front();
        }
        uint8_t* dst = ring_.data() + (size_t) t.slot * ring_stride_;
        if (t.part >= 0) assemble_part(t.layer, t.expert, t.part, dst);
        else assemble(t.layer, t.expert, dst);
        st_->busy[(size_t) t.slot].fetch_sub(1, std::memory_order_release);
    }
}

int64_t GgufExpertSource::take_slot(size_t pair) {
    const int64_t s = ring_count_ % ring_n_;
    if (st_ != nullptr)   // (a stager still on it - from rounds ago, so this does not wait in practice)
        while (st_->busy[(size_t) s].load(std::memory_order_acquire) != 0) std::this_thread::yield();
    if (ring_owner_[(size_t) s] >= 0) ring_of_[(size_t) ring_owner_[(size_t) s]] = -1;
    ring_owner_[(size_t) s] = (int64_t) pair;
    ring_stamp_[(size_t) s] = ring_count_;
    ring_of_[pair] = (int32_t) s;
    ++ring_count_;
    ++staged_;
    return s;
}

void GgufExpertSource::close() {
    if (st_ != nullptr) {
        {
            std::lock_guard<std::mutex> lk(st_->mu);
            st_->quit = true;
        }
        st_->cv.notify_all();
        for (auto& t : st_->threads) t.join();
        delete st_;
        st_ = nullptr;
    }
    gpu_res_ = nullptr;
    if (arena_ != nullptr) delete (PinnedArena*) arena_;
    arena_ = nullptr;
    res_base_ = nullptr;
    res_off_.clear();
    res_layer_start_.clear();
    dev_slice_.clear();
    registered_bytes_ = 0;
    sliced_ = false;
    resident_blobs_ = 0;
    resident_bytes_ = pinned_bytes_ = 0;
    for (Map& m : maps_) {
#if defined(_WIN32)
        if (m.base != nullptr) UnmapViewOfFile((LPCVOID) m.base);
        if (m.mapping != nullptr) CloseHandle((HANDLE) m.mapping);
        if (m.file != nullptr) CloseHandle((HANDLE) m.file);
#else
        if (m.base != nullptr) munmap((void*) m.base, (size_t) m.bytes);
        if (m.fd >= 0) ::close(m.fd);
        if (m.dfd >= 0) ::close(m.dfd);
#endif
    }
    maps_.clear();
    map_of_.clear();
    ring_.clear();
    ring_of_.clear();
    ring_owner_.clear();
    ring_stamp_.clear();
    pf_stamp_.clear();
    ring_n_ = ring_count_ = per_layer_ = 0;
    n_layers_ = n_expert_ = 0;
    reads_ = staged_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, int64_t per_layer,
                            std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.n_layers != n_layers || lay.n_expert != n_expert || n_layers <= 0 || n_expert <= 0 ||
        lay.gguf_off.size() != (size_t) (3 * n_layers) || lay.fmt.size() != (size_t) n_layers) {
        err = "GgufExpertSource: the experts' GGUF offsets are not in the pack (re-pack it with tools/iq_pack.py)";
        return false;
    }
    if (per_layer < 1) { err = "GgufExpertSource: the dispatch width is empty"; return false; }
    const size_t cut = shard1.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    map_of_.assign((size_t) n_layers, -1);
    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string path = lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()
                                     ? shard1 : dir + lay.gguf_file[(size_t) l];
        int at = -1;
        for (size_t i = 0; i < maps_.size(); ++i) if (maps_[i].path == path) at = (int) i;
        if (at < 0) {
            Map m;
            m.path = path;
#if defined(_WIN32)
            const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
            std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
            if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
            // no FILE_FLAG_RANDOM_ACCESS: retention is the whole game (see FileExpertSource::open)
            HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
            if (f == INVALID_HANDLE_VALUE) { err = "GgufExpertSource: cannot open " + path; close(); return false; }
            LARGE_INTEGER sz{};
            HANDLE h = GetFileSizeEx(f, &sz) ? CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
            void* view = h ? MapViewOfFile(h, FILE_MAP_READ, 0, 0, 0) : nullptr;
            if (view == nullptr) {
                if (h) CloseHandle(h);
                CloseHandle(f);
                err = "GgufExpertSource: cannot map " + path;
                close();
                return false;
            }
            m.file = f;
            m.mapping = h;
            m.base = (const uint8_t*) view;
            m.bytes = (uint64_t) sz.QuadPart;
#else
            const int fd = ::open(path.c_str(), O_RDONLY);
            struct stat st{};
            if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
                if (fd >= 0) ::close(fd);
                err = "GgufExpertSource: cannot open " + path;
                close();
                return false;
            }
            void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (view == MAP_FAILED) { ::close(fd); err = "GgufExpertSource: cannot map " + path; close(); return false; }
            m.fd = fd;
            m.base = (const uint8_t*) view;
            m.bytes = (uint64_t) st.st_size;
#if defined(__linux__) && defined(O_DIRECT)
            // the misses are read unbuffered, a slice per read, many at once: through the mapping a miss is a run of
            // 4 KB page faults and the file cache (small beside a model this size) churns the RAM the experts need.
            // MAYA_EXPERT_DIRECT=0 turns it off; a file system that refuses O_DIRECT keeps the mapping.
            const char* dv = std::getenv("MAYA_EXPERT_DIRECT");
            if (dv == nullptr || std::atoi(dv) != 0) {
                const int dfd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
                void* probe = nullptr;
                if (dfd >= 0 && posix_memalign(&probe, 4096, 4096) == 0) {
                    if (::pread(dfd, probe, 4096, 0) == 4096) m.dfd = dfd;
                    else ::close(dfd);
                    std::free(probe);
                } else if (dfd >= 0) {
                    ::close(dfd);
                }
            }
#endif
#endif
            maps_.push_back(m);
            at = (int) maps_.size() - 1;
        }
        map_of_[(size_t) l] = at;
        // every slice of this layer inside its file: refused at open rather than faulting at token 4000
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t blob = lay.bytes[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = lay.gguf_off[(size_t) (3 * l + r)], need = per[r] * (uint64_t) n_expert;
            if (off == 0 || per[r] == 0 || off > maps_[(size_t) at].bytes || need > maps_[(size_t) at].bytes - off) {
                err = "GgufExpertSource: layer " + std::to_string(l) + "'s experts lie outside " + path +
                      " - this is not the file the pack was made from";
                close();
                return false;
            }
        }
    }
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    per_layer_ = std::min<int64_t>(per_layer, n_expert);
    ring_n_ = 2 * per_layer_ + 8;
    ring_stride_ = ((uint64_t) lay.max_blob + 4095) & ~uint64_t{4095};
    ring_.assign((size_t) (ring_stride_ * (uint64_t) ring_n_), 0);
    ring_owner_.assign((size_t) ring_n_, -1);
    ring_stamp_.assign((size_t) ring_n_, -1);
    ring_of_.assign((size_t) (n_layers * n_expert), -1);
    res_off_.assign((size_t) (n_layers * n_expert), kNone);
    pf_stamp_.assign((size_t) (n_layers * n_expert), -1);
    // the stagers: a quarter of the hardware threads, 2 to 8 (MAYA_EXPERT_STAGERS sets it; 0: blob() assembles alone) -
    // or, where the misses are read unbuffered (Linux), the reads at once the drive serves fastest (measured once a drive)
    int stagers = std::clamp((int) std::thread::hardware_concurrency() / 4, 2, 8);
    std::string drive_said;
    if (const char* v = std::getenv("MAYA_EXPERT_STAGERS")) {
        stagers = std::clamp(std::atoi(v), 0, 64);
    } else {
#if defined(__linux__)
        const Map* big = nullptr;
        bool all_direct = !maps_.empty();
        for (const Map& m : maps_) {
            all_direct = all_direct && m.dfd >= 0;
            if (big == nullptr || m.bytes > big->bytes) big = &m;
        }
        if (all_direct && big != nullptr) {
            const int n = drive_reads_at_once(big->dfd, big->bytes, (uint64_t) lay.max_blob / 3, drive_said);
            if (n > 0) stagers = n;
        }
#endif
    }
    if (stagers > 0) {
        st_ = new Stage();
        st_->busy.reset(new std::atomic<int>[(size_t) ring_n_]);
        for (int64_t i = 0; i < ring_n_; ++i) st_->busy[(size_t) i].store(0);
        for (int i = 0; i < stagers; ++i) st_->threads.emplace_back([this] { stage_worker(); });
    }
    direct_ = st_ != nullptr && !maps_.empty();
#if !defined(_WIN32)
    for (const Map& m : maps_) if (m.dfd < 0) direct_ = false;
#else
    direct_ = false;
#endif
    if (direct_) {   // a sample of experts read both ways must agree, or the mapping stays the way the misses are read
        const auto& lay2 = strata::kernels::cpu::expert_layout();
        std::vector<uint8_t> tmp((size_t) lay2.max_blob + 4096);
        const int64_t sample[3][2] = {{0, 0}, {n_layers - 1, n_expert - 1}, {n_layers / 2, n_expert / 3}};
        for (const auto& pr : sample) {
            const uint8_t* src[3];
            uint64_t bytes[3], off[3];
            parts(pr[0], pr[1], src, bytes, off);
            const auto& fm = lay2.fmt[(size_t) pr[0]];
            for (int r = 0; r < 3 && direct_; ++r) {
                const uint64_t at = r == 0 ? 0 : r == 1 ? fm.up_off : fm.down_off;
                if (!read_part_direct(pr[0], pr[1], r, tmp.data()) || std::memcmp(tmp.data() + at, src[r], (size_t) bytes[r]) != 0)
                    direct_ = false;
            }
        }
        if (!direct_) std::fprintf(stderr, "GgufExpertSource: the unbuffered read of a sample expert differs from the mapping - the misses are read through the mapping\n");
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "experts read from the model file (%zu mapped), a staging ring of %lld (%.0f MiB), "
                  "%d stagers%s", maps_.size(), (long long) ring_n_, (double) ring_.size() / 1048576.0, stagers,
                  direct_ ? ", unbuffered slice reads" : "");
    note_ = buf;
    if (!drive_said.empty()) note_ += " - " + drive_said;
    return true;
}

void GgufExpertSource::parts(int64_t layer, int64_t expert, const uint8_t* src[3], uint64_t bytes[3],
                             uint64_t file_off[3]) const {
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    bytes[0] = fm.up_off;
    bytes[1] = fm.up_off;
    bytes[2] = blob - fm.down_off;
    const Map& m = maps_[(size_t) map_of_[(size_t) layer]];
    for (int r = 0; r < 3; ++r) {
        file_off[r] = lay.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * bytes[r];
        src[r] = m.base + file_off[r];
    }
}

void GgufExpertSource::assemble(int64_t layer, int64_t expert, uint8_t* dst) const {
    const uint8_t* src[3];
    uint64_t bytes[3], off[3];
    parts(layer, expert, src, bytes, off);
    const auto& fm = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    std::memcpy(dst, src[0], (size_t) bytes[0]);
    std::memcpy(dst + fm.up_off, src[1], (size_t) bytes[1]);
    std::memcpy(dst + fm.down_off, src[2], (size_t) bytes[2]);
}

bool GgufExpertSource::read_part_direct(int64_t layer, int64_t expert, int part, uint8_t* dst) const {
#if defined(__linux__)
    const Map& m = maps_[(size_t) map_of_[(size_t) layer]];
    if (m.dfd < 0) return false;
    const uint8_t* src[3];
    uint64_t bytes[3], off[3];
    parts(layer, expert, src, bytes, off);
    const auto& fm = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    uint8_t* to = part == 0 ? dst : part == 1 ? dst + fm.up_off : dst + fm.down_off;
    // the read is rounded out to 4 KB (O_DIRECT) into this thread's bounce buffer, then the slice copied out
    struct Bounce {
        uint8_t* p = nullptr;
        size_t cap = 0;
        ~Bounce() { std::free(p); }
    };
    static thread_local Bounce bn;
    const uint64_t a0 = off[part] & ~uint64_t{4095}, head = off[part] - a0, need = head + bytes[part];
    const uint64_t len = (need + 4095) & ~uint64_t{4095};
    if (bn.cap < len) {
        std::free(bn.p);
        bn.p = nullptr;
        bn.cap = 0;
        void* q = nullptr;
        if (posix_memalign(&q, 4096, (size_t) len + (1u << 20)) != 0) return false;
        bn.p = (uint8_t*) q;
        bn.cap = (size_t) len + (1u << 20);
    }
    uint64_t got = 0;
    while (got < need) {
        const ssize_t r = ::pread(m.dfd, bn.p + got, (size_t) (len - got), (off_t) (a0 + got));
        if (r <= 0) return false;
        got += (uint64_t) r;
        if (got < need && (got & 4095) != 0) return false;
    }
    std::memcpy(to, bn.p + head, (size_t) bytes[part]);
    return true;
#else
    (void) layer; (void) expert; (void) part; (void) dst;
    return false;
#endif
}

void GgufExpertSource::assemble_part(int64_t layer, int64_t expert, int part, uint8_t* dst) const {
    if (read_part_direct(layer, expert, part, dst)) return;
    const uint8_t* src[3];
    uint64_t bytes[3], off[3];
    parts(layer, expert, src, bytes, off);
    const auto& fm = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    std::memcpy(part == 0 ? dst : part == 1 ? dst + fm.up_off : dst + fm.down_off, src[part], (size_t) bytes[part]);
}

bool GgufExpertSource::make_resident(const std::vector<std::pair<int32_t, int32_t>>& order, uint64_t budget,
                                     uint64_t max_pinned, int threads,
                                     const std::vector<std::pair<int32_t, int32_t>>& drop, std::string& err) {
    if (maps_.empty()) { err = "GgufExpertSource: not open"; return false; }
    if (arena_ != nullptr) { err = "GgufExpertSource: the resident arena is already made"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    // ---- the budget: this PC's available RAM less a headroom for the OS and the page cache the other experts and
    // the n-gram rows are read through (12% of the RAM, at least 6 GiB); MAYA_EXPERT_RAM_GB caps it
    uint64_t avail = 0, headroom = 0;
    uint64_t room = expert_ram_room(&avail, &headroom, direct_);
    if (budget > 0) room = std::min(room, budget);
    if (const char* v = std::getenv("MAYA_EXPERT_RAM_GB")) {
        const double gb = std::atof(v);
        if (gb >= 0.0) room = std::min<uint64_t>(room, (uint64_t) (gb * 1073741824.0));
    }
    // ---- the pairs, in order, while they fit; then laid out grouped by layer (one registration slice per layer).  A
    // reservation the OS refuses (Windows' commit limit: RAM + page file) is tried again at 3/4, 1/2, 1/4 of the room,
    // then none - the experts are then all read from the file, slower, but the engine runs.
    std::vector<char> take((size_t) (n_layers_ * n_expert_), 0);
    std::vector<uint64_t> bounds;
    uint64_t at = 0;
    int64_t n = 0;
    PinnedArena* a = nullptr;
    std::string refused;
    for (const uint64_t quarter : {4u, 3u, 2u, 1u, 0u}) {
        const uint64_t limit = room / 4 * quarter;
        std::fill(take.begin(), take.end(), 0);
        std::fill(res_off_.begin(), res_off_.end(), kNone);
        uint64_t want = 0;
        n = 0;
        for (const auto& [l, e] : order) {
            if (!valid(l, e)) continue;
            const size_t i = (size_t) l * (size_t) n_expert_ + (size_t) e;
            if (take[i]) continue;
            const uint64_t b = lay.bytes[(size_t) l];
            if (want + b > limit) break;
            take[i] = 1;
            want += b;
            ++n;
        }
        bounds.clear();
        res_layer_start_.assign((size_t) n_layers_ + 1, 0);
        at = 0;
        for (int64_t l = 0; l < n_layers_; ++l) {
            res_layer_start_[(size_t) l] = at;
            bool any = false;
            for (int64_t e = 0; e < n_expert_; ++e) {
                const size_t i = (size_t) l * (size_t) n_expert_ + (size_t) e;
                if (!take[i]) continue;
                if (!any) bounds.push_back(at);
                any = true;
                res_off_[i] = at;
                at += lay.bytes[(size_t) l];
            }
        }
        res_layer_start_[(size_t) n_layers_] = at;
        bounds.push_back(at);
        if (at == 0) break;
        a = new PinnedArena(at + (uint64_t) lay.max_blob, bounds, max_pinned);
        if (a->valid()) break;
        delete a;
        a = nullptr;
        (void) cudaGetLastError();
        refused += (refused.empty() ? "" : ", ") + std::to_string(at >> 20) + " MiB";
    }
    if (a == nullptr) {   // (none at all: every expert the GPUs do not hold is read from the file)
        std::fill(res_off_.begin(), res_off_.end(), kNone);
        at = 0;
        n = 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (at > 0) {
        arena_ = a;
        res_base_ = a->data();
        registered_bytes_ = a->registered_bytes;
        sliced_ = a->slice_bytes > 0;
        if (a->registered_bytes > 0) {
            const std::vector<uint64_t> starts = sliced_ ? a->slice_starts : std::vector<uint64_t>{0};
            for (uint64_t off : starts) {
                void* d = nullptr;
                if (cudaHostGetDevicePointer(&d, (void*) (res_base_ + off), 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    dev_slice_.clear();
                    break;
                }
                dev_slice_.push_back((const uint8_t*) d);
            }
        }
        // ---- the copy, a layer per thread at a time (page faults on the mapping read the SSD)
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const int64_t l = next.fetch_add(1);
                if (l >= n_layers_) break;
                for (int64_t e = 0; e < n_expert_; ++e) {
                    const size_t i = (size_t) l * (size_t) n_expert_ + (size_t) e;
                    if (res_off_[i] == kNone) continue;
                    assemble(l, e, (uint8_t*) res_base_ + res_off_[i]);
#if !defined(_WIN32)
                    // the file pages are not needed again: give them back to the page cache's other users
                    const uint8_t* src[3];
                    uint64_t bytes[3], off[3];
                    parts(l, e, src, bytes, off);
                    const int fd = maps_[(size_t) map_of_[(size_t) l]].fd;
                    for (int r = 0; r < 3; ++r) (void) ::posix_fadvise(fd, (off_t) off[r], (off_t) bytes[r], POSIX_FADV_DONTNEED);
#endif
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < std::max(1, threads); ++t) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
#if !defined(_WIN32)
    for (const auto& [l, e] : drop) {
        if (!valid(l, e)) continue;
        const uint8_t* src[3];
        uint64_t bytes[3], off[3];
        parts(l, e, src, bytes, off);
        const int fd = maps_[(size_t) map_of_[(size_t) l]].fd;
        for (int r = 0; r < 3; ++r) (void) ::posix_fadvise(fd, (off_t) off[r], (off_t) bytes[r], POSIX_FADV_DONTNEED);
    }
#else
    (void) drop;
#endif
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    resident_blobs_ = n;
    resident_bytes_ = at;
    pinned_bytes_ = std::min(registered_bytes_, at);
    uint64_t all = 0;
    for (int64_t l = 0; l < n_layers_; ++l) all += lay.bytes[(size_t) l] * (uint64_t) n_expert_;
    all_bytes_ = all;
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "%lld experts (%.2f GiB of %.2f) resident in RAM (%.2f GiB pinned), the rest read from the model "
                  "file when routed; RAM available %.1f GiB, kept free %.1f GiB; %s",
                  (long long) n, (double) at / 1073741824.0, (double) all / 1073741824.0,
                  (double) pinned_bytes_ / 1073741824.0, (double) avail / 1073741824.0,
                  (double) headroom / 1073741824.0, arena_ ? ((PinnedArena*) arena_)->note.c_str() : "no arena");
    note_ = buf;
    if (!refused.empty()) note_ += "; the OS refused larger RAM tiers (" + refused + ")";
    return true;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (maps_.empty() || !valid(layer, expert)) return nullptr;
    ++reads_;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (res_off_[i] != kNone) return res_base_ + res_off_[i];
    // a ring slot still holding it (or a stager filling it, begin_layer's) - reused only while it cannot be refilled
    // before this layer is done
    const int32_t s0 = ring_of_[i];
    if (s0 >= 0 && ring_owner_[(size_t) s0] == (int64_t) i && ring_count_ - ring_stamp_[(size_t) s0] < ring_n_ - per_layer_) {
        if (st_ != nullptr)
            while (st_->busy[(size_t) s0].load(std::memory_order_acquire) != 0) std::this_thread::yield();
        return ring_.data() + (size_t) s0 * ring_stride_;
    }
    uint8_t* dst = ring_.data() + (size_t) take_slot(i) * ring_stride_;
    if (direct_) for (int r = 0; r < 3; ++r) assemble_part(layer, expert, r, dst);
    else assemble(layer, expert, dst);
    return dst;
}

bool GgufExpertSource::stable(int64_t layer, int64_t expert) const {
    return valid(layer, expert) && res_off_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] != kNone;
}

bool GgufExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (maps_.empty() || !valid(layer, expert) || dst == nullptr) return false;
    const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (res_off_[i] != kNone) std::memcpy(dst, res_base_ + res_off_[i], (size_t) lay.bytes[(size_t) layer]);
    else assemble(layer, expert, dst);
    return true;
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!valid(layer, expert) || registered_bytes_ == 0) return false;
    const uint64_t off = res_off_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
    return off != kNone && off + strata::kernels::cpu::expert_layout().bytes[(size_t) layer] <= registered_bytes_;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const uint64_t off = res_off_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
    if (!sliced_) return dev_slice_[0] + off;
    // one registration slice per layer group, in layer order: the slice whose start is the layer's group start
    const auto& starts = ((const PinnedArena*) arena_)->slice_starts;
    const uint64_t ls = res_layer_start_[(size_t) layer];
    const auto it = std::lower_bound(starts.begin(), starts.end(), ls);
    if (it == starts.end() || *it != ls) return nullptr;
    const size_t k = (size_t) (it - starts.begin());
    return k < dev_slice_.size() ? dev_slice_[k] + (off - ls) : nullptr;
}

void GgufExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (maps_.empty() || !valid(layer, 0) || ids == nullptr || k <= 0) return;
    // the routed experts no RAM tier holds: their three slices are queued for reading now, while the pool
    // quantizes the activation (best-effort, as FileExpertSource's)
    const int64_t epoch = ++pf_epoch_;
#if defined(_WIN32)
    struct RangeEntry { void* address; size_t bytes; };
    using PrefetchFn = BOOL (WINAPI*)(HANDLE, ULONG_PTR, RangeEntry*, ULONG);
    static const PrefetchFn prefetch = []() -> PrefetchFn {
        const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        return kernel32 ? reinterpret_cast<PrefetchFn>(GetProcAddress(kernel32, "PrefetchVirtualMemory")) : nullptr;
    }();
    RangeEntry batch[63];
    int nb = 0;
#endif
    std::vector<Stage::Task> tasks;
    for (int64_t j = 0; j < k; ++j) {
        const int64_t e = ids[j];
        if (e < 0 || e >= n_expert_) continue;
        const size_t i = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (pf_stamp_[i] == epoch || res_off_[i] != kNone) continue;
        pf_stamp_[i] = epoch;
        if (gpu_res_ != nullptr && gpu_res_[i] >= 0) continue;   // (the GPU computes it)
        const int32_t s0 = ring_of_[i];
        if (s0 >= 0 && ring_owner_[(size_t) s0] == (int64_t) i &&
            ring_count_ - ring_stamp_[(size_t) s0] < ring_n_ - per_layer_) continue;   // (staged already)
        const uint8_t* src[3];
        uint64_t bytes[3], off[3];
        parts(layer, e, src, bytes, off);
#if defined(_WIN32)
        if (prefetch != nullptr) {
            for (int r = 0; r < 3; ++r) batch[nb++] = {const_cast<uint8_t*>(src[r]), (size_t) bytes[r]};
            if (nb == 63) { (void) prefetch(GetCurrentProcess(), (ULONG_PTR) nb, batch, 0); nb = 0; }
        }
#elif defined(POSIX_FADV_WILLNEED)
        if (!direct_) {   // (unbuffered reads fill their own ring slots: nothing to read ahead into the page cache)
            const int fd = maps_[(size_t) map_of_[(size_t) layer]].fd;
            for (int r = 0; r < 3; ++r) (void) ::posix_fadvise(fd, (off_t) off[r], (off_t) bytes[r], POSIX_FADV_WILLNEED);
        }
#else
        (void) src; (void) bytes; (void) off;
#endif
        if (st_ != nullptr) {   // its slot now; a stager assembles it while the pool quantizes the activation
            const int64_t s = take_slot(i);
            if (direct_) {   // a task a slice: the SSD sees three reads for it, spread over the stagers
                st_->busy[(size_t) s].store(3, std::memory_order_release);
                for (int r = 0; r < 3; ++r) tasks.push_back({s, layer, e, r});
            } else {
                st_->busy[(size_t) s].store(1, std::memory_order_release);
                tasks.push_back({s, layer, e, -1});
            }
        }
    }
#if defined(_WIN32)
    if (nb > 0 && prefetch != nullptr) (void) prefetch(GetCurrentProcess(), (ULONG_PTR) nb, batch, 0);
#endif
    if (!tasks.empty()) {
        {
            std::lock_guard<std::mutex> lk(st_->mu);
            for (const auto& t : tasks) st_->queue.push_back(t);
        }
        st_->cv.notify_all();
    }
}

}  // namespace strata::core
