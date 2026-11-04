// src/ds4/ds4_os.hpp - the few OS services maya-ds4 needs, on Linux and on Windows: files read at an offset and
// mapped read-only (the experts read in place), page-cache hints, anonymous host memory on a NUMA node, thread pinning,
// each NUMA node's physical cores, the RAM this process can still get.  (On Windows as src/core/expert_source.cpp
// does it for GLM: no FILE_FLAG_RANDOM_ACCESS - the page cache must keep what it read.)
#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>   // (K32QueryWorkingSetEx: kernel32's, no psapi.lib)
#include "strata/platform/commit.hpp"
#else
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::ds4::os {

using fd_t = intptr_t;
constexpr fd_t kBadFd = -1;

#if defined(_WIN32)
inline std::wstring wide(const std::string& s) {   // UTF-8 -> UTF-16 (a path with non-ASCII characters)
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) (n > 0 ? n : 1), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}
#endif

/// A file opened for reading (kBadFd: it could not be).
inline fd_t open_ro(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// The same file opened for unbuffered reads (Linux: O_DIRECT - no page cache, the disk straight into the buffer;
/// kBadFd where the filesystem refuses it, or on Windows, whose unbuffered reads of a mapped file take a slow path)
inline fd_t open_direct(const std::string& path) {
#if defined(_WIN32)
    (void) path;
    return kBadFd;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// [off, off + n) through an unbuffered descriptor into dst, a kDirectAlign-aligned buffer of at least
/// direct_span(off, n) bytes: the file read from off rounded down, so the bytes land at dst + (off % kDirectAlign).
/// False if a read fails (the caller reads buffered instead).
constexpr uint64_t kDirectAlign = 4096;
inline uint64_t direct_span(uint64_t off, uint64_t n) {
    const uint64_t a0 = off & ~(kDirectAlign - 1), a1 = (off + n + kDirectAlign - 1) & ~(kDirectAlign - 1);
    return a1 - a0;
}
inline bool pread_direct(fd_t f, void* dst, uint64_t n, uint64_t off) {
#if defined(_WIN32)
    (void) f, (void) dst, (void) n, (void) off;
    return false;
#else
    const uint64_t a0 = off & ~(kDirectAlign - 1), need = direct_span(off, n), want = (off - a0) + n;
    uint8_t* p = (uint8_t*) dst;
    uint64_t got = 0;
    while (got < need) {
        const ssize_t r = ::pread((int) f, p + got, (size_t) std::min<uint64_t>(need - got, 1ull << 30), (off_t) (a0 + got));
        if (r < 0) return false;
        if (r == 0) break;   // (the file's end: the last aligned read is short)
        got += (uint64_t) r;
        if ((uint64_t) r % kDirectAlign != 0) break;
    }
    return got >= want;
#endif
}
/// Read an arbitrarily aligned expert span through O_DIRECT into a reusable aligned staging buffer,
/// then copy its exact bytes to the NUMA destination. A failure leaves the caller to use buffered I/O.
inline bool copy_direct(fd_t f, uint64_t off, uint64_t n, void* dst, std::vector<uint8_t>& staging) {
    if (f == kBadFd || n == 0 || off > UINT64_MAX - n || off + n > (uint64_t) INT64_MAX - kDirectAlign) return false;
    const uint64_t span = direct_span(off, n);
    if (span > SIZE_MAX - kDirectAlign) return false;
    staging.resize((size_t) span + kDirectAlign);
    auto* aligned = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(staging.data()) + kDirectAlign - 1) & ~(uintptr_t)(kDirectAlign - 1));
    if (!pread_direct(f, aligned, n, off)) return false;
    std::memcpy(dst, aligned + off % kDirectAlign, (size_t) n);
    return true;
}

/// Windows: the file opened for unbuffered reads (FILE_FLAG_NO_BUFFERING), for a file no view of this process maps -
/// unbuffered reads of a mapped file take a slow path there (open_direct); kBadFd elsewhere
inline fd_t open_unbuffered(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    (void) path;
    return kBadFd;
#endif
}
/// [off, off + n) through open_unbuffered's handle (one thread's) into buf, kDirectAlign-aligned and at least
/// direct_span(off, n) bytes: the bytes land at buf + off % kDirectAlign
inline bool pread_unbuffered_aligned(fd_t f, void* buf, uint64_t n, uint64_t off, uint64_t alignment) {
#if defined(_WIN32)
    if (alignment == 0 || off > UINT64_MAX - n) return false;
    const uint64_t a0 = off - off % alignment, delta = off - a0;
    if (delta > UINT64_MAX - n || delta + n > UINT64_MAX - (alignment - 1)) return false;
    const uint64_t span = (delta + n + alignment - 1) / alignment * alignment;
    if (span > MAXDWORD) return false;
    OVERLAPPED ov{};
    ov.Offset = (DWORD) a0;
    ov.OffsetHigh = (DWORD) (a0 >> 32);
    DWORD got = 0;
    return ReadFile((HANDLE) f, buf, (DWORD) span, &got, &ov) && got >= delta + n;
#else
    (void) f, (void) buf, (void) n, (void) off, (void) alignment;
    return false;
#endif
}
inline bool pread_unbuffered(fd_t f, void* buf, uint64_t n, uint64_t off) {
    return pread_unbuffered_aligned(f, buf, n, off, kDirectAlign);
}
/// The same file opened for reads from many threads at once (Windows runs the reads of a synchronous handle one at a
/// time, so an overlapped one; Linux: a plain descriptor, pread is concurrent anyway)
inline fd_t open_par(const std::string& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    return h == INVALID_HANDLE_VALUE ? kBadFd : (fd_t) h;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    return fd < 0 ? kBadFd : (fd_t) fd;
#endif
}
/// n bytes at off through open_par's handle (any thread, concurrently)
inline bool pread_par(fd_t f, void* dst, uint64_t n, uint64_t off) {
#if defined(_WIN32)
    struct Ev {   // (each thread its own, closed when it ends)
        HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        ~Ev() {
            if (h != nullptr) CloseHandle(h);
        }
    };
    static thread_local Ev ev;
    if (ev.h == nullptr) return false;
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const DWORD want = (DWORD) std::min<uint64_t>(n, 1ull << 30);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        ov.hEvent = ev.h;
        ResetEvent(ev.h);
        DWORD got = 0;
        if (!ReadFile((HANDLE) f, p, want, nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) return false;
        if (!GetOverlappedResult((HANDLE) f, &ov, &got, TRUE) || got == 0) return false;
        p += got, off += got, n -= got;
    }
    return true;
#else
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const ssize_t r = ::pread((int) f, p, (size_t) std::min<uint64_t>(n, 1ull << 30), (off_t) off);
        if (r <= 0) return false;
        p += r, off += (uint64_t) r, n -= (uint64_t) r;
    }
    return true;
#endif
}
/// `from` renamed to `to`, replacing `to` if it exists (rename(2) does; Windows' rename fails on an existing file, so
/// MoveFileExW with MOVEFILE_REPLACE_EXISTING)
inline bool replace_file(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    return MoveFileExW(wide(from).c_str(), wide(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}
inline void close_fd(fd_t f) {
    if (f == kBadFd) return;
#if defined(_WIN32)
    CloseHandle((HANDLE) f);
#else
    ::close((int) f);
#endif
}
inline uint64_t file_size(fd_t f) {
#if defined(_WIN32)
    LARGE_INTEGER sz{};
    return GetFileSizeEx((HANDLE) f, &sz) ? (uint64_t) sz.QuadPart : 0;
#else
    struct stat sb;
    return ::fstat((int) f, &sb) == 0 && sb.st_size > 0 ? (uint64_t) sb.st_size : 0;
#endif
}
/// n bytes at offset off into dst (any thread, concurrently); false if the file ends first or a read fails.
inline bool pread_full(fd_t f, void* dst, uint64_t n, uint64_t off) {
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const uint64_t want = std::min<uint64_t>(n, 1ull << 30);
#if defined(_WIN32)
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        if (!ReadFile((HANDLE) f, p, (DWORD) want, &got, &ov) || got == 0) return false;
        const uint64_t r = got;
#else
        const ssize_t rr = ::pread((int) f, p, (size_t) want, (off_t) off);
        if (rr <= 0) return false;
        const uint64_t r = (uint64_t) rr;
#endif
        p += r, off += r, n -= r;
    }
    return true;
}
/// The whole file mapped read-only (nullptr: it could not be).
#if defined(_WIN32)
/// fd -> the view map_ro returned for it (advise's Windows branch unlocks through the view; a mapping is per fd here,
/// and a stale entry can only name a dead or reused view - VirtualUnlock on it fails harmlessly).
inline std::unordered_map<fd_t, const uint8_t*>& view_of_fd() {
    static std::unordered_map<fd_t, const uint8_t*> r;
    return r;
}
inline std::mutex& view_mu() { static std::mutex m; return m; }
#endif
inline void* map_ro(fd_t f, uint64_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(_WIN32)
    HANDLE m = CreateFileMappingW((HANDLE) f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) return nullptr;
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);   // (the view keeps the mapping)
    if (v != nullptr) {
        const std::lock_guard<std::mutex> lk(view_mu());
        view_of_fd()[(size_t) f] = (const uint8_t*) v;
    }
    return v;
#else
    void* p = ::mmap(nullptr, (size_t) bytes, PROT_READ, MAP_SHARED, (int) f, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}
inline void unmap(void* p, uint64_t bytes) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    {
        const std::lock_guard<std::mutex> lk(view_mu());
        auto& r = view_of_fd();
        for (auto it = r.begin(); it != r.end();)
            it = it->second == p ? r.erase(it) : std::next(it);
    }
    UnmapViewOfFile(p);
#else
    ::munmap(p, (size_t) bytes);
#endif
}
/// Demote [p, p + n)'s whole interior pages of this process's working set to the standby list (Windows: VirtualUnlock
/// on unlocked pages removes them from the working set even though it reports ERROR_NOT_LOCKED - the working set
/// documentation; a clean file-backed page stays resident and soft-faults back with its contents, so this only makes
/// it count as available again). Returns the bytes accepted. Interior pages only: an edge page may hold a neighbour's
/// bytes. Linux has no equivalent for a shared map - its page cache already counts as available.
inline uint64_t unlock_range(const void* p, uint64_t n) {
#if defined(_WIN32)
    if (p == nullptr || n < 4096) return 0;
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uint64_t pg = si.dwPageSize ? si.dwPageSize : 4096;
    const uint64_t a = ((uint64_t) (uintptr_t) p + pg - 1) & ~(pg - 1);
    const uint64_t b = ((uint64_t) (uintptr_t) p + n) & ~(pg - 1);
    if (b <= a) return 0;
    SetLastError(ERROR_SUCCESS);
    const BOOL unlocked = VirtualUnlock((LPVOID) (uintptr_t) a, (SIZE_T) (b - a));
    // (NOT_LOCKED is the documented success path for pages this process never locked)
    return unlocked || GetLastError() == ERROR_NOT_LOCKED ? b - a : 0;
#else
    (void) p, (void) n;
    return 0;
#endif
}

/// A hint that [off, off + n) of the file is needed soon (need) or not again soon (!need); nothing where the OS has
/// no such hint for a file. Windows' !need, where this process maps the file, demotes the range's interior pages out
/// of its working set (Linux: the page cache is reclaimable and counts as available, so the hint stays a no-op there).
inline void advise(fd_t f, uint64_t off, uint64_t n, bool need) {
#if defined(_WIN32)
    if (need || f == kBadFd) return;
    const uint8_t* base = nullptr;
    {
        const std::lock_guard<std::mutex> lk(view_mu());
        const auto it = view_of_fd().find((size_t) f);
        if (it == view_of_fd().end()) return;
        base = it->second;
    }
    if (off > UINT64_MAX - n) return;
    (void) unlock_range(base + off, n);
#else
    posix_fadvise((int) f, (off_t) off, (off_t) n, need ? POSIX_FADV_WILLNEED : POSIX_FADV_DONTNEED);
#endif
}

/// This process's resident working-set bytes (the trimmer's measure; 0: it could not be read).
inline uint64_t working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pm{};
    pm.cb = sizeof pm;
    return GetProcessMemoryInfo(GetCurrentProcess(), &pm, pm.cb) ? (uint64_t) pm.WorkingSetSize : 0;
#else
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr) return 0;
    unsigned long long size = 0, resident = 0;
    const int got = std::fscanf(f, "%llu %llu", &size, &resident);
    std::fclose(f);
    return got == 2 ? resident * 4096ull : 0;
#endif
}

/// Release only whole source-file pages after their bytes have been copied to private RAM.
/// Edge pages may contain another tensor or a concurrent reader's bytes: leave them cached.
/// Windows trims those pages from this process's working set with VirtualUnlock: clean file-backed contents stay
/// recoverable by a later fault, unlike MEM_RESET. Linux advises DONTNEED. Returns bytes accepted as a hint;
/// neither interface promises global cache eviction (another process may still map the pages).
inline uint64_t release_copied_cache(fd_t f, const void* mapped_base, uint64_t off, uint64_t n) {
#if defined(_WIN32)
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uint64_t p = si.dwPageSize;
    if (mapped_base == nullptr) return 0;
#else
    const long page = sysconf(_SC_PAGESIZE);
    const uint64_t p = page > 0 ? (uint64_t) page : 0;
    (void) mapped_base;
#endif
    if (f == kBadFd || p == 0 || n == 0 || off > UINT64_MAX - n || off + n > (uint64_t) INT64_MAX - p) return 0;
    const uint64_t rem = off % p;
    const uint64_t skip = rem == 0 ? 0 : p - rem;
    if (skip >= n) return 0;
    const uint64_t begin = off + skip, end = (off + n) / p * p;
    if (end <= begin || end > (uint64_t) INT64_MAX) return 0;
#if defined(_WIN32)
    void* address = const_cast<uint8_t*>(static_cast<const uint8_t*>(mapped_base)) + begin;
    SetLastError(ERROR_SUCCESS);
    const BOOL unlocked = VirtualUnlock(address, (SIZE_T) (end - begin));
    const DWORD error = GetLastError();
    // Microsoft's Working Set documentation: unlocked pages are removed even though the call reports NOT_LOCKED.
    return unlocked || error == ERROR_NOT_LOCKED ? end - begin : 0;
#else
    return posix_fadvise((int) f, (off_t) begin, (off_t) (end - begin), POSIX_FADV_DONTNEED) == 0
               ? end - begin : 0;
#endif
}

/// Ranges of a file's mapping about to be read: the OS reads them in now, with large requests side by side, instead of
/// a page fault at a time later (Windows: PrefetchVirtualMemory - cold expert-sized pieces came in 5x faster on an NVMe;
/// Linux: madvise WILLNEED, the kernel's read-ahead of the whole range).  Best effort: a failure changes nothing.
inline void prefetch(const std::vector<std::pair<const void*, size_t>>& ranges) {
    if (ranges.empty()) return;
#if defined(_WIN32)
    std::vector<WIN32_MEMORY_RANGE_ENTRY> r(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        r[i].VirtualAddress = const_cast<void*>(ranges[i].first);
        r[i].NumberOfBytes = ranges[i].second;
    }
    PrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) r.size(), r.data(), 0);
#else
    const uintptr_t pg = 4096;
    for (const auto& x : ranges) {
        const uintptr_t a = (uintptr_t) x.first, b = a & ~(pg - 1);
        madvise((void*) b, x.second + (a - b), MADV_WILLNEED);
    }
#endif
}

/// Whether every page of [p, p + n) of a file's mapping is in the page cache (Linux mincore; elsewhere false: the
/// caller reads buffered anyway, which the cache serves)
inline bool resident(const void* p, size_t n) {
#if defined(_WIN32)
    (void) p, (void) n;
    return false;
#else
    const uintptr_t pg = 4096, a = (uintptr_t) p, b = a & ~(pg - 1);
    const size_t len = n + (a - b), np = (len + pg - 1) / pg;
    thread_local std::vector<unsigned char> v;
    if (v.size() < np) v.resize(np);
    if (mincore((void*) b, len, v.data()) != 0) return false;
    for (size_t i = 0; i < np; ++i)
        if ((v[i] & 1) == 0) return false;
    return true;
#endif
}

/// Whether a range of a file's mapping looks resident, so that prefetch() would only cost CPU time (it walks every page:
/// ~50 ns a 4 KiB page on Windows - 0.34 ms for a 6.3 MB expert matrix on a Strix Halo): `samples` pages spread over
/// it all resident (Linux: mincore, the page cache; Windows: QueryWorkingSetEx, the process's working set - a page on
/// the standby list counts as not resident, and prefetch() is what brings those back cheaply).  An expert's pages are
/// read together, so they come and go together: a wrong "yes" costs page faults, a wrong "no" a prefetch.
inline bool resident_sampled(const void* p, size_t bytes, int samples = 8) {
    if (bytes == 0) return true;
    const uintptr_t pg = 4096, a = (uintptr_t) p, span = bytes - 1;
    samples = std::max(1, std::min(samples, 16));
#if defined(_WIN32)
    PSAPI_WORKING_SET_EX_INFORMATION w[16] = {};
    for (int i = 0; i < samples; ++i)
        w[i].VirtualAddress = (PVOID) ((a + (samples > 1 ? span / (samples - 1) * i : 0)) & ~(pg - 1));
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), w, (DWORD) (samples * sizeof w[0]))) return false;
    for (int i = 0; i < samples; ++i)
        if (!w[i].VirtualAttributes.Valid) return false;
    return true;
#else
    for (int i = 0; i < samples; ++i) {
        unsigned char v = 0;
        const uintptr_t at = (a + (samples > 1 ? span / (samples - 1) * i : 0)) & ~(pg - 1);
        if (mincore((void*) at, pg, &v) != 0 || !(v & 1)) return false;
    }
    return true;
#endif
}
/// prefetch() of the ranges that do not look resident (resident_sampled) - on Windows, where PrefetchVirtualMemory walks
/// every page of a resident range (0.34 ms a 6.3 MB expert matrix on a Strix Halo); Linux's madvise(WILLNEED) on cached
/// pages costs ~1 us a matrix (a Core i5-12600T), less than the residency check (2.5 us): every range there
inline void prefetch_cold(const std::vector<std::pair<const void*, size_t>>& ranges) {
#if defined(_WIN32)
    static const bool all = [] {   // (DS4_PF_ALL=1: every range, as before)
        const char* v = std::getenv("DS4_PF_ALL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (all) return prefetch(ranges);
    std::vector<std::pair<const void*, size_t>> cold;
    for (const auto& x : ranges)
        if (!resident_sampled(x.first, x.second)) cold.push_back(x);
    prefetch(cold);
#else
    prefetch(ranges);
#endif
}

/// Whether [p, p + n) of a file's mapping is in RAM now: every page (Linux, one mincore), sampled pages of the working
/// set (Windows; a page on the standby list counts as not in RAM)
inline bool in_ram(const void* p, size_t n) {
#if defined(_WIN32)
    return resident_sampled(p, n, 8);
#else
    return resident(p, n);
#endif
}

/// (a test of DISK_SKIP_AUTO's switch back) this process's file mappings out of RAM: Linux - its pages unmapped
/// (madvise DONTNEED) and the files' cached pages dropped (posix_fadvise DONTNEED; clean pages, no root needed);
/// Windows - the mapped views out of the working set (VirtualUnlock of unlocked pages), then the rest of it
inline void drop_cached(const std::vector<std::pair<uint8_t*, size_t>>& maps, const std::vector<fd_t>& fds) {
#if defined(_WIN32)
    (void) fds;
    for (const auto& m : maps)
        if (m.first != nullptr && m.second > 0) VirtualUnlock((LPVOID) m.first, m.second);
    K32EmptyWorkingSet(GetCurrentProcess());
#else
    for (const auto& m : maps)
        if (m.first != nullptr && m.second > 0) madvise((void*) m.first, m.second, MADV_DONTNEED);
    for (fd_t f : fds)
        if (f != kBadFd) posix_fadvise((int) f, 0, 0, POSIX_FADV_DONTNEED);
#endif
}

/// Anonymous memory on NUMA node `node` of `nodes` (node < 0 or nodes < 2: wherever the first touch puts it), on huge
/// pages where the OS gives them; nullptr if it cannot be had.  Linux: the node is preferred, not required
/// (MPOL_PREFERRED) - a node that fills up (locked Engram tables, another program's memory) lends the rest from the
/// other nodes, where MPOL_BIND had the kernel end the process with RAM free on the other node (#507)
inline void* anon_alloc(size_t bytes, int node, int nodes) {
#if defined(_WIN32)
    if (node >= 0 && nodes > 1)
        return VirtualAllocExNuma(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                  (DWORD) node);
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* m = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    madvise(m, bytes, MADV_HUGEPAGE);
    if (node >= 0 && nodes > 1) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        if (syscall(SYS_mbind, m, bytes, 1 /* MPOL_PREFERRED */, mask, 256, 0) != 0) {
            // (said once: without it the per-node copies land wherever first touch puts them and nothing shows it -
            //  Docker's default seccomp profile refuses mbind to a container without CAP_SYS_NICE)
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "ds4: mbind to NUMA node %d failed (%s): the RAM copies do not prefer their node "
                                     "(a container needs SYS_NICE)\n", node, std::strerror(errno));
        }
    }
    return m;
#endif
}
inline void anon_free(void* p, size_t bytes) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, bytes);
#endif
}

/// The kernel's pool of explicit 2 MB huge pages (reserved by the admin: `sysctl vm.nr_hugepages=N`): the bytes a new
/// mapping can still take from it (free less already reserved); 0 where there is none or it cannot be read, and on
/// Windows.  MemAvailable does not count this pool.
constexpr size_t kHuge2M = (size_t) 2 << 20;
inline uint64_t huge2m_free() {
#if defined(_WIN32)
    return 0;
#else
    auto num = [](const char* path) -> uint64_t {
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) return 0;
        unsigned long long v = 0;
        const bool ok = std::fscanf(f, "%llu", &v) == 1;
        std::fclose(f);
        return ok ? (uint64_t) v : 0;
    };
    const uint64_t fr = num("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"),
                   rs = num("/sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages");
    return fr > rs ? (fr - rs) * kHuge2M : 0;
#endif
}

/// Anonymous memory on explicit 2 MB pages from that pool (MAP_HUGETLB): `bytes` rounded up to whole pages, `*got` the
/// size mapped (what anon_free takes back), NUMA node `node` of `nodes` preferred as anon_alloc does.  The pages are
/// reserved at the call, so a mapping it returns does not fault for want of them later; nullptr: the pool cannot hold
/// it (or Windows).  Unlike a transparent-huge-page arena, a pin (cudaHostRegister) of it keeps its 2 MB pages.
inline void* huge_alloc(size_t bytes, int node, int nodes, size_t* got) {
#if defined(_WIN32)
    (void) bytes, (void) node, (void) nodes, (void) got;
    return nullptr;
#else
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
    const size_t hb = (bytes + kHuge2M - 1) / kHuge2M * kHuge2M;
    void* m = ::mmap(nullptr, hb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT),
                     -1, 0);
    if (m == MAP_FAILED) return nullptr;
    if (node >= 0 && nodes > 1) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        syscall(SYS_mbind, m, hb, 1 /* MPOL_PREFERRED */, mask, 256, 0);
    }
    if (got != nullptr) *got = hb;
    return m;
#endif
}

/// The calling thread, or `t`, on logical CPU `cpu` (Windows: the first 64 CPUs, processor group 0).
inline void pin_self(int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
#endif
}
inline void pin_thread(std::thread& t, int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) SetThreadAffinityMask((HANDLE) t.native_handle(), (DWORD_PTR) 1 << cpu);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
#endif
}

/// A physical core: the logical CPUs of it this process may run on (the first one first), and its efficiency class
/// (Windows' EfficiencyClass - higher is faster; Linux: 1 for a hybrid Intel CPU's performance cores, 0 otherwise).
struct Core {
    std::vector<int> cpus;
    int eff = 0;
};

#if defined(_WIN32)
/// Each NUMA node's physical cores (their logical CPUs the process's affinity allows), processor group 0 - Windows'
/// answer to /sys/devices/system/node; empty if the OS does not say.
inline std::vector<std::vector<Core>> node_core_threads() {
    std::vector<std::vector<Core>> nodes;
    auto query = [](LOGICAL_PROCESSOR_RELATIONSHIP rel, std::vector<uint8_t>& buf) {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(rel, nullptr, &len);
        buf.resize(len);
        return len > 0 && GetLogicalProcessorInformationEx(rel, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len);
    };
    std::vector<uint8_t> nb, cb;
    if (!query(RelationNumaNode, nb) || !query(RelationProcessorCore, cb)) return nodes;
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    const KAFFINITY allowed = GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) && proc_mask != 0
                                  ? (KAFFINITY) proc_mask : ~(KAFFINITY) 0;
    std::vector<KAFFINITY> masks;
    for (size_t o = 0; o < nb.size();) {
        auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (nb.data() + o);
        if (e->NumaNode.GroupMask.Group == 0) masks.push_back(e->NumaNode.GroupMask.Mask);
        o += e->Size;
    }
    nodes.resize(masks.size());
    for (size_t o = 0; o < cb.size();) {
        auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (cb.data() + o);
        if (e->Processor.GroupCount > 0 && e->Processor.GroupMask[0].Group == 0) {
            const KAFFINITY m = e->Processor.GroupMask[0].Mask & allowed;
            Core c;
            c.eff = (int) e->Processor.EfficiencyClass;
            for (int b = 0; b < 64; ++b)
                if ((m >> b) & 1) c.cpus.push_back(b);
            for (size_t n = 0; n < masks.size() && !c.cpus.empty(); ++n)
                if ((masks[n] >> c.cpus[0]) & 1) nodes[n].push_back(c);
        }
        o += e->Size;
    }
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [](const std::vector<Core>& v) { return v.empty(); }),
                nodes.end());
    for (auto& v : nodes)
        std::sort(v.begin(), v.end(), [](const Core& a, const Core& b) { return a.cpus[0] < b.cpus[0]; });
    return nodes;
}
/// The RAM this process can still get (~0: not known).
inline uint64_t ram_avail() {
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (uint64_t) s.ullAvailPhys : ~uint64_t{0};
}
#endif

/// Windows' commit (RAM + page file, what every private allocation is charged to - on an integrated GPU the GPU's
/// own allocations too): the limit, what this process can still commit, the RAM; all ~0 where there is no such limit
/// (Linux overcommits; its RAM is ram_avail's business).
struct Commit {
    uint64_t limit = ~uint64_t{0}, avail = ~uint64_t{0}, ram = ~uint64_t{0};
};
inline Commit commit() {
    Commit c;
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        c.limit = (uint64_t) s.ullTotalPageFile;
        // (+ what a page file Windows manages may still grow by)
        c.avail = (uint64_t) s.ullAvailPageFile + strata::platform::commit_growth((uint64_t) s.ullTotalPhys);
        c.ram = (uint64_t) s.ullTotalPhys;
    }
#endif
    return c;
}

}  // namespace strata::ds4::os
