// src/mm3/mm3_os.hpp - the few OS services maya-mm3 needs, on Linux and on Windows: the model's files mapped read-only
// (maya-ds4's helpers, src/ds4/ds4_os.hpp, used as they are), the RAM this process can still get and the RAM in all,
// stdin read as it arrives, and thread pinning.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../ds4/ds4_os.hpp"

#if !defined(_WIN32)
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace mm3::os {

namespace sys = maya::ds4::os;

#if !defined(_WIN32)
inline std::string read_small(const std::string& path) {
    std::string s;
    if (std::FILE* f = std::fopen(path.c_str(), "r")) {
        char b[4096];
        size_t n;
        while ((n = std::fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
        std::fclose(f);
    }
    return s;
}
#endif

/// The RAM this process can still get, in bytes: Linux MemAvailable (the page cache counts as free) and, below it, the
/// tightest cgroup v2 limit on the way to the root; Windows the available physical memory.  0: not known.
inline uint64_t ram_avail() {
#if defined(_WIN32)
    const uint64_t a = sys::ram_avail();
    return a == ~uint64_t{0} ? 0 : a;
#else
    uint64_t avail = 0;
    {
        const std::string mi = read_small("/proc/meminfo");
        const size_t at = mi.find("MemAvailable:");
        if (at != std::string::npos) avail = std::strtoull(mi.c_str() + at + 13, nullptr, 10) << 10;
    }
    std::string cg = read_small("/proc/self/cgroup");   // "0::/user.slice/..."
    const size_t at = cg.find("0::");
    if (at == std::string::npos) return avail;
    cg = cg.substr(at + 3);
    cg = cg.substr(0, cg.find('\n'));
    for (int depth = 0; depth < 64; ++depth) {
        const std::string dir = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
        const std::string mx = read_small(dir + "/memory.max");
        if (!mx.empty() && mx.compare(0, 3, "max") != 0) {
            const uint64_t lim = std::strtoull(mx.c_str(), nullptr, 10);
            const uint64_t cur = std::strtoull(read_small(dir + "/memory.current").c_str(), nullptr, 10);
            const uint64_t left = lim > cur ? lim - cur : 0;
            if (avail == 0 || left < avail) avail = left;
        }
        if (cg == "/" || cg.empty()) break;
        const size_t sl = cg.find_last_of('/');
        cg = sl == 0 || sl == std::string::npos ? "/" : cg.substr(0, sl);
    }
    return avail;
#endif
}

/// The RAM in all (0: not known).
inline uint64_t ram_total() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (uint64_t) s.ullTotalPhys : 0;
#else
    const std::string mi = read_small("/proc/meminfo");
    const size_t at = mi.find("MemTotal:");
    return at == std::string::npos ? 0 : std::strtoull(mi.c_str() + at + 9, nullptr, 10) << 10;
#endif
}

/// Up to n bytes of stdin as they arrive (a pipe: what is there, at least one byte); 0 at its end or on an error.
inline long read_stdin(char* buf, size_t n) {
#if defined(_WIN32)
    DWORD got = 0;
    if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), buf, (DWORD) n, &got, nullptr)) return 0;
    return (long) got;
#else
    for (;;) {
        const ssize_t r = ::read(0, buf, n);
        if (r < 0 && errno == EINTR) continue;
        return r < 0 ? 0 : (long) r;
    }
#endif
}

/// A whole file mapped read-only (nullptr: it could not be; bytes its size).
inline const uint8_t* map_file(const std::string& path, size_t& bytes) {
    bytes = 0;
    const sys::fd_t f = sys::open_ro(path);
    if (f == sys::kBadFd) return nullptr;
    const uint64_t sz = sys::file_size(f);
    void* m = sz > 0 ? sys::map_ro(f, sz) : nullptr;
    sys::close_fd(f);   // (the mapping keeps the file)
    if (m != nullptr) bytes = (size_t) sz;
    return (const uint8_t*) m;
}

/// Drops the whole pages of [p, p + n) of a read-only file mapping from this process once their bytes are copied
/// elsewhere (the pinned pool, a GPU): the mapped pages would otherwise count twice in RAM (the pool's copy and the
/// mapping) and, on Windows, in the working set the available-memory figure misses.  Linux: MADV_DONTNEED (the
/// mapping's pages leave RSS), then MADV_PAGEOUT where the kernel has it (the clean page-cache copy goes too);
/// Windows: VirtualUnlock on pages that are not locked takes them out of the working set.  Partial edge pages stay
/// (a neighbour's bytes).  Advisory: a failure changes nothing, a later read just faults the page in again.
inline void drop_mapped(const void* p, size_t n) {
    const uintptr_t pg = 4096;
    const uintptr_t a = (uintptr_t) p, b = a + n;
    const uintptr_t lo = (a + pg - 1) / pg * pg, hi = b / pg * pg;
    if (hi <= lo) return;
#if defined(_WIN32)
    (void) VirtualUnlock((LPVOID) lo, (SIZE_T) (hi - lo));
#else
    (void) ::madvise((void*) lo, (size_t) (hi - lo), MADV_DONTNEED);
#if defined(MADV_PAGEOUT)
    (void) ::madvise((void*) lo, (size_t) (hi - lo), MADV_PAGEOUT);
#endif
#endif
}

/// The calling thread on logical CPU `cpu`.
inline void pin_self(int cpu) { sys::pin_self(cpu); }

/// Every page of [p, p + n) of a read-only file mapping brought into this process, `chunk` bytes at a time, until
/// `stop` is set (maya-hy4's HY4_EG_POPULATE: the Engram table in the background).  Linux: MADV_POPULATE_READ where
/// the kernel has it; Windows: PrefetchVirtualMemory reads the chunk in large requests; then (and wherever the first
/// is missing or fails) one byte a page read, so each page is in the page cache and mapped.  Returns the bytes walked
/// (n, or less when stopped).
inline size_t populate_mapped(const void* p, size_t n, const std::atomic<bool>& stop,
                              size_t chunk = (size_t) 256 << 20) {
    const size_t pg = 4096;
    if (p == nullptr || n == 0 || chunk < pg) return 0;
    const uint8_t* a = (const uint8_t*) ((uintptr_t) p & ~(uintptr_t) (pg - 1));
    const uint8_t* end = (const uint8_t*) p + n;
    const uint8_t* c = a;
    for (; c < end && !stop.load(); c += chunk) {
        const size_t m = (size_t) std::min<ptrdiff_t>((ptrdiff_t) chunk, end - c);
#if defined(__linux__) && defined(MADV_POPULATE_READ)
        if (::madvise((void*) c, m, MADV_POPULATE_READ) == 0) continue;
#elif defined(_WIN32)
        sys::prefetch({{(const void*) c, m}});
#endif
        volatile uint8_t sink = 0;
        for (size_t b = 0; b < m; b += pg) sink ^= c[b];
        (void) sink;
    }
    if (c >= end) return n;
    return c > (const uint8_t*) p ? (size_t) (c - (const uint8_t*) p) : 0;
}

}  // namespace mm3::os
