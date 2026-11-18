// include/strata/platform/huge_pages.hpp - explicit 2 MB host pages, from the reserved pool or the kernel's
// on-demand pool.  The latter spends MemAvailable; only the reserved pool is extra room for a RAM budget.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>

#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#endif

namespace strata::platform {

constexpr size_t kHugePage2M = (size_t) 2 << 20;
constexpr size_t kHugePiece = (size_t) 1 << 30;  // a block is built this much at a time
constexpr int kHugeTries = 20;                  // tries of one piece before the block is given up
constexpr int kHugePauseMs = 100;               // between tries: the kernel's reclaim catching up
constexpr int kHugeInterleave = -2;

struct HugePool {
    uint64_t reserved_free = 0;   // free reserved pages, less those already promised to a mapping
    uint64_t on_demand = 0;       // pages the kernel may still build, taken out of MemAvailable
    uint64_t total() const { return reserved_free + on_demand; }
};

inline HugePool huge_pool_from(uint64_t free_pages, uint64_t resv_pages, uint64_t overcommit_pages,
                              uint64_t surplus_pages, uint64_t mem_available_bytes) {
    HugePool pool;
    pool.reserved_free = free_pages > resv_pages ? (free_pages - resv_pages) * kHugePage2M : 0;
    pool.on_demand = overcommit_pages > surplus_pages ? (overcommit_pages - surplus_pages) * kHugePage2M : 0;
    if (mem_available_bytes != ~uint64_t{0}) {
        const uint64_t cap = mem_available_bytes / kHugePage2M * kHugePage2M;
        if (pool.on_demand > cap) pool.on_demand = cap;
    }
    return pool;
}

inline HugePool huge_pool_2m() {
#ifdef __linux__
    auto num = [](const char* path) -> uint64_t {
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) return 0;
        unsigned long long v = 0;
        const bool ok = std::fscanf(f, "%llu", &v) == 1;
        std::fclose(f);
        return ok ? (uint64_t) v : 0;
    };
    uint64_t available = ~uint64_t{0};
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                available = (uint64_t) kb * 1024;
                break;
            }
        }
        std::fclose(f);
    }
    return huge_pool_from(num("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"),
                          num("/sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages"),
                          num("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages"),
                          num("/sys/kernel/mm/hugepages/hugepages-2048kB/surplus_hugepages"), available);
#else
    return {};
#endif
}

inline uint64_t huge_round_up(uint64_t bytes) {
    if (bytes > ~uint64_t{0} - (kHugePage2M - 1)) return ~uint64_t{0};
    return (bytes + kHugePage2M - 1) / kHugePage2M * kHugePage2M;
}

inline bool huge_on(int env, const HugePool& pool, uint64_t need) {
    if (env == 0 || need == 0) return false;
    return env == 1 || pool.total() >= huge_round_up(need);
}

// Tests and diagnostics: fill replaces fallocate(fd, 0, off, len), returning 0 or -1 with errno set.
// On Linux, keep_fd receives the built file's fd for the caller to close, or -1 on failure or anonymous fallback;
// without keep_fd the file is closed here. Both optional parameters are ignored on other platforms.
inline void* huge_alloc_pieces(size_t bytes, int node, int nodes, size_t* got, size_t piece, int tries,
                               int pause_ms, int (*fill)(int fd, off_t off, off_t len) = nullptr,
                               int* keep_fd = nullptr) {
    if (got != nullptr) *got = 0;
#ifdef __linux__
    if (keep_fd != nullptr) *keep_fd = -1;
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
    if (bytes == 0 || bytes > ~size_t{0} - (kHugePage2M - 1) ||
        piece == 0 || piece > ~size_t{0} - (kHugePage2M - 1) || tries <= 0 || pause_ms < 0)
        return nullptr;
    const size_t hb = (size_t) huge_round_up(bytes);
    if (hb > huge_pool_2m().total()) return nullptr;
    piece = (size_t) huge_round_up(piece);
    int fd = -1;
    void* m = MAP_FAILED;
#ifdef SYS_memfd_create
    struct rlimit limit;
    if (hb <= (uint64_t) std::numeric_limits<off_t>::max() && ::getrlimit(RLIMIT_FSIZE, &limit) == 0 &&
        (limit.rlim_cur == RLIM_INFINITY || hb <= limit.rlim_cur)) {
        fd = (int) syscall(SYS_memfd_create, "strata-huge", MFD_CLOEXEC | MFD_HUGETLB | (21 << 26));
        if (fd >= 0) {
            if (::ftruncate(fd, (off_t) hb) == 0)
                m = ::mmap(nullptr, hb, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
            if (m == MAP_FAILED) {
                ::close(fd);
                fd = -1;
            }
        }
    }
#endif
    if (m == MAP_FAILED)
        m = ::mmap(nullptr, hb, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT), -1, 0);
    if (m == MAP_FAILED) return nullptr;
    struct ThreadPolicy {
        int mode = 0;
        unsigned long local[4] = {0, 0, 0, 0};
        std::vector<unsigned long> extra;
        unsigned long maxnode = 0;
        bool restore = false;
        ~ThreadPolicy() {
            if (restore)
                syscall(SYS_set_mempolicy, mode, extra.empty() ? local : extra.data(), maxnode);
        }
    } policy;
    try {
        if (nodes > 1 && (node >= 0 || node == kHugeInterleave)) {
            constexpr size_t bits = sizeof(unsigned long) * 8;
            const size_t count = node >= 0 ? (size_t) node + 1 : (size_t) nodes;
            // maxnode must include a spare bit: Linux reads maxnode - 1 bits from the mask.
            unsigned long local[4] = {0, 0, 0, 0};
            std::vector<unsigned long> extra;
            const size_t words = count < 4 * bits ? 4 : count / bits + 1;
            if (words > 4) extra.resize(words, 0);
            unsigned long* mask = words > 4 ? extra.data() : local;
            if (node >= 0) mask[(size_t) node / bits] = 1ul << ((size_t) node % bits);
            else
                for (int n = 0; n < nodes; ++n) mask[(size_t) n / bits] |= 1ul << ((size_t) n % bits);
            const int mode = node >= 0 ? 1 /* MPOL_PREFERRED */ : 3 /* MPOL_INTERLEAVE */;
            const auto maxnode = (unsigned long) (words * bits);
            syscall(SYS_mbind, m, hb, mode, mask, maxnode, 0);
            if (fd >= 0) {
                // hugetlbfs fallocate uses the thread's policy, not the mapping's mbind policy.
                if (words > 4) policy.extra.resize(words, 0);
                unsigned long* saved = words > 4 ? policy.extra.data() : policy.local;
                if (syscall(SYS_get_mempolicy, &policy.mode, saved, maxnode, nullptr, 0) == 0 &&
                    syscall(SYS_set_mempolicy, mode, mask, maxnode) == 0) {
                    policy.maxnode = maxnode;
                    policy.restore = true;
                }
            }
        }
        if (fd >= 0) {
            for (size_t off = 0; off < hb;) {
                const size_t len = piece < hb - off ? piece : hb - off;
                int failed = 0;
                while ((fill != nullptr ? fill(fd, (off_t) off, (off_t) len)
                                        : ::fallocate(fd, 0, (off_t) off, (off_t) len)) != 0) {
                    if (errno == EINTR) continue;
                    if (++failed >= tries) {
                        ::munmap(m, hb);
                        ::close(fd);
                        return nullptr;
                    }
                    if (pause_ms > 0) {
                        timespec pause{pause_ms / 1000, (long) (pause_ms % 1000) * 1000000L};
                        while (::nanosleep(&pause, &pause) != 0 && errno == EINTR) {}
                    }
                }
                off += len;
            }
            if (keep_fd != nullptr) *keep_fd = fd;
            else ::close(fd);
        }
    } catch (...) {
        ::munmap(m, hb);
        if (fd >= 0) ::close(fd);
        return nullptr;
    }
    if (got != nullptr) *got = hb;
    return m;
#else
    (void) bytes, (void) node, (void) nodes, (void) piece, (void) tries, (void) pause_ms;
    (void) fill, (void) keep_fd;
    return nullptr;
#endif
}

// Build a shared mapping of an unnamed hugetlbfs file in pieces: with 224 GiB cached and 2 GiB free, one
// 60 GiB reservation failed in 0.1 s; 1 GiB pieces built the same block in 24 s with 5 retries. MAP_NORESERVE
// is safe here: every page is in the file before return, so no later fault can fail for want of pages.
// These are still explicit 2 MB pages, skipped by the compactor. An unavailable file path uses the old
// anonymous mapping, which reserves the whole block without MAP_NORESERVE.
inline void* huge_alloc(size_t bytes, int node, int nodes, size_t* got) {
    return huge_alloc_pieces(bytes, node, nodes, got, kHugePiece, kHugeTries, kHugePauseMs);
}

inline void huge_free(void* p, size_t got) {
#ifdef __linux__
    if (p != nullptr) ::munmap(p, got);
#else
    (void) p, (void) got;
#endif
}

}  // namespace strata::platform
