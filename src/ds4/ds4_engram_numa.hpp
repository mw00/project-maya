// Startup-only placement of locked Engram file pages. No tensor arithmetic or file writes.
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::ds4 {
inline bool engram_numa_on() {
    const char* v = std::getenv("STRATA_DS4_ENGRAM_RAM_NUMA");
    if (!v) v = std::getenv("MAYA_DS4_ENGRAM_RAM_NUMA");
    if (!v) v = std::getenv("DS4_ENGRAM_RAM_NUMA");
    return v && std::strcmp(v, "1") == 0;
}

#if defined(__linux__)
class EngramNuma {
public:
    static constexpr size_t kStripe = size_t{2} << 20;
    using Bind = long (*)(void*, size_t, int, unsigned long, unsigned);
    explicit EngramNuma(Bind bind = system_bind) : bind_(bind) {}
    bool init(std::string& err) {
        page_ = ::sysconf(_SC_PAGESIZE);
        if (page_ <= 0 || kStripe % (size_t) page_ != 0)
            return fail(err, "unsupported page size");
        unsigned long mask[4] = {};
        if (::syscall(SYS_get_mempolicy, nullptr, mask, 256ul, nullptr, MPOL_F_MEMS_ALLOWED) != 0)
            return fail(err, "get_mempolicy", errno);
        if (mask[0] != 3ul || mask[1] || mask[2] || mask[3])
            return fail(err, "requires allowed memory nodes exactly 0 and 1");
        return true;
    }
    bool balance(uint8_t* base, size_t bytes, uint64_t file_offset, std::string& err) {
        if (page_ <= 0 || (uintptr_t) base % (size_t) page_ || file_offset % (size_t) page_ ||
            bytes > UINT64_MAX - file_offset) return fail(err, "invalid mapped range");
        for (size_t pos = 0; pos < bytes;) {
            const uint64_t off = file_offset + pos;
            const size_t extent = std::min(bytes - pos, kStripe - (size_t) (off % kStripe));
            const size_t n = (extent + (size_t) page_ - 1) / (size_t) page_;
            const int node = (int) ((off / kStripe) % 2);
            const unsigned long mask = 1ul << node;
            // Every page must already be present before STRICT scans it. No stores and no shadow copy.
            for (size_t i = 0; i < n; ++i) (void) *static_cast<volatile const uint8_t*>(base + pos + i * (size_t) page_);
            uint8_t* p = base + pos;
            // A single-node STRICT check detects misplaced warm cache pages. Unlike INTERLEAVE over{0,1},
            // it cannot accept an all-node1 range just because node1 belongs to the allowed set.
            if (bind_(p, extent, MPOL_BIND, mask, MPOL_MF_STRICT) != 0) {
                const int before_error = errno;
                if (before_error != EIO) return clear_and_fail(p, extent, err, "initial strict check", before_error);
                if (bind_(p, extent, MPOL_BIND, mask, MPOL_MF_MOVE_ALL | MPOL_MF_STRICT) != 0)
                    return clear_and_fail(p, extent, err, "strict migration incomplete; candidate refused", errno);
                ++relocated_batches;
            }
            if (bind_(p, extent, MPOL_BIND, mask, MPOL_MF_STRICT) != 0)
                return clear_and_fail(p, extent, err, "post-migration strict verification", errno);
            // Restore policy immediately: shared file pages retain migrated physical homes, but tens of
            // thousands of stripe VMAs must not accumulate, nor should future allocations be hard-bound.
            if (bind_(p, extent, MPOL_DEFAULT, 0, 0) != 0)
                return fail(err, "restore mapping policy; candidate refused", errno);
            verified[(size_t) node] += n;
            pos += extent;
        }
        return true;
    }
    uint64_t relocated_batches = 0;
    std::array<uint64_t, 2> verified{};
private:
    static long system_bind(void* p, size_t n, int mode, unsigned long mask, unsigned flags) {
        return ::syscall(SYS_mbind, p, n, mode, &mask, sizeof(mask) * 8ul, flags);
    }
    bool clear_and_fail(void* p, size_t n, std::string& err, const char* what, int code) {
        const bool cleared = bind_(p, n, MPOL_DEFAULT, 0, 0) == 0;
        fail(err, what, code);
        err += cleared ? "; policy restored; completed/partial page moves can persist"
                       : "; policy reset also failed; completed/partial page moves can persist";
        return false;
    }
    static bool fail(std::string& err, const char* what, int code = 0) {
        err = std::string("Engram RAM NUMA: ") + what;
        if (code) err += ": " + std::string(std::strerror(code));
        return false;
    }
    Bind bind_;
    long page_ = 0;
};
#endif
}  // namespace strata::ds4
