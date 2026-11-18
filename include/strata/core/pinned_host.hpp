// include/strata/core/pinned_host.hpp - large page-locked host blocks on explicit 2 MB pages when asked and
// available.  Huge blocks are always registered Mapped.  Unless the caller copies or translates the pointer,
// every visible GPU must read them at their host address (HIP: where the attribute says 0, the registered block's
// own device address decides).  A refused huge path takes cudaHostAlloc unchanged.
#pragma once

#include "strata/platform/huge_pages.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::core {

enum class PinnedBacking { None, CudaHostAlloc, HugeTlb };
// NoHostDereference: GPU access is only through copies or cudaHostGetDevicePointer, never the host address.
enum class PinnedHostAccess { HostAddress, NoHostDereference };
constexpr uint64_t kPinnedHugeMin = (uint64_t) 32 << 20;

inline int pinned_huge_env(const char* value) {
    if (value == nullptr || value[0] == '\0') return -1;
    return std::atoi(value) == 0 ? 0 : 1;
}

inline bool pinned_huge_wanted(int env, uint64_t need) {
#ifndef __linux__
    (void) env, (void) need;
    return false;
#else
    if (env == 0 || need == 0) return false;
    if (env == 1) return true;
#ifdef STRATA_USE_HIP
    return false;   // AMD: measured on one card model (Radeon AI PRO R9700) only: explicit opt-in
#else
    return need >= kPinnedHugeMin && platform::huge_on(-1, platform::huge_pool_2m(), need);
#endif
#endif
}

namespace pinned_host_detail {
struct Registry {
    std::mutex mutex;
    std::unordered_map<const void*, size_t> blocks;
};
// An inline accessor's local static is shared by all host translation units, including CUDA's host code.
inline Registry& registry() {
    static Registry r;
    return r;
}
// A block the driver will not unregister keeps its mapping: its address range must not be handed out again while
// the registration may still be live.  That block is lost until the process ends, and a line says so.
inline cudaError_t unregister(void* m, size_t got) {
    const cudaError_t err = cudaHostUnregister(m);
    if (err != cudaSuccess)
        std::fprintf(stderr, "pinned host: a %.2f GB block on 2 MB huge pages could not be unregistered (%s); it "
                             "stays mapped\n", (double) got / 1073741824.0, cudaGetErrorString(err));
    return err;
}
inline cudaError_t release(void* m, size_t got) {
    const cudaError_t err = unregister(m, got);
    if (err == cudaSuccess) platform::huge_free(m, got);
    return err;
}
#ifdef STRATA_USE_HIP
// Discrete AMD GPUs report 0 for CanUseHostPointerForRegisteredMem, yet a registered block may still be visible at
// its host address.  Check the block's actual device address on every GPU; the current device is left as it was.
// A lookup that fails is an answer (*same false).  A current device that cannot be put back is an error: the
// caller must not go on, or allocate its fallback, with another GPU current.
inline cudaError_t same_address_everywhere(void* m, size_t got, int count, bool* same_out) {
    *same_out = false;
    int cur = 0;
    if (cudaGetDevice(&cur) != cudaSuccess) {
        cudaGetLastError();   // Clear only the failed query.
        return cudaSuccess;
    }
    bool same = true;
    char* const last = (char*) m + (got - platform::kHugePage2M);   // (the block's last huge page too)
    for (int dev = 0; same && dev < count; ++dev) {
        if (cudaSetDevice(dev) != cudaSuccess) {
            cudaGetLastError();
            same = false;
            break;
        }
        for (char* h : {(char*) m, last}) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
                cudaGetLastError();   // Clear only the failed lookup.
                same = false;
                break;
            }
            if (d != (void*) h) {
                same = false;
                break;
            }
        }
    }
    const cudaError_t restored = cudaSetDevice(cur);
    if (restored != cudaSuccess) return restored;
    *same_out = same;
    return cudaSuccess;
}
#endif
}  // namespace pinned_host_detail

inline cudaError_t pinned_host_alloc(void** p, size_t bytes, unsigned flags, bool huge,
                                     PinnedHostAccess access = PinnedHostAccess::HostAddress,
                                     PinnedBacking* backing = nullptr, int node = -1, int nodes = 1) {
    if (backing != nullptr) *backing = PinnedBacking::None;
    if (huge && p != nullptr && (flags & ~(cudaHostAllocPortable | cudaHostAllocMapped)) == 0) {
        // Preserve cudaHostAlloc's host-address access unless the caller does not need it.
        bool usable = true;
#ifdef STRATA_USE_HIP
        bool check_address = false;   // (the attribute said 0; the address is checked after registration)
#endif
        int count = 0;
        if (access == PinnedHostAccess::HostAddress) {
            if (cudaGetDeviceCount(&count) != cudaSuccess) {
                cudaGetLastError();   // Clear only the failed CUDA call made here.
                usable = false;
            } else if (count == 0) usable = false;
            for (int dev = 0; usable && dev < count; ++dev) {
                int ok = 0;
                if (cudaDeviceGetAttribute(&ok, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess) {
                    cudaGetLastError();   // Clear only the failed attribute query.
                    usable = false;
                } else if (!ok) {
#ifdef STRATA_USE_HIP
                    check_address = true;
#else
                    usable = false;
#endif
                }
            }
        }
        size_t got = 0;
        if (void* m = usable ? platform::huge_alloc(bytes, node, nodes, &got) : nullptr) {
            // Map each huge page before CUDA registers it; the file path already placed the pages.
            for (size_t off = 0; off < got; off += platform::kHugePage2M) ((volatile char*) m)[off] = 0;
            unsigned reg = cudaHostRegisterMapped;
            if ((flags & cudaHostAllocPortable) != 0) reg |= cudaHostRegisterPortable;
            bool registered = cudaHostRegister(m, got, reg) == cudaSuccess;
            if (!registered) {
                cudaGetLastError();   // Clear only the failed registration.
                platform::huge_free(m, got);
            }
#ifdef STRATA_USE_HIP
            if (registered && check_address) {
                bool same = false;
                const cudaError_t restored = pinned_host_detail::same_address_everywhere(m, got, count, &same);
                cudaError_t released = cudaSuccess;
                if (!same) {
                    released = pinned_host_detail::release(m, got);
                    registered = false;
                }
                if (restored != cudaSuccess) return restored;
                if (released != cudaSuccess) return released;   // (no second block beside one that is lost)
            }
#endif
            if (registered) {
                try {
                    auto& r = pinned_host_detail::registry();
                    std::lock_guard<std::mutex> lk(r.mutex);
                    r.blocks.emplace(m, got);
                    *p = m;
                    if (backing != nullptr) *backing = PinnedBacking::HugeTlb;
                    return cudaSuccess;
                } catch (...) {
                    // No untracked registered block: a registry allocation failure takes the fallback too, unless
                    // the block could not be released (no second block beside one that is lost).
                    const cudaError_t released = pinned_host_detail::release(m, got);
                    if (released != cudaSuccess) return released;
                }
            }
        }
    }
    const cudaError_t err = cudaHostAlloc(p, bytes, flags);
    if (err == cudaSuccess && backing != nullptr) *backing = PinnedBacking::CudaHostAlloc;
    return err;
}

inline cudaError_t pinned_host_free(void* p) {
    if (p == nullptr) return cudaSuccess;
    size_t got = 0;
    auto& r = pinned_host_detail::registry();
    {
        std::lock_guard<std::mutex> lk(r.mutex);
        const auto it = r.blocks.find(p);
        if (it != r.blocks.end()) got = it->second;
    }
    if (got == 0) return cudaFreeHost(p);
    const cudaError_t err = pinned_host_detail::unregister(p, got);
    if (err != cudaSuccess) return err;   // (still recorded, still mapped)
    {   // The record goes before the mapping: the address may be handed out again at once.
        std::lock_guard<std::mutex> lk(r.mutex);
        r.blocks.erase(p);
    }
    platform::huge_free(p, got);
    return cudaSuccess;
}

inline PinnedBacking pinned_host_backing(const void* p) {
    auto& r = pinned_host_detail::registry();
    std::lock_guard<std::mutex> lk(r.mutex);
    return r.blocks.find(p) != r.blocks.end() ? PinnedBacking::HugeTlb : PinnedBacking::None;
}

}  // namespace strata::core
