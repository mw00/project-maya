// src/ds4/ds4_peer_vmm.hpp - device memory that another GPU of the same PC reaches by peer stores and loads, granted per
// allocation (CUDA's virtual memory management: cuMemCreate + cuMemSetAccess) instead of per context
// (cudaDeviceEnablePeerAccess).
//
// Why: the per-context grant maps the peer's whole context through its BAR1 aperture.  On cards whose BAR1 is smaller
// than what the engine holds on them (an RTX 4090 D: 32 GiB, DeepSeek EXL3 45-48 GB a GPU) it fails at the end of the
// load ("out of memory"), and once enabled no more than BAR1 can be allocated at all.  Granted per allocation, only these
// buffers are mapped for the peer - a few hundred MiB - and every other allocation stays the GPU's own.
//
// MAYA_DS4_PEER_VMM=1 (CUDA builds; HIP keeps its own path): the tensor-parallel pair's peer prompt all-reduce
// (MAYA_DS4_HIP_PEER_AR) and DS4_AR_P2P's buffers take this memory and never call cudaDeviceEnablePeerAccess.  The
// driver entry points are looked up at run time (the engine does not link libcuda).
#pragma once
#if !defined(MAYA_USE_HIP)
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace maya::ds4::vmm {

struct Api {
    CUresult (*granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*addr_free)(CUdeviceptr, size_t) = nullptr;
    bool ok = false;
    std::string why;
};

inline const Api& api() {
    static Api a;
    static std::once_flag once;
    std::call_once(once, [] {
        auto get = [](const char* name, void** fn, std::string& why) {
            cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            const cudaError_t e = cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q);
#else
            const cudaError_t e = cudaGetDriverEntryPoint(name, fn, cudaEnableDefault, &q);
#endif
            if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || *fn == nullptr) {
                (void) cudaGetLastError();
                if (why.empty()) why = std::string("the driver has no ") + name;
                return false;
            }
            return true;
        };
        bool ok = true;
        ok = get("cuMemGetAllocationGranularity", (void**) &a.granularity, a.why) && ok;
        ok = get("cuMemCreate", (void**) &a.create, a.why) && ok;
        ok = get("cuMemAddressReserve", (void**) &a.reserve, a.why) && ok;
        ok = get("cuMemMap", (void**) &a.map, a.why) && ok;
        ok = get("cuMemSetAccess", (void**) &a.set_access, a.why) && ok;
        ok = get("cuMemUnmap", (void**) &a.unmap, a.why) && ok;
        ok = get("cuMemRelease", (void**) &a.release, a.why) && ok;
        ok = get("cuMemAddressFree", (void**) &a.addr_free, a.why) && ok;
        a.ok = ok;
    });
    return a;
}

struct Buf {
    void* ptr = nullptr;
    size_t size = 0;
    CUmemGenericAllocationHandle h = 0;
    int dev = -1;
    bool reserved = false, mapped = false;
};

inline void release(Buf& b) {
    const Api& a = api();
    if (!a.ok) return;
    if (b.mapped) (void) a.unmap((CUdeviceptr) b.ptr, b.size);
    if (b.h != 0) (void) a.release(b.h);
    if (b.reserved) (void) a.addr_free((CUdeviceptr) b.ptr, b.size);
    b = Buf{};
}

// `bytes` of device memory on GPU `dev`, read and written by `dev` and by every GPU in `peers` (nothing else of either
// GPU's memory is mapped for the other), zeroed.  Leaves the current device as it found it.  On failure: false, `why`
// says which step, nothing left allocated.
inline bool alloc(int dev, const std::vector<int>& peers, size_t bytes, Buf& b, std::string& why) {
    const Api& a = api();
    if (!a.ok) {
        why = a.why;
        return false;
    }
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(dev);
    (void) cudaFree(nullptr);   // (the device's primary context current for the driver calls)
    auto fail = [&](const char* step, CUresult r) {
        why = std::string(step) + " failed (CUresult " + std::to_string((int) r) + ") on GPU" + std::to_string(dev);
        release(b);
        cudaSetDevice(cur);
        return false;
    };
    b = Buf{};
    b.dev = dev;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gran = 0;
    CUresult r = a.granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (r != CUDA_SUCCESS || gran == 0) return fail("cuMemGetAllocationGranularity", r);
    b.size = (bytes + gran - 1) / gran * gran;
    if ((r = a.create(&b.h, b.size, &prop, 0)) != CUDA_SUCCESS) {
        b.h = 0;
        return fail("cuMemCreate", r);
    }
    CUdeviceptr p = 0;
    if ((r = a.reserve(&p, b.size, gran, 0, 0)) != CUDA_SUCCESS) return fail("cuMemAddressReserve", r);
    b.ptr = (void*) p;
    b.reserved = true;
    if ((r = a.map(p, b.size, 0, b.h, 0)) != CUDA_SUCCESS) return fail("cuMemMap", r);
    b.mapped = true;
    std::vector<CUmemAccessDesc> acc(1 + peers.size());
    for (size_t i = 0; i < acc.size(); ++i) {
        acc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc[i].location.id = i == 0 ? dev : peers[i - 1];
        acc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    if ((r = a.set_access(p, b.size, acc.data(), acc.size())) != CUDA_SUCCESS) return fail("cuMemSetAccess", r);
    if (cudaMemset(b.ptr, 0, b.size) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return fail("cudaMemset", CUDA_ERROR_UNKNOWN);
    }
    cudaSetDevice(cur);
    return true;
}

inline bool wanted() {
    const char* e = std::getenv("MAYA_DS4_PEER_VMM");
    if (e == nullptr) e = std::getenv("DS4_PEER_VMM");
    return e != nullptr && std::atoi(e) != 0;
}

}  // namespace maya::ds4::vmm
#endif
