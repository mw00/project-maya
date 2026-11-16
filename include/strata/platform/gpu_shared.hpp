// include/strata/platform/gpu_shared.hpp - Windows' shared GPU memory: how much there is for an adapter (DXGI), and
// how much of it every GPU user holds now (the "GPU Adapter Memory" performance counters: Task Manager's "Shared GPU
// memory").
//
// On a Windows APU (a carve-out the BIOS dedicates to the GPU, plus system memory the GPU maps), every pinned host
// buffer the GPU can reach, and every device allocation that does not fit in the carve-out, is shared GPU memory.
// Windows keeps all its users together - the desktop's included - within DXGI's SharedSystemMemory (half the RAM),
// and HIP's pinning does not refuse past that: the first GPU submission that needs the buffers resident fails instead
// ("PAL failed to submit CMD" in AMD_LOG_LEVEL=1, then "unspecified launch failure", and the HIP context is lost).
// Measured on a Radeon 8065S (160 GB carve-out, 15.84 GB shared, 2.56 GB of it held by the desktop): 13.02 GB of
// pinned buffers ran and 14.02 GB failed; 159 GB of device memory (3 GB of it past the carve-out) ran with 10 GB
// pinned and failed with 11.  Free RAM was not the limit: 10 GB pinned with 7 GB of RAM free ran.
// pdh.dll and dxgi.dll are loaded here, so no program links them for this.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace strata::platform {

/// DXGI's SharedSystemMemory for the adapter with this LUID (8 bytes: a device property's luid), or - no adapter with
/// it - the one AMD adapter's; 0 when unknown, and off Windows.
inline uint64_t shared_limit_bytes(const void* luid) {
#if defined(_WIN32)
    HMODULE m = LoadLibraryA("dxgi.dll");
    if (m == nullptr) return 0;
    using Create = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = reinterpret_cast<Create>(reinterpret_cast<void*>(GetProcAddress(m, "CreateDXGIFactory1")));
    static const GUID kFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    IDXGIFactory1* f = nullptr;
    uint64_t shared = 0, amd = 0;
    int n_amd = 0;
    if (create != nullptr && SUCCEEDED(create(kFactory1, (void**) &f)) && f != nullptr) {
        IDXGIAdapter1* a = nullptr;
        for (UINT i = 0; f->EnumAdapters1(i, &a) == S_OK; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            if (SUCCEEDED(a->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                if (luid != nullptr && std::memcmp(&d.AdapterLuid, luid, sizeof(d.AdapterLuid)) == 0)
                    shared = d.SharedSystemMemory;
                if (d.VendorId == 0x1002) {
                    amd = d.SharedSystemMemory;
                    ++n_amd;
                }
            }
            a->Release();
        }
        f->Release();
    }
    FreeLibrary(m);
    return shared > 0 ? shared : n_amd == 1 ? amd : 0;   // (no LUID match: the one AMD adapter)
#else
    (void) luid;
    return 0;
#endif
}

/// The shared GPU memory all users of all adapters hold now: the sum of "\GPU Adapter Memory(*)\Shared Usage" over
/// its instances (a second GPU's share counts too, which only makes a cap smaller); -1 when it cannot be read (off
/// Windows, or no such counters).
inline int64_t shared_usage_bytes() {
#if defined(_WIN32)
    HMODULE m = LoadLibraryA("pdh.dll");
    if (m == nullptr) return -1;
    using Open = PDH_STATUS(WINAPI*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
    using Add = PDH_STATUS(WINAPI*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
    using Collect = PDH_STATUS(WINAPI*)(PDH_HQUERY);
    using Values = PDH_STATUS(WINAPI*)(PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W);
    using Close = PDH_STATUS(WINAPI*)(PDH_HQUERY);
    const auto open = reinterpret_cast<Open>(reinterpret_cast<void*>(GetProcAddress(m, "PdhOpenQueryW")));
    const auto add = reinterpret_cast<Add>(reinterpret_cast<void*>(GetProcAddress(m, "PdhAddEnglishCounterW")));
    const auto collect = reinterpret_cast<Collect>(reinterpret_cast<void*>(GetProcAddress(m, "PdhCollectQueryData")));
    const auto values = reinterpret_cast<Values>(reinterpret_cast<void*>(GetProcAddress(m, "PdhGetFormattedCounterArrayW")));
    const auto close = reinterpret_cast<Close>(reinterpret_cast<void*>(GetProcAddress(m, "PdhCloseQuery")));
    int64_t sum = -1;
    PDH_HQUERY q = nullptr;
    if (open != nullptr && add != nullptr && collect != nullptr && values != nullptr && close != nullptr &&
        open(nullptr, 0, &q) == ERROR_SUCCESS) {
        PDH_HCOUNTER c = nullptr;
        if (add(q, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &c) == ERROR_SUCCESS && collect(q) == ERROR_SUCCESS) {
            DWORD bytes = 0, n = 0;
            if (values(c, PDH_FMT_LARGE, &bytes, &n, nullptr) == (PDH_STATUS) PDH_MORE_DATA && bytes > 0) {
                std::vector<unsigned char> buf((size_t) bytes);
                auto* it = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
                if (values(c, PDH_FMT_LARGE, &bytes, &n, it) == ERROR_SUCCESS && n > 0) {
                    sum = 0;
                    for (DWORD i = 0; i < n; ++i)
                        if ((it[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA ||
                             it[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) && it[i].FmtValue.largeValue > 0)
                            sum += it[i].FmtValue.largeValue;
                }
            }
        }
        close(q);
    }
    FreeLibrary(m);
    return sum;
#else
    return -1;
#endif
}

}  // namespace strata::platform
