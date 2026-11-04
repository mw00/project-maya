// include/strata/platform/dedicated.hpp - an APU's dedicated GPU memory on Windows, as DXGI reports it.
//
// On Windows, HIP's total memory for an integrated GPU is its carve-out (the memory the BIOS dedicates to it) plus
// shared system memory: a Radeon 8065S with 160 GB of its own reports ~172.  What the engines put in the shared part
// pages under Windows - GLM's 8K-token prompts ran at 62 instead of 159 tok/s (#206), DeepSeek's Maya-S decoded at
// 2.45 tok/s on a 160 GB Strix Halo (v1.1.19) - so both keep their expert pools within the dedicated memory
// (glm_fast_path.cu pool_check, ds4_model.cu mem_info).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#endif

namespace strata::platform {

/// The dedicated video memory DXGI reports for the adapter with this LUID (8 bytes: a device property's luid), or -
/// no adapter with it - the one AMD adapter's; 0 when unknown, and off Windows.  dxgi.dll is loaded here, so no
/// program links it for this.
inline uint64_t dedicated_bytes(const void* luid) {
#if defined(_WIN32)
    HMODULE m = LoadLibraryA("dxgi.dll");
    if (m == nullptr) return 0;
    using Create = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = reinterpret_cast<Create>(reinterpret_cast<void*>(GetProcAddress(m, "CreateDXGIFactory1")));
    static const GUID kFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    IDXGIFactory1* f = nullptr;
    uint64_t ded = 0, amd = 0;
    int n_amd = 0;
    if (create != nullptr && SUCCEEDED(create(kFactory1, (void**) &f)) && f != nullptr) {
        IDXGIAdapter1* a = nullptr;
        for (UINT i = 0; f->EnumAdapters1(i, &a) == S_OK; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            if (SUCCEEDED(a->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                if (luid != nullptr && std::memcmp(&d.AdapterLuid, luid, sizeof(d.AdapterLuid)) == 0)
                    ded = d.DedicatedVideoMemory;
                if (d.VendorId == 0x1002) {
                    amd = d.DedicatedVideoMemory;
                    ++n_amd;
                }
            }
            a->Release();
        }
        f->Release();
    }
    FreeLibrary(m);
    return ded > 0 ? ded : n_amd == 1 ? amd : 0;   // (no LUID match: the one AMD adapter)
#else
    (void) luid;
    return 0;
#endif
}

}  // namespace strata::platform
