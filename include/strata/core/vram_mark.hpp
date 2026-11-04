#pragma once
// The free and total VRAM of the GPUs in use, taken before and after a prompt read and logged on ONE line next to its
// token count and time (issues #648, #438: on Windows the driver can move idle buffers of a full card to system memory,
// which slows a read; the line shows how full the card was).  Two cudaMemGetInfo calls a GPU per prompt, both from the
// request thread between the prompt's own syncs - never from a thread the device may be waiting on.
#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

struct VramMark {
    std::vector<int> dev;
    std::vector<double> free_gb, total_gb;
};

/// Free and total VRAM of each of `devs` (empty: every visible GPU).  A GPU that cannot be asked is left out.
inline VramMark vram_mark(std::vector<int> devs) {
    VramMark m;
    int cur = 0;
    if (cudaGetDevice(&cur) != cudaSuccess) {
        cudaGetLastError();
        return m;
    }
    if (devs.empty()) {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) cudaGetLastError();
        for (int i = 0; i < n; ++i) devs.push_back(i);
    }
    for (int d : devs) {
        size_t fb = 0, tb = 0;
        if (cudaSetDevice(d) != cudaSuccess || cudaMemGetInfo(&fb, &tb) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        m.dev.push_back(d);
        m.free_gb.push_back((double) fb / 1073741824.0);
        m.total_gb.push_back((double) tb / 1073741824.0);
    }
    cudaSetDevice(cur);
    return m;
}

/// "VRAM free CUDA0 1.2 of 20.0 GB before, 0.9 after" (several GPUs: "CUDA0 1.2 of 20.0, CUDA1 3.1 of 32.0 GB before,
/// 0.9, 2.8 after"); "" when nothing could be read.
inline std::string vram_mark_text(const VramMark& a, const VramMark& b) {
    if (a.dev.empty() || a.dev != b.dev) return std::string();
    char buf[96];
    std::string s = "VRAM free ";
    for (size_t i = 0; i < a.dev.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%sCUDA%d %.1f of %.1f", i ? ", " : "", a.dev[i], a.free_gb[i], a.total_gb[i]);
        s += buf;
    }
    s += " GB before, ";
    for (size_t i = 0; i < b.dev.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%s%.1f", i ? ", " : "", b.free_gb[i]);
        s += buf;
    }
    s += " after";
    return s;
}

}  // namespace strata::core
