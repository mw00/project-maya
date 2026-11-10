// ablate_capture.hpp - the read-out tools/ablate_fit.py fits a refusal direction from (MAYA_ABLATE_CAPTURE=<file>).
//
// With the variable set, the engines' refusal-ablation hook (every residual write of the main layers: the attention's
// and the FFN's output, right before it joins the residual) does not project: it appends the write's LAST row - the
// last prompt token's, in a prompt pass - to <file> instead, one record a write:
//     int32 T (rows of the pass), int32 dim, float row[dim]
// in the order the layers run (two writes a layer).  The hook runs only for a conversation that projects, so a
// capture run loads any direction of the model's hidden size (--ablate FILE) and asks for ablation; the answers it
// gives are the plain model's.  Fitting runs only: the host waits for each write.  Include after the CUDA (or HIP
// compat) runtime header.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ablcap {

inline std::FILE* file() {
    static std::FILE* f = [] {
        const char* p = std::getenv("MAYA_ABLATE_CAPTURE");
        return p != nullptr && *p ? std::fopen(p, "ab") : nullptr;
    }();
    return f;
}

inline bool on() { return file() != nullptr; }

/// Appends the last of y's T rows (dim floats each, on any GPU) as one record; waits for stream s first.
inline void put(const float* y, int T, int dim, cudaStream_t s) {
    std::FILE* f = file();
    if (f == nullptr || y == nullptr || T <= 0 || dim <= 0) return;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    std::vector<float> row((size_t) dim);
    if (cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemcpy(row.data(), y + (size_t) (T - 1) * (size_t) dim, (size_t) dim * sizeof(float),
                   cudaMemcpyDefault) != cudaSuccess) {
        std::fprintf(stderr, "MAYA_ABLATE_CAPTURE: a row could not be read back\n");
        return;
    }
    const int32_t h[2] = {(int32_t) T, (int32_t) dim};
    std::fwrite(h, sizeof(int32_t), 2, f);
    std::fwrite(row.data(), sizeof(float), (size_t) dim, f);
    std::fflush(f);
}

}  // namespace strata::ablcap
