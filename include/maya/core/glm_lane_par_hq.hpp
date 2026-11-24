// include/maya/core/glm_lane_par_hq.hpp - lock-free per-256-float-block activation quantization tracker for the
// CPU expert lane (MAYA_GLM_LANE_PAR_HQ=1).
//
// In Glm5Model::lane_experts, workers compute slices [r0, r1) of each routed expert's gate/up output `ff` before
// quantizing `ff` into `hq` for the down projection. When every 256-float block is quantized independently (which
// holds for all GGML activation formats whose block size divides 256: Q8_K=256, Q8_0=32, F16/F32=1), tracking how
// many rows of each 256-float block have completed lets the worker that finishes the 256th row quantize that block
// immediately while it is still warm in L1/L2 cache, avoiding the serial single-thread pass between gate/up and down.
// Header-only, host C++, no CUDA or ggml dependency.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>

namespace maya::core::glmlane {

class ParHqTracker {
public:
    static constexpr int kBlockSize = 256;
    static constexpr int kMaxExperts = 8;
    static constexpr int kMaxBlocksPerExpert = 16;   // covers n_ff up to 4096
    static constexpr int kMaxCounters = kMaxExperts * kMaxBlocksPerExpert;

    static constexpr bool eligible(int ne, int n_ff, size_t h_bytes) {
        if (ne < 1 || ne > kMaxExperts || n_ff <= 0 || (n_ff % kBlockSize) != 0) return false;
        const int n_hb = n_ff / kBlockSize;
        if (n_hb > kMaxBlocksPerExpert || h_bytes == 0 || (h_bytes % (size_t) n_hb) != 0) return false;
        return true;
    }

    void reset(int ne, int n_ff) {
        n_hb_ = n_ff / kBlockSize;
        const int total = ne * n_hb_;
        for (int b = 0; b < total; ++b) done_[b].store(0, std::memory_order_relaxed);
    }

    /// Record completion of rows [r0, r1) of expert `expert_idx`. For every 256-float block whose final row lands in
    /// this call, invokes `quant_block(expert_idx, block_idx)` exactly once with acquire-release visibility of all
    /// rows written to that block by other workers.
    template <class QuantBlockFn>
    void mark_rows(int expert_idx, int r0, int r1, QuantBlockFn&& quant_block) {
        if (r1 <= r0) return;
        const int b0 = r0 / kBlockSize;
        const int b1 = (r1 - 1) / kBlockSize;
        for (int b = b0; b <= b1; ++b) {
            const int br0 = std::max(r0, b * kBlockSize);
            const int br1 = std::min(r1, (b + 1) * kBlockSize);
            const int add = br1 - br0;
            if (done_[expert_idx * n_hb_ + b].fetch_add(add, std::memory_order_acq_rel) + add == kBlockSize)
                quant_block(expert_idx, b);
        }
    }

private:
    int n_hb_ = 0;
    alignas(64) std::atomic<int> done_[kMaxCounters];
};

/// Single-expert output scaling matching the serial `float m = 0.0f; m += w0 * dst[r]; out[r] = m;` reduction
/// bit for bit (including signed zero: `0.0f + (-0.0f) == +0.0f`).
inline void scale_single_rows(float* out, const float* dst, float w0, int r0, int r1) {
    for (int r = r0; r < r1; ++r) out[r] = 0.0f + w0 * dst[r];
}

}  // namespace maya::core::glmlane
