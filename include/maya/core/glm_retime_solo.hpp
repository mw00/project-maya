// include/maya/core/glm_retime_solo.hpp - contention-free ("solo") sample filter for Glm5Model::lane_retime().
//
// When multiple GPUs share one concurrent CPU worker pool (MAYA_GLM_CPU_SHARED=1), two GPUs' service threads can
// overlap on the pool during decode. The resulting wall-time queuing delay inflates per-expert CPU timing in
// `cpu_us`, which can cause `lane_retime()` to misread the CPU as slower than it is and push more experts onto PCIe.
// Filtering `lane_retime()`'s samples to calls where no other concurrent pool slot was busy before or after the call
// and no extra batch ran during it (`epoch_after - epoch_before <= max_batches`) isolates the CPU's uncontended
// expert latency without altering `cpu_experts` / `cpu_routes` / `cpu_us` telemetry.
// Header-only, host C++, no CUDA or ggml dependency.
#pragma once

#include <cstdint>

namespace maya::core::glmretime {

/// True when a `fast_cpu_experts()` call ran without concurrent pool contention from another part.
/// - On a non-concurrent pool (`!concurrent_pool`), every call is uncontended by definition.
/// - On a concurrent pool (`concurrent_pool == true`), a single decode call dispatches at most `max_batches`
///   pool batches (default 2: gate/up and down); `busy_before == 0 && busy_after == 0` and
///   `(epoch_after - epoch_before) <= max_batches` ensure no other caller overlapped on the pool.
inline bool is_solo_sample(bool concurrent_pool, int busy_before, int busy_after, uint64_t epoch_before,
                           uint64_t epoch_after, uint64_t max_batches = 2) {
    if (!concurrent_pool) return true;
    return busy_before == 0 && busy_after == 0 && (epoch_after - epoch_before) <= max_batches;
}

}  // namespace maya::core::glmretime
