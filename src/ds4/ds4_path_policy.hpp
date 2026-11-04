#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::ds4 {

// The prompt-size classes used by the path predictor: 2-31, 32-127, 128-511, and 512+ tokens.
inline int path_size_class(int tokens) {
    return tokens < 32 ? 0 : tokens < 128 ? 1 : tokens < 512 ? 2 : 3;
}

// Per-size-class flow feedback. An unmeasured prompt-size class uses the calibrated baseline factor one.
struct PathFlowCalibration {
    static constexpr int kSizeClassCount = 4;

    std::array<double, kSizeClassCount> factors{{1, 1, 1, 1}};
    std::array<int64_t, kSizeClassCount> observations{};

    // Return this prompt-size class's factor without borrowing another class's fixed-cost ratio.
    double factor_for_size_class(int size_class) const {
        if (size_class < 0 || size_class >= kSizeClassCount || observations[(size_t) size_class] == 0) return 1.0;
        return factors[(size_t) size_class];
    }

    // Learn a clipped, smoothed flow factor within this prompt-size class.
    void observe_size_class(int size_class, double ratio) {
        if (size_class < 0 || size_class >= kSizeClassCount) return;
        ratio = std::min(8.0, std::max(0.25, ratio));
        const size_t i = (size_t) size_class;
        factors[i] = observations[i] == 0 ? ratio : 0.7 * factors[i] + 0.3 * ratio;
        ++observations[i];
    }
};

// Select the flow unless windows are sufficiently faster under the existing residency safeguard.
inline bool should_use_flow_path(double predicted_flow_seconds, double predicted_window_seconds, double resident_share,
                                 double resident_threshold) {
    return !(predicted_window_seconds < predicted_flow_seconds &&
             (resident_share >= resident_threshold || 2.0 * predicted_window_seconds <= predicted_flow_seconds));
}

}  // namespace strata::ds4
