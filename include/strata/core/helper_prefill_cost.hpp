#pragma once
#include <algorithm>
#include <cmath>

namespace strata::core {
// Observed end-to-end lane costs, including contention, not isolated kernel or
// PCIe bandwidth measurements. Separate prompt sizes avoid decoding/warmup bias.
struct HelperPrefillCost {
    double row_ms = 0, mib_ms = 0;
    static int bucket(int tokens) { return tokens <= 256 ? 0 : tokens <= 2048 ? 1 : 2; }
    bool ready() const { return row_ms > 0 && mib_ms > 0; }
    void observe(int rows, double helper_ms, double mib, double copy_ms) {
        const auto update = [](double& value, double sample) {
            if (std::isfinite(sample) && sample > 0)
                value = value > 0 ? 0.8 * value + 0.2 * sample : sample;
        };
        if (rows >= 64) update(row_ms, helper_ms / rows);
        if (mib > 0) update(mib_ms, copy_ms / mib);
    }
    bool fits(int rows, double remaining_mib) const {
        // Leave headroom for result upload and interference with weight readers.
        return ready() && rows > 0 && remaining_mib > 0 &&
               rows * row_ms <= 0.85 * remaining_mib * mib_ms;
    }
};
}
