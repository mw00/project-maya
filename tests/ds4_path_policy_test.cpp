// tests/ds4_path_policy_test.cpp - CPU-only checks for the DeepSeek prompt path calibration and selection seam.
#include "../src/ds4/ds4_path_policy.hpp"

#include <cmath>
#include <cstdio>

using strata::ds4::PathFlowCalibration;
using strata::ds4::path_size_class;
using strata::ds4::should_use_flow_path;

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main() {
    CHECK(path_size_class(2) == 0 && path_size_class(31) == 0);
    CHECK(path_size_class(32) == 1 && path_size_class(127) == 1);
    CHECK(path_size_class(128) == 2 && path_size_class(511) == 2);
    CHECK(path_size_class(512) == 3 && path_size_class(2048) == 3);

    PathFlowCalibration calibration;
    CHECK(calibration.factor_for_size_class(0) == 1.0);
    calibration.observe_size_class(path_size_class(11), 3.0);
    CHECK(calibration.factor_for_size_class(0) == 3.0);
    CHECK(calibration.factor_for_size_class(path_size_class(513)) == 1.0);
    CHECK(calibration.factor_for_size_class(path_size_class(129)) == 1.0);
    // A slow 11-token first call must not turn the 513-token flow estimate from 1s into 3s,
    // wrongly selecting 2s windows. This uses the same feedback and decision as pick_path.
    CHECK(should_use_flow_path(calibration.factor_for_size_class(path_size_class(513)), 2.0, 1.0, 0.9));

    calibration.observe_size_class(path_size_class(513), 2.0);
    CHECK(calibration.factor_for_size_class(3) == 2.0);
    CHECK(calibration.factor_for_size_class(0) == 3.0);
    CHECK(calibration.factor_for_size_class(2) == 1.0);
    calibration.observe_size_class(3, 8.0);
    CHECK(std::abs(calibration.factor_for_size_class(3) - 3.8) < 1e-12);

    // Windows can still win, with or without flow observations for that size class.
    CHECK(!should_use_flow_path(calibration.factor_for_size_class(path_size_class(129)), 0.5, 0.95, 0.9));
    CHECK(!should_use_flow_path(calibration.factor_for_size_class(path_size_class(513)), 2.0, 0.95, 0.9));
    CHECK(!should_use_flow_path(4.0, 1.5, 0.5, 0.9));
    CHECK(should_use_flow_path(4.0, 3.0, 0.5, 0.9));  // cold weights keep the twofold safeguard
    CHECK(should_use_flow_path(4.0, 4.0, 0.95, 0.9)); // ties still favor flow

    PathFlowCalibration extremes;
    extremes.observe_size_class(0, 100.0);
    extremes.observe_size_class(1, 0.0);
    CHECK(extremes.factor_for_size_class(0) == 8.0);
    CHECK(extremes.factor_for_size_class(1) == 0.25);
    CHECK(extremes.factor_for_size_class(2) == 1.0);

    std::puts("ds4 path policy: class boundaries, per-class feedback, baseline fallback and window selection passed");
}
