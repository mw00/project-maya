#include "../src/ds4/ds4_pcie.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace pcie = strata::ds4::pcie;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    // A persisted device-lane verdict remains the pre-placement baseline. Experimental ZC flags are deliberately not
    // inputs to this decision: they cannot change staging allocation or the expert placement that follows it.
    CHECK(pcie::baseline_lane_off("", "off"));
    CHECK(!pcie::baseline_lane_off("", "on"));
    CHECK(!pcie::baseline_lane_off("", ""));
    CHECK(!pcie::baseline_lane_off("7", "off"));   // explicit forced routing keeps its original override

    const std::vector<char> none, zeros(4, 0), partial{0, 0, 1, 0};
    for (bool requested : {false, true})
        for (bool forced : {false, true})
            for (bool hold : {false, true})
                for (bool zc : {false, true}) {
                    const bool expected = requested && !forced && !hold && zc;
                    CHECK(pcie::zc_all_usable(requested, forced, hold, zc, partial) == expected);
                    CHECK(!pcie::zc_all_usable(requested, forced, hold, zc, none));
                    CHECK(!pcie::zc_all_usable(requested, forced, hold, zc, zeros));
                }

    // DS4_STREAM_ZC_NOWAIT consumes this same effective flag in the model, so requesting it without usable ZC-all is
    // inert too. Partial coverage permits ZC-all, while its separate layer-wide predicate still protects no-wait.
    CHECK(pcie::zc_all_usable(true, false, false, true, partial));
    CHECK(!pcie::zc_all_usable(true, false, false, false, partial));   // DS4_STREAM_ZC=0
    CHECK(!pcie::zc_all_usable(true, false, true, true, partial));    // MAYA_DS4_STREAM=2
    CHECK(!pcie::zc_all_usable(true, true, false, true, partial));    // DS4_STREAM_FORCE

    std::puts("DS4 PCIe gates: persisted lane off, no/zero/partial pinning, force and hold cases passed");
}
