#include "../src/ds4/ds4_pcie.hpp"

#include <cstdio>
#include <initializer_list>
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

    // #551: a 48 GB L40S beside 28 Xeon threads, MXFP4 experts all in RAM (--mmap-experts off): warm, the split never
    // streams; cold (a disk read on both lanes) it always does - which kept the one-GPU path on.  Where no pick can be
    // cold the warm case alone decides; where one can, as before
    pcie::Costs l40s;
    l40s.c = 0.080, l40s.oc = 0.042, l40s.p = 0.872, l40s.op = 1.199, l40s.g = 0.026, l40s.disk = 9.40;
    for (int slots : {4, 8, 16}) {
        CHECK(pcie::table(l40s, slots, 6) == "0 0 0 0 0 0 0");
        CHECK(pcie::streams_any(l40s, slots, 6));          // (some expert may be cold: unchanged)
        CHECK(pcie::streams_any(l40s, slots, 6, true));
        CHECK(!pcie::streams_any(l40s, slots, 6, false));  // (every expert in VRAM or RAM: nothing streams)
    }
    // a slow CPU lane: the warm split streams, whether or not a pick can be cold
    pcie::Costs slow;
    slow.c = 0.5, slow.oc = 0.05, slow.p = 0.3, slow.op = 0.1, slow.g = 0.02, slow.disk = 5.0;
    CHECK(pcie::split(slow, 6, 0u, 8) != 0u);
    CHECK(pcie::streams_any(slow, 8, 6, false) && pcie::streams_any(slow, 8, 6, true));
    CHECK(!pcie::streams_any(slow, 0, 6, true));   // (no staging slots: nothing streams)
    // decode's checks: the cold kind counts only where a pick can be cold (unmeasured there forever otherwise)
    CHECK(pcie::verdicts_allow(0, 0, true) && pcie::verdicts_allow(0, 0, false));
    CHECK(pcie::verdicts_allow(-1, 0, true));      // (cold requests still being measured: as before)
    CHECK(!pcie::verdicts_allow(-1, 0, false));    // (#551: warm measured slower, no cold request can come: off)
    CHECK(!pcie::verdicts_allow(-1, -1, true) && !pcie::verdicts_allow(-1, 1, false));
    CHECK(pcie::verdicts_allow(1, -1, false) && pcie::verdicts_allow(-1, 1, true));

    // decode's checks (#697): a tie keeps streaming; only 3% slower twice in a row turns it off
    const auto run = [](std::initializer_list<double> gains) {
        int wins = 0, good = 0, bad = 0, v = 0;
        for (double g : gains)
            if ((v = pcie::ab_window(wins, good, bad, g)) != 0) break;
        return v;
    };
    CHECK(run({-0.5}) == 0);                              // (the first window only warms up)
    CHECK(run({0.0, -0.001, -0.008}) == 1);               // (an RTX 3090's +0.1% / +0.8%: it keeps streaming)
    CHECK(run({0.0, 0.05, 0.04}) == 1);                   // (faster: as before)
    CHECK(run({0.0, -0.045, -0.03}) == -1);               // (slower twice in a row: off, as before)
    CHECK(run({0.0, -0.045, 0.01}) == 0);                 // (not twice in a row yet)
    CHECK(run({0.0, -0.045, 0.01, -0.04, 0.0, -0.05, 0.02, -0.06, 0.0}) == 1);   // (undecided after 9: keeps it)
    CHECK(run({0.0, -0.045, 0.01, -0.04, 0.0, -0.05, 0.02, -0.06}) == 0);
    CHECK(run({0.0, 0.01, -0.04, -0.04}) == -1);          // (a tie, then slower twice: off)

    // #920: a one-GPU PC whose CPU lane is far faster than its link for experts in RAM (the reporter's timed costs: an
    // RTX PRO 5000 over a 7.2 GB/s link, a Ryzen AI Max+ 395): nothing streams warm, only disk reads split - with most
    // of the CPU lane copied into RAM the one-GPU path ends; mostly in place, or a warm stream: kept as before
    const pcie::Costs fast_cpu{0.107, 0.037, 1.907, 0.163, 0.006, 3.54};
    CHECK(!pcie::streams_any(fast_cpu, 16, 6, false));
    CHECK(pcie::streams_any(fast_cpu, 16, 6, true));   // (the cold split: what kept the path on)
    CHECK(!pcie::one_gpu_keep(true, false, 9910, 11536));
    CHECK(!pcie::one_gpu_keep(true, false, 5768, 11536));       // (exactly half copied: still the warm rule)
    CHECK(pcie::one_gpu_keep(true, false, 5767, 11536));        // (most in place: as before)
    CHECK(pcie::one_gpu_keep(true, false, 0, 11536) && pcie::one_gpu_keep(true, false, 0, 0));
    CHECK(pcie::one_gpu_keep(true, true, 9910, 11536));         // (a warm stream: the measured case, kept)
    CHECK(!pcie::one_gpu_keep(false, true, 9910, 11536) && !pcie::one_gpu_keep(false, false, 0, 11536));
    CHECK(pcie::one_gpu_keep(true, pcie::streams_any(slow, 8, 6, false), 9910, 11536));

    std::puts("DS4 PCIe gates: persisted lane off, no/zero/partial pinning, force and hold cases passed; the warm-only "
              "decision where no pick can be cold (#551) passed; decode's checks keep streaming on a tie (#697); "
              "the one-GPU path's warm rule (#920) passed");
}
