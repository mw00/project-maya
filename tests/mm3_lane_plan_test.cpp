// tests/mm3_lane_plan_test.cpp - maya-mm3's CPU lane arithmetic (src/mm3/mm3_lane_plan.hpp): the default workers on
// hosts with and without SMT, --cpu-threads as the lane's threads, decode's timing of the lane (#552).
//     g++ -O2 -std=c++17 -Isrc/mm3 tests/mm3_lane_plan_test.cpp -o build/mm3_lane_plan_test && build/mm3_lane_plan_test
#include <cmath>
#include <cstdio>

#include "mm3_lane_plan.hpp"

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    // workers: one a physical core but the caller's; 2 cores left free where the lane would take every logical CPU
    CHECK(mm3::lane_default_workers(31, 32, 2) == 29);    // #552's VM: 32 vCPUs, no SMT -> 30 threads
    CHECK(mm3::lane_default_workers(15, 32, 2) == 15);    // 16 cores with SMT: the siblings are free, every core kept
    CHECK(mm3::lane_default_workers(39, 80, 2) == 39);    // 2 x 20 cores with SMT
    CHECK(mm3::lane_default_workers(19, 20, 2) == 17);    // a 20-core CPU without SMT (maya-ds4's Core Ultra 7 265K)
    CHECK(mm3::lane_default_workers(3, 4, 2) == 3);       // 4 cores: too few to give any away
    CHECK(mm3::lane_default_workers(31, 32, 0) == 31);    // MAYA_MM3_LANE_FREE=0: as before
    CHECK(mm3::lane_default_workers(0, 1, 2) == 0);
    CHECK(mm3::lane_default_workers(31, 0, 2) == 31);     // (logical CPUs not known: as before)
    // --cpu-threads N: N threads, the caller included
    CHECK(mm3::lane_workers_of_threads(28) == 27 && mm3::lane_workers_of_threads(1) == 0);
    CHECK(mm3::lane_workers_of_threads(0) == 0 && mm3::lane_workers_of_threads(-1) == -1);

    // decode's timing: costs that hold are never changed
    {
        mm3::LaneRetime r;
        double c1 = 0.40, add = 0.45;
        for (int i = 0; i < 2000; ++i) {
            const int n = 1 + i % 4;
            const double ms = (0.40 + (n - 1) * 0.45) * (i % 2 ? 1.08 : 0.93);   // (within 15%, noisy)
            CHECK(!r.note(n, ms, c1, add));
        }
        CHECK(c1 == 0.40 && add == 0.45 && r.changes == 0);
    }
    // a busy start (#552: 0.44 ms, +1.27 each more) against decode's 0.45 / +0.45: fitted, changed, then it holds
    {
        mm3::LaneRetime r;
        double c1 = 0.44, add = 1.27;
        int changed = 0, at = -1;
        for (int i = 0; i < 2000; ++i) {
            const int n = 1 + i % 4;
            if (r.note(n, 0.45 + (n - 1) * 0.45, c1, add)) {
                ++changed;
                if (at < 0) at = i;
            }
        }
        CHECK(changed == 1 && at > 0 && at < 200);   // (three windows of 96 experts: within ~120 calls)
        CHECK(std::fabs(c1 - 0.45) < 1e-6 && std::fabs(add - 0.45) < 1e-6);
    }
    // every call one expert (no spread to fit two costs): both scaled by the time against theirs
    {
        mm3::LaneRetime r;
        double c1 = 1.0, add = 0.5;
        bool ch = false;
        for (int i = 0; i < 400 && !ch; ++i) ch = r.note(1, 0.5, c1, add);
        CHECK(ch && std::fabs(c1 - 0.5) < 1e-9 && std::fabs(add - 0.25) < 1e-9);
    }
    // one window off and the next not: nothing changes (two in a row, the same way)
    {
        mm3::LaneRetime r;
        double c1 = 0.4, add = 0.4;
        bool ch = false;
        for (int w = 0; w < 12; ++w)
            for (int i = 0; i < 48; ++i) ch = ch || r.note(2, w % 2 ? 1.6 : 0.8, c1, add);   // (2x off, then on)
        CHECK(!ch && c1 == 0.4 && add == 0.4);
    }
    // at most kChanges changes, however far it drifts
    {
        mm3::LaneRetime r;
        double c1 = 0.4, add = 0.4;
        int changed = 0;
        for (int i = 0; i < 20000; ++i) changed += r.note(1 + i % 4, 10.0 * (1 + i % 4), c1, add);
        CHECK(changed <= mm3::LaneRetime::kChanges);
    }
    std::puts("mm3 lane plan: default workers (SMT, no SMT, LANE_FREE), --cpu-threads as threads, decode's re-timing "
              "(holds, busy start fixed once, scaled, needs two windows, capped) passed");
    return 0;
}
