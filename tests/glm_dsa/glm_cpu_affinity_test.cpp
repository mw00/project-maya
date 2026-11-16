// tests/glm_dsa/glm_cpu_affinity_test.cpp - glm-dsa B15.5 on the CPU: thread pools sized from the affinity mask.
//
// strata::core::glmaff::mask_cpus() (include/strata/core/glm_cpu_affinity.hpp), which the CPU-lane setup and the
// split search's RAM-bandwidth probe now use instead of std::thread::hardware_concurrency(): the test narrows this
// process's affinity mask with sched_setaffinity (what numactl --cpunodebind / taskset / a cgroup do) and checks that
// mask_cpus() follows it, while hardware_concurrency() keeps reporting the machine (llama.cpp #29301's bug).
#include "strata/core/glm_cpu_affinity.hpp"

#include <cstdio>
#include <thread>

int main() {
    namespace aff = strata::core::glmaff;
    int fail = 0;
#if defined(__linux__)
    cpu_set_t orig;
    CPU_ZERO(&orig);
    if (sched_getaffinity(0, sizeof orig, &orig) != 0) {
        std::printf("sched_getaffinity failed: skipped\n");
        return 77;
    }
    const int n0 = CPU_COUNT(&orig), hw = (int) std::thread::hardware_concurrency();
    std::printf("mask %d CPUs, hardware_concurrency %d, mask_cpus %d\n", n0, hw, aff::mask_cpus());
    if (aff::mask_cpus() != n0) {
        std::printf("FAIL mask_cpus() != the mask's count\n");
        ++fail;
    }
    // narrow to the first CPU of the mask, then to the first two (when there are two)
    int first[2] = {-1, -1}, found = 0;
    for (int c = 0; c < CPU_SETSIZE && found < 2; ++c)
        if (CPU_ISSET(c, &orig)) first[found++] = c;
    for (int want = 1; want <= found; ++want) {
        cpu_set_t s;
        CPU_ZERO(&s);
        for (int i = 0; i < want; ++i) CPU_SET(first[i], &s);
        if (sched_setaffinity(0, sizeof s, &s) != 0) {
            std::printf("sched_setaffinity refused: skipped\n");
            return 77;
        }
        const int got = aff::mask_cpus();
        std::printf("mask narrowed to %d: mask_cpus %d, hardware_concurrency %u\n", want, got,
                    std::thread::hardware_concurrency());
        if (got != want) {
            std::printf("FAIL mask_cpus() %d != %d\n", got, want);
            ++fail;
        }
    }
    sched_setaffinity(0, sizeof orig, &orig);
    if (aff::mask_cpus() != n0) {
        std::printf("FAIL mask_cpus() after restoring the mask\n");
        ++fail;
    }
#else
    if (aff::mask_cpus() < 1) ++fail;
#endif
    std::printf("%s (%d failures)\n", fail ? "FAILED" : "PASSED", fail);
    return fail ? 1 : 0;
}
