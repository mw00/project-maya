// Host CPU-only unit tests for hy4 TP affinity intersection (src/hy4/hy4_tp_affinity.hpp; no GPU).
#include "../src/hy4/hy4_tp_affinity.hpp"

#include <cstdio>
#include <vector>

#define CHECK(expr)                                                                     \
    do {                                                                                \
        if (!(expr)) {                                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr);                    \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

int main() {
    using namespace hy4::tp_affinity;

    CHECK((parse_cpulist("") == std::vector<int>{}));
    CHECK((parse_cpulist("3") == std::vector<int>{3}));
    CHECK((parse_cpulist("0-2,5,7-8") == std::vector<int>{0, 1, 2, 5, 7, 8}));
    CHECK((parse_cpulist("4-2,1") == std::vector<int>{1}));
    CHECK(respect_inherited_affinity() == true || respect_inherited_affinity() == false);

#if defined(__linux__)
    cpu_set_t mock;
    CPU_ZERO(&mock);
    CPU_SET(2, &mock);
    CPU_SET(4, &mock);

    const std::vector<int> raw = {0, 2, 3, 4, 6};
    CHECK((filter_cpus(raw, true, mock) == std::vector<int>{2, 4}));
    CHECK((filter_cpus(raw, false, mock) == raw));

    cpu_set_t set;
    CHECK(make_affinity_set({0, 2, 5}, true, mock, set));
    CHECK(CPU_ISSET(2, &set) && CPU_COUNT(&set) == 1);
    CHECK(!make_affinity_set({0, 1, 5}, true, mock, set));

    cpu_set_t init;
    CPU_ZERO(&init);
    CHECK(sched_getaffinity(0, sizeof init, &init) == 0);
    std::vector<int> allowed;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &init)) allowed.push_back(c);
    CHECK(!allowed.empty());

    const int keep_cpu = allowed[0];
    cpu_set_t narrow;
    CPU_ZERO(&narrow);
    CPU_SET(keep_cpu, &narrow);
    CHECK(sched_setaffinity(0, sizeof narrow, &narrow) == 0);

    cpu_set_t inherited;
    const bool has_mask = inherited_affinity_mask(inherited);
    CHECK(has_mask);

    std::vector<int> superset = {keep_cpu, keep_cpu + 1};
    if (allowed.size() > 1) superset.push_back(allowed[1]);
    CHECK(make_affinity_set(superset, has_mask, inherited, set));
    CHECK(pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0);

    cpu_set_t after;
    CPU_ZERO(&after);
    CHECK(sched_getaffinity(0, sizeof after, &after) == 0);
    CHECK(CPU_ISSET(keep_cpu, &after) && CPU_COUNT(&after) == 1);

    // Disjoint target CPUs must leave the thread affinity unchanged.
    if (allowed.size() > 1) {
        CHECK(!make_affinity_set({allowed[1]}, has_mask, inherited, set));
        CPU_ZERO(&after);
        CHECK(sched_getaffinity(0, sizeof after, &after) == 0);
        CHECK(CPU_ISSET(keep_cpu, &after) && CPU_COUNT(&after) == 1);
    }
#endif

    std::printf("hy4_tp_affinity_test: passed\n");
    return 0;
}
