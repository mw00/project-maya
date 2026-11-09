// tests/glm_dsa/glm_random_route_cpu_test.cpp - glm-dsa B15.7 on the CPU: the random-routing id generator.
//
// glmd::random_route_ids / random_route_apply (include/strata/kernels/glmd_random_route.hpp, the same code the
// glm_fast.cu route kernel runs):
//   - k distinct ids in [0, E) for every (seed, layer, counter), at E = 256 / k = 8 (GLM-5.3), E = 8 / k = 8 (a full
//     permutation) and small shapes; refusals for k > 8 or E < k;
//   - deterministic (the same inputs give the same ids), and different across layers, counters and seeds;
//   - near-uniform: over 200,000 routes every expert's count is within 5% of the mean, and the pair (rank 0, rank 1)
//     never repeats an id;
//   - apply keeps the router's weights: rank i's probability moves to its new id, so the route's weights
//     w_i = p[ids[i]] * inv are the router's, in rank order; the other probabilities are untouched;
//   - the switch is off unless STRATA_GLM_RANDOM_ROUTING is set (random_routing_seed() == 0: Flash and every normal
//     run route as before).
#include "strata/kernels/glmd_random_route.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                        \
    do {                                                     \
        if (!(c)) {                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_fail;                                        \
        }                                                    \
    } while (0)

bool distinct_in_range(const int* ids, int k, int E) {
    std::set<int> s;
    for (int i = 0; i < k; ++i) {
        if (ids[i] < 0 || ids[i] >= E) return false;
        s.insert(ids[i]);
    }
    return (int) s.size() == k;
}

void shapes() {
    int ids[8];
    for (int E : {8, 9, 16, 64, 160, 256, 512})
        for (int k = 1; k <= 8 && k <= E; ++k)
            for (uint64_t c = 0; c < 2000; ++c) {
                const bool ok = glmd::random_route_ids(1, (uint32_t) (c % 79), c, E, k, ids);
                if (!ok || !distinct_in_range(ids, k, E)) {
                    CHECK(false, "E %d k %d counter %llu: not k distinct ids in range", E, k, (unsigned long long) c);
                    return;
                }
            }
    CHECK(!glmd::random_route_ids(1, 0, 0, 256, 9, ids), "k 9 refused");
    CHECK(!glmd::random_route_ids(1, 0, 0, 4, 8, ids), "E < k refused");
    CHECK(!glmd::random_route_ids(1, 0, 0, 256, 0, ids), "k 0 refused");
    // E == k: a permutation of 0..7
    glmd::random_route_ids(7, 3, 99, 8, 8, ids);
    CHECK(distinct_in_range(ids, 8, 8), "E = k = 8: a permutation");
}

void determinism() {
    int a[8], b[8], c[8];
    glmd::random_route_ids(1, 12, 4242, 256, 8, a);
    glmd::random_route_ids(1, 12, 4242, 256, 8, b);
    bool same = true;
    for (int i = 0; i < 8; ++i) same = same && a[i] == b[i];
    CHECK(same, "the same (seed, layer, counter) gives the same ids");
    int differ_layer = 0, differ_counter = 0, differ_seed = 0;
    for (uint64_t n = 0; n < 1000; ++n) {
        glmd::random_route_ids(1, 12, n, 256, 8, a);
        glmd::random_route_ids(1, 13, n, 256, 8, b);
        glmd::random_route_ids(2, 12, n, 256, 8, c);
        int d[8];
        glmd::random_route_ids(1, 12, n + 1, 256, 8, d);
        bool l = false, s = false, q = false;
        for (int i = 0; i < 8; ++i) {
            l = l || a[i] != b[i];
            s = s || a[i] != c[i];
            q = q || a[i] != d[i];
        }
        differ_layer += l;
        differ_seed += s;
        differ_counter += q;
    }
    CHECK(differ_layer == 1000 && differ_seed == 1000 && differ_counter == 1000,
          "layers / seeds / counters give other ids (%d %d %d of 1000)", differ_layer, differ_seed, differ_counter);
}

void uniform() {
    const int E = 256, k = 8, N = 200000;
    std::vector<long> cnt(E, 0), first(E, 0);
    int ids[8];
    for (int n = 0; n < N; ++n) {
        glmd::random_route_ids(1, (uint32_t) (n % 78), (uint64_t) n / 78, E, k, ids);
        for (int i = 0; i < k; ++i) ++cnt[(size_t) ids[i]];
        ++first[(size_t) ids[0]];
    }
    const double mean = (double) N * k / E;
    double worst = 0.0;
    for (long c : cnt) worst = std::fmax(worst, std::fabs(c - mean) / mean);
    const double mean0 = (double) N / E;
    double worst0 = 0.0;
    for (long c : first) worst0 = std::fmax(worst0, std::fabs(c - mean0) / mean0);
    std::printf("uniformity over %d routes: worst expert count %.2f%% off the mean, rank 0 %.2f%%\n", N, 100.0 * worst,
                100.0 * worst0);
    CHECK(worst < 0.05, "expert counts within 5%% of the mean (worst %.3f)", worst);
    CHECK(worst0 < 0.15, "rank-0 counts within 15%% of the mean (worst %.3f)", worst0);
}

void apply_keeps_weights() {
    const int E = 256, k = 8;
    std::vector<float> p(E), p0;
    for (int e = 0; e < E; ++e) p[(size_t) e] = 1.0f / (1.0f + std::exp(-0.01f * (float) ((e * 37) % 101 - 50)));
    p0 = p;
    int ids[8] = {5, 17, 200, 3, 99, 128, 255, 0}, orig[8];
    for (int i = 0; i < k; ++i) orig[i] = ids[i];
    float w_before[8], w_after[8];
    double sum_b = 0.0, sum_a = 0.0;
    for (int i = 0; i < k; ++i) sum_b += p[(size_t) ids[i]];
    for (int i = 0; i < k; ++i) w_before[i] = (float) (p[(size_t) ids[i]] / sum_b * 2.5);
    glmd::random_route_apply(1, 7, 31337, E, k, ids, p.data());
    int expect[8];
    glmd::random_route_ids(1, 7, 31337, E, k, expect);
    bool same_ids = true;
    for (int i = 0; i < k; ++i) same_ids = same_ids && ids[i] == expect[i];
    CHECK(same_ids, "apply writes random_route_ids' ids");
    for (int i = 0; i < k; ++i) sum_a += p[(size_t) ids[i]];
    for (int i = 0; i < k; ++i) w_after[i] = (float) (p[(size_t) ids[i]] / sum_a * 2.5);
    bool same_w = true;
    for (int i = 0; i < k; ++i) same_w = same_w && w_before[i] == w_after[i];
    CHECK(same_w, "rank i keeps the router's weight");
    for (int i = 0; i < k; ++i) CHECK(p[(size_t) ids[i]] == p0[(size_t) orig[i]], "rank %d probability moved", i);
    std::set<int> touched(ids, ids + k);
    bool rest = true;
    for (int e = 0; e < E; ++e)
        if (!touched.count(e)) {
            // an original id that was not re-drawn keeps whatever it had (the route reads p only at ids[])
            bool was_orig = false;
            for (int i = 0; i < k; ++i) was_orig = was_orig || orig[i] == e;
            if (!was_orig) rest = rest && p[(size_t) e] == p0[(size_t) e];
        }
    CHECK(rest, "probabilities of experts outside both id sets are untouched");
}

}  // namespace

int main() {
    shapes();
    determinism();
    uniform();
    apply_keeps_weights();
    CHECK(std::getenv("STRATA_GLM_RANDOM_ROUTING") != nullptr || glmd::random_routing_seed() == 0,
          "off by default");
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
