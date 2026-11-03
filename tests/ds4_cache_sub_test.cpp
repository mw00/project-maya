#include "../src/ds4/ds4_cache_sub.hpp"
#include "../src/ds4/ds4_cache_sub_config.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
using strata::ds4::cache_sub::choose;

int main() {
    using strata::ds4::cache_sub::Config;
    using strata::ds4::cache_sub::parse;
    Config cfg;
    std::string err;
    CHECK(parse(nullptr, nullptr, nullptr, nullptr, cfg, err));
    CHECK(cfg.ratio == 0 && cfg.candidates == 12 && cfg.budget == .6f);
    CHECK(parse(".1", "12", "reference", ".6", cfg, err));
    for (const char* bad : {"", "nan", "inf", "-0.1", "1.1", "0.1junk", "1e-1000"}) {
        CHECK(!parse(bad, nullptr, nullptr, nullptr, cfg, err));
        CHECK(cfg.ratio == .1f && cfg.candidates == 12 && cfg.budget == .6f);
    }
    for (const char* bad : {"0", "8", "15", "12x", "99999999999999999999"})
        CHECK(!parse(nullptr, bad, nullptr, nullptr, cfg, err));
    CHECK(!parse(nullptr, nullptr, "selected", nullptr, cfg, err));
    CHECK(!parse(nullptr, nullptr, nullptr, "-1", cfg, err));
    CHECK(!parse(nullptr, nullptr, nullptr, "nan", cfg, err));
    CHECK(parse("0", "32", "reference", "0", cfg, err));

    // Biased rank zero and the strongest unbiased choice are different and both protected.
    int ids[] = {9, 8, 7, 6, 5, 4};
    float scores[] = {2, 8, 2, 1, 2, 1};
    uint8_t resident[] = {0, 0, 0, 0, 1, 1};
    int selected[8]; float spent;
    CHECK(choose(ids, scores, resident, 4, 6, .1f, true, .6f, selected, &spent));
    CHECK(selected[0] == 0 && selected[1] == 1 && selected[2] == 4 && selected[3] == 5);
    CHECK(spent <= .6f);
    CHECK(choose(ids, scores, resident, 4, 6, .1f, true, 0, selected));
    for (int i = 0; i < 4; ++i) CHECK(selected[i] == i);
    CHECK(choose(ids, scores, resident, 4, 6, 0, true, 1, selected));
    for (int i = 0; i < 4; ++i) CHECK(selected[i] == i);
    resident[2] = 1;
    CHECK(choose(ids, scores, resident, 4, 6, .1f, true, 1, selected));
    CHECK(selected[2] == 2); // native cached expert is never replaced
    resident[2] = 0;
    scores[5] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!choose(ids, scores, resident, 4, 6, .1f, true, 1, selected));
    for (int i = 0; i < 4; ++i) CHECK(selected[i] == i);
    scores[5] = 1; ids[5] = ids[4];
    CHECK(!choose(ids, scores, resident, 4, 6, .1f, true, 1, selected));
    for (int i = 0; i < 4; ++i) CHECK(selected[i] == i);
    ids[5] = 4;

    // Property checks over different rankings, cache states, shapes and thresholds.
    std::mt19937 gen(20261103);
    for (int iteration = 0; iteration < 20000; ++iteration) {
        std::array<int, 32> ix;
        std::array<float, 32> value;
        std::array<uint8_t, 32> cached;
        const int k = 1 + (int) (gen() % 8), n = k + (int) (gen() % (33-k));
        int strongest = 0;
        for (int i = 0; i < n; ++i) {
            ix[i] = 100 + i; value[i] = (float) (1 + gen()%100); cached[i] = gen()%2;
            if (i < k && value[i] > value[strongest]) strongest = i;
        }
        const float ratio = (float) (gen()%11)/10, budget = (float) (gen()%21)/10;
        CHECK(choose(ix.data(), value.data(), cached.data(), k, n, ratio, true, budget, selected, &spent));
        CHECK(selected[0] == 0 && selected[strongest] == strongest && spent <= budget);
        float mass = 0, cost = 0;
        for (int i = 0; i < k; ++i) mass += value[i];
        for (int i = 0; i < k; ++i) {
            CHECK(selected[i] >= 0 && selected[i] < n);
            for (int j = 0; j < i; ++j) CHECK(selected[i] != selected[j]);
            if (selected[i] != i) {
                CHECK(ratio > 0 && !cached[i] && selected[i] >= k && cached[selected[i]]);
                CHECK(value[selected[i]] >= ratio * value[i]);
                cost += value[i]/mass + value[selected[i]]/mass;
            }
        }
        CHECK(cost <= budget + 1e-5f);
    }
    std::puts("cache substitution configuration, protection, fallback and 20000 invariant cases passed");
}
