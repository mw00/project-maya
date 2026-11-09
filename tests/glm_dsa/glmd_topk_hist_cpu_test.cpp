// tests/glm_dsa/glmd_topk_hist_cpu_test.cpp - glm-dsa B14.15 (CPU only): the bucket (histogram) top-k selector.
//
// glmd::twin::topk_hist is the GPU kernel's algorithm (src/kernels/cuda/glmd_topk.cuh select_top_cells_hist: key
// range, histogram, boundary-bucket sort or the radix fallback, the shared compaction).  Its output must be the radix
// selector's exactly - the same cells in the same ascending order, -1 padding below top_k, ties to the lower cell -
// so the decode policy can switch selectors (STRATA_GLMD_TOPK_HIST) without changing a single selection.  Checked
// against a sort oracle (std::stable_sort by score, descending, then by cell) and against glmd::twin::topk on:
// random scores, many exact ties, all equal, two values, -inf and +-0, a heavy boundary bucket (the fallback), huge
// dynamic range, n <= top_k (identity), per-row positions, n up to 131072; plus the policy's dispatch.
#include "strata/kernels/glmd_decode_policy.hpp"
#include "strata/kernels/glmd_fast.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 7) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// the oracle: the top_k cells by score (ties to the lower cell), in ascending cell order, -1 padded
std::vector<int> oracle(const float* sc, int n, int top_k) {
    std::vector<int> out((size_t) top_k, -1);
    if (n <= top_k) {
        for (int i = 0; i < n; ++i) out[(size_t) i] = i;
        return out;
    }
    std::vector<int> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float x = sc[a] == 0.0f ? 0.0f : sc[a], y = sc[b] == 0.0f ? 0.0f : sc[b];   // -0 is +0
        return x > y;
    });
    idx.resize((size_t) top_k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

struct Case {
    std::string name;
    std::vector<float> sc;
    int top_k;
};

void run_case(const Case& c, int* fallbacks_total) {
    const int n = (int) c.sc.size();
    std::vector<int> a((size_t) c.top_k), b((size_t) c.top_k);
    int fb = 0;
    glmd::twin::topk_hist(c.sc.data(), n, n, 1, nullptr, c.top_k, a.data(), &fb);
    glmd::twin::topk(c.sc.data(), n, n, 1, nullptr, c.top_k, b.data());
    const std::vector<int> o = oracle(c.sc.data(), n, c.top_k);
    *fallbacks_total += fb;
    check(a == o && b == o, c.name + ": n " + std::to_string(n) + ", top " + std::to_string(c.top_k) +
                                (a == o ? "" : " - histogram differs from the oracle") +
                                (b == o ? "" : " - radix differs from the oracle") + (fb ? " (radix fallback)" : ""));
}

}  // namespace

int main() {
    std::printf("[the bucket selector == the radix selector == the sort oracle]\n");
    Rng r(5);
    std::vector<Case> cases;
    const int K = glmd::kTopK;
    for (int n : {2049, 3000, 9000, 32768, 65536, 131072}) {
        Case c{"random normal", std::vector<float>((size_t) n), K};
        for (auto& v : c.sc) v = (float) (std::sqrt(-2.0 * std::log(std::max(r.uni(), 1e-300))) * std::cos(6.2831853 * r.uni()));
        cases.push_back(c);
    }
    {
        Case c{"exact ties (12 distinct values)", std::vector<float>(20000), K};
        for (auto& v : c.sc) v = (float) (r.next() % 12) * 0.25f;
        cases.push_back(c);
    }
    {
        Case c{"all equal", std::vector<float>(5000, 1.5f), K};
        cases.push_back(c);
    }
    {
        Case c{"two values, the boundary inside the lower", std::vector<float>(7000), K};
        for (size_t i = 0; i < c.sc.size(); ++i) c.sc[i] = i % 7 == 0 ? 2.0f : 1.0f;
        cases.push_back(c);
    }
    {
        Case c{"-inf, +0 and -0 mixed in (ReLU-sum scores)", std::vector<float>(10000), K};
        for (size_t i = 0; i < c.sc.size(); ++i) {
            const uint64_t k = r.next() % 4;
            c.sc[i] = k == 0 ? -INFINITY : k == 1 ? 0.0f : k == 2 ? -0.0f : (float) r.uni();
        }
        cases.push_back(c);
    }
    {
        Case c{"heavy boundary bucket (a tight cluster around the cut: the fallback)", std::vector<float>(40000), K};
        for (size_t i = 0; i < c.sc.size(); ++i) c.sc[i] = i < 10 ? 1e30f : 1.0f + (float) (r.next() % 100000) * 1e-9f;
        cases.push_back(c);
    }
    {
        Case c{"huge dynamic range (1e-30 .. 1e30)", std::vector<float>(30000), K};
        for (auto& v : c.sc) v = (float) std::pow(10.0, -30.0 + 60.0 * r.uni()) * (r.next() % 3 == 0 ? -1.0f : 1.0f);
        cases.push_back(c);
    }
    {
        Case c{"ascending (ties none, boundary in one bucket)", std::vector<float>(20000), K};
        for (size_t i = 0; i < c.sc.size(); ++i) c.sc[i] = (float) i;
        cases.push_back(c);
    }
    for (int top : {1, 7, 100, 4095}) {
        Case c{"random, top " + std::to_string(top), std::vector<float>(6000), top};
        for (auto& v : c.sc) v = (float) (r.next() % 1000);
        cases.push_back(c);
    }
    for (int n : {1, 100, 2048}) {
        Case c{"n <= top_k: the identity, -1 padded", std::vector<float>((size_t) n, 3.0f), K};
        cases.push_back(c);
    }
    int fallbacks = 0;
    for (const Case& c : cases) run_case(c, &fallbacks);
    check(fallbacks >= 1, "the radix fallback ran on the clustered row(s): " + std::to_string(fallbacks));

    std::printf("[several rows with device positions and a padded stride]\n");
    {
        const int T = 5, ld = 9100, n_kv = 9000;
        std::vector<float> sc((size_t) T * ld);
        for (auto& v : sc) v = (float) (r.next() % 50);
        const int pos[T] = {100, 2047, 2048, 5000, 8999};
        std::vector<int> a((size_t) T * K), b((size_t) T * K);
        glmd::twin::topk_hist(sc.data(), ld, n_kv, T, pos, K, a.data());
        glmd::twin::topk(sc.data(), ld, n_kv, T, pos, K, b.data());
        bool same_oracle = true;
        for (int t = 0; t < T; ++t) {
            const std::vector<int> o = oracle(sc.data() + (size_t) t * ld, std::min(n_kv, pos[t] + 1), K);
            same_oracle = same_oracle && std::equal(o.begin(), o.end(), a.begin() + (size_t) t * K);
        }
        check(a == b && same_oracle, "5 rows at positions 100 .. 8999 (n_t = min(n_kv, pos + 1)): equal");
    }

    std::printf("[the decode policy's dispatch]\n");
    {
        glmd::DecodePolicy p;
        check(!p.topk_hist && !glmd::use_topk_hist(p, 100000, K), "default: off");
        unsetenv("STRATA_GLMD_TOPK_HIST");
        check(!glmd::decode_policy_env().topk_hist, "env unset: off");
        setenv("STRATA_GLMD_TOPK_HIST", "1", 1);
        glmd::DecodePolicy e = glmd::decode_policy_env();
        check(e.topk_hist, "STRATA_GLMD_TOPK_HIST=1: on");
        e.topk_mb = glmd::kSwitchOff;
        check(!glmd::use_topk_hist(e, K, K) && glmd::use_topk_hist(e, K + 1, K) && glmd::use_topk_hist(e, 100000, K),
              "selects only above top_k");
        e.topk_mb = glmd::kSwitchAuto;
        check(glmd::use_topk_hist(e, glmd::kTopkMbMin, K) && !glmd::use_topk_hist(e, glmd::kTopkMbMin + 1, K),
              "topk_mb keeps its range (auto: above 32768)");
        glmd::DecodePolicy off = e;
        off.topk_hist = false;
        check(glmd::policy_bits(e, false, false, false) != glmd::policy_bits(off, false, false, false) &&
                  (glmd::policy_bits(e, false, false, false) & glmd::kGkTopkHist),
              "the graph key carries the selector bit");
        glmd::DecodePolicy had = off;
        had.idx_had = true;
        check((glmd::policy_bits(had, false, false, false) & glmd::kGkIdxHad) &&
                  !(glmd::policy_bits(off, false, false, false) & glmd::kGkIdxHad),
              "and the B14.11 rotation bit");
        setenv("STRATA_GLMD_TOPK_HIST", "0", 1);
        check(!glmd::decode_policy_env().topk_hist, "STRATA_GLMD_TOPK_HIST=0: off");
        unsetenv("STRATA_GLMD_TOPK_HIST");
        // the twin dispatcher runs the bucket selector and returns the same cells
        std::vector<float> sc(12000);
        for (auto& v : sc) v = (float) (r.next() % 300);
        std::vector<int> a((size_t) K), b((size_t) K);
        e.topk_mb = glmd::kSwitchOff;
        glmd::twin::decode_topk(e, sc.data(), 12000, 12000, nullptr, 11999, K, a.data());
        glmd::twin::topk(sc.data(), 12000, 12000, 1, nullptr, K, b.data());
        check(a == b, "twin::decode_topk with the switch on: the radix selector's cells");
    }
    if (g_fail) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
