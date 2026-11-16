// include/strata/kernels/glmd_random_route.hpp - glm-dsa B15.7: the random-routing bench mode.
//
// MAYA_GLM_RANDOM_ROUTING=1 replaces the k experts each decode route selects with k DISTINCT pseudo-random expert
// ids and keeps the router's weights (rank i keeps the weight the router gave its rank-i expert).  Every route then
// misses the VRAM pool and the RAM tier as often as uniform routing does, independent of the prompt corpus: a
// reproducible worst-case decode floor for the BUILD-PLAN section 7 budget, which separates kernel and tier
// regressions from routing-skew drift (llama.cpp #29539, llm_graph_input_moe_ids::set_input).  Bench mode only: the
// output text is meaningless.
//
// The ids are a partial Fisher-Yates shuffle of 0..E-1 (k steps, the swaps kept sparse: no E-sized array), driven by
// splitmix64 of (seed, layer, route counter).  The route counter is the device's per-route sequence number
// (MoeDev::seq: one route per token and layer, the NextN block's included), so a run is reproducible from the load
// on; MAYA_GLM_RANDOM_ROUTING_SEED (default 1) picks another sequence.  The hook sits in glm_fast.cu
// moe_route_kernel right after the selection, so the fetch, CPU-lane, promotion and prefetch logic all see the
// random ids.  The prompt path keeps the real routing.
#pragma once

#include "strata/core/glm_env.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GLMD_RR_HD __host__ __device__
#else
#define GLMD_RR_HD
#endif

namespace glmd {

GLMD_RR_HD inline uint64_t rr_splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// k (<= 8) distinct ids in [0, E) for (seed, layer, counter): a partial Fisher-Yates shuffle of the identity
/// permutation with the touched positions kept in a sparse list.  Deterministic; ids[] is in draw order.  E < k or
/// k outside 1..8 writes nothing and returns false.
GLMD_RR_HD inline bool random_route_ids(uint64_t seed, uint32_t layer, uint64_t counter, int E, int k, int* ids) {
    if (k < 1 || k > 8 || E < k) return false;
    uint64_t st = seed * 0xD1B54A32D192ED03ull ^ ((uint64_t) layer << 40) ^ counter * 0x8CB92BA72F3D8DD7ull;
    int pos[8], val[8], n = 0;   // the sparse permutation: position pos[i] holds val[i] (n <= j + 1 <= k <= 8)
    for (int j = 0; j < k; ++j) {
        const uint64_t r = rr_splitmix64(st);
        const int pick = j + (int) (r % (uint64_t) (E - j));
        int v_pick = pick, v_j = j, i_pick = -1;
        for (int i = 0; i < n; ++i) {
            if (pos[i] == pick) { v_pick = val[i]; i_pick = i; }
            if (pos[i] == j) v_j = val[i];
        }
        ids[j] = v_pick;
        // swap: position j now holds v_pick (never read again: later steps draw from j + 1 on), pick holds v_j
        if (i_pick >= 0) val[i_pick] = v_j;
        else { pos[n] = pick; val[n] = v_j; ++n; }
    }
    return true;
}

/// The kernel hook: replaces the k selected ids with random_route_ids(seed, layer, counter) and moves each rank's
/// probability with it (p[new id] = the p of the id it replaces), so the weights the route derives from p[ids[i]] are
/// the router's.  p is the route's E-entry probability row.  A route holding an invalid id (-1: a non-finite logit)
/// is left as it is.  Single-threaded (one thread of the route block).
GLMD_RR_HD inline void random_route_apply(uint64_t seed, uint32_t layer, uint64_t counter, int E, int k, int* ids,
                                          float* p) {
    int r[8];
    for (int i = 0; i < k && i < 8; ++i)
        if (ids[i] < 0 || ids[i] >= E) return;   // an invalid route stays invalid (the kernel rejects it)
    if (!random_route_ids(seed, layer, counter, E, k, r)) return;
    float pw[8];
    for (int i = 0; i < k; ++i) pw[i] = p[ids[i]];
    for (int i = 0; i < k; ++i) {
        ids[i] = r[i];
        p[r[i]] = pw[i];
    }
}

/// 0 = off (the default); else the seed: MAYA_GLM_RANDOM_ROUTING=1, MAYA_GLM_RANDOM_ROUTING_SEED (default 1; a
/// seed of 0 is taken as 1).  Read once per process (host only).
inline uint64_t random_routing_seed() {
    static const uint64_t seed = [] {
        if (!strata::glmenv::env_flag("MAYA_GLM_RANDOM_ROUTING", false)) return (uint64_t) 0;
        uint64_t x = (uint64_t) strata::glmenv::env_int("MAYA_GLM_RANDOM_ROUTING_SEED", 1);
        if (x == 0) x = 1;
        std::fprintf(stderr, "glm: MAYA_GLM_RANDOM_ROUTING=1 (seed %llu): BENCH MODE - every decode route takes random "
                             "experts, the output is meaningless; MTP drafting is off, kept slots and prompt-cache "
                             "entries are marked (a normal load refuses them); do not save an expert profile or a usage "
                             "file from this run\n", (unsigned long long) x);
        return x;
    }();
    return seed;
}

}  // namespace glmd
