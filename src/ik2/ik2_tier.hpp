// maya-ik2's offload path: the order the VRAM tier takes the experts in, and the decode mode's estimate (no GPU here,
// tests/ik2_tier_test.cpp).
//
// The tier (ExpertCache::init's pools, then the warm) is filled once at load from the expert profile (IK2_WARM's
// picks a visit).  Two orders:
//  - picks: every expert by its picks a byte, the most first (the order of v1.1.82 and before);
//  - layers: whole layers, the most visits a byte first - a looped model's core layers (Inkling 2 Mini's 6 .. 17 run
//    three times a token) before the other NVFP4 layers, those before the Q8_0 ones - and inside a layer by picks.
//    Fewer layer visits then wait for an expert from RAM (a visit with no miss goes straight through; one with a
//    miss waits for the CPU lane or a copy), which is what costs in lane parity.
// The same order is used for KV lending (IK2_LEND) and for GPU 1's share (--gpus 2).  Only which experts sit in VRAM
// changes: the answers are those of the same decode mode either way.  The decode mode's estimate and the prompts'
// prefetch keep the profile's own counts.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <vector>

namespace ik2::tier {

// IK2_TIER_ORDER (tests): "layers" 1, "picks" 0; anything else (unset, "auto") -1 - by the decode mode
inline int forced_order(const char* s) {
    if (!s || !*s) return -1;
    if (!std::strcmp(s, "layers")) return 1;
    if (!std::strcmp(s, "picks")) return 0;
    return -1;
}

// The whole-layer order: every expert key (l * E + e) in the order the tier takes them.  visits[l]: the times a
// token runs weight layer l; slot_bytes[l]: one of its experts' slot; picks[l * E + e]: the profile's counts.  The
// layers by visits a byte (the most first), then by their picks a byte, then by index; inside a layer by picks
// (the most first; ties by expert index).
inline std::vector<int> layer_order(const std::vector<int>& visits, const std::vector<size_t>& slot_bytes,
                                    const std::vector<double>& picks, int E) {
    const int L = (int) visits.size();
    std::vector<double> tot((size_t) L, 0.0);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) tot[(size_t) l] += picks[(size_t) l * E + e];
    std::vector<int> lay((size_t) L);
    std::iota(lay.begin(), lay.end(), 0);
    std::stable_sort(lay.begin(), lay.end(), [&](int a, int b) {
        const double va = visits[(size_t) a] / (double) slot_bytes[(size_t) a];
        const double vb = visits[(size_t) b] / (double) slot_bytes[(size_t) b];
        if (va != vb) return va > vb;
        const double ta = tot[(size_t) a] / (double) slot_bytes[(size_t) a];
        const double tb = tot[(size_t) b] / (double) slot_bytes[(size_t) b];
        if (ta != tb) return ta > tb;
        return a < b;
    });
    std::vector<int> out;
    out.reserve((size_t) L * E);
    std::vector<int> ex((size_t) E);
    for (int l : lay) {
        std::iota(ex.begin(), ex.end(), 0);
        std::stable_sort(ex.begin(), ex.end(), [&](int a, int b) {
            return picks[(size_t) l * E + a] > picks[(size_t) l * E + b];
        });
        for (int e : ex) out.push_back(l * E + e);
    }
    return out;
}

// key -> its place in an order (the tier, the lending and GPU 1 sort by it, the lowest first)
inline std::vector<int> ranks(const std::vector<int>& order) {
    std::vector<int> r(order.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) r[(size_t) order[i]] = (int) i;
    return r;
}

// The tier's pools sized in an order (the offload path, ExpertCache::init): each pool starts at its minimum
// (pmin, its first pmin experts in the order), then every expert in the order takes a slot of its pool while the
// pool has fewer than pmax and the budget left holds one (a smaller slot may still fit after one that did not).
// pool_of[l]: layer l's pool (-1: not this GPU's).  Returns the bytes left.
inline size_t size_pools(const std::vector<int>& order, const std::vector<int>& pool_of, int E,
                         const std::vector<size_t>& pool_sb, const std::vector<int>& pmin,
                         const std::vector<int>& pmax, size_t budget, std::vector<int>& n_slots) {
    const size_t np = pool_sb.size();
    n_slots.assign(np, 0);
    size_t left = budget;
    for (size_t p = 0; p < np; ++p) n_slots[p] = pmin[p], left -= std::min(left, (size_t) pmin[p] * pool_sb[p]);
    std::vector<int> seen(np, 0);
    for (int key : order) {
        const int p = pool_of[(size_t) (key / E)];
        if (p < 0) continue;
        if (seen[(size_t) p]++ < pmin[(size_t) p]) continue;   // (in the pool's minimum already)
        if (n_slots[(size_t) p] < pmax[(size_t) p] && pool_sb[(size_t) p] <= left)
            ++n_slots[(size_t) p], left -= pool_sb[(size_t) p];
    }
    return left;
}

// The decode mode's estimate (IK2_LANE auto): a visit's expected misses from the share h of the profile's picks
// that the tier holds - with v1.1.82's tier (picks order) whatever order the tier is filled in, so the mode is the
// one v1.1.82 picked - then mode A's time (the copies, then the experts; the resident kernels taken as twice as fast
// as the exact ones) against lane parity's best split of the misses between the CPU lane and the bus.
struct Estimate {
    double t_a = 0, t_b = 0;
    bool lane = false;   // lane parity is the faster (by 5% or more)
};
// (the figures as the engine keeps them, floats: the same arithmetic as before, bit for bit)
inline Estimate lane_estimate(double h, int K, float lat, float cpu, float fetch, float gpu) {
    Estimate r;
    const double nm = K * (1.0 - h);
    r.t_a = nm * fetch + 0.5 * K * gpu;
    r.t_b = 1e30;
    for (int n = 0; n <= (int) std::ceil(nm); ++n) {
        const double nn = std::min<double>(n, nm);
        r.t_b = std::min(r.t_b, std::max(nn > 0 ? lat + nn * cpu : 0.0, (nm - nn) * fetch + K * gpu));
    }
    r.lane = r.t_b < 0.95 * r.t_a;
    return r;
}

}  // namespace ik2::tier
