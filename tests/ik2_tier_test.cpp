// maya-ik2's offload path: the order its VRAM tier takes the experts in (src/ik2/ik2_tier.hpp; no GPU).  Whole layers
// by visits a byte (a looped model's core first, then the other layers of its slot size, then the larger slots),
// inside a layer by picks; the pools sized in that order; the decode mode's estimate as v1.1.82 had it.
#include "../src/ik2/ik2_tier.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

// v1.1.82's estimate, as it was written in ik2_main.cu (the figures floats)
bool old_estimate(double h, int K, float lat, float cpu, float fetch, float gpu, double& t_a, double& t_b) {
    const double nm = K * (1.0 - h);
    t_a = nm * fetch + 0.5 * K * gpu;
    t_b = 1e30;
    for (int n = 0; n <= (int) std::ceil(nm); ++n) {
        const double nn = std::min<double>(n, nm);
        t_b = std::min(t_b, std::max(nn > 0 ? lat + nn * cpu : 0.0, (nm - nn) * fetch + K * gpu));
    }
    return t_b < 0.95 * t_a;
}

// a deterministic "profile": picks of expert e of layer l, skewed inside a layer (a few experts take most picks), a
// looped layer's three times as many - so a looped layer's least picked are below another layer's most picked
double picks_of(int l, int e, int visits) { return visits * 1000.0 * std::exp(-((l * 31 + e * 17) % 128) / 20.0); }

}  // namespace

int main() {
    using namespace ik2::tier;

    // IK2_TIER_ORDER
    CHECK(forced_order(nullptr) == -1);
    CHECK(forced_order("") == -1);
    CHECK(forced_order("auto") == -1);
    CHECK(forced_order("layers") == 1);
    CHECK(forced_order("picks") == 0);

    // Inkling-2-Mini's shape: 24 weight layers, 6 .. 17 run three times; 128 experts; NVFP4 slots (4.27 MiB) in
    // layers 0 .. 19, Q8_0 slots (8.07 MiB) in 20 .. 23
    const int L = 24, E = 128;
    const size_t nv = 4477440, q8 = 8462592;
    std::vector<int> visits(L, 1);
    std::vector<size_t> sb(L, nv);
    for (int l = 6; l < 18; ++l) visits[l] = 3;
    for (int l = 20; l < 24; ++l) sb[l] = q8;
    std::vector<double> picks((size_t) L * E);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) picks[(size_t) l * E + e] = picks_of(l, e, visits[l]);
    // (layer 3 busier than its class: still after every looped layer, first of the rest)
    for (int e = 0; e < E; ++e) picks[(size_t) 3 * E + e] *= 1.5;
    const auto ord = layer_order(visits, sb, picks, E);
    CHECK((int) ord.size() == L * E);
    {
        std::vector<int> seen((size_t) L * E, 0);
        for (int k : ord) CHECK(k >= 0 && k < L * E && !seen[(size_t) k]++);   // (every key once)
    }
    // whole layers in a row: the looped core first, then the other NVFP4 layers (layer 3 first), the Q8_0 ones last
    for (int i = 0; i < L; ++i) {
        const int l = ord[(size_t) i * E] / E;
        for (int j = 0; j < E; ++j) CHECK(ord[(size_t) i * E + j] / E == l);
        if (i < 12) CHECK(l >= 6 && l < 18);
        else if (i < 20) CHECK(l < 6 || l == 18 || l == 19);
        else CHECK(l >= 20);
        if (i == 12) CHECK(l == 3);
        // inside a layer: the most picked first
        for (int j = 1; j < E; ++j) CHECK(picks[(size_t) ord[(size_t) i * E + j - 1]] >= picks[(size_t) ord[(size_t) i * E + j]]);
    }
    // ties: equal picks keep the experts' order; equal layers by index
    {
        const std::vector<int> v2 = {1, 1};
        const std::vector<size_t> s2 = {256, 256};
        const std::vector<double> p2 = {5, 5, 5, 5};
        const auto o2 = layer_order(v2, s2, p2, 2);
        CHECK(o2 == (std::vector<int>{0, 1, 2, 3}));
    }
    // ranks: the inverse
    const auto rk = ranks(ord);
    for (size_t i = 0; i < ord.size(); ++i) CHECK(rk[(size_t) ord[i]] == (int) i);

    // the pools sized in the order: pool 0 the NVFP4 layers, pool 1 the Q8_0 ones; no minimum (the offload tier)
    std::vector<int> pool_of(L, 0);
    for (int l = 20; l < 24; ++l) pool_of[l] = 1;
    const std::vector<size_t> psb = {nv, q8};
    const std::vector<int> pmin = {0, 0}, pmax = {20 * E, 4 * E};
    std::vector<int> ns;
    // 7 GiB (a 12 GB card at 32K): the whole core (1,536 experts) and part of the next layer, no Q8_0 slot
    size_t left = size_pools(ord, pool_of, E, psb, pmin, pmax, (size_t) 7 << 30, ns);
    CHECK(ns[0] == (int) (((size_t) 7 << 30) / nv) && ns[1] == 0);
    CHECK(ns[0] >= 12 * E && left < nv);
    // every NVFP4 expert and some Q8_0 ones
    const size_t b2 = (size_t) 20 * E * nv + (size_t) 100 * q8 + nv;
    left = size_pools(ord, pool_of, E, psb, pmin, pmax, b2, ns);
    CHECK(ns[0] == 20 * E && ns[1] == 100 && left == nv);
    // a smaller slot after one that did not fit: the Q8_0 layers first here, the room then fits NVFP4 slots only
    {
        std::vector<int> o3;
        for (int l = 20; l < 24; ++l)
            for (int e = 0; e < E; ++e) o3.push_back(l * E + e);
        for (int l = 0; l < 20; ++l)
            for (int e = 0; e < E; ++e) o3.push_back(l * E + e);
        left = size_pools(o3, pool_of, E, psb, pmin, pmax, (size_t) 2 * q8 + nv, ns);
        CHECK(ns[1] == 2 && ns[0] == 1 && left == 0);
    }
    // a minimum: the pool's first pmin experts in the order count once
    {
        const std::vector<int> pm1 = {E, 0};
        left = size_pools(ord, pool_of, E, psb, pm1, pmax, (size_t) E * nv + 2 * nv, ns);
        CHECK(ns[0] == E + 2 && ns[1] == 0 && left == 0);
    }
    // layers not this GPU's (-1) take nothing
    {
        std::vector<int> po = pool_of;
        for (int l = 0; l < 12; ++l) po[l] = -1;
        left = size_pools(ord, po, E, psb, pmin, pmax, (size_t) 1 << 40, ns);
        CHECK(ns[0] == 8 * E && ns[1] == 4 * E);   // (layers 12 .. 19 and the Q8_0 ones only)
    }

    // With the same room, the whole-layer order leaves fewer layer visits a token waiting for RAM than picks a byte
    // (the profile spread as above: picks a byte keeps a part of every looped layer)
    {
        const size_t room = (size_t) 7 << 30;
        std::vector<int> by_picks(ord);
        std::stable_sort(by_picks.begin(), by_picks.end(), [&](int a, int b) {
            return picks[(size_t) a] / (double) sb[(size_t) (a / E)] > picks[(size_t) b] / (double) sb[(size_t) (b / E)];
        });
        auto waiting_visits = [&](const std::vector<int>& o) {
            std::vector<int> n;
            size_pools(o, pool_of, E, psb, pmin, pmax, room, n);
            std::vector<int> taken(2, 0), held_l(L, 0);
            for (int k : o) {
                const int p = pool_of[(size_t) (k / E)];
                if (taken[(size_t) p] < n[(size_t) p]) ++taken[(size_t) p], ++held_l[(size_t) (k / E)];
            }
            int w = 0;
            for (int l = 0; l < L; ++l) w += held_l[(size_t) l] < E ? visits[(size_t) l] : 0;
            return w;
        };
        const int w_layers = waiting_visits(ord), w_picks = waiting_visits(by_picks);
        std::printf("waiting visits: whole layers %d, picks a byte %d\n", w_layers, w_picks);
        // (the whole core held and layer 3 next: only 11 of the 12 other visits can wait)
        CHECK(w_layers == 48 - 36 - 1);
        CHECK(w_picks > w_layers);
    }

    // the decode mode's estimate: v1.1.82's arithmetic, bit for bit
    {
        const float lats[] = {30.0f, 12.5f}, cpus[] = {147.525f, 192.666f, 40.0f}, fetches[] = {364.416f, 365.312f, 90.0f};
        const float gpus[] = {17.156f, 28.416f, 60.0f};
        for (double h : {0.0, 0.31, 0.5, 0.75, 0.9, 1.0})
            for (int K : {8, 16})
                for (float lat : lats)
                    for (float cpu : cpus)
                        for (float fe : fetches)
                            for (float g : gpus) {
                                double a = 0, b = 0;
                                const bool lane = old_estimate(h, K, lat, cpu, fe, g, a, b);
                                const Estimate e = lane_estimate(h, K, lat, cpu, fe, g);
                                CHECK(e.lane == lane && e.t_a == a && e.t_b == b);
                            }
        // every pick in the tier: the lane has nothing to take - mode A; half of them out on a fast CPU: lane parity
        CHECK(!lane_estimate(1.0, 8, 30.0f, 147.5f, 364.4f, 17.2f).lane);
        CHECK(lane_estimate(0.5, 8, 30.0f, 147.5f, 364.4f, 17.2f).lane);
    }

    std::printf("ik2_tier_test: ok\n");
    return 0;
}
