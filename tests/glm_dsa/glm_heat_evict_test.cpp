// tests/glm_dsa/glm_heat_evict_test.cpp - glm-dsa B15.8 (CPU): heat-decay expert promotion / eviction
// (include/strata/core/glm_heat_evict.hpp).
//   1. HeatTable: 0.90 decay per tick (lazy scale == eager decay), route_layer ticks once per forward pass, the
//      renormalisation keeps values, quant() saturates;
//   2. the LRU-window victim: only the 16 oldest resident slots compete, the coldest of them wins, ties to the older;
//   3. coalescing: adjacent dst and src merge (events kept in order), a gap, another layer or a node-split class
//      (plain false) does not; defer_up / flush_ups are no-ops without a table;
//   4. the swap rule: margin and the per-layer cap of 2, Maya's 2x + 4 rule without a table;
//   5. the simulator, Maya's aged-count policy vs heat-decay on synthetic routing (16 layers x 256 experts, top-8,
//      5000 tokens, 1000 warm-up): stationary, 4 recurring topics, a gradual walk; at 64 / 96 / 160 VRAM slots a
//      layer.  The hit rates are printed (the B15.8 report) and checked: heat wins by >= 1 point on every drifting
//      trace at 64 and 96 slots and stays within 1.5 points on the stationary one; it is deterministic.
#include "strata/core/glm_heat_evict.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

using namespace strata::core::glmheat;

void test_table() {
    std::printf("[1. HeatTable]\n");
    HeatTable H;
    H.resize(4);
    H.hit(1);
    double eager = 1.0;
    for (int t = 0; t < 7; ++t) {
        H.tick();
        eager *= kDecay;
    }
    H.hit(1, 2.0);
    eager += 2.0;
    check(std::fabs(H.heat(1) - eager) < 1e-9 * eager, "lazy decay == eager decay (" + std::to_string(H.heat(1)) + ")");
    check(H.heat(0) == 0.0 && H.heat(99) == 0.0, "untouched and out-of-range keys are 0");
    for (int t = 0; t < 2000; ++t) H.tick();   // scale passes 1e30 and is renormalised
    H.hit(2);
    check(std::fabs(H.heat(2) - 1.0) < 1e-9 && H.heat(1) < 1e-80, "renormalisation keeps the values");
    HeatTable R;
    R.resize(8);
    const uint64_t t0 = R.ticks();
    for (int pass = 0; pass < 3; ++pass)
        for (int il : {3, 4, 5, 9}) R.route_layer(il);   // three forward passes (the MTP layer last)
    check(R.ticks() - t0 == 3, "route_layer: one tick per forward pass (" + std::to_string(R.ticks() - t0) + ")");
    HeatTable Q;
    Q.resize(1);
    Q.hit(0, 1e12);
    check(Q.quant(0) == 0xffffffffu, "quant saturates");
}

void test_victim() {
    std::printf("[2. the LRU-window victim]\n");
    HeatTable H;
    H.resize(64);
    const int n = 40;
    std::vector<char> st((size_t) n, 1);
    std::vector<int> key((size_t) n);
    std::vector<uint64_t> tick((size_t) n);
    for (int s = 0; s < n; ++s) {
        key[(size_t) s] = s;
        tick[(size_t) s] = 1000 - (uint64_t) s;   // slot 39 the least recently used
        H.hit((size_t) s, 10.0 + s);
    }
    H.hit(0, -9.5);   // slot 0: the coldest of all, but the most recently used: outside the window
    const int v = lru_window_victim(H, n, st.data(), 1, key.data(), tick.data(), 0);
    check(v == 24, "the coldest of the 16 oldest slots (24..39) is 24, not the globally coldest slot 0 (got " +
                       std::to_string(v) + ")");
    st[24] = 3;   // draining: not resident
    check(lru_window_victim(H, n, st.data(), 1, key.data(), tick.data(), 0) == 23,
          "a non-resident slot never competes (the window slides to 23..39)");
    H.hit(30, 25.0 - 40.0);   // slot 30 now ties 23 at heat 33? make it colder than all in the window
    check(lru_window_victim(H, n, st.data(), 1, key.data(), tick.data(), 0) == 30, "the colder slot inside wins");
    std::vector<char> none((size_t) n, 0);
    check(lru_window_victim(H, n, none.data(), 1, key.data(), tick.data(), 0) == -1, "no resident slot: -1");
}

void test_coalesce() {
    std::printf("[3. coalescing the promotion uploads]\n");
    static uint8_t vram[4096], ram[4096];
    int e[6];
    std::vector<Up> ups = {{2, vram + 300, ram + 900, 100, &e[0]}, {2, vram + 100, ram + 700, 100, &e[1]},
                           {2, vram + 200, ram + 800, 100, &e[2]}, {2, vram + 500, ram + 1000, 100, &e[3]},
                           {3, vram + 600, ram + 1100, 100, &e[4]}, {2, vram + 400, ram + 2000, 100, &e[5]}};
    const auto runs = coalesce(ups, [](int, const void*) { return true; });
    // layer 2: 100..400 contiguous (src 700..1000), then 400 (src 2000: src gap), 500 (src 1000); layer 3: 600
    bool ok = runs.size() == 4 && runs[0].il == 2 && runs[0].dst == vram + 100 && runs[0].bytes == 300 &&
              runs[0].tags.size() == 3 && runs[0].tags[0] == &e[1] && runs[0].tags[2] == &e[0] && runs[1].bytes == 100 &&
              runs[1].src == ram + 2000 && runs[2].dst == vram + 500 && runs[3].il == 3;
    check(ok, "adjacent dst + src merge (3 uploads -> 1 copy, events in order); a src gap and another layer do not (" +
                  std::to_string(runs.size()) + " runs)");
    const auto split = coalesce(ups, [](int, const void*) { return false; });
    check(split.size() == ups.size(), "a node-split class (plain false) never merges");
    check(!defer_up(nullptr, 0, vram, ram, 1, nullptr) && flush_ups(nullptr, nullptr, nullptr, nullptr).first == 0,
          "without a table (the switch off) defer_up / flush_ups do nothing");
    HeatTable H;
    H.resize(1);
    for (const Up& u : ups) defer_up(&H, u.il, u.dst, u.src, u.bytes, u.tag);
    size_t copies = 0, recorded = 0;
    const auto r = flush_ups(&H, [&](int, uint8_t*, const uint8_t*, size_t) { ++copies; }, [&](void*) { ++recorded; },
                             [](int, const void*) { return true; });
    check(r.first == 4 && r.second == 6 && copies == 4 && recorded == 6, "flush_ups: 6 uploads in 4 copies, 6 events");
    check(flush_ups(&H, [&](int, uint8_t*, const uint8_t*, size_t) { ++copies; }, [&](void*) {},
                    [](int, const void*) { return true; }).second == 0, "flushed once");
}

void test_swap_rule() {
    std::printf("[4. the swap rule]\n");
    HeatTable H;
    H.resize(1);
    H.begin_boundary();
    check(!swap_ok(&H, 16 * 3, 16 * 2, 2, 5), "heat: one route's worth above the victim is not enough (margin 2)");
    check(swap_ok(&H, 16 * 4, 16 * 2, 2, 5) && swap_ok(&H, 16 * 9, 0, 2, 5) && !swap_ok(&H, 16 * 9, 0, 2, 5),
          "heat: margin met; at most 2 swaps per layer per boundary");
    check(swap_ok(&H, 16 * 9, 0, 2, 6), "another layer has its own cap");
    H.begin_boundary();
    check(swap_ok(&H, 16 * 9, 0, 2, 5), "the cap restarts at the next boundary");
    check(swap_ok(nullptr, 24, 10, 2, 5) && !swap_ok(nullptr, 23, 10, 2, 5) && swap_ok(nullptr, 6, 0, 2, 5),
          "without a table: Maya's hot >= 2 * max(cold, 1) + 4");
}

void test_simulation() {
    std::printf("[5. simulation: Maya's aged counts vs heat decay (16 layers x 256 experts, top-8, 5000 tokens)]\n");
    struct Case {
        const char* name;
        Drift d;
        int period;
    };
    const Case cases[] = {{"stationary", Drift::Stationary, 300},
                          {"4 topics, switch every 300", Drift::Topics, 300},
                          {"4 topics, switch every 1000", Drift::Topics, 1000},
                          {"gradual walk (fast)", Drift::Gradual, 300},
                          {"gradual walk (slow)", Drift::Gradual, 1000}};
    for (int slots : {64, 96, 160}) {
        std::printf("  %d VRAM slots a layer (%.0f%% of the experts)\n", slots, 100.0 * slots / 256);
        for (const Case& cs : cases) {
            const Trace t = synth_trace(cs.d, 16, 256, 8, 5000, cs.period, 1.0, 7);
            SimConfig c;
            c.slots = slots;
            const SimResult a = simulate(t, Policy::Maya, c), b = simulate(t, Policy::Heat, c);
            const double gain = 100.0 * (b.hit_rate() - a.hit_rate());
            std::printf("    %-28s Maya %6.2f%%  heat %6.2f%%  (%+.2f points; swaps %llu vs %llu; uploads %llu in %llu "
                        "copies)\n", cs.name, 100.0 * a.hit_rate(), 100.0 * b.hit_rate(), gain,
                        (unsigned long long) a.demotions, (unsigned long long) b.demotions,
                        (unsigned long long) b.copies, (unsigned long long) b.runs);
            if (slots <= 96) {
                if (cs.d == Drift::Stationary)
                    check(gain > -1.5, std::string("stationary: heat within 1.5 points of Maya (") +
                                           std::to_string(gain) + ")");
                else
                    check(gain >= 1.0, std::string(cs.name) + ": heat wins >= 1 point (" + std::to_string(gain) + ")");
            }
            if (slots == 96 && cs.d == Drift::Topics && cs.period == 300) {
                const SimResult b2 = simulate(t, Policy::Heat, c);
                check(b2.hits == b.hits && b2.demotions == b.demotions, "deterministic");
                check(b.runs <= b.copies, "coalescing never adds copies");
            }
        }
    }
}

}  // namespace

int main() {
    std::printf("glm_heat_evict_test: heat-decay promotion / eviction (B15.8)\n");
    test_table();
    test_victim();
    test_coalesce();
    test_swap_rule();
    test_simulation();
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
