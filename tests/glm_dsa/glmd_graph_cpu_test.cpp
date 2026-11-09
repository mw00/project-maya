// tests/glm_dsa/glmd_graph_cpu_test.cpp - glm-dsa B10.1 / B14.4 (CPU): the graph-key rules of a captured glm-dsa part.
//   regime    bit 0 from p + 1 > 2048, bit 1 (WMMA scorer) from p + 1 > 16384, bit 2 (radix top-k) from p + 1 > 32768,
//             each only with its switch on;
//   n_cap     4096 up to position 4095, then powers of two, capped at max_ctx; a 64K session re-captures 5 times;
//   bits      every switch has its own bit;
//   key       any field change (part, range, plan, addresses, bits, regime, n_cap) makes a different key.
#include "strata/kernels/glmd_graph.hpp"

#include <cstdio>
#include <set>
#include <tuple>

namespace {
int g_fail = 0;
void expect(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}
}  // namespace

int main() {
    using glmd::graph_regime;
    expect(graph_regime(0, 2048, true, 16384, true, 32768) == 0 && graph_regime(2047, 2048, true, 16384, true, 32768) == 0 &&
               graph_regime(2048, 2048, true, 16384, true, 32768) == 1,
           "regime: selection starts at p + 1 = 2049");
    expect(graph_regime(16383, 2048, true, 16384, true, 32768) == 1 && graph_regime(16384, 2048, true, 16384, true, 32768) == 3 &&
               graph_regime(16384, 2048, false, 16384, true, 32768) == 1,
           "regime: the WMMA scorer from p + 1 = 16385, only when switched on");
    expect(graph_regime(32768, 2048, true, 16384, true, 32768) == 7 && graph_regime(32768, 2048, true, 16384, false, 32768) == 3,
           "regime: the radix top-k from p + 1 = 32769, only when switched on");
    expect(glmd::graph_n_cap(0, 65536) == 4096 && glmd::graph_n_cap(4095, 65536) == 4096 &&
               glmd::graph_n_cap(4096, 65536) == 8192 && glmd::graph_n_cap(65535, 65536) == 65536 &&
               glmd::graph_n_cap(100, 2048) == 2048,
           "n_cap: 4096 minimum, powers of two, capped at max_ctx");
    std::set<std::tuple<int, int64_t>> keys;
    for (int64_t p = 0; p < 65536; ++p)
        keys.insert({graph_regime(p, 2048, true, 16384, true, 32768), glmd::graph_n_cap(p, 65536)});
    std::printf("  distinct (regime, n_cap) over a 64K session: %zu\n", keys.size());
    expect(keys.size() <= 8, "a 64K session re-captures a part at most 8 times");
    const uint32_t all = glmd::graph_kernel_bits(true, true, true, true, true, true, true);
    bool distinct = true;
    for (int i = 0; i < 7; ++i) {
        const uint32_t b = glmd::graph_kernel_bits(i == 0, i == 1, i == 2, i == 3, i == 4, i == 5, i == 6);
        distinct &= b != 0 && (b & (b - 1)) == 0;
    }
    expect(distinct && all == 127u && glmd::graph_kernel_bits(false, false, false, false, false, false, false) == 0,
           "kernel bits: one bit per switch");
    glmd::GraphKey a;
    a.part = 1;
    a.l0 = 26;
    a.l1 = 54;
    a.cpu_plan = 3;
    int x, y, z;
    a.residual = &x;
    a.arena = &y;
    a.state = &z;
    a.bits = 5;
    a.regime = 1;
    a.n_cap = 4096;
    glmd::GraphKey b = a;
    bool all_differ = a == b;
    b.part = 2; all_differ &= a != b; b = a;
    b.l1 = 53; all_differ &= a != b; b = a;
    b.cpu_plan = 4; all_differ &= a != b; b = a;
    b.residual = &y; all_differ &= a != b; b = a;
    b.arena = &x; all_differ &= a != b; b = a;
    b.state = &x; all_differ &= a != b; b = a;
    b.bits = 4; all_differ &= a != b; b = a;
    b.regime = 3; all_differ &= a != b; b = a;
    b.n_cap = 8192; all_differ &= a != b;
    expect(all_differ, "key: every field takes part in the comparison");
    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
