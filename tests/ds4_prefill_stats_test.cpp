#include "../src/ds4/ds4_prefill_stats.hpp"
#include <climits>
#include <cstdio>

using strata::ds4::PrefillCounts;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    PrefillCounts c;
    // A hot resident expert serves 100 rows; a nonresident expert serves one.
    // The row hit rate is 100/101, while the unique-use hit rate is 1/2.
    const int a[] = {0, 100, 100, 101};
    CHECK(c.add_bounds(a, 3, 2));
    CHECK(c.picks == 101 && c.vram_picks == 100);
    CHECK(c.unique_uses == 2 && c.vram_unique_uses == 1);

    // Reusing an expert in another chunk counts again. Offsets may be nonzero.
    const int b[] = {7, 7, 7, 12};
    CHECK(c.add_bounds(b, 3, 2));
    CHECK(c.picks == 106 && c.vram_picks == 100);
    CHECK(c.unique_uses == 3 && c.vram_unique_uses == 1);

    // A normally resident slot lent to prefill buffers must count as streamed.
    const int lent[] = {0, 3, 3, 10};
    PrefillCounts normal, borrowed, all_streamed;
    CHECK(normal.add_bounds(lent, 3, 3));
    CHECK(borrowed.add_bounds(lent, 3, 2));
    CHECK(all_streamed.add_bounds(lent, 3, 0));
    CHECK(normal.picks == 10 && normal.vram_picks == 10);
    CHECK(borrowed.picks == 10 && borrowed.vram_picks == 3);
    CHECK(borrowed.unique_uses == 2 && borrowed.vram_unique_uses == 1);
    CHECK(all_streamed.picks == 10 && all_streamed.vram_picks == 0);

    // Invalid input is rejected atomically, even after valid leading entries.
    const int invalid[] = {0, 5, 4}, negative[] = {-1, 0};
    const auto saved = c;
    CHECK(!c.add_bounds(invalid, 2, 1));
    CHECK(!c.add_bounds(a, 3, 4));
    CHECK(!c.add_bounds(nullptr, 0, 0));
    CHECK(!c.add_bounds(negative, 1, 0));
    CHECK(c.picks == saved.picks && c.vram_picks == saved.vram_picks);
    CHECK(c.unique_uses == saved.unique_uses && c.vram_unique_uses == saved.vram_unique_uses);
    const int empty[] = {0};
    CHECK(c.add_bounds(empty, 0, 0) && c.picks == saved.picks);

    // Totals must not truncate to a signed 32-bit count across chunks/layers.
    PrefillCounts large;
    const int maximum[] = {0, INT_MAX};
    CHECK(large.add_bounds(maximum, 1, 1) && large.add_bounds(maximum, 1, 1));
    CHECK(large.picks == uint64_t(INT_MAX) * 2 && large.vram_picks == large.picks);
    std::puts("prefill statistics: weighted hits, repeated uses, lent slots, invalid bounds and 64-bit totals passed");
}
