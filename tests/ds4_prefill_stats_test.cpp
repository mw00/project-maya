#include "../src/ds4/ds4_prefill_stats.hpp"
#include <climits>
#include <cstdio>
#include <array>
#include <thread>

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

    // Each device writes only its own completed count. The owner merges after the existing completion boundary.
    // One device does no routing (a helper-only worker); another returns early (cancelled/failed partial work).
    constexpr int devices = 8, rounds = 32768;
    std::array<PrefillCounts, devices> per_device{};
    std::array<int, devices> errors{};
    std::array<std::thread, devices> workers;
    for (int i = 0; i < devices; ++i)
        workers[i] = std::thread([&, i] {
            PrefillCounts local;
            strata::ds4::PrefillCountsCapture capture{per_device[i], local};
            const int count = i == 0 ? 0 : i == devices - 1 ? rounds / 4 : rounds;
            const int bounds[] = {0, 2, 5, 10};
            for (int j = 0; j < rounds; ++j) {
                if (j == count) return;
                if (!local.add_bounds(bounds, 3, 1 + i % 3)) {
                    errors[i] = 1;
                    return;
                }
            }
        });
    for (auto& worker : workers) worker.join();
    PrefillCounts total;
    uint64_t expected_picks = 0, expected_hits = 0, expected_uses = 0, expected_vram_uses = 0;
    for (int i = 0; i < devices; ++i) {
        CHECK(errors[i] == 0);
        const uint64_t n = i == 0 ? 0 : i == devices - 1 ? rounds / 4 : rounds;
        const uint64_t hits[] = {0, 2, 5, 10};
        expected_picks += 10 * n;
        expected_hits += hits[1 + i % 3] * n;
        expected_uses += 3 * n;
        expected_vram_uses += (1 + i % 3) * n;
        total.merge(per_device[i]);
    }
    CHECK(total.picks == expected_picks && total.vram_picks == expected_hits);
    CHECK(total.unique_uses == expected_uses && total.vram_unique_uses == expected_vram_uses);
    // Later runs add to lifetime totals; a layer-major delegate must not be overwritten by its empty outer scope.
    PrefillCounts cumulative = total;
    for (const auto& count : per_device) cumulative.merge(count);
    CHECK(cumulative.picks == 2 * expected_picks && cumulative.vram_picks == 2 * expected_hits);
    PrefillCounts delegated, outer;
    {
        strata::ds4::PrefillCountsCapture outer_capture{delegated, outer};
        PrefillCounts inner;
        strata::ds4::PrefillCountsCapture inner_capture{delegated, inner};
        CHECK(inner.add_bounds(a, 3, 2));
    }
    CHECK(delegated.picks == 101 && delegated.vram_picks == 100);
    large.merge(large);
    CHECK(large.picks == uint64_t(INT_MAX) * 4 && large.vram_picks == large.picks);
    std::puts("prefill statistics: bounds, 64-bit totals, worker-local merging, partial runs and delegation passed");
}
