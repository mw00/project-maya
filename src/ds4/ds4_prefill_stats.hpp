#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::ds4 {
// Per-device executed routing, excluding rows removed by dependency pruning.
// One virtual expert may serve many rows: row hits and unique uses differ.
struct PrefillCounts {
    uint64_t picks = 0, vram_picks = 0;
    uint64_t unique_uses = 0, vram_unique_uses = 0;

    bool add_bounds(const int* bounds, size_t slots, size_t resident_slots) {
        if (!bounds || resident_slots > slots || bounds[0] < 0) return false;
        for (size_t v = 0; v < slots; ++v)
            if (bounds[v + 1] < bounds[v]) return false;
        for (size_t v = 0; v < slots; ++v) {
            const uint64_t n = uint64_t(bounds[v + 1] - bounds[v]);
            picks += n;
            unique_uses += n != 0;
            if (v < resident_slots) {
                vram_picks += n;
                vram_unique_uses += n != 0;
            }
        }
        return true;
    }
};
}
