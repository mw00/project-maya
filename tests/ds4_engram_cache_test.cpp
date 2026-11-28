// maya-ds4's Engram row cache (src/ds4/ds4_engram_io.hpp): the clock keeps the rows found since the hand last passed,
// a capacity grown keeps the rows held and evicts none until the new one is full, and the default size by the RAM the
// placed experts leave (#926). Host code only: no model, no GPU.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ds4/ds4_engram_io.hpp"

#include <cassert>
#include <cstdio>

using strata::ds4::eg_cache_default_rows;
using strata::ds4::EgRowCache;
using strata::ds4::kEgCacheDefaultMax;
using strata::ds4::kEgRowIndexBytes;

namespace {

constexpr size_t kRow = 264;   // (DeepSeek-V4.1-Flash-1017's MXFP8 rows: 256 codes, 8 scales)

void fill(uint8_t* p, uint64_t key) {
    for (size_t i = 0; i < kRow; ++i) p[i] = (uint8_t) (key * 131 + i * 7);
}
bool same(const uint8_t* p, uint64_t key) {
    if (p == nullptr) return false;
    for (size_t i = 0; i < kRow; ++i)
        if (p[i] != (uint8_t) (key * 131 + i * 7)) return false;
    return true;
}

}  // namespace

int main() {
    // the clock: a full cache's next row takes the slot of one not used since the hand last passed; the rest stay
    {
        EgRowCache c;
        c.init(4, kRow);
        for (uint64_t k = 0; k < 4; ++k) fill(c.put(k), k);
        assert(c.size() == 4 && c.capacity() == 4);
        fill(c.put(4), 4);   // (no row used yet: the hand's first slot, row 0, goes)
        assert(c.find(0) == nullptr);
        for (uint64_t k = 1; k <= 4; ++k) assert(same(c.find(k), k));
        assert(c.size() == 4);
    }
    // grow: the rows held stay where they are, new rows take new slots until the new capacity is full, and a smaller
    // capacity changes nothing
    {
        EgRowCache c;
        c.init(4, kRow);
        for (uint64_t k = 0; k < 4; ++k) fill(c.put(k), k);
        c.grow(8);
        assert(c.capacity() == 8 && c.size() == 4);
        for (uint64_t k = 4; k < 8; ++k) fill(c.put(k), k);
        assert(c.size() == 8);
        for (uint64_t k = 0; k < 8; ++k) assert(same(c.find(k), k));
        c.grow(2);
        assert(c.capacity() == 8);
        fill(c.put(8), 8);   // (full again: one row goes, the other eight's bytes unchanged)
        int held = 0;
        for (uint64_t k = 0; k <= 8; ++k) held += same(c.find(k), k) ? 1 : 0;
        assert(held == 8 && c.size() == 8);
    }
    // grow across the cache's 4096-row blocks: the rows in the old blocks stay, the new ones take new blocks
    {
        EgRowCache c;
        c.init(4000, kRow);
        for (uint64_t k = 0; k < 4000; ++k) fill(c.put(k), k);
        c.grow(10000);
        for (uint64_t k = 4000; k < 10000; ++k) fill(c.put(k), k);
        assert(c.size() == 10000 && c.capacity() == 10000);
        for (uint64_t k = 0; k < 10000; ++k) assert(same(c.find(k), k));
        fill(c.put(10000), 10000);   // (full, every row used: the hand clears them all and takes slot 0, row 0's)
        assert(c.size() == 10000 && c.find(0) == nullptr && same(c.find(10000), 10000) && same(c.find(9999), 9999));
    }
    // a capacity grown from a full cache: the clock's hand and its rows carry on
    {
        EgRowCache c;
        c.init(3, kRow);
        for (uint64_t k = 0; k < 5; ++k) fill(c.put(k), k);   // (rows 0 and 1 evicted)
        c.grow(6);
        for (uint64_t k = 5; k < 8; ++k) fill(c.put(k), k);
        for (uint64_t k = 2; k < 8; ++k) assert(same(c.find(k), k));
        assert(c.find(0) == nullptr && c.find(1) == nullptr && c.size() == 6);
    }
    // the default: a quarter of the RAM beyond the margin over a row and its index, between the load's size and 16 GiB
    {
        const uint64_t gib = uint64_t{1} << 30, margin = 8 * gib, per = kRow + kEgRowIndexBytes;
        const size_t floor = (size_t) ((256u << 20) / kRow);
        assert(eg_cache_default_rows(~uint64_t{0}, margin, kRow, floor) == floor);   // (an unknown RAM)
        assert(eg_cache_default_rows(margin, margin, kRow, floor) == floor);         // (nothing beyond the margin)
        assert(eg_cache_default_rows(margin - 1, margin, kRow, floor) == floor);
        assert(eg_cache_default_rows(margin + 64 * gib, margin, 0, floor) == floor); // (no row size)
        assert(eg_cache_default_rows(margin + 4 * per * 3000000, margin, kRow, floor) == 3000000);
        assert(eg_cache_default_rows(margin + 4 * per * 1000, margin, kRow, floor) == floor);   // (the floor wins)
        assert(eg_cache_default_rows(margin + 4096 * gib, margin, kRow, floor) == kEgCacheDefaultMax / kRow);
        // (a 160 GB container with 75 GB left beyond the margin: ~58.6M rows, 15.5 GB of rows)
        const size_t r = eg_cache_default_rows(margin + 75 * gib, margin, kRow, floor);
        assert(r == (size_t) (75 * gib / 4 / per) && r < kEgCacheDefaultMax / kRow);
    }
    std::printf("ds4_engram_cache_test: ok\n");
    return 0;
}
