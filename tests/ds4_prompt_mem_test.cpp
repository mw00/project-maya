// tests/ds4_prompt_mem_test.cpp - the DeepSeek engine's prompt-path and memory rules (src/ds4/ds4_prompt_mem.hpp),
// each against the cases it was written for.  CPU only, no model.
#include "../src/ds4/ds4_prompt_mem.hpp"

#include <cstdio>

namespace d = strata::ds4;

static int fails = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                   \
        }                                                              \
    } while (0)

int main() {
    {   // switches: unset / empty -1, else 0 or 1
        CHECK(d::tri_of(nullptr) == -1 && d::tri_of("") == -1);
        CHECK(d::tri_of("0") == 0 && d::tri_of("1") == 1 && d::tri_of("2") == 1);
    }
    {   // the path by length: off unless asked, 96 by default, length:N moves it, never below one token
        CHECK(d::prompt_pick_length(nullptr) == 0);
        CHECK(d::prompt_pick_length("") == 0);
        CHECK(d::prompt_pick_length("adaptive") == 0);
        CHECK(d::prompt_pick_length("lengthy") == 0);
        CHECK(d::prompt_pick_length("length:") == 0);
        CHECK(d::prompt_pick_length("length") == 96);
        CHECK(d::prompt_pick_length("length:400") == 400);
        CHECK(d::prompt_pick_length("length:0") == 1 && d::prompt_pick_length("length:-5") == 1);
        const int m = d::prompt_pick_length("length");
        for (int rep = 0; rep < 3; ++rep) {   // (the same prompt, the same path, every time)
            CHECK(!d::prompt_by_length(2, m));
            CHECK(!d::prompt_by_length(28, m));
            CHECK(!d::prompt_by_length(95, m));
            CHECK(d::prompt_by_length(96, m));
            CHECK(d::prompt_by_length(8192, m));
        }
    }
    {   // the chunk fallback: halves in whole 16s down to 256, then windows (0)
        CHECK(d::chunk_fallback(16384) == 8192);
        CHECK(d::chunk_fallback(4096) == 2048);
        CHECK(d::chunk_fallback(1000) == 496);
        CHECK(d::chunk_fallback(512) == 256);
        CHECK(d::chunk_fallback(511) == 0 && d::chunk_fallback(256) == 0 && d::chunk_fallback(0) == 0);
        int c = 16384, tries = 0;
        while (c > 0) c = d::chunk_fallback(c), ++tries;
        CHECK(tries == 7);   // 8192 4096 2048 1024 512 256 0
    }
    {   // the staging ring: 64 MB a GB of the device's room, 400 to 1200 MB
        const double GB = 1e9;
        CHECK(d::staging_ring_mb(0.0) == 400 && d::staging_ring_mb(-5 * GB) == 400);
        CHECK(d::staging_ring_mb(4 * GB) == 400);
        CHECK(d::staging_ring_mb(6.25 * GB) == 400);
        CHECK(d::staging_ring_mb(10 * GB) == 640);
        CHECK(d::staging_ring_mb(18.75 * GB) == 1200);
        CHECK(d::staging_ring_mb(80 * GB) == 1200);
    }
    {   // --mmap-experts auto beside an integrated GPU: none, unless DS4_COPY_ALL=1 (and they fit); =0 never
        const double GB = 1e9, spare = 8.0 * (1ull << 30);
        CHECK(d::copy_all_auto(100 * GB, 200 * GB, spare, false, -1));
        CHECK(!d::copy_all_auto(195 * GB, 200 * GB, spare, false, -1));   // (fits, but not with the spare)
        CHECK(!d::copy_all_auto(100 * GB, 200 * GB, spare, true, -1));
        CHECK(d::copy_all_auto(100 * GB, 200 * GB, spare, true, 1));
        CHECK(!d::copy_all_auto(195 * GB, 200 * GB, spare, true, 1));
        CHECK(!d::copy_all_auto(100 * GB, 200 * GB, spare, false, 0));
        CHECK(!d::copy_all_auto(100 * GB, 200 * GB, spare, true, 0));
    }
    {   // the huge-page arena: unset -> only when the reserved pool holds every copy; 0 / 1 force it
        const uint64_t GiB = 1ull << 30, page = 2ull << 20, need = 150 * GiB;
        CHECK(!d::copy_hugetlb_on(-1, 0, need));             // (no pool: the ordinary arena, as before)
        CHECK(!d::copy_hugetlb_on(-1, need - page, need));   // (one page short)
        CHECK(d::copy_hugetlb_on(-1, need, need));
        CHECK(d::copy_hugetlb_on(-1, 160 * GiB, need));
        CHECK(!d::copy_hugetlb_on(-1, 160 * GiB, 0));        // (no copies)
        CHECK(!d::copy_hugetlb_on(0, 160 * GiB, need));
        CHECK(d::copy_hugetlb_on(1, 0, need));
        CHECK(d::round_up(1, page) == page && d::round_up(page, page) == page);
        CHECK(d::round_up(page + 1, page) == 2 * page && d::round_up(0, page) == 0);
    }
    {   // an integrated GPU's room: MemAvailable less the headroom when larger, the driver's figure otherwise, at most
        // the device's total
        const uint64_t GiB = 1ull << 30, head = 6 * GiB, tot = 120 * GiB, unknown = ~uint64_t{0};
        CHECK(d::uma_free(55 * GiB, tot, 115 * GiB, head) == 109 * GiB);
        CHECK(d::uma_free(55 * GiB, tot, 60 * GiB, head) == 55 * GiB);
        CHECK(d::uma_free(55 * GiB, tot, unknown, head) == 55 * GiB);      // (MemAvailable not known)
        CHECK(d::uma_free(55 * GiB, tot, 5 * GiB, head) == 55 * GiB);      // (less than the headroom)
        CHECK(d::uma_free(55 * GiB, 100 * GiB, 200 * GiB, head) == 100 * GiB);
    }
    if (fails == 0) std::printf("ds4 prompt / memory rules: ok\n");
    return fails == 0 ? 0 : 1;
}
