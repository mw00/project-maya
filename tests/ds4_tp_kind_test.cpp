// tests/ds4_tp_kind_test.cpp - CPU-only checks for the DeepSeek engine's tensor-parallel pairing (#1276): identical
// cards pair even when the driver reserves a few MiB more on one of them; cards of different sizes do not.
#include "../src/ds4/ds4_tp_kind.hpp"

#include <cstddef>
#include <cstdio>

using maya::ds4::tp_same_kind;

#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main() {
    const std::size_t MiB = 1024ull * 1024ull;
    const std::size_t rtx3090 = 24576 * MiB - 431 * MiB, boot_vga = 24576 * MiB - 434 * MiB;   // (the report's cards)
    CHECK(tp_same_kind(rtx3090, rtx3090));
    CHECK(tp_same_kind(rtx3090, boot_vga) && tp_same_kind(boot_vga, rtx3090));
    CHECK(!tp_same_kind(24576 * MiB, 16384 * MiB));            // a 24 GB and a 16 GB card
    CHECK(!tp_same_kind(32768 * MiB, 32607 * MiB - 640 * MiB)); // two "32 GB" cards of other models (~2.4% apart)
    CHECK(!tp_same_kind(12288 * MiB, 11264 * MiB));            // 12 GB vs 11 GB
    CHECK(tp_same_kind(8192 * MiB - 300 * MiB, 8192 * MiB - 330 * MiB));
    CHECK(tp_same_kind(0, 0));
    std::puts("ds4_tp_kind_test: ok");
    return 0;
}
