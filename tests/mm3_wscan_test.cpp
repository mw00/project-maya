// tests/mm3_wscan_test.cpp - maya-mm3's look at a router that picked no expert (mm3_wscan.hpp, #672): the router's
// input / logits statistics and the scan of the model's stored weights for NaN / Inf (host code only, no model, no GPU)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "mm3_wscan.hpp"

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace ws = mm3::wscan;

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    // statistics: NaN and Inf counted apart, the finite range without them, the first bad index
    {
        const float x[6] = {1.0f, -3.0f, nan, 2.5f, -inf, 0.0f};
        const ws::Stats s = ws::finite_stats(x, 6);
        CHECK(s.n == 6 && s.nan == 1 && s.inf == 1 && s.bad() == 2 && s.first_bad == 2);
        CHECK(s.lo == -3.0f && s.hi == 2.5f && s.amax == 3.0f);
        const float y[3] = {nan, nan, nan};
        CHECK(ws::finite_stats(y, 3).bad() == 3 && ws::finite_stats(y, 3).first_bad == 0);
        CHECK(ws::finite_stats(x, 2).bad() == 0 && ws::finite_stats(x, 2).first_bad == -1);
    }
    // F16 / BF16 bits: exponent all ones is NaN / Inf, the largest finite values are not
    CHECK(ws::f16_bad(0x7c00) && ws::f16_bad(0xfc00) && ws::f16_bad(0x7e00) && !ws::f16_bad(0x7bff) && !ws::f16_bad(0));
    CHECK(ws::bf16_bad(0x7f80) && ws::bf16_bad(0x7fc0) && !ws::bf16_bad(0x7f7f) && !ws::bf16_bad(0x3f80));
    // F32 tensor: clean, then one NaN at value 5
    {
        std::vector<float> v(64, 0.5f);
        CHECK(!ws::scan_blocks("F32", (const uint8_t*) v.data(), v.size() * 4).found);
        v[5] = nan;
        const ws::Hit h = ws::scan_blocks("F32", (const uint8_t*) v.data(), v.size() * 4);
        CHECK(h.found && h.known && h.at == 20);
    }
    // block types: random-looking quant bytes never count, only the F16 scales do (d / dmin at their ggml offsets)
    struct Case {
        const char* t;
        size_t bytes;
        int d_at, dmin_at;   // (dmin -1: none)
    };
    const Case cases[] = {{"Q8_0", 34, 0, -1},     {"Q2_K", 84, 80, 82},     {"Q3_K", 110, 108, -1},
                          {"Q4_K", 144, 0, 2},     {"Q5_K", 176, 0, 2},      {"Q6_K", 210, 208, -1},
                          {"IQ1_S", 50, 0, -1},    {"IQ2_XXS", 66, 0, -1},   {"IQ2_XS", 74, 0, -1},
                          {"IQ2_S", 82, 0, -1},    {"IQ3_XXS", 98, 0, -1},   {"IQ3_S", 110, 0, -1},
                          {"IQ4_XS", 136, 0, -1}};
    for (const Case& c : cases) {
        ws::Block b;
        CHECK(ws::block_of(c.t, b) && b.bytes == c.bytes);
        const int nb = 7;
        std::vector<uint8_t> t(c.bytes * nb, 0xff);   // (0xff everywhere: every quant byte at its most)
        const auto put = [&](int blk, int at, uint16_t h) { std::memcpy(t.data() + (size_t) blk * c.bytes + at, &h, 2); };
        for (int i = 0; i < nb; ++i) {
            put(i, c.d_at, 0x3c00);   // (1.0)
            if (c.dmin_at >= 0) put(i, c.dmin_at, 0x3800);
        }
        CHECK(!ws::scan_blocks(c.t, t.data(), t.size()).found);
        put(4, c.d_at, 0x7e00);   // (block 4's d: NaN)
        ws::Hit h = ws::scan_blocks(c.t, t.data(), t.size());
        CHECK(h.found && h.at == 4 * c.bytes + (size_t) c.d_at && h.what == "d");
        put(4, c.d_at, 0x3c00);
        if (c.dmin_at >= 0) {
            put(2, c.dmin_at, 0xfc00);   // (block 2's dmin: -Inf)
            h = ws::scan_blocks(c.t, t.data(), t.size());
            CHECK(h.found && h.at == 2 * c.bytes + (size_t) c.dmin_at && h.what == "dmin");
        }
    }
    // a type it does not read: said so, never a finding
    {
        const uint8_t z[64] = {0};
        const ws::Hit h = ws::scan_blocks("IQ1_M", z, sizeof z);
        CHECK(!h.found && !h.known);
    }
    std::printf("mm3_wscan_test: ok\n");
    return 0;
}
