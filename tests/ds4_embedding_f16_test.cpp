#include "ds4_embedding_f16.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    unsigned accepted = 0, rejected = 0;
    for (uint32_t raw = 0; raw <= 0xffffu; ++raw) {
        const uint16_t h = (uint16_t) raw;
        const uint32_t f = ds4::f16_to_f32_bits(h);
        uint16_t b = 0;
        const bool ok = ds4::f16_to_exact_bf16(h, b);
        const bool finite = (f & 0x7f800000u) != 0x7f800000u;
        const bool exact = finite && (f & 0xffffu) == 0;
        if (ok != exact || (ok && ((uint32_t) b << 16) != f)) {
            std::fprintf(stderr, "F16 %04x: accepted=%d expected=%d BF16=%04x float=%08x\n",
                         (unsigned) h, ok, exact, (unsigned) b, f);
            return 1;
        }
        ok ? ++accepted : ++rejected;
    }
    // Known values: signed zero and 1.5 are exact; the next binary16 after 1.0 is not.
    uint16_t b = 0;
    if (!ds4::f16_to_exact_bf16(0x0000, b) || b != 0x0000 ||
        !ds4::f16_to_exact_bf16(0x8000, b) || b != 0x8000 ||
        !ds4::f16_to_exact_bf16(0x3e00, b) || b != 0x3fc0 ||
        ds4::f16_to_exact_bf16(0x3c01, b) ||
        ds4::f16_to_exact_bf16(0x7c00, b) || ds4::f16_to_exact_bf16(0x7e00, b)) {
        std::fprintf(stderr, "known F16/BF16 boundary failed\n");
        return 1;
    }
    std::printf("OK: all 65536 F16 bit patterns checked (%u exact finite, %u rejected)\n", accepted, rejected);
    return 0;
}
