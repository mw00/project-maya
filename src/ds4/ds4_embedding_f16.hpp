#pragma once

#include <cstdint>

namespace ds4 {

// IEEE binary16 decoded to the bit pattern of its exactly equivalent float32.
inline uint32_t f16_to_f32_bits(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) return s;
        int ee = -1;
        uint32_t mm = m;
        do {
            ++ee;
            mm <<= 1;
        } while ((mm & 0x400) == 0);
        return s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
    }
    if (e == 31) return s | 0x7f800000u | m << 13;
    return s | (e - 15 + 127) << 23 | m << 13;
}

// Accept only finite F16 values exactly representable in BF16.
inline bool f16_to_exact_bf16(uint16_t h, uint16_t& out) {
    const uint32_t bits = f16_to_f32_bits(h);
    if ((bits & 0x7f800000u) == 0x7f800000u || (bits & 0xffffu) != 0) return false;
    out = (uint16_t) (bits >> 16);
    return true;
}

} // namespace ds4
