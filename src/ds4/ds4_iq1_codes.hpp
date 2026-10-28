// src/ds4/ds4_iq1_codes.hpp - IQ1_S / IQ1_M's grid values as the int8 codes ds4_iq.cu's kernels dot, host and device
// alike (tests/ds4_iq_host_test.cpp builds IQ1_M's code table with these and checks it against ggml's own decode).
//
// iq1s_grid_gpu (ggml-common.h) holds a grid entry's 8 values as nibbles n = g + 1 (g in {-1, 0, 1}): values 0..3 in
// the low nibbles of its 4 bytes, 4..7 in the high ones.  A value's weight is d s (g +- 1/8), the sign that of its
// group's delta (IQ1S_DELTA): as the code 8 g +- 1 the scale is d s / 8, so a sub-block's sum stays an exact integer
// dot.  The 11-bit grid index and the delta's sign of a group are 12 bits of an IQ1_M block side by side (a qs byte,
// then a qh nibble: 3 index bits and the sign), so IQ1_M's codes can come from one table of 4096 entries - the grid
// lookup and the delta's bias in one load (ds4_iq.cu's persistent kernels: the table built once a block).
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_IQ1_HD __host__ __device__ __forceinline__
#else
#define DS4_IQ1_HD inline
#endif

namespace strata::ds4::gguf::iq {

// 4 nibbles times 8 (bits 3..6 of each byte: 0, 8 or 16) -> 4 codes, a byte each: bias 0x79 (delta +1/8: 8 n - 7) or
// 0x77 (-1/8: 8 n - 9) a byte, its top bit flipped back after - each byte's sum stays below 0x100, so nothing carries
// into the next
DS4_IQ1_HD uint32_t iq1_bias(uint32_t neg) { return neg ? 0x77777777u : 0x79797979u; }
DS4_IQ1_HD int iq1c(uint32_t n8, uint32_t bias) { return (int) (((n8 & 0x78787878u) + bias) ^ 0x80808080u); }

// IQ1_M's code table: entry (neg << 11) | grid index = the grid entry's values 0..3 (lo) and 4..7 (hi) as codes, the
// delta's sign neg (two words an entry: 32 KiB)
constexpr int kIq1mCodes = 4096;
DS4_IQ1_HD void iq1m_codes(uint32_t g, uint32_t neg, uint32_t& lo, uint32_t& hi) {
    const uint32_t b = iq1_bias(neg);
    lo = (uint32_t) iq1c(g << 3, b);
    hi = (uint32_t) iq1c(g >> 1, b);
}
// the table's entry for group l (0..3) of an IQ1_M sub-block: byte l of its qs word q4, and bits 4 l .. 4 l + 3 of its
// two qh bytes h (3 index bits, then the delta's sign) - the 12 bits as they stand
DS4_IQ1_HD uint32_t iq1m_entry(uint32_t q4, uint32_t h, int l) {
    return ((q4 >> (8 * l)) & 255u) | (((h >> (4 * l)) & 15u) << 8);
}
// IQ1_M's F16 block scale (its bits): the top nibbles of its 4 scale words (s01: words 0, 1; s23: words 2, 3)
DS4_IQ1_HD uint16_t iq1m_d16(uint32_t s01, uint32_t s23) {
    return (uint16_t) (((s01 >> 12) & 0xFu) | ((s01 >> 24) & 0xF0u) | ((s23 >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u));
}
// sub-block ib's two 3-bit scales as 2 s + 1: s0 for its values 0..15, s1 for 16..31
DS4_IQ1_HD void iq1m_scales(uint32_t s01, uint32_t s23, int ib, int& s0, int& s1) {
    const uint32_t w = ((ib >> 1) < 2 ? s01 : s23) >> (16 * ((ib >> 1) & 1) + 6 * (ib & 1));
    s0 = 2 * (int) (w & 7u) + 1;
    s1 = 2 * (int) ((w >> 3) & 7u) + 1;
}

}  // namespace strata::ds4::gguf::iq
