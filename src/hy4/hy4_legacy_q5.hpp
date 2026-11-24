#pragma once
#include <cstdint>
#include <cstring>
#include "ggml.h"
namespace hy4::compat {
// The original ggml Q5_0 / Q5_1 values, without requantization.
inline float q5_value(const uint8_t* block, int type, int j) {
    const int h = type == 6 ? 2 : 4;
    uint16_t dh;
    std::memcpy(&dh, block, 2);
    const float d = ggml_fp16_to_fp32(dh);
    uint32_t high;
    std::memcpy(&high, block + h, 4);
    const uint8_t* low = block + h + 4;
    const int q = (j < 16 ? low[j] & 15 : low[j - 16] >> 4) | (((high >> j) & 1u) << 4);
    if (type == 6)
        return d * (q - 16);
    uint16_t mh;
    std::memcpy(&mh, block + 2, 2);
    return d * q + ggml_fp16_to_fp32(mh);
}
} // namespace hy4::compat
