// Engram row layouts shared by the asynchronous reader and the device decoders.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strata::ds4 {

enum class EngramFormat : uint8_t { Fp8, Q6K, Iq4Nl, SplitFp8 };

/// Select a supported Engram row format, rejecting missing scales and partial blocks.
inline bool engram_row_format(std::string_view dtype, bool has_scales, int width, EngramFormat& format) {
    if (width <= 0 || width % 32 != 0) return false;
    if (dtype == "MXFP8") format = EngramFormat::Fp8;
    else if (dtype == "Q6_K" && width % 256 == 0) format = EngramFormat::Q6K;
    else if (dtype == "IQ4_NL") format = EngramFormat::Iq4Nl;
    else if (dtype == "F8_E4M3" && has_scales) format = EngramFormat::SplitFp8;
    else return false;
    return true;
}

/// Bytes read from the weight file for one Engram row; split FP8 scales are separate.
inline size_t engram_source_row_bytes(EngramFormat format, int width) {
    switch (format) {
        case EngramFormat::Fp8: return (size_t) width / 32 * 33;
        case EngramFormat::Q6K: return (size_t) width / 256 * 210;
        case EngramFormat::Iq4Nl: return (size_t) width / 32 * 18;
        case EngramFormat::SplitFp8: return (size_t) width;
    }
    return 0;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_ENGRAM_HD __host__ __device__
#else
#define DS4_ENGRAM_HD
#endif

/// Decode one IQ4_NL block value using its FP16 scale converted by the caller.
DS4_ENGRAM_HD inline float engram_iq4nl_value(const uint8_t* block, int index, float scale) {
    // ggml's IQ4_NL codebook and nibble order: low halves first, high halves second.
    constexpr int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const int code = (block[2 + index % 16] >> (index < 16 ? 0 : 4)) & 15;
    return scale * (float) values[code];
}

#undef DS4_ENGRAM_HD

}  // namespace strata::ds4
