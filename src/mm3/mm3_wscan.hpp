// src/mm3/mm3_wscan.hpp - maya-mm3's look at a router that picked no expert (#672): host code only, no GPU.
//
//   finite_stats  how many of n F32 values are NaN / Inf, the finite ones' range and largest magnitude
//   scan_blocks   the first NaN / Inf a tensor holds as stored: F32 / F16 / BF16 values, a ggml block type's F16
//                 scales (d, dmin) - the bytes a damaged download changes into NaN without changing a file's size
//
// The engine runs them only once a request has already failed (mm3_model.cu's routes_ok), to say whether the NaN
// came with the model's files or arose on the GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace mm3::wscan {

struct Stats {
    size_t n = 0, nan = 0, inf = 0;
    float lo = std::numeric_limits<float>::infinity(), hi = -std::numeric_limits<float>::infinity();
    float amax = 0.0f;
    long long first_bad = -1;   // the first NaN / Inf's index (-1: none)
    size_t bad() const { return nan + inf; }
};

inline Stats finite_stats(const float* x, size_t n) {
    Stats s;
    s.n = n;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (std::isnan(v) || std::isinf(v)) {
            if (std::isnan(v)) ++s.nan;
            else ++s.inf;
            if (s.first_bad < 0) s.first_bad = (long long) i;
            continue;
        }
        s.lo = std::min(s.lo, v);
        s.hi = std::max(s.hi, v);
        s.amax = std::max(s.amax, std::fabs(v));
    }
    return s;
}

// an F16's / BF16's bits: NaN or Inf (the exponent all ones)
inline bool f16_bad(uint16_t h) { return (h & 0x7c00u) == 0x7c00u; }
inline bool bf16_bad(uint16_t h) { return (h & 0x7f80u) == 0x7f80u; }

// a stored type's block: its bytes, its values, the byte offsets of its F16 scales (ggml-common.h's block_* structs)
struct Block {
    size_t bytes = 0;
    int values = 0;
    int n_f16 = 0;
    int f16_at[2] = {0, 0};
};
inline bool block_of(const std::string& t, Block& b) {
    const auto set = [&](size_t bytes, int values, int n, int a0, int a1 = 0) {
        b.bytes = bytes;
        b.values = values;
        b.n_f16 = n;
        b.f16_at[0] = a0;
        b.f16_at[1] = a1;
        return true;
    };
    if (t == "Q8_0") return set(34, 32, 1, 0);
    if (t == "Q4_0") return set(18, 32, 1, 0);
    if (t == "Q2_K") return set(84, 256, 2, 80, 82);     // (scales, qs, d, dmin)
    if (t == "Q3_K") return set(110, 256, 1, 108);       // (hmask, qs, scales, d)
    if (t == "Q4_K") return set(144, 256, 2, 0, 2);      // (d, dmin, scales, qs)
    if (t == "Q5_K") return set(176, 256, 2, 0, 2);      // (d, dmin, scales, qh, qs)
    if (t == "Q6_K") return set(210, 256, 1, 208);       // (ql, qh, scales, d)
    if (t == "IQ1_S") return set(50, 256, 1, 0);
    if (t == "IQ2_XXS") return set(66, 256, 1, 0);
    if (t == "IQ2_XS") return set(74, 256, 1, 0);
    if (t == "IQ2_S") return set(82, 256, 1, 0);
    if (t == "IQ3_XXS") return set(98, 256, 1, 0);
    if (t == "IQ3_S") return set(110, 256, 1, 0);
    if (t == "IQ4_XS") return set(136, 256, 1, 0);
    return false;
}

struct Hit {
    bool found = false;
    bool known = true;    // false: a type this scan does not read (IQ1_M's scales, F8, U8): nothing said about it
    uint64_t at = 0;      // the bad value's byte offset in the tensor
    std::string what;     // "value", "d", "dmin"
};

// the first NaN / Inf in a tensor of `bytes` bytes of type `dtype` at p
inline Hit scan_blocks(const std::string& dtype, const uint8_t* p, uint64_t bytes) {
    Hit h;
    if (dtype == "F32") {
        for (uint64_t i = 0; i + 4 <= bytes; i += 4) {
            uint32_t u;
            std::memcpy(&u, p + i, 4);
            if ((u & 0x7f800000u) == 0x7f800000u) return {true, true, i, "value"};
        }
        return h;
    }
    if (dtype == "F16" || dtype == "BF16") {
        const bool bf = dtype == "BF16";
        for (uint64_t i = 0; i + 2 <= bytes; i += 2) {
            uint16_t u;
            std::memcpy(&u, p + i, 2);
            if (bf ? bf16_bad(u) : f16_bad(u)) return {true, true, i, "value"};
        }
        return h;
    }
    Block b;
    if (!block_of(dtype, b)) {
        h.known = false;
        return h;
    }
    for (uint64_t o = 0; o + b.bytes <= bytes; o += b.bytes)
        for (int k = 0; k < b.n_f16; ++k) {
            uint16_t u;
            std::memcpy(&u, p + o + (uint64_t) b.f16_at[k], 2);
            if (f16_bad(u)) return {true, true, o + (uint64_t) b.f16_at[k], b.n_f16 == 2 && k == 1 ? "dmin" : "d"};
        }
    return h;
}

}  // namespace mm3::wscan
