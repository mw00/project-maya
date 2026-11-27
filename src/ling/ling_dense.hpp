// src/ling/ling_dense.hpp - full-precision dense matrices (F32, F16, BF16) for maya-ling's loader.
//
// maya-ling's matrix-vector kernels read ggml's block types (Q8_0, Q6_K, the i-quants).  A GGUF that keeps the output
// head, ssm_beta, attn_gate, eh_proj (or any dense matrix) in F32 / F16 / BF16 is turned into Q8_0 rows while it is
// loaded - ggml's own quantize_row_q8_0_ref (a 32-value block: d = max|x| / 127 kept as F16, each value
// round(x / d) as int8), the format the kernels already read for a Q8_0 file.  The result is the Q8_0 the same
// weights give in llama.cpp; matrices a file already holds in a block type are not touched by any of this.
// F16 -> F32 is exact and is used for the router (which the kernels read in F32 or BF16).
//
// Host code only, no CUDA: tests/ling_unit_test.cpp runs it.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ling::dense {

constexpr int kF32 = 0, kF16 = 1, kQ8_0 = 8, kBF16 = 30;
constexpr size_t kQ8Block = 32, kQ8Bytes = 34;

inline bool is_float_type(int t) { return t == kF32 || t == kF16 || t == kBF16; }
inline size_t element_bytes(int t) { return t == kF32 ? 4 : 2; }

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                       // subnormal: normalise
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3ffu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_f16(float f) {   // round to nearest even, as ggml's GGML_FP32_TO_FP16
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t) (sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));   // inf / nan
    if (a >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);                                     // rounds past 65504
    if (a < 0x38800000u) {                                                                         // subnormal or zero
        if (a < 0x33000000u) return (uint16_t) sign;
        const uint32_t m = (a & 0x7fffffu) | 0x800000u;
        const int sh = 126 - (int) (a >> 23);                                                      // 14 .. 24
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t) (sign | r);
    }
    uint32_t r = ((a >> 13) - (112u << 10));                  // exponent rebias, 10 mantissa bits
    const uint32_t rem = a & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return (uint16_t) (sign | r);
}

inline float bf16_to_f32(uint16_t b) {
    const uint32_t bits = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// One value of a float-typed row.
inline float load_value(int type, const uint8_t* p, size_t i) {
    if (type == kF32) {
        float f;
        std::memcpy(&f, p + i * 4, 4);
        return f;
    }
    uint16_t h;
    std::memcpy(&h, p + i * 2, 2);
    return type == kF16 ? f16_to_f32(h) : bf16_to_f32(h);
}

/// Bytes of `rows` rows of k values as Q8_0 (k a multiple of 32).
inline size_t q8_0_bytes(uint64_t rows, uint64_t k) { return (size_t) (rows * (k / kQ8Block) * kQ8Bytes); }

/// ggml's quantize_row_q8_0_ref over rows * k values of a float-typed tensor (F32, F16 or BF16, rows k apart) into
/// dst (q8_0_bytes(rows, k) bytes).  Throws when the type is not a float one, k is not a multiple of 32, or a value is
/// not finite (a block scale of inf / nan would be silent garbage).
inline void to_q8_0(int type, const uint8_t* src, uint64_t rows, uint64_t k, uint8_t* dst, const std::string& what = "") {
    if (!is_float_type(type)) throw std::runtime_error("GGUF: " + what + ": not a full-precision type");
    if (k == 0 || k % kQ8Block != 0)
        throw std::runtime_error("GGUF: " + what + ": a full-precision matrix needs rows of a multiple of 32 values");
    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t* row = src + (size_t) (r * k) * element_bytes(type);
        uint8_t* out = dst + (size_t) (r * (k / kQ8Block)) * kQ8Bytes;
        for (uint64_t b = 0; b < k / kQ8Block; ++b, out += kQ8Bytes) {
            float v[kQ8Block], amax = 0.0f;
            for (size_t j = 0; j < kQ8Block; ++j) {
                v[j] = load_value(type, row, (size_t) (b * kQ8Block + j));
                if (!std::isfinite(v[j])) throw std::runtime_error("GGUF: " + what + ": a value is not finite");
                amax = std::fmax(amax, std::fabs(v[j]));
            }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            const uint16_t dh = f32_to_f16(d);
            std::memcpy(out, &dh, 2);
            for (size_t j = 0; j < kQ8Block; ++j) out[2 + j] = (uint8_t) (int8_t) std::lround(v[j] * id);
        }
    }
}

/// The same as a vector (all of `elements` values, rows of k).
inline std::vector<uint8_t> to_q8_0(int type, const uint8_t* src, uint64_t elements, uint64_t k, const std::string& what) {
    if (k == 0 || elements % k != 0) throw std::runtime_error("GGUF: " + what + ": ragged rows");
    std::vector<uint8_t> out(q8_0_bytes(elements / k, k));
    to_q8_0(type, src, elements / k, k, out.data(), what);
    return out;
}

/// F16 (or F32, BF16) values as F32, exactly.
inline std::vector<float> to_f32(int type, const uint8_t* src, uint64_t elements) {
    std::vector<float> out((size_t) elements);
    for (uint64_t i = 0; i < elements; ++i) out[(size_t) i] = load_value(type, src, (size_t) i);
    return out;
}

}  // namespace ling::dense
