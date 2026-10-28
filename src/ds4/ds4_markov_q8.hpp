// src/ds4/ds4_markov_q8.hpp - the DSpark drafter's markov head as ggml's Q8_0 blocks (DS4_MARKOV_Q8, opt-in): its rows
// of R values (R % 32 == 0) read from the pack's F16 / BF16 / F32 bits and rounded as they load, 32 values a block of
// 34 bytes - the F16 scale d = max|x| / 127, then the 32 int8 round(x * (1 / d)), halves away from zero - in the same
// F32 steps as ggml's quantize_row_q8_0_ref, so the same bytes (tests/ds4_markov_q8_test.cpp checks them against it).
// The pack is not touched.  A row's blocks depend on that row alone: a head group member's rows (markov_share_q8) are
// the whole head's rows (markov_add_q8), byte for byte.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::ds4 {

constexpr int kMarkovQ8Bytes = 34;   // a block: the F16 scale, 32 int8
enum MarkovSrc { kMarkovF16 = 0, kMarkovBF16 = 1, kMarkovF32 = 2 };   // the pack's tensor's type

inline size_t markov_q8_row_bytes(int R) { return (size_t) (R / 32) * kMarkovQ8Bytes; }

/// 32 values into a block (quantize_row_q8_0_ref's steps)
inline void markov_q8_block(const float* x, uint8_t* out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float a = std::fabs(x[j]);
        amax = amax > a ? amax : a;   // (ggml's MAX(amax, fabsf(v)))
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t dh = strata::kernels::f16_from_f32(d);   // (round to nearest even, as GGML_FP32_TO_FP16)
    std::memcpy(out, &dh, 2);
    for (int j = 0; j < 32; ++j) out[2 + j] = (uint8_t) (int8_t) std::round(x[j] * id);
}

/// n rows of R values (src: the pack's bits of those rows, fmt a MarkovSrc) into out, n markov_q8_row_bytes(R) bytes.
inline void markov_q8_rows(const void* src, int fmt, size_t n, int R, uint8_t* out) {
    const size_t nb = n * (size_t) (R / 32);
    float x[32];
    for (size_t b = 0; b < nb; ++b) {
        for (int j = 0; j < 32; ++j) {
            const size_t i = b * 32 + (size_t) j;
            if (fmt == kMarkovF32) {
                std::memcpy(&x[j], (const uint8_t*) src + i * 4, 4);
            } else {
                uint16_t h;
                std::memcpy(&h, (const uint8_t*) src + i * 2, 2);
                if (fmt == kMarkovBF16) {
                    const uint32_t u = (uint32_t) h << 16;
                    std::memcpy(&x[j], &u, 4);
                } else
                    x[j] = strata::kernels::f32_from_f16(h);
            }
        }
        markov_q8_block(x, out + b * kMarkovQ8Bytes);
    }
}

/// The value of entry i (0 .. R - 1) of a Q8_0 row (tests): its block's scale times its int8.
inline float markov_q8_value(const uint8_t* row, int i) {
    const uint8_t* b = row + (size_t) (i / 32) * kMarkovQ8Bytes;
    uint16_t dh;
    std::memcpy(&dh, b, 2);
    return strata::kernels::f32_from_f16(dh) * (float) (int8_t) b[2 + i % 32];
}

}  // namespace strata::ds4
