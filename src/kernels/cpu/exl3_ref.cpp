// src/kernels/cpu/exl3_ref.cpp - the EXL3 format on the CPU, scalar (see include/strata/kernels/cpu/exl3_ref.hpp).
#include "strata/kernels/cpu/exl3_ref.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <vector>

namespace strata::kernels::cpu {

float exl3_decode(uint32_t state, int cb) {
    using strata::kernels::f16_from_f32;
    using strata::kernels::f32_from_f16;
    if (cb == 2) {
        // fp16(1024 + byte sum) * fp16(0x1eee) + fp16(0xc931) in one FP16 FMA: the F32 FMA is exact here (a 22-bit
        // product plus a bias of the same magnitude), so the single rounding to FP16 matches the device's
        const uint32_t x = state * 0x83DCD12Du;
        const float h = (float) (1024u + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24));
        return f32_from_f16(f16_from_f32(std::fma(h, f32_from_f16(0x1eee), f32_from_f16(0xc931))));
    }
    uint32_t x = cb == 1 ? state * 0xCBAC1FEDu : state * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    // two FP16 values within a factor of 8 of each other: their F32 sum is exact, rounded once
    return f32_from_f16(f16_from_f32(f32_from_f16((uint16_t) (x & 0xffffu)) + f32_from_f16((uint16_t) (x >> 16))));
}

void exl3_inner(const uint32_t* trellis, int ld, int k, int n, int K, int cb, float* w) {
    const int W = 8 * K;
    for (int ks = 0; ks < k / 16; ++ks)
        for (int tc = 0; tc < n / 16; ++tc) {
            const uint32_t* t = trellis + ((size_t) ks * ld + tc) * W;
            for (int p = 0; p < 256; ++p) {
                const int b1 = (p + 1 + 256) * K;
                const int i1 = (b1 - 1) >> 5, i0 = (b1 - 16) >> 5;
                const int sh = ((i1 + 1) << 5) - b1;
                const uint64_t m = ((uint64_t) t[i0 % W] << 32) | (uint64_t) t[i1 % W];
                const uint32_t st = (uint32_t) (m >> sh) & 0xffffu;
                const int lane = p >> 3, j = p & 7;
                const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = tc * 16 + (lane >> 2) + 8 * (j >> 2);
                w[(size_t) r * n + c] = exl3_decode(st, cb);
            }
        }
}

void exl3_had128(float* v, size_t n) {
    const float s = 1.0f / std::sqrt(128.0f);
    for (size_t b = 0; b < n; b += 128) {
        float* x = v + b;
        for (int h = 1; h < 128; h <<= 1)
            for (int i = 0; i < 128; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const float a = x[j], c = x[j + h];
                    x[j] = a + c;
                    x[j + h] = a - c;
                }
        for (int i = 0; i < 128; ++i) x[i] *= s;
    }
}

void exl3_mv(const float* w, const uint16_t* suh, const uint16_t* svh, int k, int n, const float* x, float* y) {
    using strata::kernels::f16_from_f32;
    using strata::kernels::f32_from_f16;
    std::vector<float> xh((size_t) k);
    for (int i = 0; i < k; ++i) xh[(size_t) i] = x[i] * f32_from_f16(suh[i]);
    exl3_had128(xh.data(), (size_t) k);
    for (int i = 0; i < k; ++i) xh[(size_t) i] = f32_from_f16(f16_from_f32(xh[(size_t) i]));
    std::vector<double> acc((size_t) n, 0.0);
    for (int r = 0; r < k; ++r) {
        const double xv = xh[(size_t) r];
        const float* row = w + (size_t) r * n;
        for (int c = 0; c < n; ++c) acc[(size_t) c] += xv * row[c];
    }
    for (int c = 0; c < n; ++c) y[c] = (float) acc[(size_t) c];
    exl3_had128(y, (size_t) n);
    for (int c = 0; c < n; ++c) y[c] *= f32_from_f16(svh[c]);
}

}  // namespace strata::kernels::cpu
