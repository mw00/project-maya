// src/mm3/mm3_gguf_host.hpp - a GGUF tensor's rows on the host for maya-mm3 (the token embedding, the indexer's Q8_0
// projections): ggml's dequantize_row_* values for Q4_K, Q5_K, Q6_K and Q8_0 (ggml-quants.c's loops, the same F32
// operations), and F32 / F16 / BF16 as they are.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace mm3::gg {

/// F16 bits -> F32, exactly (subnormals too).
inline float host_f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) mm <<= 1, ++sh;
            bits = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

namespace detail {
inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {   // ggml's get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
// one block of 256 values
inline void q4k(const uint8_t* b, float* y) {
    const float d = host_f16((uint16_t) (b[0] | b[1] << 8)), mn = host_f16((uint16_t) (b[2] | b[3] << 8));
    const uint8_t *sc = b + 4, *q = b + 16;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
        int s, m;
        scale_min_k4(is, sc, s, m);
        const float d1 = d * (float) s, m1 = mn * (float) m;
        scale_min_k4(is + 1, sc, s, m);
        const float d2 = d * (float) s, m2 = mn * (float) m;
        for (int l = 0; l < 32; ++l) {
            const float a = d1 * (float) (q[l] & 0xF);
            *y++ = a - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const float a = d2 * (float) (q[l] >> 4);
            *y++ = a - m2;
        }
    }
}
inline void q5k(const uint8_t* b, float* y) {
    const float d = host_f16((uint16_t) (b[0] | b[1] << 8)), mn = host_f16((uint16_t) (b[2] | b[3] << 8));
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    unsigned u1 = 1, u2 = 2;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
        int s, m;
        scale_min_k4(is, sc, s, m);
        const float d1 = d * (float) s, m1 = mn * (float) m;
        scale_min_k4(is + 1, sc, s, m);
        const float d2 = d * (float) s, m2 = mn * (float) m;
        for (int l = 0; l < 32; ++l) {
            const float a = d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0));
            *y++ = a - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const float a = d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0));
            *y++ = a - m2;
        }
    }
}
inline void q6k(const uint8_t* b, float* y) {
    const float d = host_f16((uint16_t) (b[208] | b[209] << 8));
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t* sc = (const int8_t*) (b + 192);
    for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * (float) sc[is + 0] * (float) q1;
            y[l + 32] = d * (float) sc[is + 2] * (float) q2;
            y[l + 64] = d * (float) sc[is + 4] * (float) q3;
            y[l + 96] = d * (float) sc[is + 6] * (float) q4;
        }
}
}  // namespace detail

/// The bytes of a row of k values of a dtype this file reads (0: none of them, or k not whole blocks).
inline size_t host_row_bytes(const std::string& dt, int64_t k) {
    if (dt == "F32") return (size_t) k * 4;
    if (dt == "F16" || dt == "BF16") return (size_t) k * 2;
    if (dt == "Q8_0") return k % 32 ? 0 : (size_t) k / 32 * 34;
    if (dt == "Q4_K") return k % 256 ? 0 : (size_t) k / 256 * 144;
    if (dt == "Q5_K") return k % 256 ? 0 : (size_t) k / 256 * 176;
    if (dt == "Q6_K") return k % 256 ? 0 : (size_t) k / 256 * 210;
    return 0;
}

/// Row r of a [rows][k] tensor of dtype dt (data: its first byte) as F32 into y - false: a dtype not read here.
inline bool host_row(const std::string& dt, const uint8_t* data, int64_t k, int64_t r, float* y) {
    const size_t rb = host_row_bytes(dt, k);
    if (rb == 0) return false;
    const uint8_t* p = data + (size_t) r * rb;
    if (dt == "F32") {
        std::memcpy(y, p, (size_t) k * 4);
    } else if (dt == "F16" || dt == "BF16") {
        for (int64_t i = 0; i < k; ++i) {
            const uint16_t b = (uint16_t) (p[2 * i] | p[2 * i + 1] << 8);
            if (dt == "F16") {
                y[i] = host_f16(b);
            } else {
                const uint32_t u = (uint32_t) b << 16;
                std::memcpy(&y[i], &u, 4);
            }
        }
    } else if (dt == "Q8_0") {
        for (int64_t blk = 0; blk < k / 32; ++blk) {
            const uint8_t* b = p + blk * 34;
            const float d = host_f16((uint16_t) (b[0] | b[1] << 8));
            for (int j = 0; j < 32; ++j) y[blk * 32 + j] = (float) (int8_t) b[2 + j] * d;
        }
    } else {
        const size_t bb = rb / (size_t) (k / 256);
        for (int64_t blk = 0; blk < k / 256; ++blk) {
            const uint8_t* b = p + (size_t) blk * bb;
            if (dt == "Q4_K") detail::q4k(b, y + blk * 256);
            else if (dt == "Q5_K") detail::q5k(b, y + blk * 256);
            else detail::q6k(b, y + blk * 256);
        }
    }
    return true;
}

}  // namespace mm3::gg
