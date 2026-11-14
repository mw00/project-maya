// Vision tensors read into the tower's existing BF16 host buffers. GGUF gate/up
// stay separate on disk; F32/F16/Q8_0 are converted only while loading into RAM.
#pragma once

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ds4 {

template <class Pack>
bool vision_read_bf16(const Pack& pk, const std::string& name, std::vector<int64_t>& shape, void* dst,
                      std::string& err) {
    const auto* t = pk.get(name);
    if (t == nullptr) {
        const auto* a = pk.get(name + ".part.0");
        const auto* b = pk.get(name + ".part.1");
        if (a == nullptr || b == nullptr || a->shape.size() != 2 || a->shape != b->shape) {
            err = "vision: " + name + ": missing tensor or mismatched gate/up parts";
            return false;
        }
        std::vector<int64_t> sa, sb;
        const size_t n = (size_t) a->shape[0] * a->shape[1];
        if (!vision_read_bf16(pk, name + ".part.0", sa, dst, err) ||
            !vision_read_bf16(pk, name + ".part.1", sb, dst ? (uint16_t*) dst + n : nullptr, err)) return false;
        shape = {sa[0] + sb[0], sa[1]};
        return true;
    }
    shape = t->shape;
    size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) { err = "vision: " + name + ": invalid shape"; return false; }
        n *= (size_t) d;
    }
    const bool bf = t->dtype == "BF16", fp16 = t->dtype == "F16", fp32 = t->dtype == "F32", q8 = t->dtype == "Q8_0";
    const size_t bytes = q8 ? n / 32 * 34 : n * (fp32 ? 4 : 2);
    if (!(bf || fp16 || fp32 || q8) || t->bytes != bytes || (q8 && n % 32)) {
        err = "vision: " + name + ": invalid " + t->dtype + " tensor (BF16/F16/F32/Q8_0 expected)";
        return false;
    }
    if (dst == nullptr) return true;
    if (bf) {
        if (pk.read(*t, dst)) return true;
        err = "vision: " + name + ": read failed";
        return false;
    }
    std::vector<uint8_t> raw(bytes);
    if (!pk.read(*t, raw.data())) { err = "vision: " + name + ": read failed"; return false; }
    uint16_t* out = (uint16_t*) dst;
    for (size_t i = 0; i < n; ++i) {
        float value;
        if (q8) {
            const uint8_t* block = raw.data() + i / 32 * 34;
            const uint16_t h = (uint16_t) (block[0] | block[1] << 8);
            value = kernels::f32_from_f16(h) * (float) (int8_t) block[2 + i % 32];
        } else if (fp16) {
            uint16_t h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            value = kernels::f32_from_f16(h);
        } else {
            std::memcpy(&value, raw.data() + i * 4, 4);
        }
        out[i] = kernels::bf16_from_f32(value);
    }
    return true;
}
}  // namespace strata::ds4
